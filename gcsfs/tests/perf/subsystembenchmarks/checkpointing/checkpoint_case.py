import contextlib
import logging
import os
import time

import fsspec
import psutil

from gcsfs.tests.perf.subsystembenchmarks._common import nodes
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.driver import CheckpointDriver
from gcsfs.tests.perf.subsystembenchmarks.dataloading import rcu
from gcsfs.tests.perf.subsystembenchmarks.dataloading.bucket import bucket_name_of
from gcsfs.tests.perf.subsystembenchmarks.dataloading.driver import assert_fsspec_gcsfs


def _terminate_process_tree(
    pid: int, grace_s: float = 10.0, *, reap_root: bool = True
) -> None:
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for proc in children + [parent]:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass
    procs = children + ([parent] if reap_root else [])
    _, alive = psutil.wait_procs(procs, timeout=grace_s)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=grace_s)


def _signal_peer_failure(ch, scope: str, exc: BaseException) -> None:
    if ch is not None and not isinstance(exc, nodes.PeerFailed):
        with contextlib.suppress(Exception):
            ch.fail(scope, repr(exc))


def _run_setup_in_phase(driver: CheckpointDriver, prefix: str, params) -> None:
    os.environ[nodes.PHASE_ENV] = "setup"
    try:
        driver.setup(prefix, params)
    except BaseException as exc:
        _signal_peer_failure(nodes.channel(), params.name, exc)
        raise


def _run_setup_process(driver: CheckpointDriver, prefix: str, params, ch) -> None:
    # We run the setup phase (which generates the checkpoint) in an isolated OS process.
    # Checkpoint generation uses ~400 GB of RAM. PyTorch's allocators and the gloo
    # multiprocessing backend leak memory and semaphores back to the parent process.
    # If we ran setup() natively, the leaked memory combined with the ~400 GB needed
    # for the subsequent read benchmark (driver.run) would exceed the VM's 732 GB limit
    # and cause an OOM crash. The OS forcefully reclaims all memory when this process joins.
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    if ch is not None:
        p = ctx.Process(target=_run_setup_in_phase, args=(driver, prefix, params))
    else:
        p = ctx.Process(target=driver.setup, args=(prefix, params))
    p.start()
    try:
        if ch is not None:
            while p.is_alive():
                p.join(timeout=1.0)
                if p.is_alive():
                    ch.check_failed(params.name)
        else:
            p.join()
    finally:
        if p.is_alive():
            _terminate_process_tree(p.pid, reap_root=False)
            p.join(timeout=10.0)
            if p.is_alive():
                p.kill()
                p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"driver.setup failed with exitcode {p.exitcode}")


def _run_case_body(
    benchmark, monitor, params, driver: CheckpointDriver, bucket_name: str, ch
) -> None:
    from gcsfs.tests.perf.subsystembenchmarks._common.benchmark_publish import (
        publish_case_metadata,
        publish_resource_metrics,
        publish_round_stats,
    )

    is_leader = ch is None or ch.group.is_leader
    params.bucket_name = bucket_name
    prefix = f"gs://{bucket_name}/checkpoint/"
    assert_fsspec_gcsfs(prefix)

    _run_setup_process(driver, prefix, params, ch)

    if is_leader and "read" in params.scenario:
        from gcsfs.tests.perf.subsystembenchmarks.dataloading import rapid_cache

        rcu.assert_storage_class(prefix, params.bucket_type)
        rapid_cache.warm_if_needed(prefix, params.bucket_type)
        rcu.warm_if_needed(prefix, params.bucket_type)

    result = None
    try:
        window_start = time.time()
        with monitor() if is_leader else contextlib.nullcontext() as m:
            result = driver.run(prefix, params)
        window_end = time.time()
        if getattr(result, "verify", None) is not None:
            result.verify()
    finally:
        if result is not None and getattr(result, "close", None) is not None:
            result.close()

    if not is_leader:
        return

    if "write" in params.scenario:
        # Saves run on *_rcu_z only, where run.py turns on RAPID writes.
        rcu.assert_storage_class(prefix, params.bucket_type)

    # Measure actual GCS files
    physical_size_bytes = 0
    try:
        fs, fs_path = fsspec.core.url_to_fs(prefix)
        logging.info("Listing GCS checkpoint prefix %s:", prefix)
        all_files = fs.find(fs_path, detail=True)
        for path, info in all_files.items():
            logging.info("  File: %s, Size: %d bytes", path, info["size"])

        # Sum up the checkpoint size
        checkpoint_files = {
            p: info for p, info in all_files.items() if driver.is_checkpoint_file(p)
        }
        physical_size_bytes = sum(info["size"] for info in checkpoint_files.values())
        logging.info("Physical checkpoint size: %d bytes", physical_size_bytes)
    except Exception as e:
        logging.error("Could not measure physical checkpoint size: %s", e)
        raise

    # Publish metrics to pytest-benchmark extra_info
    publish_case_metadata(benchmark, params, window_start, window_end)
    benchmark.extra_info.update(
        {
            "workload_family": "checkpointing",
            "checkpoint_physical_size_bytes": physical_size_bytes,
            "checkpoint_strategy": params.strategy,
            "model_id": params.model_id,
        }
    )
    benchmark.extra_info.update(params.extra_columns())
    if result.extra_columns:
        benchmark.extra_info.update(result.extra_columns)
        if "dcp_read_bytes_fetched_total" in result.extra_columns:
            benchmark.extra_info["dcp_read_amplification_planned"] = (
                result.extra_columns["dcp_read_bytes_fetched_total"]
                / physical_size_bytes
                if physical_size_bytes > 0
                else 0.0
            )

    durations = result.durations
    bytes_transferred = getattr(result, "bytes_transferred", None)
    if bytes_transferred is not None:
        size_for_throughput = bytes_transferred
    elif "read" in params.scenario:
        size_for_throughput = physical_size_bytes * driver.read_count(params)
    else:
        size_for_throughput = physical_size_bytes

    throughput = (
        sum(size_for_throughput / d for d in durations) / len(durations)
        if size_for_throughput > 0 and durations and all(durations)
        else 0.0
    )
    if "read" in params.scenario:
        benchmark.extra_info["checkpoint_read_throughput_mean_bytes_per_second"] = (
            throughput
        )
    elif "write" in params.scenario:
        benchmark.extra_info["checkpoint_write_throughput_mean_bytes_per_second"] = (
            throughput
        )
    else:
        raise ValueError(f"Unknown scenario {params.scenario}")
    publish_round_stats(benchmark, durations)
    publish_resource_metrics(benchmark, m)
    benchmark.pedantic(lambda: None, rounds=1, iterations=1, warmup_rounds=0)


def run_checkpoint_case(
    benchmark, monitor, params, driver: CheckpointDriver, *, bucket_ctx=None
):
    """Full per-case lifecycle for CheckpointDriver."""
    from gcsfs.tests.perf.subsystembenchmarks.dataloading.bucket import (
        BucketSpec,
        case_bucket,
    )

    ch = nodes.channel()
    if ch is None or ch.group.is_leader:
        spec = BucketSpec.from_env() if bucket_ctx is None else None
        ctx_factory = bucket_ctx or case_bucket
    else:
        spec = None
        ctx_factory = None

    if ch is None:
        with ctx_factory(spec, params.name) as bucket:
            bucket_name = bucket_name_of(bucket) or bucket
            _run_case_body(benchmark, monitor, params, driver, bucket_name, None)
        return

    try:
        if ch.group.is_leader:
            with ctx_factory(spec, params.name) as bucket:
                bucket_name = bucket_name_of(bucket) or bucket
                ch.publish(f"{params.name}/bucket", bucket_name)
                try:
                    _run_case_body(benchmark, monitor, params, driver, bucket_name, ch)
                except BaseException as exc:
                    _signal_peer_failure(ch, params.name, exc)
                    raise
                else:
                    ch.barrier(f"{params.name}/end")
        else:
            bucket_name = ch.get(f"{params.name}/bucket")
            _run_case_body(benchmark, monitor, params, driver, bucket_name, ch)
            ch.barrier(f"{params.name}/end")
    except BaseException as exc:
        _signal_peer_failure(ch, params.name, exc)
        raise
