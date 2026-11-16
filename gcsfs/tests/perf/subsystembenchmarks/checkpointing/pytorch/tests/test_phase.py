import contextlib
import dataclasses
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
import time

import fsspec
import pytest
import torch.distributed as dist

pytest.importorskip("transformers")

from gcsfs.tests.perf.subsystembenchmarks.checkpointing._dist import find_free_port
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.hosts import (
    HostConfig,
    local_ranks,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.phase import (
    RankContext,
    _find_phase_port,
    get_store,
    phase_durations,
    run_phase,
    shutdown_store,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.read.driver import (
    PyTorchCheckpointReadDriver,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.tests.test_parallelize import (
    _create_tiny_llama_config_dir,
    _params,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.write.driver import (
    PyTorchCheckpointWriteDriver,
)

_THIS_MODULE = (
    "gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.tests.test_phase"
)


def _rank_noop(ctx: RankContext, prefix: str, params) -> None:
    ctx.report({"durations": [(1.0, 2.0)], "rank": ctx.rank})


def _rank_report_then_wait(ctx: RankContext, prefix: str, params) -> None:
    ctx.report({"durations": [(0.0, 1.0)]})
    # Stay in the phase after reporting so the test can kill the worker mid-phase.
    time.sleep(60.0)
    dist.barrier()


def _rank_fail_after_report_on_node1(
    ctx: RankContext, prefix: str, params
) -> None:
    ctx.report({"durations": [(0.0, 1.0)]})
    if ctx.rank == 5:
        raise RuntimeError("post-report verification failed")


@contextlib.contextmanager
def _two_host_cluster(
    monkeypatch, pg_timeout_s: int = 30, precreate_store: bool = True
):
    master_port = find_free_port()
    env_vars = {
        "GCSFS_SUBSYSTEM_DEVICE": "cpu",
        "GCSFS_SUBSYSTEM_NNODES": "2",
        "GCSFS_SUBSYSTEM_NODE_RANK": "0",
        "GCSFS_SUBSYSTEM_MASTER_ADDR": "localhost",
        "GCSFS_SUBSYSTEM_MASTER_PORT": str(master_port),
        "GCSFS_SUBSYSTEM_NPROC_PER_NODE": "4",
        "GCSFS_SUBSYSTEM_PG_TIMEOUT_S": str(pg_timeout_s),
    }
    for k, v in env_vars.items():
        monkeypatch.setenv(k, v)

    leader_host = HostConfig.from_env()
    store = get_store(leader_host) if precreate_store else None

    worker_env = os.environ.copy()
    worker_env["GCSFS_SUBSYSTEM_NODE_RANK"] = "1"
    worker_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.worker",
        ],
        env=worker_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        yield leader_host, store, worker_proc
    finally:
        try:
            shutdown_store(leader_host)
        except Exception:
            pass
        if worker_proc.poll() is None:
            try:
                worker_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker_proc.kill()
                worker_proc.wait(timeout=5)


def test_two_host_fsdp8_load(tmp_path, monkeypatch):
    config_dir = _create_tiny_llama_config_dir(tmp_path, num_layers=8)
    params = _params(config_dir, strategy="fsdp", dp=8, world_size=8, rounds=2)
    prefix = (tmp_path / "ckpt").as_uri()

    with _two_host_cluster(monkeypatch) as (host, store, worker_proc):
        driver = PyTorchCheckpointReadDriver()
        driver.setup(prefix, params)
        result = driver.run(prefix, params)
        result.verify()

        assert len(result.durations) == 2
        assert all(d > 0 for d in result.durations)
        assert store.get("job/0/node/1").decode() == "ok"
        assert store.get("job/1/node/1").decode() == "ok"

        rank_results = [
            pickle.loads(store.get(f"job/1/rank/{r}")) for r in range(8)
        ]
        assert len(rank_results) == 8
        assert all(len(r["durations"]) == 2 for r in rank_results)
        assert sum(r["bytes_fetched"] for r in rank_results) == (
            result.extra_columns["dcp_read_bytes_fetched_total"]
        )

    assert worker_proc.returncode == 0


def test_two_host_spawned_setup_subprocess(tmp_path, monkeypatch):
    config_dir = _create_tiny_llama_config_dir(tmp_path, num_layers=8)
    params = _params(config_dir, strategy="fsdp", dp=8, world_size=8, rounds=1)
    prefix = (tmp_path / "ckpt").as_uri()

    with _two_host_cluster(monkeypatch, precreate_store=False) as (
        host,
        _,
        worker_proc,
    ):
        # Instantiating the driver in the parent must create & own the leader TCPStore
        # while remaining picklable for mp.get_context("spawn").Process(target=driver.setup).
        driver = PyTorchCheckpointReadDriver()

        ctx = mp.get_context("spawn")
        p = ctx.Process(target=driver.setup, args=(prefix, params))
        p.start()
        p.join()
        assert p.exitcode == 0

        # Parent's master TCPStore must still be alive for driver.run (job/1) on both hosts.
        result = driver.run(prefix, params)
        result.verify()

        store = get_store(host)
        assert store.get("job/0/node/1").decode() == "ok"
        assert store.get("job/1/node/1").decode() == "ok"
        assert len(result.durations) == 1

    assert worker_proc.returncode == 0


def test_two_host_reshard_fsdp8_to_fsdp4(tmp_path, monkeypatch):
    config_dir = _create_tiny_llama_config_dir(tmp_path, num_layers=8)
    params = _params(config_dir, strategy="fsdp", dp=4, world_size=4, rounds=1)
    params.source = {
        "strategy": "fsdp",
        "data_parallel_size": 8,
        "world_size": 8,
    }
    prefix = (tmp_path / "ckpt").as_uri()

    with _two_host_cluster(monkeypatch) as (host, store, worker_proc):
        driver = PyTorchCheckpointReadDriver()
        worker_host = dataclasses.replace(host, node_rank=1)
        assert len(local_ranks(4, worker_host)) == 0

        driver.setup(prefix, params)
        result = driver.run(prefix, params)
        result.verify()

        assert len(result.durations) == 1
        assert store.get("job/0/node/1").decode() == "ok"
        # In job/1 (world_size=4), node 1 has zero local ranks and still reports ok.
        assert store.get("job/1/node/1").decode() == "ok"
        assert store.check([f"job/1/rank/{r}" for r in range(4)])
        assert not store.check(["job/1/rank/4"])

    assert worker_proc.returncode == 0


def test_two_host_save(tmp_path, monkeypatch):
    config_dir = _create_tiny_llama_config_dir(tmp_path, num_layers=8)
    params = dataclasses.replace(
        _params(config_dir, strategy="fsdp", dp=8, world_size=8, rounds=1),
        scenario="checkpoint_write",
    )
    prefix = (tmp_path / "ckpt").as_uri()

    with _two_host_cluster(monkeypatch) as (host, store, worker_proc):
        driver = PyTorchCheckpointWriteDriver()
        driver.setup(prefix, params)
        result = driver.run(prefix, params)
        result.verify()

        assert len(result.durations) == 1
        assert result.durations[0] > 0
        assert result.extra_columns["dcp_write_bytes_per_rank_max"] > 0
        assert store.get("job/0/node/1").decode() == "ok"
        assert store.get("job/1/node/1").decode() == "ok"

        fs, root = fsspec.core.url_to_fs(prefix)
        files = fs.find(root)
        assert sum(f.endswith(".distcp") for f in files) == 8

    assert worker_proc.returncode == 0


def test_worker_post_report_rank_failure_raises_in_join(tmp_path, monkeypatch):
    with _two_host_cluster(monkeypatch, pg_timeout_s=30) as (
        host,
        store,
        worker_proc,
    ):
        handle = run_phase(
            f"{_THIS_MODULE}:_rank_fail_after_report_on_node1",
            8,
            str(tmp_path),
            None,
            host=host,
        )
        assert len(handle.results) == 8
        with pytest.raises(
            RuntimeError,
            match=r"(?s)node 1.*rank 5.*post-report verification failed",
        ):
            handle.join()


def test_worker_death_fails_case(tmp_path, monkeypatch):
    with _two_host_cluster(monkeypatch, pg_timeout_s=30) as (
        host,
        store,
        worker_proc,
    ):
        handle = run_phase(
            f"{_THIS_MODULE}:_rank_report_then_wait",
            8,
            str(tmp_path),
            None,
            host=host,
        )
        assert len(handle.results) == 8

        worker_proc.kill()
        worker_proc.wait(timeout=5)

        t0 = time.monotonic()
        with pytest.raises(RuntimeError, match="node 1"):
            handle.join()
        elapsed = time.monotonic() - t0
        assert elapsed < 60.0


def test_phase_port_differs_from_store_port(monkeypatch):
    monkeypatch.setenv("GCSFS_SUBSYSTEM_PORT_BASE", "43200")
    store_port = 43200
    phase_port = _find_phase_port(store_port)
    assert phase_port != store_port
    assert 43200 <= phase_port < 43216

    monkeypatch.delenv("GCSFS_SUBSYSTEM_PORT_BASE", raising=False)
    free_store_port = find_free_port()
    host = HostConfig(
        device="cpu",
        nnodes=1,
        node_rank=0,
        master_addr="localhost",
        master_port=free_store_port,
        nproc_per_node=4,
    )
    try:
        handle = run_phase(
            f"{_THIS_MODULE}:_rank_noop",
            2,
            "file:///tmp",
            None,
            host=host,
        )
        handle.join()
        spec = pickle.loads(get_store(host).get("job/0"))
        assert spec["port"] != host.master_port
    finally:
        shutdown_store(host)


def test_nested_fn_ref_rejected():
    def _nested(ctx, prefix, params):
        pass

    with pytest.raises(ValueError, match="module-level function"):
        run_phase(_nested, 1, "file:///tmp", None)

    with pytest.raises(ValueError, match="module-level function"):
        run_phase(f"{_THIS_MODULE}:Outer.inner", 1, "file:///tmp", None)

    host = HostConfig(nnodes=2, nproc_per_node=4)
    with pytest.raises(ValueError, match="world_size"):
        run_phase(f"{_THIS_MODULE}:_rank_noop", 0, "file:///tmp", None, host=host)
    with pytest.raises(ValueError, match="world_size"):
        run_phase(f"{_THIS_MODULE}:_rank_noop", 9, "file:///tmp", None, host=host)


def test_leader_atexit_cleanly_shuts_down_worker(tmp_path):
    master_port = find_free_port()
    base_env = os.environ.copy()
    base_env.update(
        {
            "GCSFS_SUBSYSTEM_DEVICE": "cpu",
            "GCSFS_SUBSYSTEM_NNODES": "2",
            "GCSFS_SUBSYSTEM_MASTER_ADDR": "localhost",
            "GCSFS_SUBSYSTEM_MASTER_PORT": str(master_port),
            "GCSFS_SUBSYSTEM_NPROC_PER_NODE": "2",
            "GCSFS_SUBSYSTEM_PG_TIMEOUT_S": "30",
        }
    )

    leader_env = base_env.copy()
    leader_env["GCSFS_SUBSYSTEM_NODE_RANK"] = "0"
    leader_code = (
        "from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.phase import run_phase\n"
        f"handle = run_phase('{_THIS_MODULE}:_rank_noop', 4, {str(tmp_path)!r}, None)\n"
        "handle.join()\n"
        "# Exit without calling shutdown_store() explicitly; atexit must coordinate worker shutdown.\n"
    )

    worker_env = base_env.copy()
    worker_env["GCSFS_SUBSYSTEM_NODE_RANK"] = "1"

    leader_proc = subprocess.Popen(
        [sys.executable, "-c", leader_code],
        env=leader_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    worker_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.worker",
        ],
        env=worker_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        leader_out, leader_err = leader_proc.communicate(timeout=30)
        worker_out, worker_err = worker_proc.communicate(timeout=30)
        assert leader_proc.returncode == 0, f"leader failed:\n{leader_err}"
        assert worker_proc.returncode == 0, f"worker failed:\n{worker_err}"
    finally:
        for p in (leader_proc, worker_proc):
            if p.poll() is None:
                p.kill()
                p.wait(timeout=5)


def test_durations_are_max_over_ranks():
    # Simulate ranks across two hosts with different perf_counter epochs:
    # Round 0: rank durations are 1.5, 2.25, 1.75 -> max is 2.25
    # Round 1: rank durations are 3.0, 2.0, 4.5  -> max is 4.5
    results = [
        {"durations": [(10.0, 11.5), (20.0, 23.0)]},
        {"durations": [(5000.0, 5002.25), (5010.0, 5012.0)]},
        {"durations": [(9000.0, 9001.75), (9005.0, 9009.5)]},
    ]
    assert phase_durations(results) == pytest.approx([2.25, 4.5])
    assert phase_durations([]) == []
    assert phase_durations([{}]) == []
