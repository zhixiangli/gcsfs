import contextlib
import dataclasses
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


@contextlib.contextmanager
def _two_host_cluster(monkeypatch, pg_timeout_s: int = 30):
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
    # Ensure the leader TCPStore is listening before launching the worker.
    store = get_store(leader_host)

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
    driver = PyTorchCheckpointReadDriver()

    with _two_host_cluster(monkeypatch) as (host, store, worker_proc):
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


def test_two_host_reshard_fsdp8_to_fsdp4(tmp_path, monkeypatch):
    config_dir = _create_tiny_llama_config_dir(tmp_path, num_layers=8)
    params = _params(config_dir, strategy="fsdp", dp=4, world_size=4, rounds=1)
    params.source = {
        "strategy": "fsdp",
        "data_parallel_size": 8,
        "world_size": 8,
    }
    prefix = (tmp_path / "ckpt").as_uri()
    driver = PyTorchCheckpointReadDriver()

    with _two_host_cluster(monkeypatch) as (host, store, worker_proc):
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
    driver = PyTorchCheckpointWriteDriver()

    with _two_host_cluster(monkeypatch) as (host, store, worker_proc):
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
