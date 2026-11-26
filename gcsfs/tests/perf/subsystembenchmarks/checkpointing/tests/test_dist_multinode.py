import multiprocessing as mp
import os
import socket
import time
import types

import pytest

torch = pytest.importorskip("torch")

from gcsfs.tests.perf.subsystembenchmarks._common import nodes, placement
from gcsfs.tests.perf.subsystembenchmarks.checkpointing import _dist
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch import common


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _configure_node(node_rank: int, store_port: int, per_node: int = 2) -> None:
    os.environ["GCSFS_SUBSYSTEM_DEVICE"] = "cpu"
    os.environ[placement.NODES_ENV] = "2"
    os.environ[placement.DEVICES_PER_NODE_ENV] = str(per_node)
    os.environ[nodes.NODE_RANK_ENV] = str(node_rank)
    os.environ[nodes.LEADER_ADDR_ENV] = "127.0.0.1"
    os.environ[nodes.STORE_PORT_ENV] = str(store_port)


def _allreduce_target(
    rank, world_size, port, prefix, params, q, *, weights=None, stats_dir=None
):
    import torch.distributed as dist

    _dist.setup_distributed_env(rank, world_size, port)
    try:
        tensor = torch.tensor([rank], dtype=torch.int64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        expected = sum(range(world_size))
        assert (
            int(tensor.item()) == expected
        ), f"rank {rank}: {tensor.item()} != {expected}"
        durations = []
        for _ in range(params.rounds):
            dist.barrier()
            t0 = time.perf_counter()
            time.sleep(0.02)
            dist.barrier()
            durations.append((t0, time.perf_counter()))
        if stats_dir is not None:
            common.write_stats(
                stats_dir, rank, {"rank": rank, "sum": int(tensor.item())}
            )
        q.put(durations)
    finally:
        dist.destroy_process_group()


def _leader_only_target(rank, world_size, port, prefix, params, q):
    import torch.distributed as dist

    _dist.setup_distributed_env(rank, world_size, port)
    try:
        marker = os.path.join(prefix, f"rank-{rank}")
        with open(marker, "w") as f:
            f.write(os.environ.get(nodes.NODE_RANK_ENV, ""))
        durations = []
        for _ in range(params.rounds):
            t0 = time.perf_counter()
            time.sleep(0.01)
            durations.append((t0, time.perf_counter()))
        q.put(durations)
    finally:
        dist.destroy_process_group()


def _failing_follower_target(rank, world_size, port, prefix, params, q):
    if rank >= 2:
        raise RuntimeError(f"boom on follower rank {rank}")
    import torch.distributed as dist

    _dist.setup_distributed_env(rank, world_size, port)
    try:
        dist.barrier()
        q.put([(0.0, 1.0)])
    finally:
        dist.destroy_process_group()


def _node_proc_world4(node_rank: int, store_port: int, prefix: str, out_q) -> None:
    _configure_node(node_rank, store_port, per_node=2)
    params = types.SimpleNamespace(name="w4-case", world_size=4, rounds=2)
    ranks, stats = common.spawn_with_stats(
        prefix, params, _allreduce_target, 4, weights={}
    )
    ranks.join()
    out_q.put((node_rank, ranks.durations, stats))


def _node_proc_world2(node_rank: int, store_port: int, prefix: str, out_q) -> None:
    _configure_node(node_rank, store_port, per_node=2)
    params = types.SimpleNamespace(name="w2-case", world_size=2, rounds=2)
    ranks = _dist.spawn_split(prefix, params, _leader_only_target)
    has_local_ctx = ranks._context is not None
    ranks.join()
    out_q.put((node_rank, ranks.durations, has_local_ctx))


def _node_proc_follower_fail(
    node_rank: int, store_port: int, prefix: str, out_q
) -> None:
    from torch.multiprocessing.spawn import ProcessRaisedException

    _configure_node(node_rank, store_port, per_node=2)
    params = types.SimpleNamespace(name="fail-case", world_size=4, rounds=1)
    t0 = time.monotonic()
    try:
        ranks = _dist.spawn_split(prefix, params, _failing_follower_target)
        ranks.join()
        out_q.put((node_rank, "ok", "", time.monotonic() - t0))
    except (RuntimeError, nodes.PeerFailed, ProcessRaisedException) as exc:
        out_q.put((node_rank, type(exc).__name__, str(exc), time.monotonic() - t0))


def _run_two_node_procs(target, prefix: str, timeout: float = 60.0) -> list:
    port = _free_port()
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = [
        ctx.Process(target=target, args=(rank, port, prefix, out_q)) for rank in (0, 1)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=timeout)
    try:
        for p in procs:
            assert p.exitcode == 0, f"node process exited with {p.exitcode}"
        return [out_q.get(timeout=5) for _ in range(2)]
    finally:
        for p in procs:
            if p.is_alive():
                p.kill()
                p.join(timeout=5)


def test_two_node_spawn_split_world4_all_reduce(tmp_path):
    results = {
        r[0]: (r[1], r[2])
        for r in _run_two_node_procs(_node_proc_world4, str(tmp_path))
    }
    d0, stats0 = results[0]
    d1, stats1 = results[1]
    assert len(d0) == 2
    assert d0 == pytest.approx(d1)
    assert all(d > 0 for d in d0)
    assert [s["rank"] for s in stats0] == [0, 1, 2, 3]
    assert stats0 == stats1
    assert all(s["sum"] == 6 for s in stats0)


def test_two_node_spawn_split_world2_runs_on_leader_only(tmp_path):
    results = {
        r[0]: (r[1], r[2])
        for r in _run_two_node_procs(_node_proc_world2, str(tmp_path))
    }
    d0, ctx0 = results[0]
    d1, ctx1 = results[1]
    assert ctx0 is True
    assert ctx1 is False
    assert len(d0) == 2
    assert d0 == pytest.approx(d1)
    assert (tmp_path / "rank-0").read_text() == "0"
    assert (tmp_path / "rank-1").read_text() == "0"


def test_two_node_spawn_split_follower_rank_failure_fails_both(tmp_path):
    results = {
        r[0]: r[1:]
        for r in _run_two_node_procs(
            _node_proc_follower_fail, str(tmp_path), timeout=60
        )
    }
    for node_rank in (0, 1):
        exc_type, msg, elapsed = results[node_rank]
        assert exc_type != "ok", f"node {node_rank} unexpectedly succeeded"
        assert "boom on follower" in msg
        assert elapsed < 60.0


def test_spawn_split_rejects_uneven_multinode_world(monkeypatch):
    port = _free_port()
    monkeypatch.setenv("GCSFS_SUBSYSTEM_DEVICE", "cpu")
    monkeypatch.setenv(placement.NODES_ENV, "2")
    monkeypatch.setenv(placement.DEVICES_PER_NODE_ENV, "2")
    monkeypatch.setenv(nodes.NODE_RANK_ENV, "0")
    monkeypatch.setenv(nodes.LEADER_ADDR_ENV, "127.0.0.1")
    monkeypatch.setenv(nodes.STORE_PORT_ENV, str(port))
    nodes.reset_channel()
    try:
        params = types.SimpleNamespace(name="uneven", world_size=3, rounds=1)
        with pytest.raises(ValueError, match="multiple of"):
            _dist.spawn_split("gs://b/ckpt/", params, _allreduce_target)
    finally:
        nodes.reset_channel()


def test_setup_distributed_env_sets_ipv4_interface_for_multinode(monkeypatch):
    monkeypatch.setenv("MASTER_ADDR", "pod-0.headless.default.svc.cluster.local")
    monkeypatch.delenv("GLOO_SOCKET_IFNAME", raising=False)
    monkeypatch.delenv("NCCL_SOCKET_IFNAME", raising=False)
    monkeypatch.delenv("NCCL_SOCKET_FAMILY", raising=False)
    monkeypatch.setattr(_dist, "_default_ipv4_interface", lambda: "eth0")
    monkeypatch.setattr(
        _dist.accelerator, "bind_rank", lambda *a, **kw: torch.device("cuda", 0)
    )
    monkeypatch.setattr(
        _dist.accelerator, "process_group_backend", lambda world_size=None: "nccl"
    )
    monkeypatch.setattr(torch.distributed, "init_process_group", lambda *a, **kw: None)

    _dist.setup_distributed_env(8, 16, 29500)
    assert os.environ["GLOO_SOCKET_IFNAME"] == "eth0"
    assert os.environ["NCCL_SOCKET_IFNAME"] == "eth0"
    assert os.environ["NCCL_SOCKET_FAMILY"] == "AF_INET"
