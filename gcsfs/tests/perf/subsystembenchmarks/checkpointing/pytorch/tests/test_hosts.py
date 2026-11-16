import datetime
import os
import sys
import types
from unittest import mock

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore

from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch import (
    configs,
    parallelize,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.hosts import (
    BACKENDS,
    DEFAULT_PG_TIMEOUT_S,
    HostConfig,
    init_rank,
    local_ranks,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.layout import Layout
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.model import (
    materialize,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.parallelize import (
    build_mesh,
)

_HOST_ENV_KEYS = (
    "GCSFS_SUBSYSTEM_DEVICE",
    "GCSFS_SUBSYSTEM_NNODES",
    "GCSFS_SUBSYSTEM_NODE_RANK",
    "GCSFS_SUBSYSTEM_MASTER_ADDR",
    "GCSFS_SUBSYSTEM_MASTER_PORT",
    "GCSFS_SUBSYSTEM_PORT_BASE",
    "GCSFS_SUBSYSTEM_NPROC_PER_NODE",
    "GCSFS_SUBSYSTEM_PG_TIMEOUT_S",
    "MASTER_ADDR",
    "MASTER_PORT",
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "GLOO_SOCKET_IFNAME",
    "NCCL_TUNER_PLUGIN",
    "NCCL_PROFILER_PLUGIN",
    "NCCL_NET_PLUGIN",
)


@pytest.fixture(autouse=True)
def _clean_host_env(monkeypatch):
    for key in _HOST_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _ensure_tpu_privateuse1_registered() -> None:
    if torch._C._get_privateuse1_backend_name() == "privateuseone":
        torch.utils.rename_privateuse1_backend("tpu")


def test_local_ranks_fill_hosts_in_order():
    node0 = HostConfig(nnodes=2, node_rank=0, nproc_per_node=8)
    node1 = HostConfig(nnodes=2, node_rank=1, nproc_per_node=8)

    assert list(local_ranks(16, node0)) == list(range(0, 8))
    assert list(local_ranks(16, node1)) == list(range(8, 16))

    assert list(local_ranks(8, node0)) == list(range(0, 8))
    assert list(local_ranks(8, node1)) == []
    assert len(local_ranks(8, node1)) == 0

    assert list(local_ranks(4, node0)) == list(range(0, 4))
    assert list(local_ranks(4, node1)) == []

    assert list(local_ranks(12, node0)) == list(range(0, 8))
    assert list(local_ranks(12, node1)) == list(range(8, 12))


def test_from_env_defaults(monkeypatch):
    host = HostConfig.from_env()
    assert host.device == "cpu"
    assert host.nnodes == 1
    assert host.node_rank == 0
    assert host.master_addr == "localhost"
    assert isinstance(host.master_port, int) and host.master_port > 0
    assert host.nproc_per_node == 16

    # Port base is honored when GCSFS_SUBSYSTEM_MASTER_PORT is not set.
    monkeypatch.setenv("GCSFS_SUBSYSTEM_PORT_BASE", "43100")
    host_port_base = HostConfig.from_env()
    assert 43100 <= host_port_base.master_port < 43116

    # Explicit env overrides take precedence.
    monkeypatch.setenv("GCSFS_SUBSYSTEM_DEVICE", "cuda")
    monkeypatch.setenv("GCSFS_SUBSYSTEM_NNODES", "2")
    monkeypatch.setenv("GCSFS_SUBSYSTEM_NODE_RANK", "1")
    monkeypatch.setenv("GCSFS_SUBSYSTEM_MASTER_ADDR", "10.0.0.1")
    monkeypatch.setenv("GCSFS_SUBSYSTEM_MASTER_PORT", "29505")
    monkeypatch.setenv("GCSFS_SUBSYSTEM_NPROC_PER_NODE", "8")
    overridden = HostConfig.from_env()
    assert overridden == HostConfig(
        device="cuda",
        nnodes=2,
        node_rank=1,
        master_addr="10.0.0.1",
        master_port=29505,
        nproc_per_node=8,
    )


def test_host_config_rejects_invalid_device(monkeypatch):
    with pytest.raises(ValueError, match="Unsupported device"):
        HostConfig(device="npu")

    monkeypatch.setenv("GCSFS_SUBSYSTEM_DEVICE", "npu")
    monkeypatch.setenv("GCSFS_SUBSYSTEM_NPROC_PER_NODE", "8")
    with pytest.raises(ValueError, match="Unsupported device"):
        HostConfig.from_env()


def test_from_env_device_nproc_defaults(monkeypatch):
    monkeypatch.setenv("GCSFS_SUBSYSTEM_DEVICE", "cuda")
    with mock.patch.object(torch.cuda, "device_count", return_value=4):
        assert HostConfig.from_env().nproc_per_node == 4

    monkeypatch.setenv("GCSFS_SUBSYSTEM_DEVICE", "tpu")
    fake_torch_tpu = types.ModuleType("torch_tpu")
    with (
        mock.patch.dict(sys.modules, {"torch_tpu": fake_torch_tpu}),
        mock.patch.object(torch.accelerator, "device_count", return_value=8),
    ):
        assert HostConfig.from_env().nproc_per_node == 8


def test_backend_map():
    assert BACKENDS == {"cpu": "gloo", "cuda": "nccl", "tpu": "tpu_dist"}


def test_build_mesh_uses_device_type():
    dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=4)
    try:
        layout = Layout(strategy="fsdp", world_size=4, data_parallel_size=4)
        mesh = build_mesh(layout, device_type="cpu")
        assert mesh.device_type == "cpu"
        assert mesh.mesh_dim_names == ("dp_shard",)
        assert tuple(mesh.shape) == (4,)

        with mock.patch.object(
            parallelize, "init_device_mesh", wraps=parallelize.init_device_mesh
        ) as spy_init_mesh:
            build_mesh(layout, device_type="cpu")
            spy_init_mesh.assert_called_once_with(
                "cpu", (4,), mesh_dim_names=("dp_shard",)
            )

        with mock.patch.object(parallelize, "init_device_mesh") as mock_init_mesh:
            build_mesh(layout, device_type="tpu")
            mock_init_mesh.assert_called_once_with(
                "tpu", (4,), mesh_dim_names=("dp_shard",)
            )
    finally:
        dist.destroy_process_group()


def test_materialize_uses_device():
    with torch.device("meta"):
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 8, dtype=torch.bfloat16),
            torch.nn.Linear(8, 4, dtype=torch.bfloat16),
        )
    assert all(p.is_meta for p in model.parameters())
    materialize(model, seed=7, device=torch.device("cpu"))
    assert all(
        not p.is_meta and p.device == torch.device("cpu") and p.requires_grad
        for p in model.parameters()
    )


def test_init_rank_sets_env_and_gloo_lo_only_for_localhost(monkeypatch):
    for key in (
        "MASTER_ADDR",
        "MASTER_PORT",
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "GLOO_SOCKET_IFNAME",
    ):
        monkeypatch.delenv(key, raising=False)

    with mock.patch.object(dist, "init_process_group") as mock_init_pg:
        local_host = HostConfig(
            device="cpu",
            nnodes=1,
            node_rank=0,
            master_addr="localhost",
            nproc_per_node=8,
        )
        dev = init_rank(rank=3, world_size=8, host=local_host, port=29600)
        assert dev == torch.device("cpu")
        assert os.environ["MASTER_ADDR"] == "localhost"
        assert os.environ["MASTER_PORT"] == "29600"
        assert os.environ["WORLD_SIZE"] == "8"
        assert os.environ["RANK"] == "3"
        assert os.environ["LOCAL_RANK"] == "3"
        assert os.environ["GLOO_SOCKET_IFNAME"] == "lo"
        mock_init_pg.assert_called_once_with(
            "gloo",
            rank=3,
            world_size=8,
            timeout=datetime.timedelta(seconds=DEFAULT_PG_TIMEOUT_S),
        )

    monkeypatch.delenv("GLOO_SOCKET_IFNAME", raising=False)
    with mock.patch.object(dist, "init_process_group") as mock_init_pg:
        remote_host = HostConfig(
            device="cpu",
            nnodes=2,
            node_rank=1,
            master_addr="10.0.0.1",
            nproc_per_node=8,
        )
        dev = init_rank(rank=11, world_size=16, host=remote_host, port=29601)
        assert dev == torch.device("cpu")
        assert os.environ["MASTER_ADDR"] == "10.0.0.1"
        assert os.environ["MASTER_PORT"] == "29601"
        assert os.environ["WORLD_SIZE"] == "16"
        assert os.environ["RANK"] == "11"
        assert os.environ["LOCAL_RANK"] == "3"
        assert "GLOO_SOCKET_IFNAME" not in os.environ
        mock_init_pg.assert_called_once_with(
            "gloo",
            rank=11,
            world_size=16,
            timeout=datetime.timedelta(seconds=DEFAULT_PG_TIMEOUT_S),
        )

    # Externally configured GLOO_SOCKET_IFNAME (e.g. multi-NIC GKE pod) is preserved on remote hosts.
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "eth0")
    with mock.patch.object(dist, "init_process_group"):
        init_rank(rank=11, world_size=16, host=remote_host, port=29601)
        assert os.environ["GLOO_SOCKET_IFNAME"] == "eth0"


def test_init_rank_cuda_and_tpu_binding(monkeypatch):
    monkeypatch.setenv("GCSFS_SUBSYSTEM_PG_TIMEOUT_S", "600")
    for key in ("NCCL_TUNER_PLUGIN", "NCCL_PROFILER_PLUGIN", "NCCL_NET_PLUGIN"):
        monkeypatch.delenv(key, raising=False)
    with (
        mock.patch.object(torch.cuda, "set_device") as mock_set_cuda,
        mock.patch.object(dist, "init_process_group") as mock_init_pg,
    ):
        cuda_host = HostConfig(
            device="cuda",
            nnodes=2,
            node_rank=1,
            master_addr="10.0.0.2",
            nproc_per_node=8,
        )
        dev = init_rank(rank=10, world_size=16, host=cuda_host, port=29602)
        assert dev == torch.device("cuda", 2)
        assert os.environ["NCCL_TUNER_PLUGIN"] == "none"
        assert os.environ["NCCL_PROFILER_PLUGIN"] == "none"
        assert os.environ["NCCL_NET_PLUGIN"] == "none"
        mock_set_cuda.assert_called_once_with(2)
        mock_init_pg.assert_called_once_with(
            "nccl",
            rank=10,
            world_size=16,
            timeout=datetime.timedelta(seconds=600),
        )

    _ensure_tpu_privateuse1_registered()
    fake_torch_tpu = types.ModuleType("torch_tpu")
    with (
        mock.patch.dict(sys.modules, {"torch_tpu": fake_torch_tpu}),
        mock.patch.object(dist, "init_process_group") as mock_init_pg,
    ):
        tpu_host = HostConfig(
            device="tpu",
            nnodes=2,
            node_rank=1,
            master_addr="10.0.0.3",
            nproc_per_node=4,
        )
        dev = init_rank(rank=6, world_size=8, host=tpu_host, port=29603)
        assert dev == torch.device("tpu")
        assert os.environ["RANK"] == "6"
        assert os.environ["LOCAL_RANK"] == "2"
        mock_init_pg.assert_called_once_with(
            "tpu_dist",
            rank=6,
            world_size=8,
            timeout=datetime.timedelta(seconds=600),
        )


def test_configs_uses_host_config_from_env(monkeypatch):
    assert configs.HostConfig is HostConfig
    monkeypatch.setenv("GCSFS_SUBSYSTEM_NNODES", "1")
    monkeypatch.setenv("GCSFS_SUBSYSTEM_NPROC_PER_NODE", "8")
    params = configs.case_params("checkpoint_read")
    skipped = [p.id for p in params if p.marks]
    assert "load-llama_3_1_8b-pp-hsdp-tp-pp2dpr2dp2tp2-ws16-reg" in skipped
