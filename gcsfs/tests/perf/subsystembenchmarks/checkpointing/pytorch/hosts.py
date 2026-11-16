from __future__ import annotations

import dataclasses
import datetime
import os

try:
    import torch
    import torch.distributed as dist
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    dist = None  # type: ignore[assignment]

from gcsfs.tests.perf.subsystembenchmarks.checkpointing._dist import find_free_port

BACKENDS: dict[str, str] = {
    "cpu": "gloo",
    "cuda": "nccl",
    "tpu": "tpu_dist",
}

DEFAULT_PG_TIMEOUT_S = 1800


def _default_nproc_per_node(device: str) -> int:
    if device == "cpu":
        return 16
    if device == "cuda":
        return torch.cuda.device_count()
    if device == "tpu":
        import torch_tpu  # noqa: F401

        tpu_mod = getattr(torch, "tpu", None) or getattr(torch_tpu, "tpu", torch_tpu)
        count = tpu_mod.device_count()
        return count if count > 0 else 8
    raise ValueError(f"Unsupported device {device!r}; expected one of {tuple(BACKENDS)}")


@dataclasses.dataclass(frozen=True)
class HostConfig:
    device: str = "cpu"
    nnodes: int = 1
    node_rank: int = 0
    master_addr: str = "localhost"
    master_port: int = 29500
    nproc_per_node: int = 16

    @classmethod
    def from_env(cls) -> HostConfig:
        device = os.environ.get("GCSFS_SUBSYSTEM_DEVICE", "cpu")
        nnodes = int(os.environ.get("GCSFS_SUBSYSTEM_NNODES", "1"))
        node_rank = int(os.environ.get("GCSFS_SUBSYSTEM_NODE_RANK", "0"))
        master_addr = os.environ.get("GCSFS_SUBSYSTEM_MASTER_ADDR", "localhost")
        port_env = os.environ.get("GCSFS_SUBSYSTEM_MASTER_PORT")
        master_port = int(port_env) if port_env else find_free_port()
        nproc_env = os.environ.get("GCSFS_SUBSYSTEM_NPROC_PER_NODE")
        nproc_per_node = (
            int(nproc_env) if nproc_env else _default_nproc_per_node(device)
        )
        return cls(
            device=device,
            nnodes=nnodes,
            node_rank=node_rank,
            master_addr=master_addr,
            master_port=master_port,
            nproc_per_node=nproc_per_node,
        )


def local_ranks(world_size: int, host: HostConfig) -> range:
    start = host.node_rank * host.nproc_per_node
    stop = min((host.node_rank + 1) * host.nproc_per_node, world_size)
    return range(start, max(start, stop))


def init_rank(rank: int, world_size: int, host: HostConfig, port: int) -> torch.device:
    if host.device not in BACKENDS:
        raise ValueError(
            f"Unsupported device {host.device!r}; expected one of {tuple(BACKENDS)}"
        )

    local_rank = rank - host.node_rank * host.nproc_per_node
    os.environ["MASTER_ADDR"] = host.master_addr
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    if host.master_addr in ("localhost", "127.0.0.1"):
        os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    else:
        os.environ.pop("GLOO_SOCKET_IFNAME", None)

    if host.device == "cpu":
        device = torch.device("cpu")
    elif host.device == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        os.environ.setdefault("TPU_VISIBLE_CHIPS", str(local_rank))
        os.environ.setdefault("CLOUD_TPU_TASK_ID", str(rank))
        import torch_tpu  # noqa: F401

        if torch._C._get_privateuse1_backend_name() == "privateuseone":
            torch.utils.rename_privateuse1_backend("tpu")
        tpu_mod = getattr(torch, "tpu", None) or getattr(torch_tpu, "tpu", None)
        if tpu_mod is not None and hasattr(tpu_mod, "set_device"):
            tpu_mod.set_device("tpu")
        device = torch.device("tpu")

    timeout_s = int(
        os.environ.get("GCSFS_SUBSYSTEM_PG_TIMEOUT_S", str(DEFAULT_PG_TIMEOUT_S))
    )
    dist.init_process_group(
        BACKENDS[host.device],
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=timeout_s),
    )
    return device
