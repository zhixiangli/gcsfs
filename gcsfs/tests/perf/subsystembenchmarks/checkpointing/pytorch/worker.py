from __future__ import annotations

import argparse
import datetime
import os
import pickle
import socket
import threading
import time

import torch.distributed as dist

from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.hosts import (
    DEFAULT_PG_TIMEOUT_S,
    HostConfig,
    local_ranks,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.phase import (
    _terminate_spawn_context,
    spawn_local_ranks,
)

_HEARTBEAT_INTERVAL_S = 1.0
_POLL_INTERVAL_S = 0.05


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-host worker for checkpointing/pytorch phase runner"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--nnodes", type=int, default=None)
    parser.add_argument("--node-rank", type=int, default=None)
    parser.add_argument("--master-addr", default=None)
    parser.add_argument("--master-port", type=int, default=None)
    parser.add_argument("--nproc-per-node", type=int, default=None)
    return parser.parse_args(argv)


def _apply_cli_overrides(args: argparse.Namespace) -> None:
    mapping = {
        "device": "GCSFS_SUBSYSTEM_DEVICE",
        "nnodes": "GCSFS_SUBSYSTEM_NNODES",
        "node_rank": "GCSFS_SUBSYSTEM_NODE_RANK",
        "master_addr": "GCSFS_SUBSYSTEM_MASTER_ADDR",
        "master_port": "GCSFS_SUBSYSTEM_MASTER_PORT",
        "nproc_per_node": "GCSFS_SUBSYSTEM_NPROC_PER_NODE",
    }
    for attr, env_key in mapping.items():
        val = getattr(args, attr, None)
        if val is not None:
            os.environ[env_key] = str(val)


def _heartbeat_loop(
    store: dist.TCPStore, node_rank: int, stop_event: threading.Event
) -> None:
    seq = 1
    hb_key = f"node/{node_rank}/heartbeat"
    while not stop_event.wait(_HEARTBEAT_INTERVAL_S):
        try:
            store.set(hb_key, str(seq))
            seq += 1
        except Exception:
            break


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _apply_cli_overrides(args)
    host = HostConfig.from_env()

    timeout_s = int(
        os.environ.get("GCSFS_SUBSYSTEM_PG_TIMEOUT_S", str(DEFAULT_PG_TIMEOUT_S))
    )
    store = dist.TCPStore(
        host_name=host.master_addr,
        port=host.master_port,
        is_master=False,
        wait_for_workers=False,
        timeout=datetime.timedelta(seconds=timeout_s),
    )

    if host.master_addr in ("localhost", "127.0.0.1"):
        local_addr = "localhost"
    else:
        try:
            local_addr = socket.gethostbyname(socket.gethostname())
        except OSError:
            local_addr = socket.gethostname()
    store.set(f"node/{host.node_rank}/addr", local_addr)
    store.set(f"node/{host.node_rank}/heartbeat", "0")
    stop_event = threading.Event()
    hb_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(store, host.node_rank, stop_event),
        daemon=True,
    )
    hb_thread.start()

    job_id = 0
    try:
        while True:
            job_key = f"job/{job_id}"
            while True:
                if store.check([job_key]):
                    break
                if store.check(["shutdown"]):
                    stop_event.set()
                    hb_thread.join(timeout=1.0)
                    try:
                        store.set(f"node/{host.node_rank}/shutdown_ack", "1")
                    except Exception:
                        pass
                    return 0
                time.sleep(_POLL_INTERVAL_S)

            spec = pickle.loads(store.get(job_key))
            world_size = int(spec["world_size"])
            ranks = local_ranks(world_size, host)
            node_key = f"job/{job_id}/node/{host.node_rank}"
            abort_key = f"job/{job_id}/abort"

            if len(ranks) == 0:
                store.set(node_key, "ok")
            else:
                context = spawn_local_ranks(job_id, spec, host)
                aborted = False
                try:
                    while not context.join(timeout=_POLL_INTERVAL_S, grace_period=0.5):
                        if store.check([abort_key]) or store.check(["shutdown"]):
                            _terminate_spawn_context(context)
                            aborted = True
                            break
                    if aborted:
                        store.set(node_key, f"node {host.node_rank} aborted")
                    else:
                        store.set(node_key, "ok")
                except BaseException as exc:
                    _terminate_spawn_context(context)
                    err_idx = getattr(exc, "error_index", 0)
                    rank = (
                        ranks.start + err_idx
                        if 0 <= err_idx < len(ranks)
                        else ranks.start
                    )
                    store.set(
                        node_key, f"node {host.node_rank} rank {rank}:\n{exc}"
                    )

            job_id += 1
    finally:
        stop_event.set()


if __name__ == "__main__":
    raise SystemExit(main())
