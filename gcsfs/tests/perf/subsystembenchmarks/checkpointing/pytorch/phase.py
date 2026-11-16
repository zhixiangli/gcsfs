from __future__ import annotations

import atexit
import ctypes
import dataclasses
import datetime
import gc
import importlib
import os
import pickle
import signal
import socket
import sys
import time
import traceback
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing.spawn import ProcessRaisedException

from gcsfs.tests.perf.subsystembenchmarks.checkpointing._dist import (
    PORT_BLOCK_SIZE,
    _port_is_free,
    find_free_port,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.hosts import (
    DEFAULT_PG_TIMEOUT_S,
    HostConfig,
    init_rank,
    local_ranks,
)

_HEARTBEAT_STALE_S = 10.0
_POLL_INTERVAL_S = 0.05

# Per-process cache of leader/client TCPStore objects keyed by (pid, master_addr, master_port).
_STORES: dict[tuple[int, str, int], dist.TCPStore] = {}
# Maps (master_addr, master_port) -> PID of the process that created the master TCPStore.
_STORE_OWNER_PID: dict[tuple[str, int], int] = {}
# Maps (master_addr, master_port) -> nnodes recorded for the master TCPStore.
_STORE_OWNER_NNODES: dict[tuple[str, int], int] = {}
# Per-process default master port when GCSFS_SUBSYSTEM_MASTER_PORT is not set in os.environ.
_DEFAULT_MASTER_PORTS: dict[tuple[int, str | None], int] = {}
# Ports that ensure_leader_store populated into os.environ["GCSFS_SUBSYSTEM_MASTER_PORT"].
_AUTO_SET_ENV_PORTS: set[str] = set()
_ATEXIT_REGISTERED = False


class PhaseError(RuntimeError, ProcessRaisedException):
    """Exception raised when any rank or worker fails during a phase."""

    def __init__(self, msg: str, error_index: int = 0, error_pid: int = 0):
        RuntimeError.__init__(self, msg)
        self.error_index = error_index
        self.pid = error_pid


@dataclasses.dataclass
class RankContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    local_world_size: int = 1
    _store: dist.TCPStore | None = dataclasses.field(default=None, repr=False)
    _job_id: int = dataclasses.field(default=0, repr=False)
    _reported: bool = dataclasses.field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.local_world_size <= 0:
            self.local_world_size = max(1, self.world_size)

    def report(self, result: dict[str, Any]) -> None:
        if self._reported:
            raise RuntimeError(
                f"rank {self.rank}: RankContext.report() called more than once"
            )
        self._reported = True
        if self._store is not None:
            self._store.set(
                f"job/{self._job_id}/rank/{self.rank}", pickle.dumps(result)
            )


@dataclasses.dataclass
class PhaseHandle:
    results: list[dict[str, Any]]
    _join_fn: Callable[[], None] = dataclasses.field(repr=False)

    def join(self) -> None:
        self._join_fn()


def phase_durations(results: list[dict[str, Any]]) -> list[float]:
    if not results:
        return []
    rounds_per_rank = [r.get("durations") for r in results]
    if any(not d for d in rounds_per_rank):
        return []
    num_rounds = len(rounds_per_rank[0])
    return [
        max(
            float(rounds_per_rank[rank][round_idx][1])
            - float(rounds_per_rank[rank][round_idx][0])
            for rank in range(len(results))
        )
        for round_idx in range(num_rounds)
    ]


def _pg_timeout_s() -> int:
    return int(
        os.environ.get("GCSFS_SUBSYSTEM_PG_TIMEOUT_S", str(DEFAULT_PG_TIMEOUT_S))
    )


def _owner_pid_for(master_addr: str, master_port: int) -> int | None:
    addr_port = (master_addr, master_port)
    if addr_port in _STORE_OWNER_PID:
        return _STORE_OWNER_PID[addr_port]
    env_owner = os.environ.get("_GCSFS_STORE_OWNER")
    if env_owner:
        try:
            env_addr, env_port_s, env_pid_s = env_owner.rsplit(":", 2)
            if env_addr == master_addr and int(env_port_s) == master_port:
                return int(env_pid_s)
        except ValueError:
            pass
    return None


def _atexit_shutdown() -> None:
    pid = os.getpid()
    owned_hosts = [
        HostConfig(
            nnodes=_STORE_OWNER_NNODES.get((addr, port), 1),
            master_addr=addr,
            master_port=port,
        )
        for (addr, port), owner_pid in list(_STORE_OWNER_PID.items())
        if owner_pid == pid
    ]
    for h in owned_hosts:
        try:
            shutdown_store(h)
        except Exception:
            pass


def _register_atexit() -> None:
    global _ATEXIT_REGISTERED
    if not _ATEXIT_REGISTERED:
        atexit.register(_atexit_shutdown)
        _ATEXIT_REGISTERED = True


def _resolve_host(host: HostConfig | None) -> HostConfig:
    if host is not None:
        return host
    if os.environ.get("GCSFS_SUBSYSTEM_MASTER_PORT"):
        return HostConfig.from_env()
    key = (os.getpid(), os.environ.get("GCSFS_SUBSYSTEM_PORT_BASE"))
    port = _DEFAULT_MASTER_PORTS.get(key)
    if port is None:
        resolved = HostConfig.from_env()
        _DEFAULT_MASTER_PORTS[key] = resolved.master_port
        return resolved
    return dataclasses.replace(HostConfig.from_env(), master_port=port)


def ensure_leader_store(host: HostConfig | None = None) -> dist.TCPStore | None:
    resolved = _resolve_host(host)
    if resolved.node_rank != 0:
        return None
    if "GCSFS_SUBSYSTEM_MASTER_PORT" not in os.environ:
        port_str = str(resolved.master_port)
        os.environ["GCSFS_SUBSYSTEM_MASTER_PORT"] = port_str
        _AUTO_SET_ENV_PORTS.add(port_str)
    store = get_store(resolved)
    return store


def get_store(host: HostConfig | None = None) -> dist.TCPStore:
    resolved = _resolve_host(host)
    pid = os.getpid()
    key = (pid, resolved.master_addr, resolved.master_port)
    addr_port = (resolved.master_addr, resolved.master_port)
    store = _STORES.get(key)
    if store is not None:
        if _STORE_OWNER_PID.get(addr_port) == pid:
            _STORE_OWNER_NNODES[addr_port] = max(
                _STORE_OWNER_NNODES.get(addr_port, 1), resolved.nnodes
            )
        return store

    timeout = datetime.timedelta(seconds=_pg_timeout_s())
    owner_pid = _owner_pid_for(resolved.master_addr, resolved.master_port)

    # Connect as a client on workers or when another process (e.g. parent pytest process) owns the store.
    if (
        resolved.node_rank != 0
        or (owner_pid is not None and owner_pid != pid)
        or not _port_is_free(resolved.master_port)
    ):
        store = dist.TCPStore(
            host_name=resolved.master_addr,
            port=resolved.master_port,
            is_master=False,
            wait_for_workers=False,
            timeout=timeout,
        )
        _STORES[key] = store
        return store

    try:
        store = dist.TCPStore(
            host_name=resolved.master_addr,
            port=resolved.master_port,
            is_master=True,
            wait_for_workers=False,
            timeout=timeout,
        )
        _STORE_OWNER_PID[addr_port] = pid
        _STORE_OWNER_NNODES[addr_port] = max(
            _STORE_OWNER_NNODES.get(addr_port, 1), resolved.nnodes
        )
        os.environ["_GCSFS_STORE_OWNER"] = (
            f"{resolved.master_addr}:{resolved.master_port}:{pid}"
        )
        if "GCSFS_SUBSYSTEM_MASTER_PORT" not in os.environ:
            port_str = str(resolved.master_port)
            os.environ["GCSFS_SUBSYSTEM_MASTER_PORT"] = port_str
            _AUTO_SET_ENV_PORTS.add(port_str)
        _register_atexit()
    except Exception:
        store = dist.TCPStore(
            host_name=resolved.master_addr,
            port=resolved.master_port,
            is_master=False,
            wait_for_workers=False,
            timeout=timeout,
        )
    _STORES[key] = store
    return store


def shutdown_store(host: HostConfig | None = None) -> None:
    pid = os.getpid()
    if host is not None:
        keys = [(pid, host.master_addr, host.master_port)]
        base_nnodes = host.nnodes
    else:
        keys = [k for k in _STORES if k[0] == pid]
        base_nnodes = int(os.environ.get("GCSFS_SUBSYSTEM_NNODES", "1"))

    for key in keys:
        _, addr, port = key
        addr_port = (addr, port)
        owner_pid = _owner_pid_for(addr, port)
        recorded_nnodes = _STORE_OWNER_NNODES.pop(addr_port, 1)
        nnodes = max(
            base_nnodes,
            recorded_nnodes,
            int(os.environ.get("GCSFS_SUBSYSTEM_NNODES", "1")),
        )
        store = _STORES.pop(key, None)
        for def_key, def_port in list(_DEFAULT_MASTER_PORTS.items()):
            if def_key[0] == pid and def_port == port:
                _DEFAULT_MASTER_PORTS.pop(def_key, None)
        if store is None:
            continue
        if owner_pid is None or owner_pid == pid:
            try:
                store.set("shutdown", "1")
                if nnodes > 1:
                    ack_keys = [f"node/{k}/shutdown_ack" for k in range(1, nnodes)]
                    deadline = time.monotonic() + 2.0
                    while time.monotonic() < deadline:
                        if store.check(ack_keys):
                            break
                        time.sleep(_POLL_INTERVAL_S)
            except Exception:
                pass
            _STORE_OWNER_PID.pop(addr_port, None)
            if os.environ.get("_GCSFS_STORE_OWNER") == f"{addr}:{port}:{pid}":
                os.environ.pop("_GCSFS_STORE_OWNER", None)
            port_str = str(port)
            if port_str in _AUTO_SET_ENV_PORTS:
                _AUTO_SET_ENV_PORTS.discard(port_str)
                if os.environ.get("GCSFS_SUBSYSTEM_MASTER_PORT") == port_str:
                    os.environ.pop("GCSFS_SUBSYSTEM_MASTER_PORT", None)
        del store

    if host is None:
        for k in [k for k in _DEFAULT_MASTER_PORTS if k[0] == pid]:
            _DEFAULT_MASTER_PORTS.pop(k, None)
    gc.collect()


def _find_phase_port(store_port: int) -> int:
    port = find_free_port()
    if port != store_port:
        return port

    base = os.environ.get("GCSFS_SUBSYSTEM_PORT_BASE")
    if base:
        for candidate in range(int(base), int(base) + PORT_BLOCK_SIZE):
            if candidate != store_port and _port_is_free(candidate):
                return candidate
        raise RuntimeError(
            f"no free phase port != {store_port} in "
            f"[{base}, {int(base) + PORT_BLOCK_SIZE}) (GCSFS_SUBSYSTEM_PORT_BASE)"
        )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as hold:
        try:
            hold.bind(("", store_port))
        except OSError:
            pass
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


def _set_pdeathsig(parent_pid: int) -> None:
    if sys.platform != "linux":
        return
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG = 1
        if os.getppid() != parent_pid:
            os.kill(os.getpid(), signal.SIGKILL)
    except Exception:
        pass


def _validate_fn_ref(fn_ref: str | Callable[..., None]) -> str:
    if callable(fn_ref):
        qualname = getattr(fn_ref, "__qualname__", "")
        if not qualname or "." in qualname or "<" in qualname:
            raise ValueError(
                f"fn_ref must be a module-level function, got {fn_ref!r}"
            )
        fn_ref = f"{fn_ref.__module__}:{qualname}"
    if ":" not in fn_ref:
        raise ValueError(
            f"fn_ref must be 'module:function', got {fn_ref!r}"
        )
    _, func_name = fn_ref.split(":", 1)
    if not func_name or "." in func_name or "<" in func_name:
        raise ValueError(
            f"fn_ref must reference a module-level function, got {fn_ref!r}"
        )
    return fn_ref


def _resolve_fn(fn_ref: str) -> Callable[..., None]:
    validated = _validate_fn_ref(fn_ref)
    module_name, func_name = validated.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        if not module_name.startswith("gcsfs."):
            module = importlib.import_module(f"gcsfs.tests.perf.{module_name}")
        else:
            raise
    return getattr(module, func_name)


def _rank_entry(
    local_idx: int,
    ranks_start: int,
    local_world_size: int,
    job_id: int,
    spec: dict[str, Any],
    host: HostConfig,
    parent_pid: int,
) -> None:
    _set_pdeathsig(parent_pid)
    rank = ranks_start + local_idx
    local_rank = rank - host.node_rank * host.nproc_per_node
    world_size = int(spec["world_size"])
    port = int(spec["port"])

    store = dist.TCPStore(
        host_name=host.master_addr,
        port=host.master_port,
        is_master=False,
        wait_for_workers=False,
        timeout=datetime.timedelta(seconds=_pg_timeout_s()),
    )
    try:
        device = init_rank(rank, world_size, host, port)
        ctx = RankContext(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=device,
            local_world_size=local_world_size,
            _store=store,
            _job_id=job_id,
        )
        fn = _resolve_fn(spec["fn_ref"])
        fn(ctx, spec["prefix"], spec["params"], **spec["kwargs"])
        if not ctx._reported:
            ctx.report({})
    except BaseException:
        tb = traceback.format_exc()
        try:
            store.set(
                f"job/{job_id}/rank_error/{rank}",
                f"node {host.node_rank} rank {rank}:\n{tb}",
            )
        except Exception:
            pass
        raise
    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


def spawn_local_ranks(
    job_id: int, spec: dict[str, Any], host: HostConfig
) -> Any | None:
    ranks = local_ranks(int(spec["world_size"]), host)
    if len(ranks) == 0:
        return None
    return mp.spawn(
        _rank_entry,
        args=(ranks.start, len(ranks), job_id, spec, host, os.getpid()),
        nprocs=len(ranks),
        join=False,
    )


def _terminate_spawn_context(
    context: Any | None,
    store: dist.TCPStore | None = None,
    job_id: int | None = None,
) -> None:
    if store is not None and job_id is not None:
        try:
            store.set(f"job/{job_id}/abort", "1")
        except Exception:
            pass
    if context is None:
        return
    for proc in getattr(context, "processes", []):
        if proc.is_alive():
            try:
                proc.kill()
            except Exception:
                pass
    for proc in getattr(context, "processes", []):
        try:
            proc.join(timeout=1.0)
        except Exception:
            pass


class _HeartbeatMonitor:
    def __init__(self, store: dist.TCPStore, host: HostConfig):
        self._store = store
        self._host = host
        now = time.monotonic()
        self._last_val: dict[int, bytes | None] = {
            k: None for k in range(1, host.nnodes)
        }
        self._last_change: dict[int, float] = {
            k: now for k in range(1, host.nnodes)
        }

    def check_dead_worker(self, job_id: int) -> str | None:
        now = time.monotonic()
        for k in range(1, self._host.nnodes):
            if self._store.check([f"job/{job_id}/node/{k}"]):
                continue
            hb_key = f"node/{k}/heartbeat"
            if self._store.check([hb_key]):
                val = self._store.get(hb_key)
                if val != self._last_val[k]:
                    self._last_val[k] = val
                    self._last_change[k] = now
                elif now - self._last_change[k] > _HEARTBEAT_STALE_S:
                    return (
                        f"node {k} worker died during job {job_id} "
                        f"(heartbeat stalled for >{_HEARTBEAT_STALE_S:.1f}s)"
                    )
        return None


def _collect_failure_message(
    store: dist.TCPStore,
    job_id: int,
    world_size: int,
    host: HostConfig,
    local_exc: BaseException | None = None,
    extra_reason: str | None = None,
) -> str:
    if host.nnodes > 1:
        # Give surviving workers a brief window to post their node status.
        wait_until = time.monotonic() + 0.25
        node_keys = [f"job/{job_id}/node/{k}" for k in range(1, host.nnodes)]
        while time.monotonic() < wait_until:
            if store.check(node_keys):
                break
            time.sleep(_POLL_INTERVAL_S)

    parts: list[str] = []
    if extra_reason:
        parts.append(extra_reason)

    for r in range(world_size):
        err_key = f"job/{job_id}/rank_error/{r}"
        if store.check([err_key]):
            parts.append(store.get(err_key).decode("utf-8", errors="replace"))

    if local_exc is not None:
        ranks = local_ranks(world_size, host)
        err_idx = getattr(local_exc, "error_index", 0)
        rank = (
            ranks.start + err_idx
            if 0 <= err_idx < len(ranks)
            else ranks.start
        )
        parts.append(f"node {host.node_rank} rank {rank}: {local_exc}")

    for k in range(1, host.nnodes):
        node_key = f"job/{job_id}/node/{k}"
        if store.check([node_key]):
            status = store.get(node_key).decode("utf-8", errors="replace")
            if status != "ok":
                parts.append(f"node {k} error:\n{status}")
        else:
            parts.append(
                f"node {k} did not report completion for job {job_id} "
                "(worker died or timed out)"
            )

    if not parts:
        parts.append(f"job {job_id} failed on node {host.node_rank}")
    return "\n".join(parts)


def run_phase(
    fn_ref: str | Callable[..., None],
    world_size: int,
    prefix: str,
    params: Any,
    host: HostConfig | None = None,
    **kwargs: Any,
) -> PhaseHandle:
    fn_ref = _validate_fn_ref(fn_ref)
    resolved_host = _resolve_host(host)
    max_ranks = resolved_host.nnodes * resolved_host.nproc_per_node
    if world_size < 1 or world_size > max_ranks:
        raise ValueError(
            f"world_size={world_size} must be in [1, {max_ranks}] "
            f"(nnodes={resolved_host.nnodes}, nproc_per_node={resolved_host.nproc_per_node})"
        )
    store = get_store(resolved_host)
    port = _find_phase_port(resolved_host.master_port)
    job_id = store.add("job/next_id", 1) - 1

    spec = {
        "fn_ref": fn_ref,
        "world_size": world_size,
        "prefix": prefix,
        "params": params,
        "kwargs": kwargs,
        "port": port,
    }
    store.set(f"job/{job_id}", pickle.dumps(spec))

    context = spawn_local_ranks(job_id, spec, resolved_host)
    if context is None:
        store.set(f"job/{job_id}/node/{resolved_host.node_rank}", "ok")

    timeout_s = _pg_timeout_s()
    deadline = time.monotonic() + timeout_s
    hb_monitor = _HeartbeatMonitor(store, resolved_host)
    all_rank_keys = [f"job/{job_id}/rank/{r}" for r in range(world_size)]
    local_joined = context is None

    try:
        while not store.check(all_rank_keys):
            for k in range(1, resolved_host.nnodes):
                node_key = f"job/{job_id}/node/{k}"
                if store.check([node_key]):
                    status = store.get(node_key).decode("utf-8", errors="replace")
                    if status != "ok":
                        raise PhaseError(
                            _collect_failure_message(
                                store,
                                job_id,
                                world_size,
                                resolved_host,
                                extra_reason=f"node {k} failed before reporting:\n{status}",
                            )
                        )

            if not local_joined:
                try:
                    if context.join(timeout=0, grace_period=0.5):
                        local_joined = True
                        store.set(
                            f"job/{job_id}/node/{resolved_host.node_rank}", "ok"
                        )
                        local_rank_keys = [
                            f"job/{job_id}/rank/{r}"
                            for r in local_ranks(world_size, resolved_host)
                        ]
                        if not store.check(local_rank_keys):
                            raise PhaseError(
                                _collect_failure_message(
                                    store,
                                    job_id,
                                    world_size,
                                    resolved_host,
                                    extra_reason=(
                                        f"node {resolved_host.node_rank} ranks exited "
                                        "without reporting results"
                                    ),
                                )
                            )
                except PhaseError:
                    raise
                except BaseException as exc:
                    store.set(
                        f"job/{job_id}/node/{resolved_host.node_rank}", str(exc)
                    )
                    raise PhaseError(
                        _collect_failure_message(
                            store,
                            job_id,
                            world_size,
                            resolved_host,
                            local_exc=exc,
                        ),
                        error_index=getattr(exc, "error_index", 0),
                        error_pid=getattr(exc, "pid", 0),
                    ) from exc

            dead_msg = hb_monitor.check_dead_worker(job_id)
            if dead_msg is not None:
                raise PhaseError(
                    _collect_failure_message(
                        store,
                        job_id,
                        world_size,
                        resolved_host,
                        extra_reason=dead_msg,
                    )
                )

            if time.monotonic() > deadline:
                raise PhaseError(
                    _collect_failure_message(
                        store,
                        job_id,
                        world_size,
                        resolved_host,
                        extra_reason=(
                            f"job {job_id} timed out after {timeout_s}s waiting for "
                            "ranks to report"
                        ),
                    )
                )

            time.sleep(_POLL_INTERVAL_S)
    except BaseException:
        _terminate_spawn_context(context, store=store, job_id=job_id)
        raise

    results = [
        pickle.loads(store.get(f"job/{job_id}/rank/{r}"))
        for r in range(world_size)
    ]

    def _join() -> None:
        nonlocal local_joined
        join_deadline = time.monotonic() + timeout_s
        worker_ranks = range(1, resolved_host.nnodes)
        verified_ok_workers: set[int] = set()
        try:
            while True:
                if not local_joined:
                    try:
                        if context.join(timeout=_POLL_INTERVAL_S, grace_period=0.5):
                            local_joined = True
                            store.set(
                                f"job/{job_id}/node/{resolved_host.node_rank}", "ok"
                            )
                    except BaseException as exc:
                        store.set(
                            f"job/{job_id}/node/{resolved_host.node_rank}", str(exc)
                        )
                        raise PhaseError(
                            _collect_failure_message(
                                store,
                                job_id,
                                world_size,
                                resolved_host,
                                local_exc=exc,
                            ),
                            error_index=getattr(exc, "error_index", 0),
                            error_pid=getattr(exc, "pid", 0),
                        ) from exc

                for k in worker_ranks:
                    if k in verified_ok_workers:
                        continue
                    node_key = f"job/{job_id}/node/{k}"
                    if store.check([node_key]):
                        status = store.get(node_key).decode(
                            "utf-8", errors="replace"
                        )
                        if status != "ok":
                            raise PhaseError(
                                _collect_failure_message(
                                    store,
                                    job_id,
                                    world_size,
                                    resolved_host,
                                    extra_reason=f"node {k} failed:\n{status}",
                                )
                            )
                        verified_ok_workers.add(k)

                if local_joined and len(verified_ok_workers) == len(worker_ranks):
                    return

                dead_msg = hb_monitor.check_dead_worker(job_id)
                if dead_msg is not None:
                    raise PhaseError(
                        _collect_failure_message(
                            store,
                            job_id,
                            world_size,
                            resolved_host,
                            extra_reason=dead_msg,
                        )
                    )

                if time.monotonic() > join_deadline:
                    raise PhaseError(
                        _collect_failure_message(
                            store,
                            job_id,
                            world_size,
                            resolved_host,
                            extra_reason=(
                                f"job {job_id} timed out after {timeout_s}s in join()"
                            ),
                        )
                    )

                if local_joined:
                    time.sleep(_POLL_INTERVAL_S)
        except BaseException:
            _terminate_spawn_context(context, store=store, job_id=job_id)
            raise

    return PhaseHandle(results=results, _join_fn=_join)
