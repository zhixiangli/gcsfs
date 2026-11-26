import contextlib
import functools
import os
import socket
import sys

from gcsfs.tests.perf.subsystembenchmarks._common import (
    accelerator,
    nodes,
    placement,
    spawn,
)

# Cap on spawned ranks to prevent OOM on test-c4-highmem.
MAX_RANKS = accelerator.CPU_MAX_RANKS


# Concurrent benchmark processes on one VM (cloudbuild/subsystembenchmarks/scripts/run-ci.sh)
# each get a disjoint block of ports via GCSFS_SUBSYSTEM_PORT_BASE. Without it, an
# OS-assigned free port is probed and only bound later by the TCPStore, so two processes
# can pick the same port and one fails with EADDRINUSE.
PORT_BLOCK_SIZE = 16


def _port_is_free(port: int) -> bool:
    # A TCPStore listens dual-stack on [::]:port, which also makes this IPv4 bind fail;
    # binding without SO_REUSEADDR also rejects a port that holds a server-side TIME_WAIT.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
        except OSError:
            return False
    return True


def find_free_port() -> int:
    """Finds an available TCP port for the Gloo distributed backend."""
    base = os.environ.get("GCSFS_SUBSYSTEM_PORT_BASE")
    if base:
        # The first free port in the block. A checkpoint read case's setup (in its own
        # spawned process) and run (in the parent) may both get the same port; that is safe
        # because setup's process group is gone before run probes the block.
        for port in range(int(base), int(base) + PORT_BLOCK_SIZE):
            if _port_is_free(port):
                return port
        raise RuntimeError(
            f"no free port in [{base}, {int(base) + PORT_BLOCK_SIZE}) "
            "(GCSFS_SUBSYSTEM_PORT_BASE)"
        )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def run_split(prefix, params, target_fn, world_size_override=None) -> list[float]:
    """Spawns processes to run the distributed benchmark and gathers timing."""
    ranks = spawn_split(prefix, params, target_fn, world_size_override)
    ranks.join()
    return ranks.durations


class SpawnedRanks:
    """Ranks spawned by spawn_split.

    ``durations`` is reduced as soon as every rank has reported, so a caller
    can stop timing or monitoring while the ranks do post-run work (such as
    verifying loaded bytes). ``join`` waits for the ranks to exit and re-raises
    a rank failure; ``close`` terminates any remaining workers and shuts down
    the SyncManager idempotently.
    """

    def __init__(
        self,
        durations,
        context,
        manager,
        *,
        channel=None,
        sync_key: str | None = None,
    ):
        self.durations = durations
        self.sync_key = sync_key
        self._context = context
        self._manager = manager
        self._channel = channel
        self._join_key = f"{sync_key}/joined" if sync_key else None
        self._scope = nodes.NodeChannel._scope(sync_key) if sync_key else None
        self._closed = False

    def allgather_stats(self, local_stats: list) -> list:
        """Allgather per-rank stats across nodes when running multi-node."""
        if self._channel is not None and self.sync_key is not None:
            return self._channel.allgather(f"{self.sync_key}/stats", local_stats)
        return local_stats

    def _signal_failure(self, exc: BaseException) -> None:
        if (
            self._channel is not None
            and self._scope is not None
            and not isinstance(exc, nodes.PeerFailed)
        ):
            with contextlib.suppress(Exception):
                self._channel.fail(self._scope, repr(exc))

    def join(self) -> None:
        if self._closed:
            return
        try:
            if self._context is not None:
                while not self._context.join(timeout=1.0 if self._channel else None):
                    if self._channel is not None and self._scope is not None:
                        self._channel.check_failed(self._scope)
        except BaseException as exc:
            spawn.terminate(self._context)
            self._signal_failure(exc)
            raise
        finally:
            self._closed = True
            if self._manager is not None:
                self._manager.shutdown()
        if self._channel is not None and self._join_key is not None:
            self._channel.barrier(self._join_key)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            spawn.terminate(self._context)
        finally:
            if self._manager is not None:
                self._manager.shutdown()
        if self._channel is not None and self._join_key is not None:
            active_exc = sys.exc_info()[1]
            if active_exc is not None:
                self._signal_failure(active_exc)
            else:
                self._channel.barrier(self._join_key)


def _multinode_rank_wrapper(
    local_rank,
    node_rank,
    per_node,
    leader_addr,
    target_fn,
    world_size,
    port,
    prefix,
    params,
    q,
):
    global_rank = node_rank * per_node + local_rank
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["MASTER_ADDR"] = leader_addr
    target_fn(global_rank, world_size, port, prefix, params, q)


def _normalize_local_results(local_results, rounds: int):
    """Rebase per-node perf_counter pairs to the node's earliest round begin.

    Different hosts have unrelated monotonic-clock epochs, so raw timestamps
    cannot be compared across nodes. Every timed round is bracketed by
    ``dist.barrier()``, so rebasing each node's round timestamps by that node's
    earliest start preserves intra-node skew while remaining epoch-invariant.
    """
    if not local_results or all(r is None for r in local_results):
        return local_results
    normalized = [[] for _ in local_results]
    for r in range(rounds):
        min_begin = min(res[r][0] for res in local_results)
        for i, res in enumerate(local_results):
            normalized[i].append([res[r][0] - min_begin, res[r][1] - min_begin])
    return normalized


def local_nprocs_for_world_size(
    world_size: int, node_group: nodes.NodeGroup | None = None
) -> int:
    """Return how many local ranks this node spawns for ``world_size``."""
    if node_group is None:
        if accelerator.device_kind() == "cpu":
            return min(world_size, MAX_RANKS)
        return world_size
    per_node = placement.devices_per_node()
    if world_size <= per_node:
        return world_size if node_group.is_leader else 0
    if world_size % per_node != 0:
        raise ValueError(
            f"world_size {world_size} must be a multiple of "
            f"devices_per_node {per_node} for multi-node runs"
        )
    active_nodes = world_size // per_node
    if active_nodes > node_group.num_nodes:
        raise ValueError(
            f"world_size {world_size} needs {active_nodes} nodes; "
            f"job has {node_group.num_nodes}"
        )
    return per_node if node_group.node_rank < active_nodes else 0


def _spawn_split_multinode(
    ch, prefix, params, target_fn, world_size_override=None
) -> SpawnedRanks:
    case_name = getattr(params, "name", "case")
    key = ch.next_key(case_name)
    world_size = (
        world_size_override if world_size_override is not None else params.world_size
    )
    per_node = placement.devices_per_node()
    local_nprocs = local_nprocs_for_world_size(world_size, ch.group)

    if local_nprocs > 0 and accelerator.device_kind() != "cpu":
        local_devices = accelerator.local_device_count()
        if local_nprocs > local_devices:
            raise ValueError(
                f"local ranks {local_nprocs} exceeds {local_devices} devices"
            )

    manager = None
    context = None
    try:
        if ch.group.is_leader:
            port = find_free_port()
            ch.publish(f"{key}/port", str(port))
        else:
            port = int(ch.get(f"{key}/port"))

        local_results = []
        if local_nprocs > 0:
            local_results, context, manager = spawn.spawn_local_ranks(
                _multinode_rank_wrapper,
                lambda q: (
                    ch.group.node_rank,
                    per_node,
                    ch.group.leader_addr,
                    target_fn,
                    world_size,
                    port,
                    prefix,
                    params,
                    q,
                ),
                local_nprocs,
                world_size=world_size,
                channel=ch,
                scope=case_name,
            )

        normalized = _normalize_local_results(local_results, params.rounds)
        all_results = ch.allgather(f"{key}/results", normalized)
    except BaseException as exc:
        try:
            spawn.terminate(context)
        finally:
            if manager is not None:
                manager.shutdown()
        if not isinstance(exc, nodes.PeerFailed):
            with contextlib.suppress(Exception):
                ch.fail(case_name, repr(exc))
        raise

    return SpawnedRanks(
        reduce_durations(all_results, world_size, params.rounds),
        context,
        manager,
        channel=ch,
        sync_key=key,
    )


def spawn_split(prefix, params, target_fn, world_size_override=None) -> SpawnedRanks:
    """Spawns the ranks and returns once every rank has put its result."""
    ch = nodes.channel()
    if ch is not None:
        return _spawn_split_multinode(
            ch, prefix, params, target_fn, world_size_override
        )

    world_size = (
        world_size_override if world_size_override is not None else params.world_size
    )
    if accelerator.device_kind() == "cpu":
        world_size = min(world_size, MAX_RANKS)
    else:
        local_devices = accelerator.local_device_count()
        if world_size > local_devices:
            raise ValueError(f"world_size {world_size} exceeds {local_devices} devices")
    port = find_free_port()
    results, context, manager = spawn.spawn_local_ranks(
        target_fn,
        lambda q: (world_size, port, prefix, params, q),
        world_size,
        world_size=world_size,
    )
    return SpawnedRanks(
        reduce_durations(results, world_size, params.rounds), context, manager
    )


def reduce_durations(results, world_size, rounds) -> list[float]:
    if all(r is None for r in results):
        return []

    # We reduce across ranks for each round.
    durations = []
    for r in range(rounds):
        begins = [results[rank][r][0] for rank in range(world_size)]
        ends = [results[rank][r][1] for rank in range(world_size)]
        durations.append(max(ends) - min(begins))
    return durations


_reduce = reduce_durations


_LOCAL_MASTER_ADDRS = frozenset({"localhost", "127.0.0.1", "::1"})


@functools.lru_cache(maxsize=1)
def _default_ipv4_interface() -> str | None:
    """Return the interface carrying the default IPv4 route, or ``eth0`` if present."""
    try:
        with open("/proc/net/route") as f:
            next(f, None)
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[1] == "00000000" and parts[0] != "lo":
                    return parts[0]
    except OSError:
        pass
    if os.path.isdir("/sys/class/net/eth0"):
        return "eth0"
    return None


def setup_distributed_env(rank: int, world_size: int, port: int):
    """Bind this rank's accelerator and initialize the distributed process group."""
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    master_addr = os.environ.get("MASTER_ADDR") or "localhost"
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(port)
    if master_addr in _LOCAL_MASTER_ADDRS:
        os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    else:
        iface = _default_ipv4_interface()
        if iface:
            os.environ.setdefault("GLOO_SOCKET_IFNAME", iface)
            os.environ.setdefault("NCCL_SOCKET_IFNAME", iface)
        os.environ.setdefault("NCCL_SOCKET_FAMILY", "AF_INET")
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)

    device = accelerator.bind_rank(local_rank, global_rank=rank, world_size=world_size)
    backend = accelerator.process_group_backend(world_size)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device
