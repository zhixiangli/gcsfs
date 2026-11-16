import os
import time

import fsspec
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader, FsspecWriter

from gcsfs.tests.perf.subsystembenchmarks.checkpointing.driver import (
    CheckpointDriver,
    CheckpointResult,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.common import (
    RUN_SEED,
    SETUP_SEED,
    RecordingLoadPlanner,
    build,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.phase import (
    RankContext,
    ensure_leader_store,
    phase_durations,
    run_phase,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.state import (
    expected_bytes_requested,
    get_fqn_state_dict,
    verify_state,
)


def _rank_setup_checkpoint(ctx: RankContext, prefix: str, params) -> None:
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // ctx.local_world_size))
    model, optim = build(
        params, params.source_layout(), SETUP_SEED, device=ctx.device
    )
    state = get_fqn_state_dict(model, optim)
    dcp.save(state, storage_writer=FsspecWriter(prefix))


def _rank_load(ctx: RankContext, prefix: str, params) -> None:
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // ctx.local_world_size))
    model, optim = build(
        params, params.target_layout(), RUN_SEED, device=ctx.device
    )
    state = get_fqn_state_dict(model, optim)
    bytes_expected = expected_bytes_requested(state)
    # Build the fsspec/gcsfs instance and fetch a token before the timed rounds.
    fs, root = fsspec.core.url_to_fs(prefix)
    fs.exists(root)

    durations = []
    for _ in range(params.rounds):
        planner = RecordingLoadPlanner()
        dist.barrier()
        begin = time.perf_counter()
        dcp.load(state, storage_reader=FsspecReader(prefix), planner=planner)
        dist.barrier()
        durations.append((begin, time.perf_counter()))

    # Report before the checks below so the parent can close its timed
    # and monitored window while the ranks verify the loaded bytes.
    ctx.report(
        {
            "durations": durations,
            "read_items": int(planner.read_items),
            "bytes_fetched": int(planner.bytes_fetched),
        }
    )

    # The planner check catches wrong shapes; verify_state catches wrong
    # or missing bytes, which DCP does not verify (no CRC), so a bad
    # reader would otherwise just look faster.
    if planner.bytes_requested != bytes_expected:
        raise RuntimeError(
            f"rank {ctx.rank}: DCP requested {planner.bytes_requested} tensor bytes, "
            f"expected {bytes_expected}"
        )

    bad = verify_state(get_fqn_state_dict(model, optim), SETUP_SEED)
    if bad:
        raise RuntimeError(
            f"rank {ctx.rank}: value mismatch in {len(bad)} tensors, e.g. {bad[0]}"
        )


_SETUP_FN = f"{__name__}:_rank_setup_checkpoint"
_LOAD_FN = f"{__name__}:_rank_load"


class PyTorchCheckpointReadDriver(CheckpointDriver):
    def __init__(self) -> None:
        ensure_leader_store()

    def setup(self, prefix, params):
        run_phase(
            _SETUP_FN,
            params.source_layout().world_size,
            prefix,
            params,
        ).join()

    def run(self, prefix, params):
        phase = run_phase(
            _LOAD_FN,
            params.target_layout().world_size,
            prefix,
            params,
        )
        read_items = [int(r["read_items"]) for r in phase.results]
        bytes_fetched = [int(r["bytes_fetched"]) for r in phase.results]
        total_fetched = sum(bytes_fetched)
        return CheckpointResult(
            durations=phase_durations(phase.results),
            bytes_transferred=total_fetched,
            extra_columns={
                "dcp_read_items_per_rank_max": max(read_items),
                "dcp_read_bytes_fetched_total": total_fetched,
                "dcp_read_bytes_fetched_per_rank_max": max(bytes_fetched),
            },
            verify=phase.join,
        )

    def is_checkpoint_file(self, path: str) -> bool:
        return path.endswith((".distcp", ".metadata"))
