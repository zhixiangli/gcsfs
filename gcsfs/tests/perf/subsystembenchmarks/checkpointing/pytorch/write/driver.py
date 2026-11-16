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
    RecordingSavePlanner,
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


def _rank_save(ctx: RankContext, prefix: str, params) -> None:
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // ctx.local_world_size))
    model, optim = build(
        params, params.target_layout(), SETUP_SEED, device=ctx.device
    )
    state = get_fqn_state_dict(model, optim)
    # Build the fsspec/gcsfs instance and fetch a token before the timed rounds.
    fs, root = fsspec.core.url_to_fs(prefix)
    fs.exists(root)

    durations = []
    for _ in range(params.rounds):
        if ctx.rank == 0:
            fs.invalidate_cache()
            if fs.exists(root):
                fs.rm(root, recursive=True)
        planner = RecordingSavePlanner()
        dist.barrier()
        fs.invalidate_cache()
        begin = time.perf_counter()
        dcp.save(state, storage_writer=FsspecWriter(prefix), planner=planner)
        dist.barrier()
        durations.append((begin, time.perf_counter()))

    ctx.report(
        {
            "durations": durations,
            "bytes_written": int(planner.bytes_written),
        }
    )


def _rank_verify_saved(ctx: RankContext, prefix: str, params) -> None:
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // ctx.local_world_size))
    model, optim = build(
        params, params.target_layout(), RUN_SEED, device=ctx.device
    )
    state = get_fqn_state_dict(model, optim)
    bytes_expected = expected_bytes_requested(state)
    planner = RecordingLoadPlanner()
    dcp.load(state, storage_reader=FsspecReader(prefix), planner=planner)
    ctx.report({})

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


_SAVE_FN = f"{__name__}:_rank_save"
_VERIFY_FN = f"{__name__}:_rank_verify_saved"


class PyTorchCheckpointWriteDriver(CheckpointDriver):
    def __init__(self) -> None:
        ensure_leader_store()

    def setup(self, prefix, params):
        pass

    def run(self, prefix, params):
        phase = run_phase(
            _SAVE_FN,
            params.target_layout().world_size,
            prefix,
            params,
        )
        bytes_written = [int(r["bytes_written"]) for r in phase.results]
        phase.join()

        def _verify():
            run_phase(
                _VERIFY_FN,
                params.target_layout().world_size,
                prefix,
                params,
            ).join()

        return CheckpointResult(
            durations=phase_durations(phase.results),
            extra_columns={
                "dcp_write_bytes_per_rank_max": max(bytes_written),
            },
            verify=_verify,
        )

    def is_checkpoint_file(self, path: str) -> bool:
        return path.endswith((".distcp", ".metadata"))
