import functools
import json
import os
import tempfile
import time

import fsspec
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.fsspec_filesystem import FsspecReader, FsspecWriter

from gcsfs.tests.perf.subsystembenchmarks.checkpointing._dist import (
    run_split,
    setup_distributed_env,
    spawn_split,
)
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
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.state import (
    expected_bytes_requested,
    get_fqn_state_dict,
    verify_state,
)


def _rank_save(rank, world_size, port, prefix, params, q, *, stats_dir):
    setup_distributed_env(rank, world_size, port)
    try:
        torch.set_num_threads(max(1, (os.cpu_count() or 1) // world_size))
        model, optim = build(params, params.target_layout(), SETUP_SEED)
        state = get_fqn_state_dict(model, optim)
        # Build the fsspec/gcsfs instance and fetch a token before the timed rounds.
        fs, root = fsspec.core.url_to_fs(prefix)
        fs.exists(root)

        durations = []
        for _ in range(params.rounds):
            if rank == 0:
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

        with open(os.path.join(stats_dir, f"rank{rank}.json"), "w") as f:
            json.dump({"bytes_written": planner.bytes_written}, f)
        q.put(durations)
    finally:
        dist.destroy_process_group()


def _rank_verify_saved(rank, world_size, port, prefix, params, q):
    setup_distributed_env(rank, world_size, port)
    try:
        torch.set_num_threads(max(1, (os.cpu_count() or 1) // world_size))
        model, optim = build(params, params.target_layout(), RUN_SEED)
        state = get_fqn_state_dict(model, optim)
        bytes_expected = expected_bytes_requested(state)
        planner = RecordingLoadPlanner()
        dcp.load(state, storage_reader=FsspecReader(prefix), planner=planner)
        q.put(None)

        if planner.bytes_requested != bytes_expected:
            raise RuntimeError(
                f"rank {rank}: DCP requested {planner.bytes_requested} tensor bytes, "
                f"expected {bytes_expected}"
            )

        bad = verify_state(state, SETUP_SEED)
        if bad:
            raise RuntimeError(
                f"rank {rank}: value mismatch in {len(bad)} entries, e.g. {bad[0]}"
            )
    finally:
        dist.destroy_process_group()


class PyTorchCheckpointWriteDriver(CheckpointDriver):
    def setup(self, prefix, params):
        pass

    def run(self, prefix, params):
        with tempfile.TemporaryDirectory() as stats_dir:
            ranks = spawn_split(
                prefix, params, functools.partial(_rank_save, stats_dir=stats_dir)
            )
            bytes_written = []
            for name in os.listdir(stats_dir):
                with open(os.path.join(stats_dir, name)) as f:
                    data = json.load(f)
                    bytes_written.append(int(data["bytes_written"]))
            ranks.join()

        def _verify():
            run_split(
                prefix,
                params,
                _rank_verify_saved,
                world_size_override=params.target_layout().world_size,
            )

        return CheckpointResult(
            durations=ranks.durations,
            extra_columns={
                "dcp_write_bytes_per_rank_max": max(bytes_written),
            },
            verify=_verify,
        )

    def is_checkpoint_file(self, path: str) -> bool:
        return path.endswith((".distcp", ".metadata"))
