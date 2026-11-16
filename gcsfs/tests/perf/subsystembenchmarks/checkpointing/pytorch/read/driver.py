import functools
import json
import os
import tempfile
import time

import fsspec
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader, FsspecWriter

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
    build,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.state import (
    expected_bytes_requested,
    get_fqn_state_dict,
    verify_state,
)


def _rank_setup_checkpoint(rank, world_size, port, prefix, params, q):
    setup_distributed_env(rank, world_size, port)
    try:
        torch.set_num_threads(max(1, (os.cpu_count() or 1) // world_size))
        model, optim = build(params, params.target_layout(), SETUP_SEED)
        state = get_fqn_state_dict(model, optim)
        dcp.save(state, storage_writer=FsspecWriter(prefix))
        q.put(None)
    finally:
        dist.destroy_process_group()


def _rank_load(rank, world_size, port, prefix, params, q, *, stats_dir):
    setup_distributed_env(rank, world_size, port)
    try:
        torch.set_num_threads(max(1, (os.cpu_count() or 1) // world_size))
        model, optim = build(params, params.target_layout(), RUN_SEED)
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

        with open(os.path.join(stats_dir, f"rank{rank}.json"), "w") as f:
            json.dump(
                {
                    "read_items": planner.read_items,
                    "bytes_fetched": planner.bytes_fetched,
                },
                f,
            )
        # Report before the checks below so the parent can close its timed
        # and monitored window while the ranks verify the loaded bytes.
        q.put(durations)

        # The planner check catches wrong shapes; verify_state catches wrong
        # or missing bytes, which DCP does not verify (no CRC), so a bad
        # reader would otherwise just look faster.
        if planner.bytes_requested != bytes_expected:
            raise RuntimeError(
                f"rank {rank}: DCP requested {planner.bytes_requested} tensor bytes, "
                f"expected {bytes_expected}"
            )

        bad = verify_state(get_fqn_state_dict(model, optim), SETUP_SEED)
        if bad:
            raise RuntimeError(
                f"rank {rank}: value mismatch in {len(bad)} tensors, e.g. {bad[0]}"
            )
    finally:
        dist.destroy_process_group()


class PyTorchCheckpointReadDriver(CheckpointDriver):
    def setup(self, prefix, params):
        run_split(prefix, params, _rank_setup_checkpoint)

    def run(self, prefix, params):
        with tempfile.TemporaryDirectory() as stats_dir:
            ranks = spawn_split(
                prefix, params, functools.partial(_rank_load, stats_dir=stats_dir)
            )
            read_items = []
            bytes_fetched = []
            for name in os.listdir(stats_dir):
                with open(os.path.join(stats_dir, name)) as f:
                    data = json.load(f)
                    read_items.append(int(data["read_items"]))
                    bytes_fetched.append(int(data["bytes_fetched"]))

        fs, fs_path = fsspec.core.url_to_fs(prefix)
        all_files = fs.find(fs_path, detail=True)
        physical_size = sum(
            info["size"]
            for path, info in all_files.items()
            if self.is_checkpoint_file(path)
        )
        total_fetched = sum(bytes_fetched)
        amplification = total_fetched / physical_size if physical_size > 0 else 0.0

        return CheckpointResult(
            durations=ranks.durations,
            bytes_transferred=total_fetched,
            extra_columns={
                "dcp_read_items_per_rank_max": max(read_items),
                "dcp_read_bytes_fetched_total": total_fetched,
                "dcp_read_bytes_fetched_per_rank_max": max(bytes_fetched),
                "dcp_read_amplification_planned": amplification,
            },
            verify=ranks.join,
        )

    def is_checkpoint_file(self, path: str) -> bool:
        return path.endswith((".distcp", ".metadata"))
