import math

import torch
import torch.nn as nn
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.checkpoint.planner import LoadItemType, LoadPlan, SavePlan

from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.layout import Layout
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.model import (
    build_model,
    build_optimizer,
    materialize,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.parallelize import (
    build_mesh,
    parallelize,
)

# Run seeds differently from setup so every tensor starts with known-wrong
# bytes; a reader that skips items would otherwise pass verify_state.
SETUP_SEED = 0
RUN_SEED = 1
_CPU = torch.device("cpu")


def build(
    params,
    layout: Layout,
    seed: int,
    device: torch.device | str = _CPU,
) -> tuple[nn.Module, torch.optim.AdamW]:
    dev = torch.device(device)
    mesh = build_mesh(layout, device_type=dev.type)
    model = parallelize(build_model(params), mesh, layout)
    materialize(model, seed, device=dev)
    return model, build_optimizer(model, seed)


class RecordingLoadPlanner(DefaultLoadPlanner):
    """Records item count, logical tensor bytes, and storage bytes fetched on this rank."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read_items: int = 0
        self.bytes_requested: int = 0
        self.bytes_fetched: int = 0

    def _record(self, plan: LoadPlan) -> LoadPlan:
        md = self.metadata.state_dict_metadata if self.metadata is not None else {}
        storage = (
            self.metadata.storage_data
            if self.metadata is not None and self.metadata.storage_data is not None
            else {}
        )
        self.read_items = len(plan.items)
        self.bytes_requested = sum(
            math.prod(item.lengths) * md[item.dest_index.fqn].properties.dtype.itemsize
            for item in plan.items
            if item.type == LoadItemType.TENSOR and item.dest_index.fqn in md
        )
        storage_indices = {
            item.storage_index
            for item in plan.items
            if item.type in (LoadItemType.TENSOR, LoadItemType.BYTE_IO)
        }
        self.bytes_fetched = sum(
            storage[idx].length for idx in storage_indices if idx in storage
        )
        return plan

    def create_local_plan(self) -> LoadPlan:
        return self._record(super().create_local_plan())

    def finish_plan(self, new_plan: LoadPlan) -> LoadPlan:
        return self._record(super().finish_plan(new_plan))


class RecordingSavePlanner(DefaultSavePlanner):
    """Records tensor chunk bytes written on this rank."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bytes_written: int = 0

    def finish_plan(self, new_plan: SavePlan) -> SavePlan:
        plan = super().finish_plan(new_plan)
        self.bytes_written = sum(
            math.prod(item.tensor_data.chunk.sizes)
            * item.tensor_data.properties.dtype.itemsize
            for item in plan.items
            if item.tensor_data is not None
        )
        return plan
