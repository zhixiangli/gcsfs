import dataclasses
import os
from collections.abc import Mapping
from typing import Optional

from gcsfs.tests.perf.subsystembenchmarks.checkpointing.configurator import (
    _BUCKET,
    CheckpointParameters,
    _model_id_slug,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.hosts import HostConfig
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.layout import (
    STRATEGY_MESH_DIMS,
    Layout,
)

__all__ = ["STRATEGY_MESH_DIMS", "PyTorchCheckpointParameters"]


@dataclasses.dataclass
class PyTorchCheckpointParameters(CheckpointParameters):
    """Parameters for a native PyTorch checkpoint benchmark case."""

    data_parallel_replicate_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1
    source: Optional[dict] = None
    source_device: Optional[str] = None

    def __post_init__(self):
        if (
            isinstance(self.source, Mapping)
            and "strategy" in self.source
            and "world_size" in self.source
        ):
            src = self.source_layout()
            self.setup_world_size = src.world_size
            self.setup_tensor_parallel_size = src.tensor_parallel_size
            self.setup_data_parallel_size = src.data_parallel_size
        super().__post_init__()

    def extra_columns(self):
        cols = super().extra_columns()
        src = self.source_layout()
        host = HostConfig.from_env()
        source_device = self.source_device
        if source_device is None:
            source_root = os.environ.get("GCSFS_SUBSYSTEM_CHECKPOINT_SOURCE_ROOT")
            if source_root:
                from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.handoff import (
                    read_manifest,
                )

                manifest = read_manifest(source_root, self.source_key())
                if manifest is not None:
                    source_device = manifest.device
        if source_device is None:
            source_device = host.device

        cols.update(
            {
                "device": host.device,
                "nnodes": host.nnodes,
                "source_device": source_device,
                "data_parallel_replicate_size": self.data_parallel_replicate_size,
                "pipeline_parallel_size": self.pipeline_parallel_size,
                "expert_parallel_size": self.expert_parallel_size,
                "source_strategy": src.strategy,
                "setup_world_size": src.world_size,
                "setup_tensor_parallel_size": src.tensor_parallel_size,
                "setup_data_parallel_size": src.data_parallel_size,
                "setup_data_parallel_replicate_size": src.data_parallel_replicate_size,
                "setup_pipeline_parallel_size": src.pipeline_parallel_size,
                "setup_expert_parallel_size": src.expert_parallel_size,
            }
        )
        return cols

    def target_layout(self) -> Layout:
        return Layout(
            strategy=self.strategy,
            world_size=self.world_size,
            data_parallel_replicate_size=self.data_parallel_replicate_size,
            data_parallel_size=self.data_parallel_size,
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            expert_parallel_size=self.expert_parallel_size,
        )

    def source_layout(self) -> Layout:
        if self.source is None:
            return self.target_layout()
        return Layout.from_mapping(self.source)

    def source_key(self) -> str:
        return f"{_model_id_slug(self.model_id)}-{self.source_layout().token()}"

    def mesh_dims(self) -> list[tuple[str, int]]:
        return self.target_layout().mesh_dims()

    def benchmark_name(self):
        op = "load" if "read" in self.scenario else "save"
        target = self.target_layout()
        source = self.source_layout()
        parts = [
            op,
            _model_id_slug(self.model_id),
            target.token(),
        ]
        if source != target:
            parts.append(f"from-{source.token()}")
        parts.append(_BUCKET[self.bucket_type])
        return "-".join(parts)
