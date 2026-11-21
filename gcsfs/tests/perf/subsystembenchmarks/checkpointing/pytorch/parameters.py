import dataclasses

from gcsfs.tests.perf.subsystembenchmarks.checkpointing.configurator import (
    _BUCKET,
    CheckpointParameters,
    _model_id_slug,
)
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

    def extra_columns(self):
        cols = super().extra_columns()
        cols.update(
            {
                "data_parallel_replicate_size": self.data_parallel_replicate_size,
                "pipeline_parallel_size": self.pipeline_parallel_size,
                "expert_parallel_size": self.expert_parallel_size,
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

    def mesh_dims(self) -> list[tuple[str, int]]:
        return self.target_layout().mesh_dims()

    def benchmark_name(self):
        op = "load" if "read" in self.scenario else "save"
        return "-".join(
            [
                op,
                _model_id_slug(self.model_id),
                self.target_layout().token(),
                _BUCKET[self.bucket_type],
            ]
        )
