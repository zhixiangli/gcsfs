from gcsfs.tests.perf.subsystembenchmarks.checkpointing._dist import MAX_RANKS
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.configurator import (
    OneFactorCheckpointConfigurator,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.parameters import (
    PyTorchCheckpointParameters,
)


class PyTorchCheckpointConfigurator(OneFactorCheckpointConfigurator):
    FRAMEWORK = "pytorch"
    PARAMS_CLASS = PyTorchCheckpointParameters

    def validate_case(self, p):
        super().validate_case(p)
        p.target_layout().validate()
        # run_split clamps to MAX_RANKS; fail here rather than in
        # init_device_mesh after the per-case bucket exists.
        if p.world_size > MAX_RANKS:
            raise ValueError(
                f"world_size={p.world_size} exceeds MAX_RANKS={MAX_RANKS}"
            )
