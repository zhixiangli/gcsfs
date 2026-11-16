import dataclasses
import os
from collections.abc import Mapping
from typing import Optional

import pytest

from gcsfs.tests.perf.subsystembenchmarks.checkpointing.configurator import (
    OneFactorCheckpointConfigurator,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.hosts import HostConfig
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.layout import Layout
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.parameters import (
    PyTorchCheckpointParameters,
)


def placement_skip_reason(
    p: PyTorchCheckpointParameters,
    host: Optional[HostConfig] = None,
) -> Optional[str]:
    if host is None:
        host = HostConfig.from_env()
    capacity = host.nnodes * host.nproc_per_node
    needed = p.target_layout().world_size
    if not os.environ.get("GCSFS_SUBSYSTEM_CHECKPOINT_SOURCE_ROOT"):
        needed = max(needed, p.source_layout().world_size)
    if needed > capacity:
        return f"needs {needed} ranks, run has {host.nnodes}x{host.nproc_per_node}"
    return None


def case_params(scenario: str, host: Optional[HostConfig] = None) -> list:
    if host is None:
        host = HostConfig.from_env()
    params = []
    for case in PyTorchCheckpointConfigurator(__file__).generate_cases():
        if case.scenario != scenario:
            continue
        reason = placement_skip_reason(case, host)
        marks = [pytest.mark.skip(reason=reason)] if reason else ()
        params.append(pytest.param(case, id=case.name, marks=marks))
    return params


class PyTorchCheckpointConfigurator(OneFactorCheckpointConfigurator):
    FRAMEWORK = "pytorch"
    PARAMS_CLASS = PyTorchCheckpointParameters

    def build_cases(self, scenario, common_config, include_disabled=False):
        if "variants" in scenario:
            flat = []
            for v in scenario["variants"]:
                if isinstance(v, list):
                    flat.extend(v)
                else:
                    flat.append(v)
            scenario = {**scenario, "variants": flat}
        return super().build_cases(scenario, common_config, include_disabled)

    def validate_case(self, p):
        super().validate_case(p)
        if p.source is not None:
            if not isinstance(p.source, Mapping):
                raise ValueError(
                    f"source layout must be a mapping, got {type(p.source).__name__}"
                )
            allowed = {f.name for f in dataclasses.fields(Layout)}
            unknown = sorted(set(p.source) - allowed)
            if unknown:
                raise ValueError(f"Unknown keys in source layout: {unknown}")
            missing = sorted({"strategy", "world_size"} - set(p.source))
            if missing:
                raise ValueError(f"Missing required keys in source layout: {missing}")
        p.target_layout().validate()
        p.source_layout().validate()
