import dataclasses
import pathlib

import pytest
import yaml

from gcsfs.tests.perf.subsystembenchmarks.checkpointing.configurator import (
    _model_id_slug,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch import configs
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.configs import (
    PyTorchCheckpointConfigurator,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.layout import Layout


def test_token_matches_existing_ids():
    yaml_path = pathlib.Path(configs.__file__).with_name("configs.yaml")
    raw = yaml.safe_load(yaml_path.read_text())
    common = raw["common"]
    baseline = common["baseline"]
    read_scenario = next(
        s for s in raw["scenarios"] if s["scenario"] == "checkpoint_read"
    )
    rows = [baseline, *({**baseline, **v} for v in read_scenario["variants"])]
    cases = PyTorchCheckpointConfigurator(configs.__file__).generate_cases()

    assert len(rows) == len(cases)
    for row, case in zip(rows, cases):
        slug = _model_id_slug(row.get("model_id", common["model_id"]))
        assert (
            f"load-{slug}-{Layout.from_mapping(row).token()}-reg"
            == case.benchmark_name()
        )


def test_layout_is_frozen_and_orders_mesh_dims():
    layout = Layout(
        strategy="pp_hsdp_tp",
        world_size=16,
        pipeline_parallel_size=2,
        data_parallel_replicate_size=2,
        data_parallel_size=2,
        tensor_parallel_size=2,
    )
    assert layout.mesh_dims() == [
        ("pp", 2),
        ("dp_replicate", 2),
        ("dp_shard", 2),
        ("tp", 2),
    ]
    with pytest.raises(dataclasses.FrozenInstanceError):
        layout.world_size = 8


def test_validate_rejects_bad_mesh_product():
    layout = Layout(
        strategy="fsdp_tp",
        world_size=8,
        data_parallel_size=2,
        tensor_parallel_size=2,
    )
    with pytest.raises(
        ValueError,
        match=r"Mesh product \(dpr=1 \* dp=2 \* tp=2 \* pp=1\) = 4 must equal world_size=8",
    ):
        layout.validate()


def test_validate_rejects_ep_size_mismatch():
    ep_mismatch = Layout(
        strategy="ep",
        world_size=8,
        data_parallel_size=8,
        expert_parallel_size=4,
    )
    with pytest.raises(
        ValueError,
        match="expert_parallel_size=4 must be 8 for strategy 'ep'",
    ):
        ep_mismatch.validate()

    non_ep_mismatch = Layout(
        strategy="fsdp",
        world_size=8,
        data_parallel_size=8,
        expert_parallel_size=2,
    )
    with pytest.raises(
        ValueError,
        match="expert_parallel_size=2 must be 1 for strategy 'fsdp'",
    ):
        non_ep_mismatch.validate()


def test_validate_rejects_unsupported_strategy_and_wrong_dims():
    with pytest.raises(
        ValueError,
        match="Unsupported strategy 'ddp' for framework 'pytorch'",
    ):
        Layout(strategy="ddp", world_size=8, data_parallel_size=8).validate()

    with pytest.raises(
        ValueError,
        match=r"Strategy 'tp' needs non-unit sizes for exactly \('tp',\)",
    ):
        Layout(strategy="tp", world_size=8, data_parallel_size=8).validate()


def test_validate_allows_large_world_size():
    # MAX_RANKS is a host placement concern, not a Layout invariant.
    Layout(strategy="fsdp", world_size=64, data_parallel_size=64).validate()
