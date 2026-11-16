import json
import pathlib

import pytest

from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch import configs
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.configs import (
    HostConfig,
    PyTorchCheckpointConfigurator,
    case_params,
    placement_skip_reason,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.parameters import (
    PyTorchCheckpointParameters,
)

CONFIG = configs.__file__

_EXPECTED_LOAD_NAMES = [
    "load-llama_3_1_8b-fsdp-ws8-reg",
    "load-llama_3_1_8b-hsdp-dpr2dp4-ws8-reg",
    "load-llama_3_1_8b-hsdp-dpr4dp2-ws8-reg",
    "load-llama_3_1_8b-tp-tp8-ws8-reg",
    "load-llama_3_1_8b-fsdp-tp-dp2tp4-ws8-reg",
    "load-llama_3_1_8b-hsdp-tp-dpr2dp2tp2-ws8-reg",
    "load-llama_3_1_8b-pp-pp8-ws8-reg",
    "load-llama_3_1_8b-pp-fsdp-pp2dp4-ws8-reg",
    "load-llama_3_1_8b-pp-fsdp-tp-pp2dp2tp2-ws8-reg",
    "load-llama_3_1_8b-pp-hsdp-tp-pp2dpr2dp2tp2-ws16-reg",
    "load-olmoe_1b_7b_0924-fsdp-ws8-reg",
    "load-olmoe_1b_7b_0924-ep-dp8ep8-ws8-reg",
]

_EXPECTED_RESHARD_NAMES = [
    "load-llama_3_1_8b-fsdp-ws4-from-fsdp-ws8-reg",
    "load-llama_3_1_8b-fsdp-ws16-from-fsdp-ws8-reg",
    "load-llama_3_1_8b-hsdp-dpr2dp8-ws16-from-fsdp-ws8-reg",
    "load-llama_3_1_8b-tp-tp4-ws4-from-tp-tp8-ws8-reg",
    "load-llama_3_1_8b-fsdp-ws8-from-tp-tp8-ws8-reg",
    "load-llama_3_1_8b-fsdp-tp-dp4tp2-ws8-from-fsdp-tp-dp2tp4-ws8-reg",
    "load-llama_3_1_8b-fsdp-ws8-from-fsdp-tp-dp2tp4-ws8-reg",
    "load-llama_3_1_8b-hsdp-tp-dpr2dp4tp2-ws16-from-hsdp-tp-dpr2dp2tp2-ws8-reg",
    "load-llama_3_1_8b-fsdp-tp-dp4tp2-ws8-from-pp-fsdp-tp-pp2dp2tp2-ws8-reg",
    "load-olmoe_1b_7b_0924-ep-dp4ep4-ws4-from-ep-dp8ep8-ws8-reg",
    "load-olmoe_1b_7b_0924-fsdp-ws8-from-ep-dp8ep8-ws8-reg",
]


def _cases():
    return PyTorchCheckpointConfigurator(CONFIG).generate_cases()


def test_existing_load_ids_unchanged():
    read_cases = [
        c
        for c in _cases()
        if c.scenario == "checkpoint_read" and c.sweep_axis != "reshard"
    ]
    assert [c.name for c in read_cases] == _EXPECTED_LOAD_NAMES


def test_reshard_ids():
    reshard_cases = [c for c in _cases() if c.sweep_axis == "reshard"]
    assert [c.name for c in reshard_cases] == _EXPECTED_RESHARD_NAMES


def test_source_equal_to_target_has_no_from_token(tmp_path):
    p = PyTorchCheckpointParameters(
        name="",
        bucket_name="",
        bucket_type="regional",
        rounds=1,
        scenario="checkpoint_read",
        framework="pytorch",
        model_id="gs://gcs-aiml-huggingface-model-weights/Llama-3.1-8B",
        strategy="tp",
        tensor_parallel_size=4,
        data_parallel_size=1,
        world_size=4,
        source={"strategy": "tp", "tensor_parallel_size": 4, "world_size": 4},
    )
    assert p.source_layout() == p.target_layout()
    assert "-from-" not in p.benchmark_name()
    assert p.benchmark_name() == "load-llama_3_1_8b-tp-tp4-ws4-reg"

    # A row whose source equals its own target does not collide with a different
    # baseline layout, and has no -from- token.
    non_colliding = tmp_path / "configs.yaml"
    non_colliding.write_text(
        """
common:
  rounds: 1
  model_id: "gs://gcs-aiml-huggingface-model-weights/Llama-3.1-8B"
  baseline:
    strategy: "fsdp"
    data_parallel_size: 8
    world_size: 8
scenarios:
  - name: "checkpoint_read"
    scenario: "checkpoint_read"
    variants:
      - axis: "reshard"
        strategy: "tp"
        tensor_parallel_size: 4
        data_parallel_size: 1
        world_size: 4
        source: {strategy: "tp", tensor_parallel_size: 4, world_size: 4}
"""
    )
    built = PyTorchCheckpointConfigurator(str(tmp_path / "configs.py")).generate_cases()
    assert [c.name for c in built] == [
        "load-llama_3_1_8b-fsdp-ws8-reg",
        "load-llama_3_1_8b-tp-tp4-ws4-reg",
    ]

    # If it is genuinely the same case as the plain load row, duplicate-id check rejects it.
    colliding = tmp_path / "configs.yaml"
    colliding.write_text(
        """
common:
  rounds: 1
  model_id: "gs://gcs-aiml-huggingface-model-weights/Llama-3.1-8B"
  baseline:
    strategy: "fsdp"
    data_parallel_size: 8
    world_size: 8
scenarios:
  - name: "checkpoint_read"
    scenario: "checkpoint_read"
    variants:
      - axis: "reshard"
        strategy: "fsdp"
        data_parallel_size: 8
        world_size: 8
        source: {strategy: "fsdp", data_parallel_size: 8, world_size: 8}
"""
    )
    with pytest.raises(ValueError, match="two cases share the benchmark id"):
        PyTorchCheckpointConfigurator(str(tmp_path / "configs.py")).generate_cases()


def test_every_reshard_source_has_a_write_row():
    cases = _cases()
    write_keys = {c.source_key() for c in cases if c.scenario == "checkpoint_write"}
    write_layouts = {
        (c.model_id, c.target_layout())
        for c in cases
        if c.scenario == "checkpoint_write"
    }
    reshard_cases = [c for c in cases if c.sweep_axis == "reshard"]
    assert len(reshard_cases) == 11
    for c in reshard_cases:
        assert c.source_key() in write_keys
        assert (c.model_id, c.source_layout()) in write_layouts


def test_write_rows_mirror_read_rows():
    cases = _cases()
    read_cases = [
        c
        for c in cases
        if c.scenario == "checkpoint_read" and c.sweep_axis != "reshard"
    ]
    write_cases = [c for c in cases if c.scenario == "checkpoint_write"]
    assert len(write_cases) == len(read_cases) == 12
    for r, w in zip(read_cases, write_cases):
        assert w.model_id == r.model_id
        assert w.target_layout() == r.target_layout()
        assert w.source_layout() == w.target_layout()
        assert w.sweep_axis == r.sweep_axis
        assert w.name == r.name.replace("load-", "save-", 1)


def test_oversized_case_is_skipped_not_error(monkeypatch):
    # validate_case no longer raises on world_size > MAX_RANKS; host placement skips instead.
    big = PyTorchCheckpointParameters(
        name="c",
        bucket_name="",
        bucket_type="regional",
        rounds=1,
        scenario="checkpoint_read",
        framework="pytorch",
        model_id="m",
        strategy="fsdp",
        data_parallel_size=32,
        world_size=32,
    )
    PyTorchCheckpointConfigurator(CONFIG).validate_case(big)

    host_1x8 = HostConfig(nnodes=1, nproc_per_node=8)
    params = case_params("checkpoint_read", host=host_1x8)
    assert len(params) == 23

    skipped_ids = []
    for param in params:
        case = param.values[0]
        needs_16 = (
            case.target_layout().world_size > 8 or case.source_layout().world_size > 8
        )
        if needs_16:
            skipped_ids.append(param.id)
            assert len(param.marks) == 1
            mark = param.marks[0]
            assert mark.name == "skip"
            assert "needs 16 ranks" in mark.kwargs["reason"]
            assert mark.kwargs["reason"] == "needs 16 ranks, run has 1x8"
        else:
            assert list(param.marks) == []

    assert skipped_ids == [
        "load-llama_3_1_8b-pp-hsdp-tp-pp2dpr2dp2tp2-ws16-reg",
        "load-llama_3_1_8b-fsdp-ws16-from-fsdp-ws8-reg",
        "load-llama_3_1_8b-hsdp-dpr2dp8-ws16-from-fsdp-ws8-reg",
        "load-llama_3_1_8b-hsdp-tp-dpr2dp4tp2-ws16-from-hsdp-tp-dpr2dp2tp2-ws8-reg",
    ]

    # Source world_size is only checked when GCSFS_SUBSYSTEM_CHECKPOINT_SOURCE_ROOT is unset.
    src_16_to_8 = PyTorchCheckpointParameters(
        name="c",
        bucket_name="",
        bucket_type="regional",
        rounds=1,
        scenario="checkpoint_read",
        framework="pytorch",
        model_id="gs://gcs-aiml-huggingface-model-weights/Llama-3.1-8B",
        strategy="fsdp",
        data_parallel_size=8,
        world_size=8,
        source={"strategy": "fsdp", "data_parallel_size": 16, "world_size": 16},
    )
    monkeypatch.delenv("GCSFS_SUBSYSTEM_CHECKPOINT_SOURCE_ROOT", raising=False)
    assert (
        placement_skip_reason(src_16_to_8, host_1x8)
        == "needs 16 ranks, run has 1x8"
    )
    monkeypatch.setenv("GCSFS_SUBSYSTEM_CHECKPOINT_SOURCE_ROOT", "gs://b/root")
    assert placement_skip_reason(src_16_to_8, host_1x8) is None


def test_reshard_source_columns():
    reshard_cases = [c for c in _cases() if c.sweep_axis == "reshard"]
    assert len(reshard_cases) == 11
    for c in reshard_cases:
        extra = c.extra_columns()
        src = c.source_layout()
        tgt = c.target_layout()
        assert extra["source_strategy"] == src.strategy
        assert extra["setup_world_size"] == src.world_size
        assert extra["setup_tensor_parallel_size"] == src.tensor_parallel_size
        assert extra["setup_data_parallel_size"] == src.data_parallel_size
        assert (
            extra["setup_data_parallel_replicate_size"]
            == src.data_parallel_replicate_size
        )
        assert extra["setup_pipeline_parallel_size"] == src.pipeline_parallel_size
        assert extra["setup_expert_parallel_size"] == src.expert_parallel_size
        assert extra["world_size"] == tgt.world_size
        assert extra["tensor_parallel_size"] == tgt.tensor_parallel_size
        assert extra["data_parallel_size"] == tgt.data_parallel_size
        assert (
            extra["data_parallel_replicate_size"] == tgt.data_parallel_replicate_size
        )
        assert extra["pipeline_parallel_size"] == tgt.pipeline_parallel_size
        assert extra["expert_parallel_size"] == tgt.expert_parallel_size

    by_name = {c.name: c for c in reshard_cases}
    pp_to_2d = by_name[
        "load-llama_3_1_8b-fsdp-tp-dp4tp2-ws8-from-pp-fsdp-tp-pp2dp2tp2-ws8-reg"
    ]
    cols = pp_to_2d.extra_columns()
    assert cols["source_strategy"] == "pp_fsdp_tp"
    assert cols["setup_pipeline_parallel_size"] == 2
    assert cols["setup_data_parallel_size"] == 2
    assert cols["setup_tensor_parallel_size"] == 2
    assert cols["setup_world_size"] == 8
    assert cols["pipeline_parallel_size"] == 1
    assert cols["data_parallel_size"] == 4
    assert cols["tensor_parallel_size"] == 2


_OLMOE = "gs://gcs-aiml-huggingface-model-weights/OLMoE-1B-7B-0924"


@pytest.mark.parametrize("model_id", [_OLMOE, "gs://b/Llama-3.1-70B"])
def test_model_id_override_skips_rows_that_pin_model_id(monkeypatch, model_id):
    # The OLMoE rows pin their own model_id. With --model-id they are skipped,
    # so the override means "exactly this model" (and OLMoE as the override
    # does not collide with the OLMoE fsdp row on the same benchmark id).
    monkeypatch.setenv("GCSFS_SUBSYSTEM_MODEL_ID", model_id)
    cases = _cases()
    plain_read = [
        c
        for c in cases
        if c.scenario == "checkpoint_read" and c.sweep_axis != "reshard"
    ]
    assert len(plain_read) == 10
    assert {c.model_id for c in cases} == {model_id}
    assert not any("olmoe" in c.name for c in cases if model_id != _OLMOE)


def test_mesh_product_equals_world_size():
    cases = _cases()
    assert len(cases) == 35
    for c in cases:
        for layout in (c.target_layout(), c.source_layout()):
            # expert_parallel_size is not a mesh factor: EP reuses the dp_shard ranks.
            mesh_product = (
                layout.data_parallel_replicate_size
                * layout.data_parallel_size
                * layout.tensor_parallel_size
                * layout.pipeline_parallel_size
            )
            assert mesh_product == layout.world_size
            assert layout.world_size in (4, 8, 16)


@pytest.mark.parametrize(
    "sizes",
    [
        pytest.param(dict(strategy="tp", data_parallel_size=8), id="tp-without-tp"),
        pytest.param(
            dict(strategy="fsdp", data_parallel_replicate_size=2, data_parallel_size=4),
            id="fsdp-with-replicate",
        ),
        pytest.param(
            dict(
                strategy="fsdp",
                data_parallel_size=8,
                source=dict(strategy="tp", data_parallel_size=8, world_size=8),
            ),
            id="bad-source-layout",
        ),
    ],
)
def test_validate_case_rejects_sizes_unused_by_strategy(sizes):
    p = PyTorchCheckpointParameters(
        name="c",
        bucket_name="",
        bucket_type="regional",
        rounds=1,
        scenario="checkpoint_read",
        framework="pytorch",
        model_id="m",
        world_size=8,
        **sizes,
    )
    with pytest.raises(ValueError, match="needs non-unit sizes"):
        PyTorchCheckpointConfigurator(CONFIG).validate_case(p)


def test_baseline_is_fsdp():
    cases = _cases()
    read_baselines = [
        c
        for c in cases
        if c.scenario == "checkpoint_read" and c.sweep_axis == "baseline"
    ]
    assert len(read_baselines) == 1
    baseline = read_baselines[0]
    assert baseline.strategy == "fsdp"
    assert baseline.name == "load-llama_3_1_8b-fsdp-ws8-reg"
    assert baseline.scenario == "checkpoint_read"

    strategy_cases = [
        c
        for c in cases
        if c.scenario == "checkpoint_read" and c.sweep_axis == "strategy"
    ]
    assert len(strategy_cases) == 10
    for c in strategy_cases:
        assert c.scenario == "checkpoint_read"


def test_olmoe_cases():
    olmoe = "gs://gcs-aiml-huggingface-model-weights/OLMoE-1B-7B-0924"
    cases = {c.name: c for c in _cases() if c.model_id == olmoe}
    fsdp = cases["load-olmoe_1b_7b_0924-fsdp-ws8-reg"]
    ep = cases["load-olmoe_1b_7b_0924-ep-dp8ep8-ws8-reg"]
    assert (fsdp.sweep_axis, fsdp.strategy, fsdp.expert_parallel_size) == (
        "model",
        "fsdp",
        1,
    )
    assert (ep.sweep_axis, ep.strategy, ep.expert_parallel_size) == (
        "strategy",
        "ep",
        8,
    )
    for c in (fsdp, ep):
        assert (c.data_parallel_size, c.world_size) == (8, 8)


def test_expert_parallel_size_must_match_dp(tmp_path):
    config = tmp_path / "configs.yaml"
    config.write_text(
        """
common:
  rounds: 1
  model_id: "gs://gcs-aiml-huggingface-model-weights/OLMoE-1B-7B-0924"
  baseline:
    strategy: "ep"
    data_parallel_size: 8
    expert_parallel_size: 4
    world_size: 8
scenarios:
  - name: "checkpoint_read"
    scenario: "checkpoint_read"
"""
    )
    with pytest.raises(ValueError, match="expert_parallel_size"):
        PyTorchCheckpointConfigurator(str(tmp_path / "configs.py")).generate_cases()


def test_extra_columns():
    cases = _cases()
    for c in cases:
        extra = c.extra_columns()
        assert "data_parallel_replicate_size" in extra
        assert "pipeline_parallel_size" in extra
        assert "expert_parallel_size" in extra
        assert "source_strategy" in extra
        assert "setup_data_parallel_replicate_size" in extra
        assert "setup_pipeline_parallel_size" in extra
        assert "setup_expert_parallel_size" in extra
        assert extra["data_parallel_replicate_size"] == c.data_parallel_replicate_size
        assert extra["pipeline_parallel_size"] == c.pipeline_parallel_size
        assert extra["expert_parallel_size"] == c.expert_parallel_size


def test_columns_in_schema():
    root = pathlib.Path(__file__).resolve().parents[7]
    schema_path = (
        root / "cloudbuild/subsystembenchmarks/subsystembenchmarks_schema.json"
    )
    with open(schema_path) as f:
        schema_fields = {f["name"] for f in json.load(f)["schema"]["fields"]}

    cases = _cases()
    for c in cases:
        for col in c.extra_columns():
            assert col in schema_fields, f"param column {col} missing from schema"

    # Columns emitted by PyTorchCheckpointReadDriver.run / run_checkpoint_case
    driver_cols = [
        "dcp_read_items_per_rank_max",
        "dcp_read_bytes_fetched_total",
        "dcp_read_bytes_fetched_per_rank_max",
        "dcp_read_amplification_planned",
    ]
    for col in driver_cols:
        assert col in schema_fields, f"driver column {col} missing from schema"
