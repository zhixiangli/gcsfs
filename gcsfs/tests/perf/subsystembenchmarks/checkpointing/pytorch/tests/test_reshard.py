import contextlib
import dataclasses

import fsspec
import pytest
from torch.distributed.checkpoint.fsspec_filesystem import FsspecReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

pytest.importorskip("transformers")

from gcsfs.tests.perf.subsystembenchmarks.checkpointing import checkpoint_case
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.read.driver import (
    PyTorchCheckpointReadDriver,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.tests.test_parallelize import (
    _create_tiny_llama_config_dir,
    _create_tiny_olmoe_config_dir,
    _params,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.tests.test_checkpoint_case import (
    _Bench,
    _Monitor,
)

# All 11 spec reshard pairs (Spec §2 / configs.yaml), scaled to fit within 8 ranks
# (ws16 -> ws8, ws8 -> ws4, ws4 -> ws2).
_RESHARD_PAIRS = [
    pytest.param(
        "llama",
        dict(strategy="fsdp", data_parallel_size=4, world_size=4),
        dict(strategy="fsdp", dp=2, world_size=2),
        id="fsdp-shrink",
    ),
    pytest.param(
        "llama",
        dict(strategy="fsdp", data_parallel_size=4, world_size=4),
        dict(strategy="fsdp", dp=8, world_size=8),
        id="fsdp-grow",
    ),
    pytest.param(
        "llama",
        dict(strategy="fsdp", data_parallel_size=4, world_size=4),
        dict(strategy="hsdp", dpr=2, dp=4, world_size=8),
        id="fsdp-to-hsdp",
    ),
    pytest.param(
        "llama",
        dict(strategy="tp", tensor_parallel_size=4, world_size=4),
        dict(strategy="tp", tp=2, world_size=2),
        id="tp-shrink",
    ),
    pytest.param(
        "llama",
        dict(strategy="tp", tensor_parallel_size=4, world_size=4),
        dict(strategy="fsdp", dp=4, world_size=4),
        id="tp-to-fsdp",
    ),
    pytest.param(
        "llama",
        dict(
            strategy="fsdp_tp",
            data_parallel_size=2,
            tensor_parallel_size=4,
            world_size=8,
        ),
        dict(strategy="fsdp_tp", dp=4, tp=2, world_size=8),
        id="fsdp_tp-reshape",
    ),
    pytest.param(
        "llama",
        dict(
            strategy="fsdp_tp",
            data_parallel_size=2,
            tensor_parallel_size=2,
            world_size=4,
        ),
        dict(strategy="fsdp", dp=4, world_size=4),
        id="fsdp_tp-to-fsdp",
    ),
    # Scaled down from the 8->16 hsdp_tp grow in configs.yaml: at 4 ranks
    # hsdp_tp (3 non-unit dims >= 8 ranks) is represented by fsdp_tp (dp=2, tp=2).
    pytest.param(
        "llama",
        dict(
            strategy="fsdp_tp",
            data_parallel_size=2,
            tensor_parallel_size=2,
            world_size=4,
        ),
        dict(strategy="hsdp_tp", dpr=2, dp=2, tp=2, world_size=8),
        id="hsdp_tp-grow",
    ),
    pytest.param(
        "llama",
        dict(
            strategy="pp_fsdp_tp",
            pipeline_parallel_size=2,
            data_parallel_size=2,
            tensor_parallel_size=2,
            world_size=8,
        ),
        dict(strategy="fsdp_tp", dp=2, tp=2, world_size=4),
        id="pp_fsdp_tp-to-fsdp_tp",
    ),
    pytest.param(
        "olmoe",
        dict(
            strategy="ep",
            data_parallel_size=4,
            expert_parallel_size=4,
            world_size=4,
        ),
        dict(strategy="ep", dp=2, ep=2, world_size=2),
        id="ep-shrink",
    ),
    pytest.param(
        "olmoe",
        dict(
            strategy="ep",
            data_parallel_size=4,
            expert_parallel_size=4,
            world_size=4,
        ),
        dict(strategy="fsdp", dp=4, world_size=4),
        id="ep-to-fsdp",
    ),
]


def _assert_saved_source_layout(prefix: str, params) -> None:
    """Verifies that setup() saved the checkpoint using params.source_layout()."""
    src = params.source_layout()
    fs, root = fsspec.core.url_to_fs(prefix)
    distcp_files = [f for f in fs.find(root) if f.endswith(".distcp")]
    assert len(distcp_files) == src.world_size

    md = FsspecReader(prefix).read_metadata().state_dict_metadata
    down_key = "model.model.layers.0.mlp.experts.down_proj"
    if down_key in md:
        down = md[down_key]
        assert down.chunks[0].sizes[0] == down.size[0] // src.expert_parallel_size
    else:
        o_proj = md["model.model.layers.0.self_attn.o_proj.weight"]
        assert len(o_proj.chunks) == src.data_parallel_size * src.tensor_parallel_size
        assert tuple(o_proj.chunks[0].sizes) == (
            o_proj.size[0] // src.data_parallel_size,
            o_proj.size[1] // src.tensor_parallel_size,
        )


@pytest.mark.parametrize("model_kind, source, target", _RESHARD_PAIRS)
def test_reshard_roundtrip(tmp_path, model_kind, source, target):
    if model_kind == "olmoe":
        config_dir = _create_tiny_olmoe_config_dir(tmp_path, num_experts=16)
    else:
        config_dir = _create_tiny_llama_config_dir(tmp_path, num_layers=8)

    params = dataclasses.replace(_params(config_dir, **target), source=dict(source))
    params.source_layout().validate()
    params.target_layout().validate()
    assert params.source_layout() != params.target_layout()

    prefix = (tmp_path / "ckpt").as_uri()
    driver = PyTorchCheckpointReadDriver()

    driver.setup(prefix, params)
    _assert_saved_source_layout(prefix, params)

    result = driver.run(prefix, params)
    result.verify()
    assert len(result.durations) == params.rounds


class _LocalReadDriver(PyTorchCheckpointReadDriver):
    """Redirects gs:// case-bucket prefixes to a local file:// URI for unit testing."""

    def __init__(self, local_prefix: str):
        self.local_prefix = local_prefix

    def setup(self, prefix, params):
        super().setup(self.local_prefix, params)

    def run(self, prefix, params):
        return super().run(self.local_prefix, params)


def test_reshard_tp_to_fsdp_amplification(tmp_path, monkeypatch):
    # Tiny Llama constants (matches _create_tiny_llama_config_dir).
    num_layers = 8
    hidden = 64
    intermediate = 128
    vocab = 256
    elem_bytes = 2  # bfloat16
    tensors_per_param = 3  # weight, exp_avg, exp_avg_sq
    world_size = 4

    config_dir = _create_tiny_llama_config_dir(tmp_path, num_layers=num_layers)
    params = dataclasses.replace(
        _params(config_dir, strategy="fsdp", dp=world_size, world_size=world_size),
        source={
            "strategy": "tp",
            "tensor_parallel_size": world_size,
            "world_size": world_size,
        },
    )

    ckpt_dir = tmp_path / "ckpt"
    local_prefix = ckpt_dir.as_uri()

    monkeypatch.setattr(checkpoint_case, "assert_fsspec_gcsfs", lambda p: None)
    monkeypatch.setattr(
        checkpoint_case.rcu, "assert_storage_class", lambda prefix, bt: None
    )
    monkeypatch.setattr(checkpoint_case.rcu, "warm_if_needed", lambda prefix, bt: None)

    orig_url_to_fs = fsspec.core.url_to_fs

    def mock_url_to_fs(url, **kwargs):
        if url.startswith("gs://"):
            return orig_url_to_fs(local_prefix, **kwargs)
        return orig_url_to_fs(url, **kwargs)

    monkeypatch.setattr(fsspec.core, "url_to_fs", mock_url_to_fs)

    @contextlib.contextmanager
    def bucket_ctx(spec, case_id, **kw):
        yield "test-bucket"

    bench = _Bench()
    driver = _LocalReadDriver(local_prefix)
    checkpoint_case.run_checkpoint_case(
        bench, _Monitor(), params, driver, bucket_ctx=bucket_ctx
    )

    # Hand-computed logical tensor bytes from tiny Llama parameter shapes:
    # - Colwise under TP (Shard(0), aligned with FSDP Shard(0)):
    #   embed_tokens (V*H), lm_head (V*H), and per-layer q/k/v (3*H*H) +
    #   gate/up (2*I*H).
    # - Rowwise under TP (Shard(1), orthogonal to FSDP Shard(0)):
    #   per-layer o_proj (H*H) + down_proj (H*I).
    # - Replicated norms under TP:
    #   (2*L + 1) layernorm vectors of length H.
    expected_colwise_tensor_bytes = (
        tensors_per_param
        * elem_bytes
        * (
            2 * vocab * hidden
            + num_layers * (3 * hidden * hidden + 2 * intermediate * hidden)
        )
    )
    expected_rowwise_tensor_bytes = (
        tensors_per_param
        * elem_bytes
        * num_layers
        * (hidden * hidden + hidden * intermediate)
    )
    expected_norm_tensor_bytes = (
        tensors_per_param * elem_bytes * (2 * num_layers + 1) * hidden
    )

    metadata = FsspecReader(local_prefix).read_metadata()
    colwise_names = (
        "embed_tokens",
        "lm_head",
        "q_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
    )
    rowwise_names = ("o_proj", "down_proj")

    colwise_tensor_bytes = 0
    rowwise_tensor_bytes = 0
    norm_tensor_bytes = 0
    for fqn, item_md in metadata.state_dict_metadata.items():
        if not isinstance(item_md, TensorStorageMetadata) or fqn.endswith(".step"):
            continue
        nbytes = sum(
            c.sizes.numel() * item_md.properties.dtype.itemsize for c in item_md.chunks
        )
        if any(name in fqn for name in rowwise_names):
            assert len(item_md.chunks) == world_size
            assert item_md.chunks[0].sizes == (
                item_md.size[0],
                item_md.size[1] // world_size,
            )
            rowwise_tensor_bytes += nbytes
        elif any(name in fqn for name in colwise_names):
            assert len(item_md.chunks) == world_size
            assert item_md.chunks[0].sizes == (
                item_md.size[0] // world_size,
                item_md.size[1],
            )
            colwise_tensor_bytes += nbytes
        else:
            assert len(item_md.chunks) == 1
            norm_tensor_bytes += nbytes

    assert colwise_tensor_bytes == expected_colwise_tensor_bytes
    assert rowwise_tensor_bytes == expected_rowwise_tensor_bytes
    assert norm_tensor_bytes == expected_norm_tensor_bytes

    # Storage bytes fetched across all W FSDP ranks when loading a TP-W checkpoint:
    # - Each saved chunk of a colwise (Shard(0)) tensor is fetched by 1 FSDP rank.
    # - Each saved chunk of a rowwise (Shard(1)) tensor overlaps every FSDP rank's
    #   row slice and is fetched in full by all W FSDP ranks.
    # - Each deduplicated replicated item (norms, step, BYTE_IO param_groups) is
    #   fetched by all W FSDP ranks.
    colwise_storage_bytes = 0
    rowwise_storage_bytes = 0
    replicated_storage_bytes = 0
    for idx, info in metadata.storage_data.items():
        fqn = idx.fqn
        if (
            not fqn.endswith(".step")
            and not fqn.startswith("optim.param_groups.")
            and any(name in fqn for name in rowwise_names)
        ):
            rowwise_storage_bytes += info.length
        elif (
            not fqn.endswith(".step")
            and not fqn.startswith("optim.param_groups.")
            and any(name in fqn for name in colwise_names)
        ):
            colwise_storage_bytes += info.length
        else:
            replicated_storage_bytes += info.length

    expected_fetched_total = 1 * colwise_storage_bytes + world_size * (
        rowwise_storage_bytes + replicated_storage_bytes
    )
    physical_size = sum(
        p.stat().st_size
        for p in ckpt_dir.iterdir()
        if driver.is_checkpoint_file(p.name)
    )
    expected_amplification = expected_fetched_total / physical_size

    assert bench.extra_info["checkpoint_physical_size_bytes"] == physical_size
    assert bench.extra_info["dcp_read_bytes_fetched_total"] == expected_fetched_total
    assert bench.extra_info["dcp_read_amplification_planned"] == pytest.approx(
        expected_amplification
    )
    assert 1.0 < bench.extra_info["dcp_read_amplification_planned"] < world_size
