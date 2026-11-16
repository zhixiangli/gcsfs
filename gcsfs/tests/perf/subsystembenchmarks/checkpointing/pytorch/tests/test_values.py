import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor
from torch.testing._internal.distributed.fake_pg import FakeStore

pytest.importorskip("transformers")

from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch import values
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
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.state import (
    get_fqn_state_dict,
    verify_state,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.tests.test_parallelize import (
    _create_tiny_llama_config_dir,
    _params,
)


def test_expected_is_deterministic():
    a = values.expected(
        (4, 8),
        (16, 8),
        (4, 0),
        seed=42,
        fqn="model.layers.0.self_attn.q_proj.weight",
    )
    b = values.expected(
        (4, 8),
        (16, 8),
        (4, 0),
        seed=42,
        fqn="model.layers.0.self_attn.q_proj.weight",
    )
    assert a.device.type == "cpu"
    assert a.dtype == torch.bfloat16
    assert a.shape == (4, 8)
    assert torch.equal(a, b)
    key = values.fqn_key(42, "model.layers.0.self_attn.q_proj.weight")
    assert isinstance(key, int)
    assert -(1 << 63) <= key < (1 << 63)
    assert key == values.fqn_key(42, "model.layers.0.self_attn.q_proj.weight")


def test_expected_differs_by_seed_and_fqn():
    base = values.expected((4, 8), (4, 8), (0, 0), seed=0, fqn="w")
    diff_seed = values.expected((4, 8), (4, 8), (0, 0), seed=1, fqn="w")
    diff_fqn = values.expected((4, 8), (4, 8), (0, 0), seed=0, fqn="w.exp_avg")
    diff_fqn2 = values.expected((4, 8), (4, 8), (0, 0), seed=0, fqn="w.exp_avg_sq")
    assert not torch.equal(base, diff_seed)
    assert not torch.equal(base, diff_fqn)
    assert not torch.equal(base, diff_fqn2)
    assert not torch.equal(diff_fqn, diff_fqn2)


def test_no_nan_inf():
    for seed in (0, 1, 999):
        t = values.expected(
            (256, 256), (256, 256), (0, 0), seed=seed, fqn=f"layer.{seed}.weight"
        )
        assert not torch.isnan(t).any()
        assert not torch.isinf(t).any()
        mags = t.to(torch.float32).abs()
        assert (mags >= 2**-7).all()
        assert (mags < 2**-6).all()
        assert (t > 0).any() and (t < 0).any()


def test_shards_assemble_to_same_full_tensor(monkeypatch):
    global_shape = (10, 6)
    seed = 7
    fqn = "model.layers.3.mlp.down_proj.weight"
    full = values.expected(global_shape, global_shape, (0, 0), seed=seed, fqn=fqn)

    # Shard(0) over 4 (uneven: 3, 3, 3, 1 rows at offsets 0, 3, 6, 9)
    from_shard0 = torch.empty_like(full)
    for r0, rows in ((0, 3), (3, 3), (6, 3), (9, 1)):
        shard = values.expected((rows, 6), global_shape, (r0, 0), seed=seed, fqn=fqn)
        from_shard0[r0 : r0 + rows, :] = shard
    assert torch.equal(from_shard0, full)

    # Shard(1) over 2 (3, 3 cols at offsets 0, 3)
    from_shard1 = torch.empty_like(full)
    for c0, cols in ((0, 3), (3, 3)):
        shard = values.expected((10, cols), global_shape, (0, c0), seed=seed, fqn=fqn)
        from_shard1[:, c0 : c0 + cols] = shard
    assert torch.equal(from_shard1, full)

    # Multi-chunk path produces identical global tensor when chunk size is tiny.
    monkeypatch.setattr(values, "_CHUNK_ELEMENTS", 11)
    chunked_full = values.expected(
        global_shape, global_shape, (0, 0), seed=seed, fqn=fqn
    )
    assert torch.equal(chunked_full, full)


def test_dtensor_local_region_uneven():
    full = values.expected((10, 6), (10, 6), (0, 0), seed=13, fqn="uneven.weight")
    row_offsets = []
    local_shapes = []
    shards = []

    for rank in range(4):
        dist.init_process_group("fake", store=FakeStore(), rank=rank, world_size=4)
        try:
            mesh = init_device_mesh("cpu", (4,), mesh_dim_names=("dp_shard",))
            dt = distribute_tensor(
                torch.empty(10, 6, dtype=torch.bfloat16),
                mesh,
                [Shard(0)],
            )
            global_shape, offset = values.local_region(dt)
            assert global_shape == (10, 6)
            assert offset[1] == 0
            row_offsets.append(offset[0])
            local_shapes.append(tuple(dt.to_local().shape))

            values.fill_(dt, seed=13, fqn="uneven.weight")
            shards.append(dt.to_local().clone())
        finally:
            dist.destroy_process_group()

    assert row_offsets == [0, 3, 6, 9]
    assert local_shapes == [(3, 6), (3, 6), (3, 6), (1, 6)]
    assert torch.equal(torch.cat(shards, dim=0), full)


def test_verify_state_names_mismatch(tmp_path):
    params = _params(
        _create_tiny_llama_config_dir(tmp_path), "fsdp", dp=4, world_size=4
    )
    layout = Layout(strategy="fsdp", world_size=4, data_parallel_size=4)

    dist.init_process_group("fake", store=FakeStore(), rank=1, world_size=4)
    try:
        mesh = build_mesh(layout)
        model = parallelize(build_model(params), mesh, layout)
        materialize(model, seed=5)
        optim = build_optimizer(model, seed=5)
        state = get_fqn_state_dict(model, optim)

        assert verify_state(state, seed=5) == []

        q_fqn = "model.layers.0.self_attn.q_proj.weight"
        q_local = state["model"][q_fqn].to_local()
        orig = q_local[0, 0].item()
        with torch.no_grad():
            q_local[0, 0] = -q_local[0, 0]
        assert verify_state(state, seed=5) == [f"model.{q_fqn}"]
        with torch.no_grad():
            q_local[0, 0] = orig
        assert verify_state(state, seed=5) == []

        exp_avg_local = state["optim"][f"state.{q_fqn}.exp_avg"].to_local()
        orig_ea = exp_avg_local[0, 1].item()
        with torch.no_grad():
            exp_avg_local[0, 1] = -exp_avg_local[0, 1]
        assert verify_state(state, seed=5) == [f"optim.state.{q_fqn}.exp_avg"]
        with torch.no_grad():
            exp_avg_local[0, 1] = orig_ea

        step_t = state["optim"][f"state.{q_fqn}.step"]
        step_t.add_(1)
        assert verify_state(state, seed=5) == [f"optim.state.{q_fqn}.step"]
        step_t.sub_(1)
        assert verify_state(state, seed=5) == []
    finally:
        dist.destroy_process_group()


def test_expected_accepts_device_argument():
    cpu_t = values.expected(
        (4, 8), (8, 8), (4, 0), seed=9, fqn="layer.0.weight", device="cpu"
    )
    assert cpu_t.device.type == "cpu"
    assert torch.equal(
        cpu_t,
        values.expected((4, 8), (8, 8), (4, 0), seed=9, fqn="layer.0.weight"),
    )

