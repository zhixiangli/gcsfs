"""Tests for GPU/TPU device placement, Ray actor resources, and TPU env lifecycle in ray_pytorch."""

import contextlib
import dataclasses
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ray")

from gcsfs.tests.perf.subsystembenchmarks._common import (  # noqa: E402
    accelerator,
    ray_runtime,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.ray_pytorch import (  # noqa: E402
    common,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.ray_pytorch.read import (  # noqa: E402
    driver as read_driver,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.ray_pytorch.write import (  # noqa: E402
    driver as write_driver,
)


def _dummy_model():
    payload = SimpleNamespace(
        model=SimpleNamespace(layers=[torch.nn.Linear(4, 4, bias=False)])
    )
    probe = torch.nn.Linear(4, 4, bias=False)
    return SimpleNamespace(payload=payload, probe=probe)


def test_parallelize_model_uses_device_type(monkeypatch):
    import torch.distributed.device_mesh as device_mesh_mod
    import torch.distributed.fsdp as fsdp_mod

    mesh_calls = []

    class _FakeMesh:
        def __getitem__(self, key):
            return f"submesh:{key}"

    def fake_init_device_mesh(device_type, mesh_shape, *, mesh_dim_names=None):
        mesh_calls.append((device_type, mesh_shape, mesh_dim_names))
        return _FakeMesh()

    monkeypatch.setattr(device_mesh_mod, "init_device_mesh", fake_init_device_mesh)
    monkeypatch.setattr(fsdp_mod, "fully_shard", lambda *a, **k: None)
    monkeypatch.setattr(common, "apply_llama_tp", lambda *a, **k: None)

    fsdp_params = SimpleNamespace(
        strategy="fsdp_sharded",
        world_size=4,
        data_parallel_size=4,
        tensor_parallel_size=1,
    )
    common.parallelize_model(_dummy_model(), fsdp_params, "cuda")
    common.parallelize_model(_dummy_model(), fsdp_params)

    mp_params = SimpleNamespace(
        strategy="model_parallel_sharded",
        world_size=8,
        data_parallel_size=2,
        tensor_parallel_size=4,
    )
    common.parallelize_model(_dummy_model(), mp_params, "tpu")

    assert mesh_calls == [
        ("cuda", (4,), None),
        ("cpu", (4,), None),
        ("tpu", (2, 4), ("dp", "tp")),
    ]


@pytest.mark.parametrize(
    "kind,dev_count,world_size,expected_opts",
    [
        ("cuda", 8, 2, {"num_gpus": 1.0}),
        ("tpu", 8, 4, {"resources": {"TPU": 1.0}}),
        ("cpu", 16, 2, {}),
    ],
)
def test_actor_options_request_accelerator(
    monkeypatch, kind, dev_count, world_size, expected_opts
):
    monkeypatch.setattr(accelerator, "device_kind", lambda: kind)
    monkeypatch.setattr(accelerator, "local_device_count", lambda: dev_count)
    monkeypatch.setattr(accelerator, "prepare_launch", lambda ws: None)

    assert accelerator.ray_resources(world_size) == expected_opts

    def fake_record_actors(cls_name):
        recorded = []

        class _FakeHandle:
            def __init__(self, *args):
                self.args = args

            def __getattr__(self, name):
                return SimpleNamespace(remote=lambda *a, **k: (cls_name, name))

        class _FakeActorCls:
            def options(self, **kwargs):
                recorded.append(kwargs)
                return SimpleNamespace(remote=lambda *args: _FakeHandle(*args))

            def remote(self, *args):
                recorded.append(None)
                return _FakeHandle(*args)

        return recorded, _FakeActorCls()

    @dataclasses.dataclass
    class _Params:
        world_size: int
        rounds: int = 1
        setup_world_size: int = 1
        setup_tensor_parallel_size: int = 1
        setup_data_parallel_size: int = 1
        tensor_parallel_size: int = 1
        data_parallel_size: int = 1
        strategy: str = "fsdp_sharded"

    params = _Params(
        world_size=world_size,
        setup_world_size=world_size,
        setup_data_parallel_size=world_size,
        data_parallel_size=world_size,
    )

    monkeypatch.setattr(common, "ensure_ray_initialized", lambda ws=None: None)
    monkeypatch.setattr(common, "kill_actors", lambda workers: None)
    monkeypatch.setattr(common, "find_free_port", lambda: 29500)

    # Load driver
    load_opts, fake_load_cls = fake_record_actors("load")
    monkeypatch.setattr(read_driver, "RayCheckpointLoadWorker", fake_load_cls)
    monkeypatch.setattr(
        common.ray,
        "get",
        lambda calls: (
            [[(0.0, 1.0)] for _ in range(world_size)]
            if calls and calls[0][1] == "load_rounds"
            else None
        ),
    )
    read_driver.run_ray_load("gs://b/ckpt", params)
    assert load_opts == [expected_opts] * world_size

    # Setup worker
    setup_opts, fake_setup_cls = fake_record_actors("setup")
    monkeypatch.setattr(read_driver, "RayCheckpointSetupWorker", fake_setup_cls)
    monkeypatch.setattr(read_driver, "shutdown_ray", lambda: None)
    read_driver.RayCheckpointReadDriver().setup("gs://b/ckpt", params)
    assert setup_opts == [expected_opts] * world_size

    # Save driver
    save_opts, fake_save_cls = fake_record_actors("save")
    monkeypatch.setattr(write_driver, "RayCheckpointSaveWorker", fake_save_cls)
    monkeypatch.setattr(
        common.ray,
        "get",
        lambda calls: (
            [[(0.0, 1.0)] for _ in range(world_size)]
            if calls and calls[0][1] == "save_rounds"
            else None
        ),
    )
    write_driver.run_ray_save("gs://b/ckpt", params)
    assert save_opts == [expected_opts] * world_size


def test_states_on_param_device(monkeypatch):
    meta_model = torch.nn.Linear(6, 4, dtype=torch.bfloat16, device="meta")
    meta_opt = torch.optim.AdamW(meta_model.parameters(), lr=1e-3)
    common.materialize_adamw_states(meta_opt)
    for p in meta_model.parameters():
        state = meta_opt.state[p]
        assert state["exp_avg"].device == p.device == torch.device("meta")
        assert state["exp_avg_sq"].device == p.device == torch.device("meta")
        assert state["exp_avg"].dtype == p.dtype

    events = []
    fake_device = torch.device("cuda", 1)

    class _TrackedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.bfloat16))

        def to(self, dev):
            events.append(("to", dev))
            return self

    fake_model = _TrackedModel()
    monkeypatch.setattr(common, "load_benchmark_model", lambda p: fake_model)
    monkeypatch.setattr(
        common,
        "parallelize_model",
        lambda m, p, device_type="cpu": events.append(("parallelize", device_type))
        or m,
    )
    monkeypatch.setattr(
        common, "materialize_adamw_states", lambda opt: events.append(("materialize",))
    )

    params = SimpleNamespace(strategy="fsdp_sharded")
    common.setup_model_and_optimizer(params, fake_device)
    assert events == [("to", fake_device), ("parallelize", "cuda"), ("materialize",)]


def test_ensure_ray_initialized_prepares_and_restores_tpu_env(monkeypatch):
    events = []
    monkeypatch.setenv("TORCH_TPU_TOPOLOGY", "preset-topo")
    for var in (
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_XPROF_SESSION_ID",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
    ):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setattr(ray_runtime, "_SAVED_TPU_LAUNCH_ENV", None)
    monkeypatch.setattr(ray_runtime, "_ACTIVE_TPU_WORLD_SIZE", None)
    initialized = {"state": False}
    monkeypatch.setattr(common.ray, "is_initialized", lambda: initialized["state"])
    monkeypatch.setattr(accelerator, "device_kind", lambda: "tpu")
    monkeypatch.setattr(accelerator, "local_device_count", lambda: 8)

    def fake_prepare(ws):
        events.append(("prepare", ws))
        os.environ["TORCH_TPU_TOPOLOGY"] = "1,2,1,2"
        os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = "localhost:1,localhost:2"
        os.environ["TORCH_TPU_XPROF_SESSION_ID"] = "999"
        os.environ["RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS"] = "1"

    def fake_init(**kwargs):
        events.append(("ray.init", os.environ.get("TORCH_TPU_TOPOLOGY")))
        initialized["state"] = True

    def fake_shutdown():
        events.append(("ray.shutdown",))
        initialized["state"] = False

    monkeypatch.setattr(accelerator, "prepare_launch", fake_prepare)
    monkeypatch.setattr(common.ray, "init", fake_init)
    monkeypatch.setattr(common.ray, "shutdown", fake_shutdown)

    common.ensure_ray_initialized(4)
    assert events == [("prepare", 4), ("ray.init", "1,2,1,2")]
    assert os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] == "localhost:1,localhost:2"

    common.shutdown_ray()
    assert events == [("prepare", 4), ("ray.init", "1,2,1,2"), ("ray.shutdown",)]
    assert os.environ["TORCH_TPU_TOPOLOGY"] == "preset-topo"
    assert "TORCH_TPU_SLICEBUILDER_ADDRESSES" not in os.environ


def test_kill_actors_invokes_ray_kill_no_restart(monkeypatch):
    killed = []
    monkeypatch.setattr(common.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(common.ray, "kill", lambda w, **kw: killed.append((w, kw)))
    common.kill_actors(["actor-0", "actor-1"])
    assert killed == [
        ("actor-0", {"no_restart": True}),
        ("actor-1", {"no_restart": True}),
    ]


def test_load_round_synchronizes_before_end(monkeypatch):
    events = []
    fake_device = torch.device("cuda", 0)

    monkeypatch.setattr(
        read_driver,
        "setup_distributed_env",
        lambda rank, ws, port: events.append(("setup_env", rank, ws, port))
        or fake_device,
    )
    monkeypatch.setattr(
        read_driver,
        "setup_model_and_optimizer",
        lambda params, dev=None: events.append(("setup_model", dev))
        or (object(), object()),
    )
    monkeypatch.setattr(
        read_driver,
        "resolve_storage",
        lambda prefix: (MagicMock(), MagicMock(), "b/ckpt"),
    )
    monkeypatch.setattr(
        read_driver.dist, "broadcast_object_list", lambda lst, src: None
    )
    monkeypatch.setattr(read_driver.dist, "barrier", lambda: events.append("barrier"))
    monkeypatch.setattr(read_driver.dist, "destroy_process_group", lambda: None)
    monkeypatch.setattr(read_driver, "get_state_dict", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(
        read_driver.dcp, "load", lambda *a, **k: events.append("dcp.load")
    )
    monkeypatch.setattr(
        read_driver, "set_state_dict", lambda *a, **k: events.append("set_state_dict")
    )
    monkeypatch.setattr(
        accelerator,
        "synchronize",
        lambda dev: events.append(("synchronize", dev)),
    )

    ticks = iter([10.0, 25.0])
    monkeypatch.setattr(
        read_driver.time,
        "perf_counter",
        lambda: events.append("perf_counter") or next(ticks),
    )

    class _FakeCkpt:
        def __init__(self, **kwargs):
            pass

        def as_directory(self):
            return contextlib.nullcontext("/tmp/fake")

    monkeypatch.setattr(read_driver.ray.train, "Checkpoint", _FakeCkpt)

    params = SimpleNamespace(strategy="fsdp_sharded", rounds=1)
    worker_cls = read_driver.RayCheckpointLoadWorker.__ray_metadata__.modified_class
    worker = worker_cls(0, 2, 29500, "gs://b/ckpt", params)
    worker.setup()
    assert worker.device == fake_device

    durations = worker.load_rounds()
    assert durations == [(10.0, 25.0)]
    assert events == [
        ("setup_env", 0, 2, 29500),
        ("setup_model", fake_device),
        "barrier",
        "perf_counter",
        "dcp.load",
        "set_state_dict",
        ("synchronize", fake_device),
        "barrier",
        "perf_counter",
    ]


def test_save_round_synchronizes_before_end(monkeypatch):
    events = []
    fake_device = torch.device("cuda", 0)

    monkeypatch.setattr(
        write_driver,
        "setup_distributed_env",
        lambda rank, ws, port: events.append(("setup_env", rank, ws, port))
        or fake_device,
    )
    monkeypatch.setattr(
        write_driver,
        "setup_model_and_optimizer",
        lambda params, dev=None: events.append(("setup_model", dev))
        or (object(), object()),
    )
    monkeypatch.setattr(
        write_driver,
        "resolve_storage",
        lambda prefix: (MagicMock(), MagicMock(), "b/ckpt"),
    )
    monkeypatch.setattr(
        write_driver,
        "stage_checkpoint_locally",
        lambda *a, **k: events.append("stage") and None,
    )
    monkeypatch.setattr(write_driver.dist, "barrier", lambda: events.append("barrier"))
    monkeypatch.setattr(
        write_driver,
        "upload_checkpoint",
        lambda *a, **k: events.append("upload"),
    )
    monkeypatch.setattr(
        accelerator,
        "synchronize",
        lambda dev: events.append(("synchronize", dev)),
    )
    monkeypatch.setattr(write_driver.dist, "destroy_process_group", lambda: None)

    ticks = iter([10.0, 25.0])
    monkeypatch.setattr(
        write_driver.time,
        "perf_counter",
        lambda: events.append("perf_counter") or next(ticks),
    )

    params = SimpleNamespace(strategy="fsdp_sharded", rounds=1)
    worker_cls = write_driver.RayCheckpointSaveWorker.__ray_metadata__.modified_class
    worker = worker_cls(0, 2, 29501, "gs://b/ckpt", params)
    worker.setup()
    assert worker.device == fake_device

    durations = worker.save_rounds()
    assert durations == [(10.0, 25.0)]
    assert events == [
        ("setup_env", 0, 2, 29501),
        ("setup_model", fake_device),
        "stage",
        "barrier",
        "perf_counter",
        "upload",
        ("synchronize", fake_device),
        "perf_counter",
    ]


@pytest.mark.parametrize("world_size,expected_broadcast", [(1, False), (4, True)])
def test_load_full_state_dict_broadcast_matches_world_size(
    monkeypatch, world_size, expected_broadcast
):
    seen_options = []
    fake_device = torch.device("meta")

    monkeypatch.setattr(
        read_driver, "setup_distributed_env", lambda rank, ws, port: fake_device
    )
    monkeypatch.setattr(
        read_driver,
        "setup_model_and_optimizer",
        lambda params, dev=None: (object(), object()),
    )
    monkeypatch.setattr(
        read_driver,
        "resolve_storage",
        lambda prefix: (MagicMock(), MagicMock(), "b/ckpt"),
    )
    monkeypatch.setattr(
        read_driver.dist, "broadcast_object_list", lambda lst, src: None
    )
    monkeypatch.setattr(read_driver.dist, "barrier", lambda: None)
    monkeypatch.setattr(read_driver.dist, "destroy_process_group", lambda: None)
    monkeypatch.setattr(
        read_driver.torch,
        "load",
        lambda *a, **k: {"model": {}, "optimizer": {}},
    )
    monkeypatch.setattr(
        read_driver,
        "set_model_state_dict",
        lambda m, sd, *, options: seen_options.append(
            ("model", options.broadcast_from_rank0)
        ),
    )
    monkeypatch.setattr(
        read_driver,
        "set_optimizer_state_dict",
        lambda m, o, sd, *, options: seen_options.append(
            ("opt", options.broadcast_from_rank0)
        ),
    )
    monkeypatch.setattr(accelerator, "synchronize", lambda dev: None)

    class _FakeCkpt:
        def __init__(self, **kwargs):
            pass

        def as_directory(self):
            return contextlib.nullcontext("/tmp/fake")

    monkeypatch.setattr(read_driver.ray.train, "Checkpoint", _FakeCkpt)

    params = SimpleNamespace(strategy="single" if world_size == 1 else "ddp", rounds=1)
    worker_cls = read_driver.RayCheckpointLoadWorker.__ray_metadata__.modified_class
    worker = worker_cls(0, world_size, 29502, "gs://b/ckpt", params)
    worker.setup()
    worker.load_rounds()

    assert seen_options == [
        ("model", expected_broadcast),
        ("opt", expected_broadcast),
    ]
