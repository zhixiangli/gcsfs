"""Tests for PyTorch Lightning GPU support and stated TPU skip."""

from types import SimpleNamespace

import pytest

L = pytest.importorskip("lightning.pytorch")
torch = pytest.importorskip("torch")

from gcsfs.tests.perf.subsystembenchmarks._common import (  # noqa: E402
    accelerator,
    placement,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch_lightning import (  # noqa: E402
    common,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch_lightning.read import (  # noqa: E402
    driver as read_driver,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch_lightning.write import (  # noqa: E402
    driver as write_driver,
)


def _make_params(strategy="ddp", world_size=4, tp=2, dp=2, rounds=1):
    return SimpleNamespace(
        strategy=strategy,
        world_size=world_size,
        tensor_parallel_size=tp,
        data_parallel_size=dp,
        rounds=rounds,
        model_id="unused",
        max_world_size=lambda: world_size,
    )


def test_trainer_device_args(monkeypatch):
    monkeypatch.setattr(accelerator, "device_kind", lambda: "cpu")
    assert common.trainer_device_args(4) == {"accelerator": "cpu", "devices": 4}

    monkeypatch.setattr(accelerator, "device_kind", lambda: "cuda")
    assert common.trainer_device_args(8) == {"accelerator": "gpu", "devices": 8}

    monkeypatch.setattr(accelerator, "device_kind", lambda: "tpu")
    with pytest.raises(ValueError, match="tpu"):
        common.trainer_device_args(8)


@pytest.mark.parametrize(
    "strategy",
    [
        "ddp",
        "fsdp_sharded",
        "fsdp_full",
        "model_parallel_sharded",
        "model_parallel_full",
    ],
)
@pytest.mark.parametrize(
    "kind,expected_backend",
    [("cpu", "gloo"), ("cuda", "nccl")],
)
def test_strategy_backend_matches_accelerator(
    monkeypatch, strategy, kind, expected_backend
):
    monkeypatch.setattr(accelerator, "device_kind", lambda: kind)
    strat = common.get_strategy(_make_params(strategy=strategy))
    assert strat._process_group_backend == expected_backend


def test_root_device_override_cpu_only(monkeypatch):
    for cls, kwargs in (
        (common.HostAwareFSDPStrategy, {}),
        (
            common.HostAwareModelParallelStrategy,
            {"tensor_parallel_size": 2, "data_parallel_size": 2},
        ),
    ):
        strat = cls(**kwargs)
        monkeypatch.setattr(accelerator, "device_kind", lambda: "cpu")
        assert strat.root_device == torch.device("cpu")

        monkeypatch.setattr(accelerator, "device_kind", lambda: "cuda")
        fake_gpu = torch.device("cuda", 3)
        strat.parallel_devices = [fake_gpu]
        strat.cluster_environment = SimpleNamespace(local_rank=lambda: 0)
        assert strat.root_device == fake_gpu


def test_lightning_skipped_on_tpu(monkeypatch):
    monkeypatch.delenv(placement.NODES_ENV, raising=False)
    monkeypatch.delenv(placement.DEVICES_PER_NODE_ENV, raising=False)
    monkeypatch.delenv(placement.COLLECTED_CASES_FILE_ENV, raising=False)
    params = _make_params(strategy="ddp", world_size=8)

    monkeypatch.setattr(accelerator, "device_kind", lambda: "tpu")
    monkeypatch.setattr(accelerator, "local_device_count", lambda: 8)
    reason = placement.skip_reason(params, "checkpointing/pytorch_lightning")
    assert reason is not None
    assert "limitations.md#lightning-tpu" in reason

    monkeypatch.setattr(accelerator, "device_kind", lambda: "cuda")
    monkeypatch.setattr(accelerator, "local_device_count", lambda: 8)
    assert placement.skip_reason(params, "checkpointing/pytorch_lightning") is None


def test_setup_read_trainer_calls_model_to_device_when_not_restoring_after_setup(
    monkeypatch,
):
    events = []

    class _FakeStrategy:
        restore_checkpoint_after_setup = False

        def connect(self, model):
            events.append("connect")

        def setup_environment(self):
            events.append("setup_environment")

        def setup(self, trainer):
            events.append("setup")

        def model_to_device(self):
            events.append("model_to_device")

    recorded_trainer_args = {}

    class _FakeTrainer:
        def __init__(self, **kwargs):
            recorded_trainer_args.update(kwargs)
            self.strategy = _FakeStrategy()
            self.state = SimpleNamespace(fn=None)

    fake_model = SimpleNamespace(to=lambda dtype: fake_model, trainer=None)
    monkeypatch.setattr(accelerator, "device_kind", lambda: "cuda")
    monkeypatch.setattr(read_driver, "DummyModel", lambda params: fake_model)
    monkeypatch.setattr(read_driver.L, "Trainer", _FakeTrainer)
    monkeypatch.setattr(
        read_driver, "_call_configure_model", lambda t: events.append("configure_model")
    )

    read_driver._setup_read_trainer("gs://b/p", _make_params(), world_size=4)
    assert recorded_trainer_args["accelerator"] == "gpu"
    assert recorded_trainer_args["devices"] == 4
    assert recorded_trainer_args["logger"] is False
    assert events == [
        "connect",
        "setup_environment",
        "configure_model",
        "model_to_device",
    ]


def test_write_driver_single_uses_trainer_device_args_and_syncs(monkeypatch):
    events = []
    fake_dev = torch.device("cuda", 0)
    monkeypatch.setattr(accelerator, "device_kind", lambda: "cuda")
    monkeypatch.setattr(
        accelerator, "bind_rank", lambda r: events.append(("bind", r)) or fake_dev
    )
    monkeypatch.setattr(
        accelerator, "synchronize", lambda d: events.append(("sync", d))
    )

    fake_model = SimpleNamespace(to=lambda dtype: fake_model)
    recorded_args = {}

    class _FakeTrainer:
        def __init__(self, **kwargs):
            recorded_args.update(kwargs)

        def fit(self, model, train_dataloaders=None):
            events.append("fit")

        def save_checkpoint(self, path):
            events.append(("save", path))

    monkeypatch.setattr(write_driver, "DummyModel", lambda params: fake_model)
    monkeypatch.setattr(write_driver.L, "Trainer", _FakeTrainer)
    res = write_driver.PLCheckpointWriteDriver().run(
        "gs://b/p", _make_params(strategy="single", world_size=1, rounds=1)
    )
    assert len(res.durations) == 1
    assert recorded_args["accelerator"] == "gpu"
    assert recorded_args["devices"] == 1
    assert recorded_args["logger"] is False
    assert events == [
        ("bind", 0),
        "fit",
        ("save", "gs://b/p/model.ckpt"),
        ("sync", fake_dev),
    ]
