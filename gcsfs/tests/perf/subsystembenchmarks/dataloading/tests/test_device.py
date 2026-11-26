"""Shared CPU/GPU/TPU device helpers for data-loading drivers.

The ``meta`` device stands in for an accelerator so copy paths run on CPU-only hosts;
``accelerator`` calls that need a real device are replaced with recorders.
"""

import itertools
import os
import types

import pytest

torch = pytest.importorskip("torch")

from gcsfs.tests.perf.subsystembenchmarks._common import accelerator
from gcsfs.tests.perf.subsystembenchmarks.dataloading import device as device_lib
from gcsfs.tests.perf.subsystembenchmarks.dataloading import driver as dl_driver

FAKE_GPU = torch.device("meta")
FAKE_TPU = torch.device("meta")


@pytest.fixture(autouse=True)
def _clean_accelerator(monkeypatch):
    accelerator._resolve_device_kind.cache_clear()
    monkeypatch.delenv(accelerator.DEVICE_ENV, raising=False)
    monkeypatch.delenv("TORCH_DEVICE_BACKEND_AUTOLOAD", raising=False)
    yield
    accelerator._resolve_device_kind.cache_clear()


@pytest.fixture
def sync_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(accelerator, "synchronize", calls.append)
    return calls


@pytest.fixture
def on_cpu(monkeypatch):
    monkeypatch.setattr(accelerator, "device_kind", lambda: "cpu")


@pytest.fixture
def on_gpu(monkeypatch):
    def configure(device_count):
        monkeypatch.setattr(accelerator, "device_kind", lambda: "cuda")
        monkeypatch.setattr(accelerator, "local_device_count", lambda: device_count)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: device_count)

    return configure


@pytest.fixture
def on_tpu(monkeypatch):
    def configure(device_count=8):
        monkeypatch.setattr(accelerator, "device_kind", lambda: "tpu")
        monkeypatch.setattr(accelerator, "local_device_count", lambda: device_count)

    return configure


@pytest.fixture
def warmups(monkeypatch):
    """Record the context warm-up allocation instead of touching a real GPU."""
    devices = []
    monkeypatch.setattr(torch.cuda, "set_device", lambda index: None)
    monkeypatch.setattr(
        torch, "zeros", lambda *shape, device=None: devices.append(device)
    )
    return devices


def test_rank_device_is_none_on_cpu(on_cpu):
    assert device_lib.rank_device(3) is None


def test_rank_device_binds_rank_modulo_device_count(on_gpu, warmups, monkeypatch):
    bound = []
    on_gpu(4)
    monkeypatch.setattr(torch.cuda, "set_device", bound.append)

    assert device_lib.rank_device(6) == torch.device("cuda", 2)
    assert bound == [2]


def test_rank_device_creates_the_cuda_context_before_timed_rounds(on_gpu, warmups):
    on_gpu(4)

    device = device_lib.rank_device(6)

    assert warmups == [device]


def test_pin_memory_matches_device_presence():
    assert device_lib.pin_memory(None) is False
    assert device_lib.pin_memory(FAKE_GPU) is True


def test_to_device_moves_nested_tensors_and_passes_other_values_through():
    batch = [
        {"images": [torch.ones(3, 2, 2), torch.ones(3, 4, 4)], "text": "cap"},
        {"images": [b"raw-jpeg"], "text": None, "ids": (torch.ones(2), 7)},
    ]

    out = device_lib.to_device(batch, FAKE_GPU)

    assert [t.device for t in out[0]["images"]] == [FAKE_GPU, FAKE_GPU]
    assert out[0]["images"][1].shape == (3, 4, 4)
    assert out[0]["text"] == "cap"
    assert out[1]["images"] == [b"raw-jpeg"]
    assert out[1]["text"] is None
    assert isinstance(out[1]["ids"], tuple)
    assert out[1]["ids"][0].device == FAKE_GPU
    assert out[1]["ids"][1] == 7


def test_to_device_returns_same_object_when_already_on_device_or_no_tensors():
    plain = {"text": ["a", "b"], "meta": (1, None)}
    assert device_lib.to_device(plain, FAKE_GPU) is plain

    already = {"t": torch.ones(2, device=FAKE_GPU), "s": "ok"}
    assert device_lib.to_device(already, FAKE_GPU) is already


def test_feed_returns_loader_unchanged_on_cpu(on_cpu):
    loader = [{"label": torch.tensor([1])}]
    assert device_lib.feed(loader, None) is loader


def test_feed_moves_batches_and_synchronizes_once_per_pass(on_gpu, sync_calls):
    on_gpu(8)
    feed = device_lib.feed([{"label": torch.tensor([1])}] * 3, FAKE_GPU)

    first = list(feed)
    list(feed)

    assert [b["label"].device for b in first] == [FAKE_GPU] * 3
    assert sync_calls == [FAKE_GPU, FAKE_GPU]


def test_feed_synchronizes_once_per_pass_on_tpu(on_tpu, sync_calls):
    on_tpu(8)
    env_inside = []

    class _Loader:
        def __iter__(self):
            env_inside.append(os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD"))
            yield {"label": torch.tensor([1])}
            yield {"label": torch.tensor([2])}

    feed = device_lib.feed(_Loader(), FAKE_TPU)
    out = list(feed)
    assert [b["label"].device for b in out] == [FAKE_TPU, FAKE_TPU]
    assert env_inside == ["0"]
    assert "TORCH_DEVICE_BACKEND_AUTOLOAD" not in os.environ
    assert sync_calls == [FAKE_TPU]


def test_feed_sets_worker_env_on_tpu_even_without_device(on_tpu, sync_calls):
    on_tpu(8)
    env_inside = []

    class _Loader:
        def __iter__(self):
            env_inside.append(os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD"))
            yield {"raw": b"jpeg-bytes"}

    feed = device_lib.feed(_Loader(), None)
    assert list(feed) == [{"raw": b"jpeg-bytes"}]
    assert env_inside == ["0"]
    assert "TORCH_DEVICE_BACKEND_AUTOLOAD" not in os.environ
    assert sync_calls == []


def test_feed_synchronizes_when_a_pass_yields_no_batches(on_gpu, sync_calls):
    on_gpu(8)
    assert list(device_lib.feed([], FAKE_GPU)) == []
    assert sync_calls == [FAKE_GPU]


def test_feed_synchronizes_when_iterator_is_closed_early(on_gpu, sync_calls):
    on_gpu(8)
    iterator = iter(device_lib.feed([{"label": torch.tensor([1])}] * 3, FAKE_GPU))

    next(iterator)
    assert sync_calls == []
    iterator.close()

    assert sync_calls == [FAKE_GPU]


def test_rounds_stopped_at_a_target_synchronize_before_their_end_timestamp(
    on_gpu, monkeypatch
):
    """measure_epochs breaks at the target; the feed must sync before the end
    timestamp (CPython closes the abandoned generator at the break)."""
    on_gpu(8)
    clock = itertools.count()
    synced_at = []
    monkeypatch.setattr(dl_driver, "timestamp", lambda: next(clock))
    monkeypatch.setattr(
        accelerator, "synchronize", lambda device: synced_at.append(next(clock))
    )
    feed = device_lib.feed([{"label": torch.ones(2)}] * 10, FAKE_GPU)

    per_epoch, _ = dl_driver.measure_epochs(feed, 2, lambda batch: 2, target=4)

    assert [rows for _, _, rows in per_epoch] == [4, 4]
    assert len(synced_at) == 2
    for (begin, end, _), sync in zip(per_epoch, synced_at):
        assert begin < sync < end


def test_ray_task_resources(on_cpu, on_gpu, on_tpu, monkeypatch):
    monkeypatch.setattr(accelerator, "device_kind", lambda: "cpu")
    assert device_lib.ray_task_resources(8, has_features=False) == {}
    assert device_lib.ray_task_resources(8, has_features=True) == {}

    on_gpu(8)
    assert device_lib.ray_task_resources(16, has_features=False) == {"num_gpus": 0.5}
    assert device_lib.ray_task_resources(8, has_features=False) == {"num_gpus": 1.0}
    assert device_lib.ray_task_resources(8, has_features=True) == {}

    on_tpu(8)
    assert device_lib.ray_task_resources(8, has_features=False) == {
        "resources": {"TPU": 1.0}
    }
    # Ray only accepts whole TPU chips, so a fractional share is refused.
    with pytest.raises(ValueError, match="exceeds 8 tpu devices"):
        device_lib.ray_task_resources(16, has_features=False)
    assert device_lib.ray_task_resources(8, has_features=True) == {}


def test_spawn_rank_epochs_prepares_launch(monkeypatch):
    from gcsfs.tests.perf.subsystembenchmarks._common import spawn

    calls = []

    def fake_spawn_local_ranks(fn, spawn_args_fn, nprocs, *, world_size, **kwargs):
        args = spawn_args_fn("q")
        calls.append(("spawn", nprocs, world_size, kwargs, args[0], args[4]))
        return [([(0.0, 1.0, 10)], 0.1, 0.2)] * nprocs, "ctx", "mgr"

    monkeypatch.setattr(spawn, "spawn_local_ranks", fake_spawn_local_ranks)
    monkeypatch.setattr(
        spawn, "join_local_ranks", lambda ctx, mgr: calls.append(("join", ctx, mgr))
    )
    params = types.SimpleNamespace(world_size=8, rounds=1)
    durations, rows, ttfb, build_s = dl_driver.spawn_rank_epochs(
        lambda *a: None, "gs://b/p", params
    )
    assert calls == [
        ("spawn", 8, 8, {"prepare_launch": True}, 8, "q"),
        ("join", "ctx", "mgr"),
    ]
    assert durations == [1.0]
    assert rows == [80]
    assert ttfb == pytest.approx(0.1)
    assert build_s == pytest.approx(0.2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_feed_delivers_tensors_to_a_real_gpu():
    device = device_lib.rank_device(0)
    batches = list(device_lib.feed([{"label": torch.arange(4)}], device))
    assert batches[0]["label"].device == device
