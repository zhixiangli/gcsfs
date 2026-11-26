import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("transformers")

from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch import (  # noqa: E402
    common,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.layout import (  # noqa: E402
    Layout,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.tests.tiny import (  # noqa: E402
    make_params,
)


def test_stats_roundtrip_in_rank_order(tmp_path):
    for rank in (1, 0):
        common.write_stats(str(tmp_path), rank, {"rank": rank})
    assert common.read_stats(str(tmp_path)) == [{"rank": 0}, {"rank": 1}]


def test_spawn_with_stats_returns_ranks_and_stats(monkeypatch):
    fake_ranks = MagicMock()
    fake_ranks.allgather_stats.side_effect = lambda s: s

    def _fake_spawn(prefix, params, target_fn, world_size_override=None):
        assert world_size_override == 2
        target_fn(stats=7)
        return fake_ranks

    def _target(*, weights, stats_dir, stats):
        common.write_stats(stats_dir, 0, {"weights": sorted(weights), "n": stats})

    monkeypatch.setattr(common, "spawn_split", _fake_spawn)
    weights = {"w": 1}
    ranks, stats = common.spawn_with_stats("p", None, _target, 2, weights)
    assert ranks is fake_ranks
    assert stats == [{"weights": ["w"], "n": 7}]
    assert weights == {"w": 1}
    fake_ranks.join.assert_not_called()


def test_spawn_with_stats_closes_ranks_and_releases_weights_on_error(monkeypatch):
    fake_ranks = MagicMock()
    monkeypatch.setattr(common, "spawn_split", lambda *a, **kw: fake_ranks)
    monkeypatch.setattr(
        common, "read_stats", MagicMock(side_effect=OSError("corrupt stats_dir"))
    )
    weights = {"w": 1}
    with pytest.raises(OSError, match="corrupt stats_dir"):
        common.spawn_with_stats("p", None, lambda **kw: None, 2, weights)
    fake_ranks.close.assert_called_once_with()
    fake_ranks.join.assert_not_called()
    assert weights == {}


def test_timed_synchronizes_before_end_barrier(monkeypatch):
    events = []
    tpu_stub = SimpleNamespace(type="tpu")
    monkeypatch.setattr(common.dist, "barrier", lambda: events.append("barrier"))
    monkeypatch.setattr(
        common.accelerator,
        "synchronize",
        lambda dev: events.append(("synchronize", dev)),
    )

    begin, end = common.timed(tpu_stub, lambda x: events.append(("load", x)), 1)
    assert begin <= end
    assert events == ["barrier", ("load", 1), ("synchronize", tpu_stub), "barrier"]


def test_warm_up_touches_storage_then_synchronizes(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(common.accelerator, "synchronize", events.append)
    common.warm_up(tmp_path.as_uri(), torch.device("cuda", 0))
    assert events == [torch.device("cuda", 0)]


def test_prepare_rejects_bad_layouts_before_staging(tmp_path, tiny_models, monkeypatch):
    staged = []
    monkeypatch.setattr(common, "stage_model", staged.append)
    driver = common.DCPDriver()
    fsdp = Layout("fsdp", world_size=2, data_parallel_size=2)
    tp = Layout("tp", world_size=2, tensor_parallel_size=2)

    tied_dir = tmp_path / "tied"
    tied_dir.mkdir()
    with open(f"{tiny_models['llama']}/config.json") as f:
        cfg = json.load(f)
    cfg["tie_word_embeddings"] = True
    (tied_dir / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="tie_word_embeddings"):
        driver.prepare(make_params(str(tied_dir)), fsdp)

    olmoe = make_params(tiny_models["olmoe"])
    for layouts in ((tp,), (fsdp, tp)):
        with pytest.raises(ValueError, match="MoE"):
            driver.prepare(olmoe, *layouts)
    assert staged == []

    driver.prepare(olmoe, fsdp, fsdp)
    assert staged == [tiny_models["olmoe"]]


def test_is_checkpoint_file():
    driver = common.DCPDriver()
    assert driver.is_checkpoint_file("ckpt/__0_0.distcp")
    assert driver.is_checkpoint_file("ckpt/.metadata")
    assert not driver.is_checkpoint_file("ckpt/rank0.json")


def test_rank_session_sets_threads_only_on_cpu(monkeypatch):
    calls = SimpleNamespace(setup=[], destroyed=0, threads=[])
    monkeypatch.setattr(
        common.torch, "set_num_threads", lambda n: calls.threads.append(n)
    )
    monkeypatch.setattr(
        common.dist,
        "destroy_process_group",
        lambda: setattr(calls, "destroyed", calls.destroyed + 1),
    )

    monkeypatch.setattr(
        common,
        "setup_distributed_env",
        lambda *a, **kw: calls.setup.append((a, kw)) or torch.device("cpu"),
    )
    with pytest.raises(RuntimeError, match="boom"):
        with common.rank_session(1, 2, 1234) as device:
            assert device == torch.device("cpu")
            raise RuntimeError("boom")
    assert calls.setup == [((1, 2, 1234), {})]
    assert len(calls.threads) == 1
    assert calls.destroyed == 1

    calls.threads.clear()
    cuda_dev = torch.device("cuda", 0)
    monkeypatch.setattr(
        common,
        "setup_distributed_env",
        lambda *a, **kw: cuda_dev,
    )
    with common.rank_session(0, 2, 1235) as device:
        assert device == cuda_dev
    assert calls.threads == []
    assert calls.destroyed == 2


def test_import_fsspec_filesystem_falls_back_to_private_module(monkeypatch):
    import sys

    fake_reader = object()
    fake_writer = object()
    fake_private = SimpleNamespace(FsspecReader=fake_reader, FsspecWriter=fake_writer)
    monkeypatch.setitem(
        sys.modules, "torch.distributed.checkpoint.fsspec_filesystem", None
    )
    monkeypatch.setitem(
        sys.modules, "torch.distributed.checkpoint._fsspec_filesystem", fake_private
    )
    reader, writer = common.import_fsspec_filesystem()
    assert reader is fake_reader
    assert writer is fake_writer
