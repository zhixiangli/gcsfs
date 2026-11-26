"""Ray Data read driver wiring for CPU/GPU delivery."""

import itertools

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ray")

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from gcsfs.tests.perf.subsystembenchmarks._common import (  # noqa: E402
    accelerator,
    ray_runtime,
)
from gcsfs.tests.perf.subsystembenchmarks.dataloading.ray_data.parameters import (  # noqa: E402
    RayDataReadParameters,
)
from gcsfs.tests.perf.subsystembenchmarks.dataloading.ray_data.read import (  # noqa: E402
    driver,
)

FAKE_GPU = torch.device("meta")


def _params(**overrides):
    values = dict(
        name="device",
        bucket_name="",
        bucket_type="regional",
        rounds=2,
        scenario="read",
        framework="ray_data",
        fmt="pretok_parquet",
        file_count=4,
        rows_per_file=5,
        access="sequential",
        num_workers=0,
        batch_size=4,
        prefetch_factor=2,
        split_by_node=False,
        world_size=1,
    )
    values.update(overrides)
    return RayDataReadParameters(**values)


@pytest.fixture(scope="module")
def ray_cluster():
    driver.ensure_ray_initialized()
    yield
    driver.shutdown_ray()


def _write_shards(root, fmt):
    for i in range(4):
        if fmt == "pretok_parquet":
            table = pa.table({"tokens": [[1, 2, 3]] * 5, "label": [i] * 5})
        else:
            table = pa.table({"text": [f"doc {i}"] * 5, "label": [i] * 5})
        pq.write_table(table, root / f"shard_{i:05d}.parquet")
    return str(root)


@pytest.fixture
def sync_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(accelerator, "synchronize", calls.append)
    return calls


@pytest.fixture
def cpu_batches_from_ray(monkeypatch):
    """Ray itself stays on CPU: pinning pretok batches needs a real GPU."""
    real_kwargs = driver._torch_batch_kwargs

    def kwargs(params, shuffle_seed, collate_fn, device):
        return real_kwargs(params, shuffle_seed, collate_fn, None)

    monkeypatch.setattr(driver, "_torch_batch_kwargs", kwargs)


def test_torch_batch_kwargs_on_cpu_match_todays_arguments():
    kwargs = driver._torch_batch_kwargs(_params(), None, None, None)
    assert kwargs["device"] == torch.device("cpu")
    assert kwargs["pin_memory"] is False


def test_torch_batch_kwargs_on_gpu_pin_pretok_batches():
    kwargs = driver._torch_batch_kwargs(_params(), None, None, FAKE_GPU)
    assert kwargs["device"] == FAKE_GPU
    assert kwargs["pin_memory"] is True


def test_torch_batch_kwargs_on_gpu_skip_pinning_with_custom_collate():
    collate = driver._TextParquetCollateFn()
    kwargs = driver._torch_batch_kwargs(
        _params(fmt="text_parquet"), None, collate, FAKE_GPU
    )
    assert kwargs["device"] == FAKE_GPU
    assert kwargs["pin_memory"] is False
    assert kwargs["collate_fn"] is collate


def test_text_parquet_labels_reach_device_and_strings_survive(
    ray_cluster, tmp_path, sync_calls
):
    # Unmocked: Ray gets device=FAKE_GPU but no pinning, and leaves this
    # non-tensor batch on the host for feed to move.
    params = _params(fmt="text_parquet")
    arrow_fs, paths = driver.resolve_parquet_source(
        _write_shards(tmp_path, "text_parquet")
    )
    dataset = driver.build_dataset(arrow_fs, paths, params)

    batches = list(
        driver._iter_batches(
            dataset, params, None, driver._TextParquetCollateFn(), FAKE_GPU
        )
    )

    assert sum(len(b["text"]) for b in batches) == 20
    assert all(isinstance(t, str) for b in batches for t in b["text"])
    assert all(b["label"].device == FAKE_GPU for b in batches)
    assert sync_calls == [FAKE_GPU]


def test_single_rank_binds_rank0_and_synchronizes_each_round(
    ray_cluster, tmp_path, monkeypatch, sync_calls, cpu_batches_from_ray
):
    ranks = []
    monkeypatch.setattr(
        driver.device_lib, "rank_device", lambda rank: ranks.append(rank) or FAKE_GPU
    )
    params = _params()
    arrow_fs, paths = driver.resolve_parquet_source(
        _write_shards(tmp_path, "pretok_parquet")
    )

    _, rows, _ = driver.run_single_rank(
        driver.build_dataset(arrow_fs, paths, params), params
    )

    assert rows == [20, 20]
    assert ranks == [0]
    assert sync_calls == [FAKE_GPU, FAKE_GPU]


def test_consume_shard_binds_its_gpu_and_synchronizes(
    ray_cluster, tmp_path, monkeypatch, sync_calls, cpu_batches_from_ray
):
    """The task body, run locally: Ray's num_gpus scoping makes rank 0 its GPU."""
    ranks = []
    monkeypatch.setattr(
        driver.device_lib, "rank_device", lambda rank: ranks.append(rank) or FAKE_GPU
    )
    params = _params()
    arrow_fs, paths = driver.resolve_parquet_source(
        _write_shards(tmp_path, "pretok_parquet")
    )
    (split,) = driver.build_dataset(arrow_fs, paths, params).streaming_split(
        n=1, equal=False
    )

    _, _, rows = driver._consume_shard._function(split, params, None, None)

    assert rows == 20
    assert ranks == [0]
    assert sync_calls == [FAKE_GPU]


def test_multi_rank_tasks_request_their_accelerator_resources(
    ray_cluster, tmp_path, monkeypatch
):
    requested = []
    real_consume = driver._consume_shard

    class _RecordingRemote:
        def options(self, **options):
            requested.append(options)
            # No accelerators in the test cluster: schedule on CPU after recording.
            return real_consume.options(num_gpus=0, resources={})

    monkeypatch.setattr(driver, "_consume_shard", _RecordingRemote())
    monkeypatch.setattr(
        driver.device_lib,
        "ray_task_resources",
        lambda ws, has_features: {} if has_features else {"num_gpus": 0.375},
    )
    params = _params(split_by_node=True, world_size=2)
    arrow_fs, paths = driver.resolve_parquet_source(
        _write_shards(tmp_path, "pretok_parquet")
    )

    _, rows, _ = driver.run_multi_rank(
        driver.build_dataset(arrow_fs, paths, params), params
    )

    assert rows == [20, 20]
    # One task per rank per round.
    assert requested == [{"num_gpus": 0.375}] * (params.world_size * params.rounds)


def test_feature_columns_skip_the_accelerator(
    ray_cluster, tmp_path, monkeypatch, sync_calls
):
    ranks, resource_calls = [], []
    monkeypatch.setattr(
        driver.device_lib, "rank_device", lambda rank: ranks.append(rank) or FAKE_GPU
    )
    real_resources = driver.device_lib.ray_task_resources

    def spy_resources(ws, has_features):
        res = real_resources(ws, has_features)
        resource_calls.append((ws, has_features, res))
        return res

    monkeypatch.setattr(driver.device_lib, "ray_task_resources", spy_resources)
    monkeypatch.setattr(accelerator, "device_kind", lambda: "cuda")
    monkeypatch.setattr(accelerator, "local_device_count", lambda: 8)
    params = _params(seq_len=2, feature_column_count=6, projected_feature_count=3)
    prefix = f"{tmp_path}/features/"
    params.ingest(prefix)
    arrow_fs, paths = driver.resolve_parquet_source(prefix)
    dataset = driver.build_dataset(arrow_fs, paths, params)

    _, rows, _ = driver.run_single_rank(dataset, params)
    (split,) = dataset.streaming_split(n=1, equal=False)
    _, _, shard_rows = driver._consume_shard._function(split, params, None, None)
    multi = _params(
        seq_len=2,
        feature_column_count=6,
        projected_feature_count=3,
        split_by_node=True,
        world_size=2,
    )
    _, multi_rows, _ = driver.run_multi_rank(dataset, multi)

    assert rows == [20, 20] and shard_rows == 20 and multi_rows == [20, 20]
    assert ranks == [] and sync_calls == []
    assert resource_calls == [(2, True, {})]


class _FakeActor:
    def __init__(self, log, rank, source, opts):
        self.log, self.rank = log, rank
        log["inits"].append((rank, source, opts))

    class _Method:
        def __init__(self, fn):
            self._fn = fn

        def remote(self, *args, **kwargs):
            return self._fn(*args, **kwargs)

    @property
    def ready(self):
        return self._Method(lambda: self.log["ready"].append(self.rank) or True)

    @property
    def consume(self):
        def run(seed, collate):
            self.log["consume"].append((self.rank, seed))
            return (10.0 + self.rank, 20.0 + self.rank, 5)

        return self._Method(run)


@pytest.fixture
def fake_tpu_actors(monkeypatch):
    """Simulated TPU host whose _ShardConsumer actors run in-process."""
    log = {"inits": [], "ready": [], "consume": [], "killed": []}

    class _ConsumerCls:
        def options(self, **opts):
            class _Bound:
                def remote(self, rank, params, source):
                    return _FakeActor(log, rank, source, opts)

            return _Bound()

    monkeypatch.setattr(accelerator, "device_kind", lambda: "tpu")
    monkeypatch.setattr(accelerator, "local_device_count", lambda: 8)
    monkeypatch.setattr(driver, "_ShardConsumer", _ConsumerCls())
    monkeypatch.setattr(driver, "timestamp", itertools.count(0.0).__next__)
    monkeypatch.setattr(
        driver.ray, "get", lambda refs: list(refs) if isinstance(refs, list) else refs
    )
    monkeypatch.setattr(
        driver.ray,
        "kill",
        lambda actor, no_restart=True: log["killed"].append((actor.rank, no_restart)),
    )
    return log


class _CountingDataset:
    """Dataset stand-in that counts streaming_split calls (fake or real splits)."""

    def __init__(self, dataset=None):
        self._dataset = dataset
        self.split_calls = 0

    def streaming_split(self, n, equal=False):
        self.split_calls += 1
        if self._dataset is None:
            return [f"split-{i}" for i in range(n)]
        return self._dataset.streaming_split(n, equal=equal)


def test_tpu_multi_rank_gives_each_actor_its_split_once(fake_tpu_actors):
    dataset = _CountingDataset()
    params = _params(split_by_node=True, world_size=4, rounds=2)

    durations, rows, _ = driver.run_multi_rank(dataset, params)

    assert dataset.split_calls == 1
    assert fake_tpu_actors["inits"] == [
        (r, f"split-{r}", {"resources": {"TPU": 1.0}}) for r in range(4)
    ]
    assert fake_tpu_actors["ready"] == [0, 1, 2, 3]
    assert len(fake_tpu_actors["consume"]) == 8
    assert rows == [20, 20]
    assert len(durations) == 2 and all(d > 0 for d in durations)
    assert fake_tpu_actors["killed"] == [(r, True) for r in range(4)]


def test_tpu_single_rank_gives_the_actor_its_dataset_once(fake_tpu_actors, monkeypatch):
    parent_binds = []
    monkeypatch.setattr(driver.device_lib, "rank_device", parent_binds.append)
    params = _params(split_by_node=False, world_size=1, rounds=2)

    durations, rows, _ = driver.run_single_rank("fake-ds", params)

    assert parent_binds == []
    assert fake_tpu_actors["inits"] == [(0, "fake-ds", {"resources": {"TPU": 1.0}})]
    assert fake_tpu_actors["consume"] == [(0, None), (0, None)]
    assert rows == [5, 5]
    assert len(durations) == 2 and all(d > 0 for d in durations)
    assert fake_tpu_actors["killed"] == [(0, True)]


def _write_labeled_shards(root, count):
    for i in range(count):
        table = pa.table({"tokens": [[1, 2, 3]] * 5, "label": [i] * 5})
        pq.write_table(table, root / f"shard_{i:05d}.parquet")
    return str(root)


def _shard_order(labels):
    order = []
    for label in labels:
        if not order or order[-1] != label:
            order.append(label)
    return order


@pytest.fixture
def preserve_order():
    ctx = driver.ray.data.DataContext.get_current()
    previous = ctx.execution_options.preserve_order
    ctx.execution_options.preserve_order = True
    yield
    ctx.execution_options.preserve_order = previous


class _RecordingSource:
    """Dataset or split wrapper that logs the shard labels each pass reads.

    It travels to Ray tasks and actors exactly like the real source, so the
    per-round file order is observed through Ray's own pickling.
    """

    def __init__(self, source, log_dir, tag="rank0"):
        self._source, self._log_dir, self._tag = source, log_dir, tag
        self.split_calls = 0

    def streaming_split(self, n, equal=False):
        self.split_calls += 1
        splits = self._source.streaming_split(n, equal=equal)
        return [
            _RecordingSource(split, self._log_dir, f"rank{i}")
            for i, split in enumerate(splits)
        ]

    def iter_torch_batches(self, **kwargs):
        import json
        import os
        import time

        labels = []
        for batch in self._source.iter_torch_batches(**kwargs):
            labels.extend(int(v) for v in batch["label"])
            yield batch
        path = os.path.join(self._log_dir, f"{time.time_ns()}-{self._tag}.json")
        with open(path, "w") as f:
            json.dump(labels, f)


def _round_orders(log_dir, passes_per_round):
    import json

    paths = sorted(log_dir.iterdir())
    rounds = []
    for i in range(0, len(paths), passes_per_round):
        group = []
        for path in paths[i : i + passes_per_round]:
            tag = path.stem.split("-", 1)[1]
            group.append((tag, tuple(_shard_order(json.loads(path.read_text())))))
        rounds.append(tuple(sorted(group)))
    return rounds


@pytest.mark.parametrize("path", ["cpu-multi", "tpu-single", "tpu-multi"])
def test_shuffled_rounds_read_files_in_a_new_order(
    ray_cluster, tmp_path, monkeypatch, preserve_order, request, path
):
    """Real Ray tasks/actors: every round reshuffles files, with exact rows."""
    import sys

    from ray import cloudpickle

    # Ray workers cannot import this test module, so ship _RecordingSource by value.
    module = sys.modules[__name__]
    cloudpickle.register_pickle_by_value(module)
    request.addfinalizer(lambda: cloudpickle.unregister_pickle_by_value(module))
    multi = path.endswith("multi")
    params = _params(
        access="shuffled",
        file_shuffle=True,
        rounds=3,
        file_count=6,
        split_by_node=multi,
        world_size=2 if multi else 1,
    )
    if path.startswith("tpu"):
        # The TPU actor path, scheduled on this CPU-only test cluster.
        monkeypatch.setattr(accelerator, "device_kind", lambda: "tpu")
        monkeypatch.setattr(
            driver.device_lib, "ray_task_resources", lambda ws, has_features: {}
        )
    shards = tmp_path / "shards"
    shards.mkdir()
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    arrow_fs, paths = driver.resolve_parquet_source(_write_labeled_shards(shards, 6))
    source = _RecordingSource(driver.build_dataset(arrow_fs, paths, params), log_dir)

    run = driver.run_multi_rank if multi else driver.run_single_rank
    _, rows, _ = run(source, params)

    assert rows == [30, 30, 30]
    assert source.split_calls == (1 if multi else 0)
    rounds = _round_orders(log_dir, passes_per_round=2 if multi else 1)
    assert len(rounds) == 3
    for round_orders in rounds:
        assert sorted(s for _, order in round_orders for s in order) == list(range(6))
    assert len(set(rounds)) == 3, rounds


_FEATURES = dict(seq_len=2, feature_column_count=6, projected_feature_count=3)


@pytest.mark.parametrize(
    "overrides, failure, expected_ensure, expected_shutdowns",
    [
        (dict(split_by_node=True, world_size=4), None, [4], 1),
        (dict(split_by_node=False, world_size=4), None, [1], 1),
        (dict(split_by_node=True, world_size=4), "run", [4], 1),
        (dict(split_by_node=True, world_size=4), "ensure", [4], 1),
        (dict(split_by_node=True, world_size=4, **_FEATURES), None, [None], 0),
    ],
    ids=["multi", "single", "run-fails", "init-fails", "features"],
)
def test_run_read_tpu_lifecycle(
    monkeypatch, overrides, failure, expected_ensure, expected_shutdowns
):
    monkeypatch.setattr(accelerator, "device_kind", lambda: "tpu")
    ensured, shutdowns = [], []

    def fake_ensure(world_size=None):
        ensured.append(world_size)
        if failure == "ensure":
            raise RuntimeError("init failed")

    def fake_run(dataset, params):
        if failure == "run":
            raise RuntimeError("run failed")
        return [1.0], [20], 0.1

    monkeypatch.setattr(driver, "ensure_ray_initialized", fake_ensure)
    monkeypatch.setattr(driver, "shutdown_ray", lambda: shutdowns.append(True))
    monkeypatch.setattr(driver, "resolve_parquet_source", lambda *a: ("fs", ["p"]))
    monkeypatch.setattr(driver, "build_dataset", lambda *a: "dataset")
    monkeypatch.setattr(driver, "run_multi_rank", fake_run)
    monkeypatch.setattr(driver, "run_single_rank", fake_run)

    if failure:
        with pytest.raises(RuntimeError, match="failed"):
            driver.RayDataReadDriver().run_read("gs://b/p/", _params(**overrides), None)
    else:
        result = driver.RayDataReadDriver().run_read(
            "gs://b/p/", _params(**overrides), None
        )
        assert result.rows_per_epoch == [20]

    assert ensured == expected_ensure
    assert len(shutdowns) == expected_shutdowns


def test_tpu_ensure_ray_initialized_rejects_world_size_above_host_tpus(monkeypatch):
    monkeypatch.setattr(accelerator, "device_kind", lambda: "tpu")
    monkeypatch.setattr(accelerator, "local_device_count", lambda: 4)
    prepared, inits = [], []
    monkeypatch.setattr(accelerator, "prepare_launch", prepared.append)
    monkeypatch.setattr(driver.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(driver.ray, "init", lambda **kw: inits.append(kw))
    monkeypatch.setattr(ray_runtime, "_SAVED_TPU_LAUNCH_ENV", None)
    monkeypatch.setattr(ray_runtime, "_ACTIVE_TPU_WORLD_SIZE", None)
    monkeypatch.setenv("RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS", "unset-me")
    monkeypatch.delenv("RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS")

    with pytest.raises(ValueError, match="exceeds 4"):
        driver.ensure_ray_initialized(world_size=8)

    assert prepared == []
    assert inits == []
