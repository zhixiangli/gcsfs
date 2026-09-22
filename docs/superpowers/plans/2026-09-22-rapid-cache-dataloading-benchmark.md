# Rapid Cache Dataloading Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `rapid_cache_cold` and `rapid_cache_warm` bucket types to the `subsystembenchmarks` dataloading suite so every WebDataset (and shared dataloading) benchmark case can run 3 epochs against a freshly provisioned per-case cold or warm GCS Rapid Cache.

**Architecture:** We introduce a small `dataloading/rapid_cache.py` helper module that calls the GCS JSON API (`b/{bucket}/anywhereCaches`) via `gcsfs.GCSFileSystem.call` to create, wait until `RUNNING`, warm (by reading all ingested objects before timing), and disable a Rapid Cache instance per case bucket. `dataloading/bucket.py`, `dataloading/read_case.py`, `dataloading/configurator.py`, `checkpointing/configurator.py`, `run.py`, and `cloudbuild/subsystembenchmarks/` are updated to accept `rapid_cache_cold` and `rapid_cache_warm` as `--bucket-type` values requiring `--zone`.

**Tech Stack:** Python 3, `gcsfs`, `pytest`, `pytest-benchmark`, Google Cloud Storage JSON API (`anywhereCaches`), Cloud Build YAML.

**Spec:** `docs/superpowers/specs/2026-09-22-rapid-cache-dataloading-benchmark-design.md`

## Global Constraints

- Rapid Cache is a provisioning dimension (`--bucket-type`), not a `configs.yaml` variant axis.
- `BUCKET_TYPES` must be `("regional", "zonal", "hns", "rapid_cache_cold", "rapid_cache_warm")` with ID tokens `rccold` and `rcwarm`.
- Both `rapid_cache_cold` and `rapid_cache_warm` require `--zone` (`GCSFS_SUBSYSTEM_ZONE`) and create a standard regional bucket (`bucket_kwargs(spec) == {}`) paired with a zonal Rapid Cache in `spec.zone`.
- `rapid_cache_cold` creates the cache with `ingestOnWrite=False` and runs the configured 3 epochs without a warmup pass (Epoch 1 misses, Epochs 2–3 hit).
- `rapid_cache_warm` creates the cache with `ingestOnWrite=True` AND performs an untimed read pass (`warm_if_needed`) over all ingested objects under `prefix` after `params.ingest(prefix)` and before `window_start = time.time()`, then runs the configured 3 epochs (Epochs 1–3 hit).
- No new BigQuery schema columns are added; only the `bucket_type` description in `cloudbuild/subsystembenchmarks/subsystembenchmarks_schema.json` is updated.
- All unit tests must run offline without live GCS via `pytest gcsfs/tests/perf/subsystembenchmarks --run-benchmarks-infra`.

## Review Focus

1. **Terminal / unexpected cache states during `wait_running`**: If `GET b/{bucket}/anywhereCaches/{zone}` returns `DISABLED` or `PAUSED` instead of `CREATING` or `RUNNING`, `wait_running` must fail immediately with an actionable `RuntimeError` rather than polling until the 1800s timeout expires.
2. **Case failure during corpus ingestion or read**: If `params.ingest(prefix)` or `driver.run_read()` raises an exception inside `with case_bucket(...)`, teardown must still disable the Rapid Cache (`POST .../disable`), delete objects, and attempt `rmdir` without masking the original exception.
3. **Disable API failure at teardown**: If `POST b/{bucket}/anywhereCaches/{zone}/disable` raises an exception during teardown, `_delete` must suppress/log it and still execute `fs.rm(f"{name}/", recursive=True)` so billable bucket objects are never stranded.
4. **Warm pass partial/empty object list**: If `warm_if_needed` is called for `rapid_cache_warm` on a `gs://` prefix where `fs.find(prefix)` returns no files (e.g., broken ingest), it must raise a `RuntimeError` rather than silently succeeding as a fake warm cache run.
5. **Non-GCS local prefix in `warm_if_needed`**: When `run_read_case` runs against a `local_case_bucket` (`/tmp/.../data/`) even with `params.bucket_type == "rapid_cache_warm"`, `warm_if_needed` must be a no-op so unit tests using `local_case_bucket` do not instantiate `GCSFileSystem`.

---

### Task 1: Create `dataloading/rapid_cache.py` helper module and unit tests

**Files:**
- Create: `gcsfs/tests/perf/subsystembenchmarks/dataloading/rapid_cache.py`
- Create: `gcsfs/tests/perf/subsystembenchmarks/dataloading/tests/test_rapid_cache.py`

**Interfaces:**
- Produces:
  - `RAPID_CACHE_BUCKET_TYPES = ("rapid_cache_cold", "rapid_cache_warm")`
  - `DEFAULT_TIMEOUT_SECONDS = 1800`
  - `DEFAULT_POLL_SECONDS = 10`
  - `is_rapid_cache_bucket_type(bucket_type: str) -> bool`
  - `ingest_on_write_for(bucket_type: str) -> bool`
  - `create(fs, bucket: str, zone: str, *, ingest_on_write: bool) -> dict`
  - `wait_running(fs, bucket: str, zone: str, *, timeout: float = DEFAULT_TIMEOUT_SECONDS, poll: float = DEFAULT_POLL_SECONDS, sleep=time.sleep, clock=time.monotonic) -> dict`
  - `disable(fs, bucket: str, zone: str) -> None`
  - `warm_if_needed(prefix: str, bucket_type: str, *, fs=None) -> int`

- [ ] **Step 1: Write the failing unit tests in `test_rapid_cache.py`**

```python
import pytest

from gcsfs.tests.perf.subsystembenchmarks.dataloading import rapid_cache


class _FakeCacheFS:
    def __init__(self, states=("CREATING", "RUNNING"), files=None, fail_disable=False):
        self.calls = []
        self._states = list(states)
        self._files = dict(files or {})
        self._fail_disable = fail_disable
        self.cat_calls = []

    def call(self, method, path, *, json=None, json_out=False):
        self.calls.append((method, path, json, json_out))
        if method == "POST" and path.endswith("/anywhereCaches"):
            return {"state": "CREATING", "zone": json["zone"]}
        if method == "GET" and "/anywhereCaches/" in path:
            state = self._states.pop(0) if len(self._states) > 1 else self._states[0]
            return {"state": state}
        if method == "POST" and path.endswith("/disable"):
            if self._fail_disable:
                raise RuntimeError("disable failed")
            return {"state": "DISABLED"}
        raise AssertionError(f"unexpected call: {method} {path}")

    def find(self, prefix):
        return sorted(self._files.keys())

    def cat_file(self, path):
        self.cat_calls.append(path)
        return self._files[path]


def test_is_rapid_cache_and_ingest_on_write():
    assert rapid_cache.is_rapid_cache_bucket_type("rapid_cache_cold") is True
    assert rapid_cache.is_rapid_cache_bucket_type("rapid_cache_warm") is True
    assert rapid_cache.is_rapid_cache_bucket_type("regional") is False
    assert rapid_cache.ingest_on_write_for("rapid_cache_cold") is False
    assert rapid_cache.ingest_on_write_for("rapid_cache_warm") is True


def test_create_posts_anywhere_cache_body():
    fs = _FakeCacheFS()
    rapid_cache.create(fs, "my-bucket", "us-central1-a", ingest_on_write=True)
    assert fs.calls == [
        (
            "POST",
            "b/my-bucket/anywhereCaches",
            {"zone": "us-central1-a", "ingestOnWrite": True},
            True,
        )
    ]


def test_wait_running_polls_until_running():
    fs = _FakeCacheFS(states=["CREATING", "CREATING", "RUNNING"])
    sleeps = []
    t = [0.0]

    def clock():
        return t[0]

    def sleep(dt):
        sleeps.append(dt)
        t[0] += dt

    resp = rapid_cache.wait_running(
        fs,
        "my-bucket",
        "us-central1-a",
        timeout=60,
        poll=5,
        sleep=sleep,
        clock=clock,
    )
    assert resp["state"] == "RUNNING"
    assert sleeps == [5, 5]


def test_wait_running_fails_fast_on_terminal_state():
    fs = _FakeCacheFS(states=["DISABLED"])
    with pytest.raises(RuntimeError, match="DISABLED"):
        rapid_cache.wait_running(
            fs,
            "my-bucket",
            "us-central1-a",
            timeout=60,
            poll=5,
            sleep=lambda _: None,
        )


def test_wait_running_times_out_with_bucket_and_zone():
    fs = _FakeCacheFS(states=["CREATING"])
    ticks = iter([0.0, 10.0, 25.0])
    with pytest.raises(TimeoutError, match="my-bucket.*us-central1-a"):
        rapid_cache.wait_running(
            fs,
            "my-bucket",
            "us-central1-a",
            timeout=20,
            poll=10,
            sleep=lambda _: None,
            clock=lambda: next(ticks),
        )


def test_disable_suppresses_errors():
    fs = _FakeCacheFS(fail_disable=True)
    rapid_cache.disable(fs, "my-bucket", "us-central1-a")
    assert fs.calls == [
        ("POST", "b/my-bucket/anywhereCaches/us-central1-a/disable", None, True)
    ]


def test_warm_if_needed_reads_all_objects_only_for_warm_gcs_prefix():
    fs = _FakeCacheFS(
        files={
            "my-bucket/data/shard_00000.tar": b"abc",
            "my-bucket/data/shard_00001.tar": b"defg",
        }
    )
    assert (
        rapid_cache.warm_if_needed("gs://my-bucket/data/", "rapid_cache_cold", fs=fs)
        == 0
    )
    assert fs.cat_calls == []

    assert (
        rapid_cache.warm_if_needed("/tmp/local/data/", "rapid_cache_warm", fs=fs) == 0
    )
    assert fs.cat_calls == []

    total = rapid_cache.warm_if_needed(
        "gs://my-bucket/data/", "rapid_cache_warm", fs=fs
    )
    assert total == 7
    assert fs.cat_calls == [
        "gs://my-bucket/data/shard_00000.tar",
        "gs://my-bucket/data/shard_00001.tar",
    ]


def test_warm_if_needed_raises_when_warm_prefix_has_no_objects():
    fs = _FakeCacheFS(files={})
    with pytest.raises(RuntimeError, match="no objects found to warm"):
        rapid_cache.warm_if_needed("gs://my-bucket/data/", "rapid_cache_warm", fs=fs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest gcsfs/tests/perf/subsystembenchmarks/dataloading/tests/test_rapid_cache.py --run-benchmarks-infra -v`
Expected: FAIL with `ImportError: cannot import name 'rapid_cache'`

- [ ] **Step 3: Write minimal implementation in `gcsfs/tests/perf/subsystembenchmarks/dataloading/rapid_cache.py`**

```python
"""Per-case GCS Rapid Cache (Anywhere Cache) lifecycle and warmup helpers."""

import logging
import os
import time

RAPID_CACHE_BUCKET_TYPES = ("rapid_cache_cold", "rapid_cache_warm")
DEFAULT_TIMEOUT_SECONDS = 1800
DEFAULT_POLL_SECONDS = 10
_PENDING_STATES = ("CREATING", "PROVISIONING", "")


def is_rapid_cache_bucket_type(bucket_type):
    """Return True if bucket_type provisions a per-case Rapid Cache."""
    return bucket_type in RAPID_CACHE_BUCKET_TYPES


def ingest_on_write_for(bucket_type):
    """Return True if the Rapid Cache should enable ingestOnWrite."""
    if bucket_type == "rapid_cache_warm":
        return True
    if bucket_type == "rapid_cache_cold":
        return False
    raise ValueError(f"not a Rapid Cache bucket_type: {bucket_type!r}")


def timeout_from_env():
    """Return Rapid Cache creation timeout in seconds from env or default."""
    return int(
        os.environ.get(
            "GCSFS_SUBSYSTEM_RAPID_CACHE_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS)
        )
    )


def create(fs, bucket, zone, *, ingest_on_write):
    """Initiate Rapid Cache creation on bucket in zone via GCS JSON API."""
    return fs.call(
        "POST",
        f"b/{bucket}/anywhereCaches",
        json={"zone": zone, "ingestOnWrite": bool(ingest_on_write)},
        json_out=True,
    )


def wait_running(
    fs,
    bucket,
    zone,
    *,
    timeout=DEFAULT_TIMEOUT_SECONDS,
    poll=DEFAULT_POLL_SECONDS,
    sleep=time.sleep,
    clock=time.monotonic,
):
    """Poll GET anywhereCaches/{zone} until state == 'RUNNING'."""
    deadline = clock() + timeout
    last_state = "UNKNOWN"
    while True:
        resp = fs.call("GET", f"b/{bucket}/anywhereCaches/{zone}", json_out=True) or {}
        state = str(resp.get("state", "")).upper()
        last_state = state or last_state
        if state == "RUNNING":
            return resp
        if state not in _PENDING_STATES:
            raise RuntimeError(
                f"Rapid Cache for bucket {bucket!r} in zone {zone!r} entered "
                f"unexpected state {state!r}: {resp}"
            )
        if clock() >= deadline:
            raise TimeoutError(
                f"Timed out after {timeout}s waiting for Rapid Cache on bucket "
                f"{bucket!r} in zone {zone!r} to reach RUNNING (last state: {last_state!r})"
            )
        sleep(poll)


def disable(fs, bucket, zone):
    """Best-effort disable of the Rapid Cache on bucket in zone."""
    try:
        fs.call("POST", f"b/{bucket}/anywhereCaches/{zone}/disable", json_out=True)
    except Exception as exc:
        logging.warning(
            "could not disable Rapid Cache on bucket %s (%s): %s", bucket, zone, exc
        )


def warm_if_needed(prefix, bucket_type, *, fs=None):
    """Read every object under prefix once (untimed) when bucket_type is rapid_cache_warm."""
    if bucket_type != "rapid_cache_warm" or not str(prefix).startswith("gs://"):
        return 0
    if fs is None:
        import gcsfs

        fs = gcsfs.GCSFileSystem()
    objects = sorted(fs.find(prefix))
    if not objects:
        raise RuntimeError(f"no objects found to warm under {prefix!r}")
    total_bytes = 0
    for obj in objects:
        url = obj if str(obj).startswith("gs://") else f"gs://{obj}"
        total_bytes += len(fs.cat_file(url))
    return total_bytes
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest gcsfs/tests/perf/subsystembenchmarks/dataloading/tests/test_rapid_cache.py --run-benchmarks-infra -v`
Expected: PASS (all 7 tests pass)

- [ ] **Step 5: Commit**

```bash
git add gcsfs/tests/perf/subsystembenchmarks/dataloading/rapid_cache.py gcsfs/tests/perf/subsystembenchmarks/dataloading/tests/test_rapid_cache.py
git commit -m "feat(subsystembenchmarks): add per-case Rapid Cache API and warmup helpers"
```

---

### Task 2: Wire `rapid_cache_cold` and `rapid_cache_warm` into `dataloading/bucket.py` and `dataloading/read_case.py`

**Files:**
- Modify: `gcsfs/tests/perf/subsystembenchmarks/dataloading/bucket.py:14-127`
- Modify: `gcsfs/tests/perf/subsystembenchmarks/dataloading/read_case.py:65-75`
- Test: `gcsfs/tests/perf/subsystembenchmarks/dataloading/tests/test_bucket.py`
- Test: `gcsfs/tests/perf/subsystembenchmarks/dataloading/tests/test_read_case.py`

**Interfaces:**
- Consumes: `rapid_cache.is_rapid_cache_bucket_type`, `rapid_cache.ingest_on_write_for`, `rapid_cache.create`, `rapid_cache.wait_running`, `rapid_cache.disable`, `rapid_cache.timeout_from_env`, `rapid_cache.warm_if_needed`
- Produces:
  - `BUCKET_TYPES = ("regional", "zonal", "hns", "rapid_cache_cold", "rapid_cache_warm")`
  - `case_bucket(spec, case_id, *, fs=None)` creates and waits for Rapid Cache when `is_rapid_cache_bucket_type(spec.bucket_type)` is True, and disables it in `_delete(fs, name, spec)` before removing objects.
  - `run_read_case` invokes `rapid_cache.warm_if_needed(prefix, params.bucket_type)` after `params.ingest(prefix)` and before `window_start = time.time()`.

- [ ] **Step 1: Write the failing tests in `test_bucket.py` and `test_read_case.py`**

Add to `gcsfs/tests/perf/subsystembenchmarks/dataloading/tests/test_bucket.py`:
```python
@pytest.mark.parametrize("bucket_type", ["rapid_cache_cold", "rapid_cache_warm"])
def test_rapid_cache_types_require_zone_and_use_regional_bucket_body(bucket_type):
    with pytest.raises(ValueError, match="GCSFS_SUBSYSTEM_ZONE"):
        _spec(bucket_type=bucket_type).validate()
    spec = _spec(bucket_type=bucket_type, zone="us-central1-a")
    spec.validate()
    assert bucket.bucket_kwargs(spec) == {}


def test_case_bucket_creates_waits_and_disables_rapid_cache(monkeypatch):
    monkeypatch.setenv("GCSFS_SUBSYSTEM_RAPID_CACHE_TIMEOUT", "30")
    fs = _FakeFS()
    spec = _spec(bucket_type="rapid_cache_warm", zone="us-central1-a")
    with bucket.case_bucket(spec, "read-wds-x", fs=fs) as prefix:
        name = bucket.bucket_name_of(prefix)
        assert ("POST", f"b/{name}/anywhereCaches", {"zone": "us-central1-a", "ingestOnWrite": True}) in fs.api_calls
        assert ("GET", f"b/{name}/anywhereCaches/us-central1-a", None) in fs.api_calls
    assert ("POST", f"b/{name}/anywhereCaches/us-central1-a/disable", None) in fs.api_calls
    assert fs.removed == [f"{name}/"]
```
And update `_FakeFS` in `test_bucket.py` to record `call(self, method, path, *, json=None, json_out=False)` in `self.api_calls` and return `{"state": "RUNNING"}`.

Add to `gcsfs/tests/perf/subsystembenchmarks/dataloading/tests/test_read_case.py`:
```python
def test_run_read_case_invokes_warm_if_needed_before_timing(tmp_path, monkeypatch):
    monkeypatch.setattr(read_case, "assert_fsspec_gcsfs", lambda prefix: None)
    order = []

    def fake_warm(prefix, bucket_type):
        order.append(("warm", bucket_type))
        return 0

    monkeypatch.setattr(read_case.rapid_cache, "warm_if_needed", fake_warm)

    class _OrderDriver(_FakeDriver):
        def run_read(self, prefix, params, manifest):
            order.append(("run_read", params.bucket_type))
            return super().run_read(prefix, params, manifest)

    read_case.run_read_case(
        _Bench(),
        _Monitor(),
        _params(bucket_type="rapid_cache_warm"),
        _OrderDriver(rows=10),
        bucket_ctx=_local_bucket_ctx(tmp_path),
    )
    assert order == [("warm", "rapid_cache_warm"), ("run_read", "rapid_cache_warm")]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest gcsfs/tests/perf/subsystembenchmarks/dataloading/tests/test_bucket.py gcsfs/tests/perf/subsystembenchmarks/dataloading/tests/test_read_case.py --run-benchmarks-infra -v`
Expected: FAIL (`unknown bucket_type 'rapid_cache_cold'`, `AttributeError: module 'read_case' has no attribute 'rapid_cache'`)

- [ ] **Step 3: Implement changes in `dataloading/bucket.py` and `dataloading/read_case.py`**

In `gcsfs/tests/perf/subsystembenchmarks/dataloading/bucket.py`:
```python
from gcsfs.tests.perf.subsystembenchmarks.dataloading import rapid_cache

BUCKET_TYPES = ("regional", "zonal", "hns", *rapid_cache.RAPID_CACHE_BUCKET_TYPES)
```
In `BucketSpec.validate(self)`:
```python
        if (
            self.bucket_type in ("zonal", *rapid_cache.RAPID_CACHE_BUCKET_TYPES)
            and not self.zone
        ):
            raise ValueError(
                f"{self.bucket_type} buckets need GCSFS_SUBSYSTEM_ZONE (the placement zone)"
            )
```
In `bucket_kwargs(spec)`:
```python
def bucket_kwargs(spec):
    """buckets.insert body for this bucket type (gcsfs.mkdir forwards these verbatim)."""
    if spec.bucket_type in ("regional", *rapid_cache.RAPID_CACHE_BUCKET_TYPES):
        return {}
```
In `_delete(fs, name, spec=None)` and `case_bucket(spec, case_id, *, fs=None)`:
```python
def _delete(fs, name, spec=None):
    """Best-effort teardown; cloudbuild sweeps the prefix at the end as the safety net."""
    if spec is not None and rapid_cache.is_rapid_cache_bucket_type(spec.bucket_type):
        rapid_cache.disable(fs, name, spec.zone)
    try:
        fs.rm(f"{name}/", recursive=True)
    except FileNotFoundError:
        pass
    except Exception as exc:
        logging.warning("could not empty benchmark bucket %s: %s", name, exc)
    with contextlib.suppress(Exception):
        fs.rmdir(name)


@contextlib.contextmanager
def case_bucket(spec, case_id, *, fs=None):
    """Create this case's bucket, yield its corpus prefix, delete it on the way out."""
    if fs is None:
        import gcsfs

        fs = gcsfs.GCSFileSystem(project=spec.project)
    name = case_bucket_name(spec.prefix, case_id)
    fs.mkdir(name, location=spec.location, **bucket_kwargs(spec))
    try:
        if rapid_cache.is_rapid_cache_bucket_type(spec.bucket_type):
            rapid_cache.create(
                fs,
                name,
                spec.zone,
                ingest_on_write=rapid_cache.ingest_on_write_for(spec.bucket_type),
            )
            rapid_cache.wait_running(
                fs,
                name,
                spec.zone,
                timeout=rapid_cache.timeout_from_env(),
            )
        yield f"gs://{name}/data/"
    finally:
        _delete(fs, name, spec=spec)
```

In `gcsfs/tests/perf/subsystembenchmarks/dataloading/read_case.py`:
Import `rapid_cache` from `gcsfs.tests.perf.subsystembenchmarks.dataloading` at module level and call `rapid_cache.warm_if_needed(prefix, params.bucket_type)` immediately after `manifest = params.ingest(prefix)` (before `window_start = time.time()`):
```python
    with bucket_ctx(params.name) as prefix:
        params.bucket_name = bucket_name_of(prefix)
        assert_fsspec_gcsfs(prefix)
        manifest = params.ingest(prefix)
        rapid_cache.warm_if_needed(prefix, params.bucket_type)

        expected_rows = manifest["sample_count"]
        window_start = time.time()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest gcsfs/tests/perf/subsystembenchmarks/dataloading/tests/test_bucket.py gcsfs/tests/perf/subsystembenchmarks/dataloading/tests/test_read_case.py --run-benchmarks-infra -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gcsfs/tests/perf/subsystembenchmarks/dataloading/bucket.py gcsfs/tests/perf/subsystembenchmarks/dataloading/read_case.py gcsfs/tests/perf/subsystembenchmarks/dataloading/tests/test_bucket.py gcsfs/tests/perf/subsystembenchmarks/dataloading/tests/test_read_case.py
git commit -m "feat(subsystembenchmarks): wire per-case cold and warm Rapid Cache into bucket and read_case"
```

---

### Task 3: Add `rccold`/`rcwarm` ID tokens, CLI arguments, Cloud Build support, and schema doc update

**Files:**
- Modify: `gcsfs/tests/perf/subsystembenchmarks/dataloading/configurator.py:14`
- Modify: `gcsfs/tests/perf/subsystembenchmarks/checkpointing/configurator.py:8`
- Modify: `gcsfs/tests/perf/subsystembenchmarks/run.py:25-102`
- Modify: `cloudbuild/subsystembenchmarks/subsystembenchmarks-cloudbuild.yaml:6,64-67,114-127`
- Modify: `cloudbuild/subsystembenchmarks/subsystembenchmarks_schema.json:22`
- Test: `gcsfs/tests/perf/subsystembenchmarks/dataloading/tests/test_configurator.py`
- Test: `gcsfs/tests/perf/subsystembenchmarks/tests/test_run_groups.py`

**Interfaces:**
- Consumes: `BUCKET_TYPES` and `RAPID_CACHE_BUCKET_TYPES` from `dataloading/bucket.py` / `dataloading/rapid_cache.py`
- Produces:
  - `_BUCKET` mapping `"rapid_cache_cold": "rccold"` and `"rapid_cache_warm": "rcwarm"` in `dataloading/configurator.py` and `checkpointing/configurator.py`
  - `run.py` `--bucket-type` choices `("regional", "zonal", "hns", "rapid_cache_cold", "rapid_cache_warm")` requiring `--zone` for `zonal`, `rapid_cache_cold`, and `rapid_cache_warm`, plus `--rapid-cache-timeout` (default `1800`) exported as `GCSFS_SUBSYSTEM_RAPID_CACHE_TIMEOUT`
  - `subsystembenchmarks-cloudbuild.yaml` accepting `rapid_cache_cold|rapid_cache_warm` in `_BUCKET_TYPE` and sweeping leaked age-expired benchmark buckets in `cleanup-leaked-resources`

- [ ] **Step 1: Write the failing tests in `test_configurator.py` and `test_run_groups.py`**

Add to `gcsfs/tests/perf/subsystembenchmarks/dataloading/tests/test_configurator.py`:
```python
@pytest.mark.parametrize(
    ("bucket_type", "token"),
    [("rapid_cache_cold", "rccold"), ("rapid_cache_warm", "rcwarm")],
)
def test_rapid_cache_bucket_types_produce_expected_id_tokens(
    tmp_path, monkeypatch, bucket_type, token
):
    monkeypatch.setenv("GCSFS_SUBSYSTEM_BUCKET_TYPE", bucket_type)
    text = _YAML.replace(
        '      - {axis: "bucket_type", bucket_type: "hns"}   # run-level key -> must reject\n',
        "",
    )
    cases = _write(tmp_path, text).generate_cases()
    assert cases[0].name == f"read-fk-ptpq-seq-nw8-fc8x4096-{token}"
    assert cases[0].rounds == 3
```

Add to `gcsfs/tests/perf/subsystembenchmarks/tests/test_run_groups.py`:
```python
@pytest.mark.parametrize("bucket_type", ["rapid_cache_cold", "rapid_cache_warm"])
def test_parse_args_requires_zone_for_rapid_cache_bucket_types(capsys, bucket_type):
    with pytest.raises(SystemExit):
        run.parse_args(
            [
                "--group=dataloading/webdataset",
                f"--bucket-type={bucket_type}",
            ]
            + _REQUIRED
        )
    assert "--zone is required" in capsys.readouterr().err


@pytest.mark.parametrize("bucket_type", ["rapid_cache_cold", "rapid_cache_warm"])
def test_parse_args_accepts_rapid_cache_with_zone_and_timeout(monkeypatch, bucket_type):
    args = run.parse_args(
        [
            "--group=dataloading/webdataset",
            f"--bucket-type={bucket_type}",
            "--zone=us-central1-a",
            "--rapid-cache-timeout=900",
        ]
        + _REQUIRED
    )
    run._setup_environment(args)
    assert os.environ["GCSFS_SUBSYSTEM_BUCKET_TYPE"] == bucket_type
    assert os.environ["GCSFS_SUBSYSTEM_ZONE"] == "us-central1-a"
    assert os.environ["GCSFS_SUBSYSTEM_RAPID_CACHE_TIMEOUT"] == "900"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest gcsfs/tests/perf/subsystembenchmarks/dataloading/tests/test_configurator.py gcsfs/tests/perf/subsystembenchmarks/tests/test_run_groups.py --run-benchmarks-infra -v`
Expected: FAIL (`KeyError: 'rapid_cache_cold'`, `invalid choice: 'rapid_cache_cold'`)

- [ ] **Step 3: Implement updates in `configurator.py`, `run.py`, and `cloudbuild/subsystembenchmarks/`**

1. In `gcsfs/tests/perf/subsystembenchmarks/dataloading/configurator.py` and `gcsfs/tests/perf/subsystembenchmarks/checkpointing/configurator.py`:
```python
_BUCKET = {
    "regional": "reg",
    "zonal": "zon",
    "hns": "hns",
    "rapid_cache_cold": "rccold",
    "rapid_cache_warm": "rcwarm",
}
```
2. In `gcsfs/tests/perf/subsystembenchmarks/run.py`:
- Update `--bucket-type` choices to `("regional", "zonal", "hns", "rapid_cache_cold", "rapid_cache_warm")`.
- Add `--rapid-cache-timeout` (type `int`, default `1800`).
- Export `os.environ["GCSFS_SUBSYSTEM_RAPID_CACHE_TIMEOUT"] = str(args.rapid_cache_timeout)` in `_setup_environment(args)`.
- In `parse_args`, require `args.zone` when `args.bucket_type in ("zonal", "rapid_cache_cold", "rapid_cache_warm")` and require `args.rapid_cache_timeout > 0`.
3. In `cloudbuild/subsystembenchmarks/subsystembenchmarks-cloudbuild.yaml`:
- Update the `_BUCKET_TYPE` comment and `case "$${_BUCKET_TYPE}"` check to `regional|zonal|hns|rapid_cache_cold|rapid_cache_warm)`.
- In `cleanup-leaked-resources`, also sweep any leaked buckets matching `^gs://${_INFRA_PREFIX}-` older than `THRESHOLD` so buckets whose disabled Rapid Cache 1-hour grace period expired after an earlier build are cleaned up.
4. In `cloudbuild/subsystembenchmarks/subsystembenchmarks_schema.json`:
- Update the `bucket_type` description to: `"Storage tier of the case bucket: regional, zonal (RAPID), hns, rapid_cache_cold, or rapid_cache_warm."`

- [ ] **Step 4: Run the entire subsystembenchmarks test suite to verify all tests pass**

Run: `pytest gcsfs/tests/perf/subsystembenchmarks --run-benchmarks-infra -v`
Expected: PASS (all unit and config tests across `dataloading`, `webdataset`, `ray_data`, `huggingface_datasets`, and `checkpointing` pass)

- [ ] **Step 5: Commit**

```bash
git add gcsfs/tests/perf/subsystembenchmarks/dataloading/configurator.py gcsfs/tests/perf/subsystembenchmarks/checkpointing/configurator.py gcsfs/tests/perf/subsystembenchmarks/run.py gcsfs/tests/perf/subsystembenchmarks/dataloading/tests/test_configurator.py gcsfs/tests/perf/subsystembenchmarks/tests/test_run_groups.py cloudbuild/subsystembenchmarks/subsystembenchmarks-cloudbuild.yaml cloudbuild/subsystembenchmarks/subsystembenchmarks_schema.json
git commit -m "feat(subsystembenchmarks): expose rapid_cache_cold and rapid_cache_warm in CLI and Cloud Build"
```
