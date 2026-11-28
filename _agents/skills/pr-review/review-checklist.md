# GCSFS Review Checklist

Each item is phrased as the question a `gcsfs` maintainer would ask. Report only items actually violated in the diff. Items marked **(diff check)** are mechanical: search every added/modified line for them on every review.

Cross-references:
- Security guardrails (mandatory): [security.md](../../rules/security.md) and [gcsfs_threat_surface.md](../security-audit/references/gcsfs_threat_surface.md)
- Subsystem invariants: [gcsfs-invariants.md](references/gcsfs-invariants.md)
- Upstream library contracts: [upstream-contracts.md](references/upstream-contracts.md)

## 1. Necessity, Design & Upstream Layer

- [ ] **Concrete problem on `main`.** "Without this PR, what exact call/input fails on `main` today?" Require a minimal failing example in the description or your analysis before polishing lines.
- [ ] **Root cause vs. symptom.** Is this patching a symptom (e.g., adding a retry, knob, or fallback) where fixing the ordering, precondition, or state ownership eliminates the bug class?
- [ ] **Right layer (`gcsfs` vs. upstream).** Does this belong in `fsspec`, `google-cloud-storage`, `google-cloud-storage-control`, `google-api-core`, or `grpcio`? Check [upstream-contracts.md §5](references/upstream-contracts.md#5-upstream-vs-gcsfs-bug-ownership-matrix).
- [ ] **Private/experimental upstream APIs (diff check).** Search added lines for `._`, `_experimental`, `google.cloud._storage_v2`, or undocumented `grpc.*` channel args. Any temporary workaround requires a `TODO` with an upstream issue/PR link AND justification why fixing upstream isn't cheaper.
- [ ] **`fsspec` CI-on-HEAD vs. `2026.9.0` skew (diff check).** CI installs `filesystem_spec` git HEAD while PyPI users run `fsspec>=2026.9.0`. Search added lines for post-`2026.9.0` kwargs/features (e.g., `_expand_path(..., withdirs=)`, removing the `AdaptiveReadaheadCache` loop-rebind workaround, `DirFileSystem` HEAD fixes) per [upstream-contracts.md §1](references/upstream-contracts.md#ci-on-head-vs-fsspec202690-skew-table).
- [ ] **Dual import-time mode (`GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT=false/true`).** Does the change work cleanly under both `CoreGCSFileSystem` (`false`) and `ExtendedGcsFileSystem` (`true`)? Remember the class swap in `gcsfs/__init__.py` is import-time only.
- [ ] **Existing counterpart.** Grep for existing helpers before accepting a second definition: `_dircache.py`, `_validate_grpc_write_kms_key_name`, `_classify_layout_error`, `as_byte_buffer`, `is_empty_range`, `_resolve_generation_match`, `zb_hns_utils.download_range`, `poll_lro`, `fsspec.asyn.sync_teardown` / `zb_hns_utils.sync_teardown`, `tests/layouts.py`, `tests/lro_fakes.py`.
- [ ] **One-sided design.** Is the fix/feature added only to `GCSFileSystem` or only `ExtendedGcsFileSystem`, only `GCSFile` or only `ZonalFile`, only one write entry point (`_put_file` / `_pipe_file` / `_perform_parallel_upload` / `simple_upload` / `initiate_upload`), only Flat/HNS/Zonal/RCU (`rcu_rapid_write=False` vs `True`), or only sync vs native `asynchronous=True`? Ask why it isn't symmetric.
- [ ] **Value & performance evidence.** Does a new public API, knob, or extension point have a concrete in-repo consumer or named user? Do hot-path performance PRs adding complexity include multi-run benchmark numbers beating noise (and does the diff actually contain a plausible mechanism for the claimed gain)?

## 2. Complexity, Minimalism & Configuration

- [ ] **Justified net LoC.** For large additions with no deletions, ask where the complexity comes from and write out the smallest alternative with a rough LoC estimate.
- [ ] **Challenge every new flag, knob, or env var (diff check).** Search added lines for `os.environ` / `os.getenv` / new `__init__` params. Prefer `fsspec` config (`FSSPEC_GCS_*`) parsed with `_parse_bool_flag` / `_parse_int_flag` (handles JSON booleans/ints AND INI strings) over bespoke env vars. If always constant, hardcode; if unsupported, raise early.
- [ ] **No dead abstractions or single-use wrappers.** Inline single-use helpers unless they isolate a clear concurrency/teardown invariant; delete code made dead by the PR in the same PR.
- [ ] **Dependency policy (diff check).** Check `pyproject.toml` and `environment_gcsfs.yaml`:
  - No upper bounds (`<...`) on runtime dependencies (`fsspec` uses monthly CalVer; upper bounds break ecosystem solvers).
  - Raising the `fsspec` lower bound requires naming the exact `fsspec` feature/fix per `CONTRIBUTING.md` / `docs/source/release.rst` (and a maintainer call).
  - Keep `environment_gcsfs.yaml` synced with `pyproject.toml` (prefer conda packages over pip where available).

## 3. Public API, Backward Compatibility & Placement

- [ ] **`fsspec` signature & kwargs parity.** Check return-type contracts (`cat("b/f")` returns `bytes`, `cat(["b/f"])` returns `dict`; never pass an already-expanded list into `super()._cat`), kwargs forwarding (`_rm` `batch_size` vs `batchsize`, `_mv_file` fallback kwargs, `_find` `**kwargs`, `_ls` `detail`), and glob-metacharacter safety (`*`, `?`, `[` in object names).
- [ ] **Pickling & Dask/Ray/DataLoader worker safety (diff check).** Check `storage_options`, `GCSFile.__reduce__`, and `ZonalFile._reopen_kwargs`: preserve `generation`, `_reopen_cache_options`, `consistency`, `timeout`, `pool_size`, and `cache_type=None` for default; never serialize loops, connectors, gRPC channels, or unredacted `token=`. Check `HttpError`/`GenerationMismatchError` `__reduce__` if adding new exceptions.
- [ ] **Trailing-slash & `dircache` contracts.** `_strip_protocol` deliberately keeps trailing `"/"` for 0-byte directory markers — verify new path logic handles `"b/d/"` cleanly (no self-counting `"dir/"` placeholders, no `_parent`/`_glob`/`_walk`/`DirFileSystem` regressions). Remember `GCSFileSystem.invalidate_cache(path)` clears ancestors only, not descendants.
- [ ] **Module placement.** Keep HTTP/core in `core.py`, HNS/Zonal/RCU orchestration in `extended_gcsfs.py`, gRPC pool/`ctypes`/teardown primitives in `zb_hns_utils.py`, Zonal file semantics in `zonal_file.py`, dircache updaters in `_dircache.py`, LRO polling in `poller.py`, parallel uploads in `parallel_upload.py`, retries in `retry.py`, and telemetry in `telemetry/`.

## 4. Correctness: Bucket Layouts, Concurrency, Teardown, Telemetry & Performance

### Bucket Layouts & RCU ([gcsfs-invariants.md §5](references/gcsfs-invariants.md#5-bucket-layout-classification--rcu-lifecycle))
- [ ] **All 7 `BucketLayout`s + `UNKNOWN`.** Verify behavior across `STANDARD`, `STANDARD_HIERARCHICAL`, `ZONAL_HIERARCHICAL`, `STANDARD_RCU`, `HIERARCHICAL_RCU`, `STANDARD_RAPID_CACHE`, `HIERARCHICAL_RAPID_CACHE`, and `UNKNOWN`.
- [ ] **Layout fallback cache separation.** Never write `BucketLayout.UNKNOWN` into `_storage_layout_cache`; fallbacks belong in `_storage_layout_fallbacks` (300s TTL). `_classify_layout_error` must raise `NotFound` immediately and fall back on `Unauthenticated` only for `AnonymousCredentials` / `unverified_client_cert`.
- [ ] **RCU disablement recovery.** `_rcu_disabled_after_error` returns `bool`: `_init_aaow` raises `RapidCacheUltraDisabledError` (HTTP fallback across `_put_file`, `_pipe_file`, `initiate_upload`, `simple_upload`), while `_perform_parallel_upload` retries `upload(None)` without `storage_class="RAPID"`. Verify all write entry points stay consistent.

### Zonal & RCU gRPC Read Path ([gcsfs-invariants.md §7](references/gcsfs-invariants.md#7-zonal--rcu-grpc-read-path-mrdpool-cat_ranges-ctypes-zero-copy))
- [ ] **`download_ranges` zero-copy & EOF guard (diff check).** Every `mrd.download_ranges(...)` call MUST (1) pass `**_download_ranges_kwargs(type(mrd))`, (2) guard `length == 0` before calling (`read_length=0` reads to EOF in GCS!), and (3) respect `MRD_MAX_RANGES = 1000`. Prefer reusing `zb_hns_utils.download_range(s)`.
- [ ] **MRD pool health & staleness (diff check).** Any PR touching `MRDPool` / `MRDPoolCache` acquisition, release, or error recovery must verify:
  - `mrd.is_stream_open` is checked before recycling (on `DataCorruption`, `AsyncMultiRangeDownloader` closes itself; recycling a closed MRD poisons the pool!).
  - `_is_stale` rejects MRDs with `persisted_size < pool.persisted_size` (Zonal appends grow `persisted_size` at the same generation), and redirected opens with `read_handle` have `persisted_size is None`.
  - Stream creation happens outside `self._cond`; waiters wake via `asyncio.shield(self._notify_all())`; pool closes use `_close_pool(pool)`.
  - Round-robin sharers respect `_StreamMultiplexer.reopen_stream` non-retryable `"Stream reopening"` broadcast hazard.
- [ ] **`ReadAheadChunked` Zonal-only.** Never wire `ReadAheadChunked` to `GCSFile._fetch_range` (lacks `chunk_lengths` -> `TypeError`).

### Zonal & RCU gRPC Write Path ([gcsfs-invariants.md §8](references/gcsfs-invariants.md#8-zonal--rcu-grpc-write-path-zonalfile-aaow-parallel-upload))
- [ ] **Eager validation before open.** Validate `if_generation_match` and `kms_key_name` and call `as_byte_buffer(data)` BEFORE opening `MRDPool` or `AAOW` (`AAOW.open()` immediately overwrites an existing object!).
- [ ] **Failure/close symmetry.** `finalize_on_close` defaults `False` (Zonal) / `True` (RCU). `_abort_write()` sets `finalize_on_close=False, forced=True` without masking exceptions; `mrd_pool` and `aaow` close independently; `close_aaow` re-raises only when `finalize_on_close=True`. Remember `fsspec.open()` and `TextIOWrapper` bypass `ZonalFile.__exit__`. Do not rely on `AAOW.append()` retry for correctness (upstream `seek(persisted_size)` bug).
- [ ] **Parallel upload eligibility.** `_may_use_parallel_upload` requires `if_generation_match is None`, known `storage_class`, finalized gRPC writes, non-empty metadata keys/values on HTTP, `os.path.isfile(lpath)` (rejects `/dev/null`/FIFOs), and `size >= threshold`.

### HNS & Directory Cache ([gcsfs-invariants.md §4, §6](references/gcsfs-invariants.md#6-hns-folder-operations--lro-poller))
- [ ] **File/folder coexistence & placeholders.** File `"a"` and folder `"a/"` can coexist in HNS: key by `(name, type)`, check `_info` on folder create, delete deepest-first, preserve `delete_placeholder=not (recursive and maxdepth is None)`, and filter self-entry `"path/"` placeholders in directory listings/counts.
- [ ] **LRO poller & `rename_folder`.** Use `poll_lro` (`_wait_keeping_cancel`, `operation.done(retry=None)`, `unwrap_operation_result`, `_BACKGROUND_TASKS` strong set, 15s `cancel()` cap). `rename_folder` fallback to copy+delete after `rename_submitted` is legal **only** when `operation.operation.done and operation.operation.HasField("error")` — never after `TimeoutError`.
- [ ] **`HnsDirCacheUpdater`.** Uses only `_is_bucket_hns_cached` (no post-op RPC), normalizes with `_strip_protocol(...).rstrip("/")`, and `_rm_files` (`core.py`) passes ONLY `deleted + confirmed_absent` to `_rm_files_cache_update`. Never poison `self.dircache[path]` with filtered listings (e.g., altered `includeFoldersAsPrefixes`).

### Asyncio, Event Loops, `fork()`, gRPC & Teardown ([gcsfs-invariants.md §9](references/gcsfs-invariants.md#9-file-lifecycle-adaptive-prefetcher-pickling--teardown), [upstream-contracts.md §4](references/upstream-contracts.md#4-grpcio--grpcaio--c-core))
- [ ] **Cross-thread teardown & finalizers (diff check).** Search added lines for `loop.create_task` in `@staticmethod`, `__del__`, or `weakref.finalize` callbacks. Never call `loop.create_task` from a non-loop thread or discard unreferenced tasks — use `_on_loop_thread` + `_defer_task` / `asyncio.run_coroutine_threadsafe` / `sync_teardown`, guard `sys.is_finalizing()`, and close `_connector` when no loop is running.
- [ ] **Cancellation & interrupt cleanup (diff check).** Search for `except Exception:` in `__init__` / open / teardown paths that own pools, caches, or FDs — must catch `BaseException` (and re-raise) so `KeyboardInterrupt` and `asyncio.CancelledError` clean up `handoff_lock` / `mrd_pool` / `cache = prefetcher = None` and drain `os.pwrite` futures before `os.close(fd)`.
- [ ] **Native async & multi-loop binding.** Under `asynchronous=True`, `self._loop` is `None`. Verify gRPC channels bind to the active loop (Python 3.12+/3.14 `get_working_loop()` trap) and are closed on that loop. Never hold a blocking `threading.Lock` across `create_task()` (Python 3.12+ eager task deadlock).
- [ ] **`fork()` safety.** Preserve `bind_gcs_sync_methods` (ref-cycle protects `fork()`+`exec()` from `grpc.aio` `PollerCompletionQueue.shutdown()` hang) and `os.register_at_fork` detector resets.
- [ ] **gRPC channel & interceptor contracts (diff check).** Check `attempt_direct_path=False` on custom endpoints (prevents `:port` strip to `google-c2p:///host`), `max_send/receive_message_length=-1` on pre-built channels (prevents silent 4 MiB cap), `grpc.http2.ping_timeout_ms` for keepalive (`writing.cc` bug), no `http://`/`https://` scheme in gRPC targets, explicit `:port` on local endpoints (default is 443), and no single interceptor class inheriting multiple `grpc.aio` interceptor ABCs (`Channel.__init__` `elif` trap).

### Truthiness, Validation & Retries ([gcsfs-invariants.md §2, §3](references/gcsfs-invariants.md#3-paths-versioning-preconditions--consistency-checkers))
- [ ] **Zero/bool/empty traps (diff check).** Search added lines for `if generation:`, `if if_generation_match:`, `not generation`, or `isinstance(..., int)` without `not isinstance(..., bool)`. `0` is the create-only precondition; `True` is an `int` in Python!
- [ ] **Retry & consistency invariants.** Preserve `validate_response` ignoring `499` and excluding `401` from `"invalid"` -> `ValueError`; keep resumable non-final chunks aligned to `256 KiB..256 MiB`; raise `NotImplementedError` on missing requested hash in `ConsistencyChecker` (never downgrade `md5` -> `size`).

### Usage Telemetry ([gcsfs-invariants.md §10](references/gcsfs-invariants.md#10-usage-telemetry-architecture--wire-schema))
- [ ] **Wire IDs & sync wrappers (diff check).** Never reuse/renumber reserved `GcsfsMethod` IDs (`2, 42, 43, 45`) or assign uncoordinated ad-hoc wire numbers; token names must match lowercased enum suffix. Any new GCSFS-only `async def _foo` needs `foo = asyn.sync_wrapper(_foo)` on both `GCSFileSystem` and `ExtendedGcsFileSystem` (otherwise `mirror_gcs_methods` skips it and `test_telemetry.py` fails).
- [ ] **Privacy & non-blocking.** Allowlisted param flags record boolean truthiness ONLY (never values/paths/tokens); `DO_NOT_TRACK`/`GCSFS_NO_TELEMETRY` short-circuit immediately; `wrap_file_methods` wraps only `cls.__dict__`; lowercase `GRPC_METADATA_KEY`; `EnvironmentDetector` never blocks and uses `trust_env=False`.

### Hot-Path Performance
- [ ] **Avoid allocations & loop stalls (diff check).** Flag extra payload copies (`bytes(view)`, `b"".join` on hot read/write paths), losing the exact `type(data) is bytes` whole-buffer fastpath in `DirectMemmoveBuffer`/`PartialView`, per-request `os.environ` reads or locks, blocking CPU work on the asyncio loop, and N+1 RPCs (`_du`, unbatched deletes).

## 5. Security, Auth, TLS & `ctypes` Zero-Copy

Enforce every rule in [_agents/rules/security.md](../../rules/security.md) (and consult [gcsfs_threat_surface.md](../security-audit/references/gcsfs_threat_surface.md) when touching auth, URLs, local writes, TLS/mTLS, `ctypes`, pickling, or CI):

- [ ] **REST URL segment quoting (diff check).** Search added lines for `f"b/`, `"b/" +`, `f"o/`, or unquoted `{bucket}`/`{key}` in URLs. Every path segment must pass via positional `*args` to `_call`/`_request`/`_format_path` or explicit `quote(val)` (`safe=""`).
- [ ] **Local filesystem containment.** Any `_get_file` / local cache writer must reject `..`, leading `/`, drive prefixes, and pre-existing local symlinks (`os.path.realpath`, since `fsspec`'s `check_contained` uses `abspath`).
- [ ] **Credential & key redaction (diff check).** Search added `logger.*`, exceptions, `headers=`, telemetry, and `__reduce__`/`to_dict` for `Authorization`, `Proxy-Authorization`, `X-Goog-Encryption-Key*`, tokens, or signed URL query strings. Wrap raw-token HTTP errors with `from None`. Never pass auth/CSEK headers into `_call(..., headers=...)` (it debug-logs `kwargs.get("headers")`).
- [ ] **TLS, mTLS, endpoints & anonymous gRPC (diff check).** Search for `ssl=False`, `verify=False`, `CERT_NONE`, plaintext `http://` outside emulator/`endpoint_url`/metadata server, or `AnonymousCredentials()` hitting `AsyncGrpcClient` (silently opens plaintext `insecure_channel`!). Local credential/cert files must use `0o600` or `TemporaryDirectory()`. Check `GOOGLE_CLOUD_UNIVERSE_DOMAIN=""` -> `"googleapis.com"` handling.
- [ ] **`ctypes` / zero-copy buffer safety (diff check).** Search `zb_hns_utils.py` edits for:
  - Shared `ctypes.pythonapi` mutation (must use private `_pythonapi = ctypes.PyDLL(...)`).
  - `PyBytes_FromStringAndSize.restype` != `ctypes.py_object`.
  - `PyObject_GetBuffer` without `PyBuffer_Release` in `finally`.
  - Removing or weakening `_total_bytes_written < expected_size` (leaks uninitialized heap memory!), `_allocated_intervals` non-overlap, `owned_view.release()` in `finally`, or `close()` waiting on `_done_event`.
- [ ] **CMEK `kms_key_name` fail-closed (diff check).** Reject `/cryptoKeyVersions/` (`ValueError`), reject RCU gRPC writes (`NotImplementedError`), and in `ab` append mode resolve via HTTP `objects.get` (never `_info`) with `if_generation_match=0` retry.
- [ ] **Timeouts, subprocesses, deserialization & CI supply chain (diff check).**
  - Search for `aiohttp.ClientTimeout(total=...)` with numeric `total` on data transfers (caps large transfers AND raises non-retriable `TimeoutError`!), or `requests`/`aiohttp` calls missing `timeout=`.
  - No `shell=True`, `yaml.load` (use `safe_load`), or `pickle.loads`/`eval`/`exec` on remote/untrusted data.
  - GitHub Actions & `.pre-commit-config.yaml`: 40-char commit SHA pins, `permissions: contents: read`, `${{ github.event.* }}` via `env:`, SHA-256 verification on downloaded binaries.

## 6. Tests, CI, Cloud Build & Benchmarks

- [ ] **Mutation-resistance check.** Ask: *"If I revert the core line of this PR, does at least one new/updated test fail?"* Flag bare `mock.Mock()` (prefer `autospec=True`, real protos, `tests/layouts.py`, `tests/lro_fakes.py`), vacuous `all(r is not None for r in [])` on empty lists, tests bypassing public entry points, and `0 failed, N skipped` local test claims.
- [ ] **Broken pytest marker trap (diff check).** Search added lines for `@pytest.mark.requires_real_gcs`, `@pytest.mark.requires_hns`, or `@pytest.mark.requires_rapid` — these silently do NOT skip! Must import `skipif` markers from `gcsfs.tests.conftest`.
- [ ] **Core unit-test gRPC isolation (diff check).** Search core unit tests for `gcsfs.GCSFileSystem(` (pulls `ExtendedGcsFileSystem` under default flag, triggering `_forbid_remote_grpc_connections` / CI fake-SA failures). Use `CoreGCSFileSystem(token="anon", skip_instance_cache=True)` or stub gRPC clients.
- [ ] **Test hygiene (diff check).** Search for real wall-clock `asyncio.sleep(` (use `fake_clock` or `sleep(0)`), real-GCS temporary bucket names missing the `gcsfs-test-` IAM prefix (or hardcoded without UUID/cleanup), protobuf reloads missing from `test_init.py` `_KEEP_MODULES`, `memoryview` mocks missing `mock.reset_mock()`, unisolated `default_usage_tracker`, or adaptive tests omitting `cache_type="adaptive"`.
- [ ] **CI, Codecov, Benchmarks & Cloud Build wiring (diff check).**
  - New tests under `gcsfs/tests/perf/**`: `conftest.py`'s `pytest_ignore_collect` skips them by default, and microbenchmark infra unit tests must live at top-level `gcsfs/tests/perf/microbenchmarks/test_*.py` (never inside a benchmark group subdir; subsystem/macrobenchmark infra tests live under their `tests/` subdirs) and run under `.github/workflows/ci.yml` `--run-benchmarks-infra`.
  - Codecov enforces 80% patch coverage (0% drop) — flag dead/untested branches.
  - `cloudbuild/` E2E scripts: worker counts must match across `cloudbuild/create_buckets.sh`, `cloudbuild/run_tests.sh`, and `cloudbuild/cleanup.sh`; quote shell vars (`"$VAR"`); use Helm `--set-literal` for comma/bracket values.

## 7. Scope, Naming, Description, Docs & Changelog

- [ ] **One concern per PR.** Flag drive-by renames, unrelated formatting, comment rewrites, or bundled unrelated fixes.
- [ ] **PR title & description match HEAD.** In `fsspec/gcsfs`, the PR title/body becomes the squash commit message: verify defaults, method signatures, concrete before/after behavior, benchmark evidence, and what was actually tested (emulator vs real GCS) match the diff. (Skip PR-description findings in local branch mode when no description exists.)
- [ ] **`docs/source/changelog.rst` & out-of-diff docs.** User-facing changes need an accurate `Unreleased` entry (bold lead-in why it matters, exact exceptions/opt-out, `(#PR)`). Check whether `docs/source/*.rst`, `README.md`, `cloudbuild/*/README.md`, or docstrings went stale, keeping user docs (`docs/source/*.rst`, `README.md`) separate from developer internals (`docs/source/developer.rst`).
- [ ] **Actionable errors & comments.** Error messages name what went wrong and what the caller should do instead; magic numbers carry a justifying comment; temporary workarounds carry a `TODO` + upstream issue link.
- [ ] **Silent non-conflicting merge hazard.** Check recent `main` commits and open PRs for new helpers (e.g., `_download_ranges_kwargs`, `_validate_grpc_write_kms_key_name`, `_close_pool`) that a new call site in this PR forgot to adopt.
