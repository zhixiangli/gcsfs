# GCSFS Codebase Invariants

Symbol-indexed reference of `gcsfs` production invariants, subtle traps, and test/CI wiring rules. Load only the sections matching files touched by the PR.

## 1. Module Map & Import-Time Class Swap

- **Import-time swap (`gcsfs/__init__.py`)**: `GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT` (default `"true"`) controls whether `gcsfs.GCSFileSystem` is `ExtendedGcsFileSystem` (`gcsfs/extended_gcsfs.py`) or core `GCSFileSystem` (`gcsfs/core.py`). The swap happens **once at import time**; changing the env var after importing `gcsfs` does nothing unless modules are reloaded.
- **Dual-mode CI**: GitHub Actions runs unit tests with both `GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT=false` and the default (`true`). Every public/protected override on `ExtendedGcsFileSystem` must preserve `GCSFileSystem` behavior when disabled, and every new `GCSFileSystem` method must work under both classes.
- **Module ownership**:
  - `gcsfs/core.py`: `GCSFileSystem`, `GCSFile`, HTTP/aiohttp transport, path/generation/precondition helpers, resumable/simple uploads, adaptive prefetcher wiring, deferred-close worker.
  - `gcsfs/extended_gcsfs.py`: `ExtendedGcsFileSystem`, `BucketLayout` classification, RCU lifecycle, HNS folder/rename/rm/find/mkdir, gRPC `MRD`/`AAOW` orchestration, `_cat_ranges` D'Hondt scheduler, gRPC upload entry points.
  - `gcsfs/zb_hns_utils.py`: `MRDPool`, `MRDPoolCache`, `init_mrd`, `init_aaow`, `download_range(s)`, `_download_ranges_kwargs`, `DirectMemmoveBuffer`, `PartialView`, `sync_teardown` / `_on_loop_thread` / `_defer_task`.
  - `gcsfs/zonal_file.py`: `ZonalFile` (read/write/append/create over `MRDPool` + `AAOW`, CMEK validation, `_abort_write`).
  - `gcsfs/_dircache.py`: `DirCacheUpdater` (flat) and `HnsDirCacheUpdater` (HNS).
  - `gcsfs/parallel_upload.py`: parallel file & bytes upload (`async_transfer_manager` over `AsyncGrpcClient`).
  - `gcsfs/poller.py`: generic async poller and `poll_lro` for HNS long-running operations.
  - `gcsfs/retry.py`: `HttpError`, `ChecksumError`, `GenerationMismatchError`, `NonRetryableError`, `validate_response`, `retry_request`, `get_storage_control_retry_config`.
  - `gcsfs/credentials.py`: `GoogleCredentials`, `_get_creds_from_raw_token`, token caching/refresh.
  - `gcsfs/checkers.py`: `MD5Checker`, `SizeChecker`, `Crc32cChecker`, `get_consistency_checker`.
  - `gcsfs/caching.py`: `ReadAheadChunked` (Zonal-only cache class).
  - `gcsfs/utils.py`: `is_empty_range`, `as_byte_buffer`.
  - `gcsfs/telemetry/`: `context.py`, `detector.py`, `features_proto.py`, `grpc.py`, `manager.py`, `schema.py`, `wrapper.py`.
  - `gcsfs/mapping.py`, `gcsfs/inventory_report.py`, `gcsfs/dask_link.py`, `gcsfs/concurrency.py`, `gcsfs/cli/gcsfuse.py`.

## 2. HTTP Transport, Timeouts, Retries & Auth/mTLS

- **`DEFAULT_REQUESTS_TIMEOUT` (`core.py`)**: `aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=60)`.
  - `sock_connect` / `sock_read` raise `aiohttp.ServerTimeoutError` (a subclass of `aiohttp.ClientError`, which is in `RETRIABLE_EXCEPTIONS` and retried by `retry_request`).
  - Setting a numeric `total` timeout caps legitimate large uploads/downloads AND raises `asyncio.TimeoutError` (NOT a `ClientError`, NOT retried by `retry_request`; converted to `FSTimeoutError` in sync mode). Never recommend or accept a fixed `total` timeout on data-transfer `aiohttp` calls.
- **`validate_response` & `retry_request` (`retry.py`)**:
  - `status == 499` is ignored on cancelled resumable uploads (do not turn 499 into an error).
  - `status == 401` is explicitly excluded from the `"invalid" in str(msg)` -> `ValueError` rule so `"Invalid Credentials: invalid_token"` remains a retriable `HttpError` (triggers credential refresh + retry).
  - `HttpError` and `GenerationMismatchError` define `__reduce__` so exceptions survive multiprocessing/Dask/Ray worker pickling.
- **URL quoting & header redaction (`core.py`)**:
  - `GCSFileSystem._format_path` runs `quote(p)` (`safe=""`) **only** on positional `*args` passed to `_call` / `_request`. Unquoted f-strings or string concatenation (`f"b/{bucket}/o/{key}"`) bypass quoting and allow `/`, `?`, `#`, or `..` injection. (Pre-existing legacy unquoted sites exist in `_make_bucket_requester_pays`, `_rmdir`, `_info` — never copy them or introduce new ones.)
  - `_call` logs `kwargs.get("headers")` at `DEBUG` **before** `_get_headers` injects `Authorization`. Never pass `Authorization`, `Proxy-Authorization`, or CSEK (`X-Goog-Encryption-Key*`) inside `headers=`.
- **Credentials (`credentials.py`)**:
  - `_get_creds_from_raw_token` uses a context-managed `requests.Session()`, checks `LOCAL_REFRESH_BUFFER = 300`s unless `FETCH_RAW_TOKEN_EXPIRY=false`, and wraps network/5xx errors into `OSError(...) from None` so raw token strings never leak in chained tracebacks.
  - Legacy `~/.gcs_tokens` cache uses `pickle`; never expand its scope or add new pickle deserialization paths.
  - Any new credential/key/cert file must use `0o600` (`os.open(..., 0o600)`) or live inside a `tempfile.TemporaryDirectory()` (`0o700`) and be cleaned up promptly.
- **mTLS & universe domain (`core.py`)**:
  - mTLS uses `google.auth.transport.mtls` and `_client_cert_ssl_context` (temporary cert/key in `TemporaryDirectory`, `http/1.1` ALPN).
  - `ExtendedGcsFileSystem._grpc_location` deliberately omits the mTLS endpoint because `AsyncGrpcClient` does not present a client certificate.
  - `_gcp_universe_domain()` normalizes empty string `""` to `"googleapis.com"` and passes it explicitly to `ClientOptions` so GAPIC does not raise `EmptyUniverseError`.

## 3. Paths, Versioning, Preconditions & Consistency Checkers

- **`_strip_protocol` trailing-slash quirk (`core.py`)**:
  - `_strip_protocol` intentionally preserves trailing `"/"` so 0-byte directory marker objects (`"b/d/"`) remain distinguishable from files (`"b/d"`).
  - Known fallout to watch in any path-manipulating PR: `_parent("b/d/")` returns `"b/d"` (self), `glob("b/*/")` can return `[]`, `walk("b/d/")` can emit self-children, `DirFileSystem` hits `AssertionError` on trailing slashes, and `FSMap.dirfs` fails. Always check how a PR normalizes trailing `"/"`.
- **Generations & preconditions (`core.py`)**:
  - `_coalesce_generation`, `_resolve_generation_match`, and `_resolve_open_generation_match` must:
    - Reject `bool` explicitly (`isinstance(True, int)` is `True` in Python!).
    - Reject negative values and non-ASCII digits.
    - Check `is not None` (never truthiness `if gen:` — `if_generation_match=0` is the create-only precondition; dropping `0` silently turns exclusive create into unconditional overwrite).
  - `_translate_generation_mismatch` maps HTTP 412 (`FileExistsError`) to `GenerationMismatchError` when `if_generation_match > 0` while keeping `FileExistsError` for `if_generation_match == 0`.
  - `_strip_generation_match` strips `&ifGenerationMatch=\d+` before issuing `DELETE` on resumable upload session URLs.
- **Resumable uploads & zero-copy buffers (`core.py`, `utils.py`)**:
  - Non-final chunks of a resumable upload must be multiples of `GCS_MIN_BLOCK_SIZE = 2**18` (256 KiB) and at most `GCS_MAX_BLOCK_SIZE = 2**28` (256 MiB).
  - `as_byte_buffer` avoids copying `bytes` and handles zero-sized multi-dimensional NumPy arrays (e.g. `np.empty((0, 3))` where `memoryview.cast("B")` raises `TypeError`/`ValueError`).
- **Consistency checkers (`checkers.py`)**:
  - Zonal (`RAPID`) objects have no `md5Hash` header/metadata. When a caller explicitly requests `consistency="md5"` and the hash is absent, the checker must raise `NotImplementedError` — never silently downgrade `"md5"` to `"size"`.

## 4. Directory Cache (dircache & HnsDirCacheUpdater)

- **Flat cache (`DirCacheUpdater`, `_dircache.py`)**:
  - `_write_file_cache_update` pops the immediate `parent` entry if cached, and otherwise calls broad `invalidate_cache(parent)` because GCS writes can implicitly materialize missing intermediate virtual directories.
- **HNS cache (`HnsDirCacheUpdater`, `_dircache.py`)**:
  - Uses **only** `_is_bucket_hns_cached` — never sends a layout RPC after a mutation.
  - Normalizes `path1` and `path2` via `_strip_protocol(...).rstrip("/")`.
  - `_rm_files` (`gcsfs/core.py`) must pass **only** paths in `deleted + confirmed_absent` (HTTP 200/204/404 or gRPC `NOT_FOUND`) to `HnsDirCacheUpdater._rm_files_cache_update` (`gcsfs/_dircache.py`). Never mutate the HNS cache in place for failed or ambiguous deletes (unlike flat invalidation, stale in-place edits do not self-correct).
- **Known `invalidate_cache` quirk (`core.py`)**:
  - `GCSFileSystem.invalidate_cache(path)` walks **up** ancestor directories only (it does not clear cached descendants under `path`) and does not call `super().invalidate_cache(path)`. PRs deleting or renaming directory subtrees must explicitly clear descendants if needed.

## 5. Bucket Layout Classification & RCU Lifecycle

- **7 real `BucketLayout` combinations + `UNKNOWN` (`extended_gcsfs.py`, `tests/layouts.py`)**:
  - `STANDARD`, `STANDARD_HIERARCHICAL`, `ZONAL_HIERARCHICAL`, `STANDARD_RCU`, `HIERARCHICAL_RCU`, `STANDARD_RAPID_CACHE`, `HIERARCHICAL_RAPID_CACHE`, plus `BucketLayout.UNKNOWN`.
  - Every new branch on bucket type must be checked across all 7 layouts + `UNKNOWN`.
- **Cache separation (`extended_gcsfs.py`)**:
  - `_storage_layout_cache` stores **only** authoritative answers from `get_storage_layout`.
  - `_storage_layout_fallbacks` stores `BucketLayout.UNKNOWN` fallbacks separately with `_LAYOUT_FALLBACK_TTL = 300.0`s so transient errors or temporary permission issues self-heal instead of permanently poisoning the bucket layout. Never write `UNKNOWN` into `_storage_layout_cache`.
- **`_classify_layout_error` (`extended_gcsfs.py`)**:
  - `NotFound` -> `MISSING` (raised immediately, never cached).
  - `PermissionDenied`, `InvalidArgument`, `FailedPrecondition`, `MethodNotImplemented` -> `FALLBACK`.
  - `Unauthenticated` -> `FALLBACK` **only** when using `AnonymousCredentials` or `unverified_client_cert` (otherwise raise — never silently swallow broken credentials).
  - Other errors -> `RAISE` if `serves_control_api` (default endpoint or `_control_api_learned` is `True`), else `FALLBACK` (allowing one unretried probe against custom `https://` endpoints). Plain `http://` endpoints skip Storage Control lookup entirely.
- **RCU (`is_rcu`) lifecycle (`extended_gcsfs.py`)**:
  - Reads on RCU buckets always use gRPC (`MRD`).
  - Writes on RCU buckets use HTTP by default; gRPC `AAOW` (`storage_class="RAPID"`, `finalize_on_close=True`) is used only when `rcu_rapid_write=True`.
  - On RCU `GoogleAPICallError`, `_rcu_disabled_after_error` re-checks the bucket layout once and returns `bool`: `_init_aaow` raises `RapidCacheUltraDisabledError` (caught by `_put_file`, `_pipe_file`, `initiate_upload`, and `simple_upload` to retry over HTTP, and surfaced by `ZonalFile`), while `_perform_parallel_upload` retries `upload(None)` without `storage_class="RAPID"`. **Every** write entry point must stay consistent with this recovery path.

## 6. HNS Folder Operations & LRO Poller

- **Coexisting file and folder (`extended_gcsfs.py`)**:
  - In GCS HNS, a file `"a"` and a folder `"a/"` can coexist under the same parent.
  - `_create_hns_folder` checks `_info(path)` first to preserve file/folder collision semantics.
  - `_find_hns_entries` and `_expand_path_with_details` key entries by `(name, type)` (never `name` alone, which drops one of the two).
  - `_rm_bucket_paths` deletes deepest-first by slash depth, handles coexisting objects + folders, and passes `delete_placeholder=not (recursive and maxdepth is None)` because boundary directories under a finite `maxdepth` have their `"dir/"` placeholder filtered out by `_find`.
- **LRO poller (`poller.py`)**:
  - `_wait_keeping_cancel` uses `asyncio.wait` rather than `asyncio.wait_for` to avoid the Python 3.10/3.11 lost-cancellation bug.
  - Polls with `operation.done(retry=None)` so GAPIC does not nest internal retries inside the poller's retry loop.
  - `unwrap_operation_result` converts untyped base `GoogleAPICallError` from `AsyncOperation` into typed exceptions via `from_grpc_status`.
  - On `asyncio.CancelledError`, dispatches `asyncio.wait_for(operation.cancel(), 15)` held in the module-level `_BACKGROUND_TASKS` strong set so GC cannot destroy the cancellation task mid-flight.
- **`rename_folder` fallback guard (`extended_gcsfs.py`)**:
  - Falling back from `rename_folder` to copy+delete after `rename_submitted` is permitted **only** when `operation.operation.done and operation.operation.HasField("error")`. Never fall back after `TimeoutError` — the server-side rename may still be running and complete concurrently, corrupting data.

## 7. Zonal & RCU gRPC Read Path (MRDPool, cat_ranges, ctypes Zero-Copy)

- **`MRDPool` & `MRDPoolCache` (`zb_hns_utils.py`)**:
  - Reserves slots (`_active_count` / `_creating_count`) under `self._cond`, opens gRPC streams **outside** the lock, and wakes waiters via `asyncio.shield(self._notify_all())` so cancellation cannot strand waiting tasks.
  - Round-robin sharers use a lock-free `_inflight` counter on each MRD wrapper.
  - `_is_stale` rejects idle cached MRDs whose `persisted_size < pool.persisted_size` (Zonal appends and finalization preserve object generation while growing `persisted_size`!) and parks them in `_stale_mrds` for deferred close.
  - **Known gap to watch**: neither `MRDPool` nor `MRDPoolCache.release` currently checks `mrd.is_stream_open`. After `DataCorruption`, `AsyncMultiRangeDownloader` closes itself; recycling a closed MRD poisons subsequent reads in the pool. Any PR touching MRD acquisition/release/recovery should guard `is_stream_open`.
  - `MRDPoolCache` holds `weakref.ref(gcsfs)` to avoid reference cycles, keys pools by `(bucket, object, str(gen) if gen is not None else None, cache_type)`, and protects active keys from LRU eviction via `_incref`/`_decref`. Use `extended_gcsfs._close_pool(pool)` (shielded, defined in `gcsfs/extended_gcsfs.py`) on cancellation-sensitive teardown paths.
- **`download_ranges` & `_cat_ranges` (`zb_hns_utils.py`, `extended_gcsfs.py`)**:
  - Respect `MRD_MAX_RANGES = 1000` per `download_ranges` call.
  - Guard `length == 0` **before** calling `download_ranges`: GCS `BidiReadObject` interprets `read_length=0` as "read from offset to EOF", causing silent full-file reads / OOM.
  - Pass `**_download_ranges_kwargs(type(mrd))` (`accepts_memoryview=True` when supported) on **every** `mrd.download_ranges(...)` call site. Missing it silently disables zero-copy on that path (known silent non-conflicting merge hazard).
  - `init_mrd` calls `_enable_zero_copy_bidi_read()` (`os.environ.setdefault("GCS_PYTHON_CLIENT_ZERO_COPY_BIDI_READ", "1")`).
  - `_cat_ranges` uses D'Hondt stream allocation (`_allocate_streams`), LPT bin-packing (`_split_ranges_across_streams`), and `asyncio.wait` (not `asyncio.gather`) for deterministic single cancellation and clean pool teardown.
- **`DirectMemmoveBuffer` & `PartialView` (`zb_hns_utils.py`)**:
  - Bind CPython C-API symbols **only** on the module's private `_pythonapi = ctypes.PyDLL(...)`. Never mutate `argtypes`/`restype` on shared `ctypes.pythonapi`.
  - `PyBytes_FromStringAndSize.restype` must remain `ctypes.py_object` so a `NULL` return raises `MemoryError` instead of returning `0`.
  - Every `PyObject_GetBuffer` must pair with `PyBuffer_Release` in a `finally` block.
  - `THRESHOLD_BYTES_FOR_SCHEDULING = 128 KiB`: below threshold, copies run inline; above threshold, copies offload to the thread pool.
  - Whole-buffer fastpath is restricted to exact `type(data) is bytes` (small `memoryview` covering the whole buffer copies via `.tobytes()` to hit the `bytes` fastpath).
  - `owned_view.release()` must run in `_do_memmove`'s `finally` block AND on any exception before `_submit_write`.
  - Preserve `_allocated_intervals` non-overlap checks, `close()` waiting on `_done_event.wait()` before releasing buffer ownership, and `get_value()` rejecting `_total_bytes_written < expected_size` (short-read guard against returning uninitialized memory).
- **`ReadAheadChunked` (`caching.py`)**:
  - Registered only when `zonal_file.py` is imported. Calls `fetcher(start=..., chunk_lengths=...)`, which **only** `ZonalFile._fetch_range` supports (`GCSFile._fetch_range` raises `TypeError`).

## 8. Zonal & RCU gRPC Write Path (ZonalFile, AAOW, Parallel Upload)

- **`ZonalFile` (`zonal_file.py`)**:
  - Validates `_resolve_open_generation_match` and `_validate_grpc_write_kms_key_name` **before** opening an MRD pool or `AAOW` (avoiding resource leaks and premature object overwrite on invalid inputs).
  - Uses `handoff_lock` during open so interrupting `asyn.sync` closes any opened MRD pool cleanly.
  - Omits `self.size` from `is_empty_range` checks because `self.size` can be stale on unfinalized/appended Zonal objects.
  - Streams `write()` directly to `aaow.append(as_byte_buffer(data))` (call `as_byte_buffer` before opening/mutating writer state).
  - `finalize_on_close` defaults to `False` on Zonal buckets and `True` on RCU buckets.
  - `_abort_write()` sets `finalize_on_close=False, forced=True` on error or exception inside a `with` block. `ZonalFile.__exit__` skips opening a writer if `aaow is None` and aborts without masking the caller's exception. **Caveat**: `fsspec.open()` (`OpenFile`) and `io.TextIOWrapper` bypass `ZonalFile.__exit__` and call `close()` directly.
  - `_close_impl` invalidates directory cache on failed flush and tears down `mrd_pool` and `aaow` independently (an error closing one must not skip closing the other).
  - `close_aaow` re-raises close errors **only** when `finalize_on_close=True`.
- **CMEK fail-closed rules (`extended_gcsfs.py`, `zonal_file.py`)**:
  - `_validate_grpc_write_kms_key_name` rejects RCU gRPC writes with `NotImplementedError`, requires `zb_hns_utils.Preconditions` SDK support (`NotImplementedError` if absent), and rejects `/cryptoKeyVersions/` paths with `ValueError`.
  - `ab` append mode uses `_resolve_kms_append` via direct HTTP `objects.get` (never `_info`, which can resolve to an HNS folder) and retries creation with `if_generation_match=0` on 404 race.
- **Parallel upload (`parallel_upload.py`, `extended_gcsfs.py`)**:
  - Dual import: tries `google.cloud.storage.asyncio.async_transfer_manager`, then falls back to `google.cloud.storage._experimental.asyncio.async_transfer_manager`.
  - `_may_use_parallel_upload` requires: `if_generation_match is None`, `storage_class != StorageClass.UNKNOWN`, finalized writes on gRPC path, non-empty metadata keys/values on HTTP path, `os.path.isfile(lpath)` (rejects `/dev/null`, FIFOs, pipes), and `size >= threshold` (`DEFAULT_PARALLEL_UPLOAD_THRESHOLD = 128 MiB`).
  - Config flags must use `_parse_bool_flag` / `_parse_int_flag` to handle both JSON-decoded `FSSPEC_GCS_*` values (`bool`/`int`) and INI string values (`"true"`/`"128"`).

## 9. File Lifecycle, Adaptive Prefetcher, Pickling & Teardown

- **`GCSFile` & `AdaptiveReadaheadCache` (`core.py`)**:
  - Default `cache_type="adaptive"` (from `fsspec>=2026.9.0`); falls back to `"readahead"` with empty `{}` `cache_options` when `USE_EXPERIMENTAL_ADAPTIVE_PREFETCHING=false`.
  - Rebinds `BackgroundPrefetcher` to `gcsfs.loop` if `prefetcher.loop is not fs_loop` (workaround for `fsspec==2026.9.0` before `filesystem_spec#2292`), wires `prefetcher.producer.fetcher = self._async_fetch_range`, and disarms `cache.close = lambda: None` so GC does not block on the event loop.
  - `_abort_open` explicitly clears local references (`cache = prefetcher = None`) before re-raising so exception tracebacks do not pin `/tmp` file descriptors.
- **Deferred close & concurrent download teardown (`core.py`, `zb_hns_utils.py`)**:
  - `GCSFile.__del__` and on-loop-thread closes (`_on_loop_thread(loop)`) offload teardown to the single daemon thread `"gcsfs-deferred-close"` (`_defer_close`) to avoid blocking the event loop or deadlocking GC.
  - `_get_file_concurrent` must drain all `pending_writes` (`os.pwrite` worker futures) even under `asyncio.CancelledError` **before** calling `os.close(fd)` (preventing EBADF or writing to a recycled FD).
  - **Known teardown traps**:
    - `GCSFileSystem.close_session` currently calls non-threadsafe `loop.create_task(session.close())` from a `weakref.finalize` callback — never copy this pattern into new code; use `_on_loop_thread` + `_defer_task` / `asyncio.run_coroutine_threadsafe` / `sync_teardown` and close `_connector` when no running loop exists.
    - `ExtendedGcsFileSystem._close_resources()` is not registered on GC/atexit; PRs adding gRPC clients/pools must verify lifecycle ownership.
    - Never hold a blocking `threading.Lock` across `loop.create_task()` (Python 3.12+ eager task factories run the coroutine synchronously to the first `await`, causing self-deadlock).
- **Pickling (`core.py`, `zonal_file.py`)**:
  - `GCSFile.__reduce__` and `ZonalFile._reopen_kwargs` must preserve `generation`, `_reopen_cache_options`, `consistency`, `timeout`, `pool_size`, and keep `cache_type=None` when using the default so unpickling respects target worker defaults.
  - Never include live loops, aiohttp connectors, gRPC channels, or unredacted tokens in pickled state (`storage_options`, `__reduce__`, `__getstate__`).

## 10. Usage Telemetry Architecture & Wire Schema

- **Wire format (`telemetry/context.py`, `telemetry/features_proto.py`, `telemetry/schema.py`, `telemetry/grpc.py`)**:
  - Header/metadata keys: `HTTP_HEADER_KEY = "X-Goog-Storage-Gcsfs-Features"` (Title-Case for HTTP) and `GRPC_METADATA_KEY = "x-goog-storage-gcsfs-features"` (strictly lowercase; gRPC rejects uppercase metadata keys).
  - Payload: URL-safe base64-encoded `GcsfsFeatures` protobuf.
  - **Reserved `GcsfsMethod` wire IDs (`2, 42, 43, 45`)**: NEVER reuse or renumber existing or reserved enum values.
  - Every tracked token name in `schema.py` / `wrapper.py` must match the lowercased enum suffix in `features_proto.py`.
- **Method wrapping & 6 execution flows (`telemetry/wrapper.py`)**:
  - `wrapper.py` handles 6 execution flows: tracked async methods, synced wrappers, inherited `fsspec` methods, generator methods (`_snapshot_generator_tokens`), file-object methods (`wrap_file_methods`), and mirrored methods (`mirror_gcs_methods`).
  - Any new GCSFS-only `async def _foo` must define `foo = asyn.sync_wrapper(_foo)` AND exist on both `GCSFileSystem` and `ExtendedGcsFileSystem` (or be inherited cleanly).
  - `wrap_file_methods` wraps only `cls.__dict__` (not inherited methods).
  - Mixins providing tracked methods must precede `GCSFileSystem` in the MRO.
- **`fork()` + `exec()` safety (`telemetry/wrapper.py`)**:
  - `bind_gcs_sync_methods(self)` intentionally creates an instance reference cycle. Do **not** "clean it up" or delete it: in `fork()`+`exec()` workflows (`subprocess.Popen(..., preexec_fn=...)`, e.g. `ray.init()`), `fsspec` clears `cls._cache` post-fork; the ref-cycle prevents immediate refcount-zero destruction of inherited `grpc.aio` channels (which would hang the child inside `PollerCompletionQueue.shutdown()` before `exec()`).
- **Privacy & non-blocking detection (`telemetry/detector.py`, `telemetry/manager.py`)**:
  - Only boolean truthiness of allowlisted flags (`recursive`, `detail`, `withdirs`, `prefix`, `cache_options`) may be recorded — never raw values, bucket/object paths, or credentials.
  - `DO_NOT_TRACK` and `GCSFS_NO_TELEMETRY` must short-circuit at zero cost before creating contexts or starting probes.
  - `EnvironmentDetector` must never block callers, must set `trust_env=False` on metadata server probes, must use `fsspec.asyn.get_loop()`, and must preserve `os.register_at_fork` (`_reset_detectors_after_fork`) to reset locks and generation counters.

## 11. Tests, CI, Cloud Build & Benchmark Invariants

- **Broken pytest marks trap (`gcsfs/tests/conftest.py`)**:
  - `@pytest.mark.requires_real_gcs`, `@pytest.mark.requires_hns`, and `@pytest.mark.requires_rapid` **silently do NOT skip** (there is no `pytest_collection_modifyitems` hook wiring them). Tests MUST import the `pytest.mark.skipif` decorators `requires_real_gcs`, `requires_hns`, and `requires_rapid` from `gcsfs.tests.conftest` (whereas `skip_if_zonal(gcs)` and `skip_if_grpc_read(gcs)` are runtime helpers called inside test bodies, not decorators).
- **Autouse gRPC guard (`_forbid_remote_grpc_connections`, `conftest.py`)**:
  - Unit tests forbid opening remote gRPC channels because concurrent gRPC C-core `getenv()` calls racing with `monkeypatch.setenv()` cause glibc `SIGSEGV` on Ubuntu 24.04 CI.
  - Core unit tests must instantiate `CoreGCSFileSystem(token="anon", skip_instance_cache=True)` (from `gcsfs.core import GCSFileSystem as CoreGCSFileSystem`) or stub gRPC clients — using `gcsfs.GCSFileSystem()` pulls `ExtendedGcsFileSystem` under default `GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT=true`.
- **Emulator cleanup & real-GCS IAM scope (`conftest.py`)**:
  - `final_cleanup` skips mocked unit tests without Docker unless `STORAGE_EMULATOR_HOST` is set.
  - Any temporary bucket created against real GCS MUST start with `gcsfs-test-` (IAM permissions are scoped to that prefix).
- **Specific test traps**:
  - `test_init.py` keeps `gcsfs.telemetry.features_proto` in `_KEEP_MODULES` because reloading protobuf modules triggers descriptor-pool double-registration errors.
  - Call `mock.reset_mock()` whenever mocks capture `memoryview` arguments (otherwise held views keep buffers pinned / trigger `BufferError`).
  - Use `fake_clock` or `asyncio.sleep(0)` instead of real wall-clock `asyncio.sleep(...)`.
  - Adaptive prefetcher tests should pass `cache_type="adaptive"` explicitly; telemetry tests must isolate `default_usage_tracker`.
- **CI, Codecov, Benchmarks & Cloud Build**:
  - `conftest.py`'s `pytest_ignore_collect` skips **all** of `gcsfs/tests/perf/**` unless `--run-benchmarks` or `--run-benchmarks-infra` is passed. Any new benchmark infra unit test MUST be wired into `.github/workflows/ci.yml` (`--run-benchmarks-infra`).
  - Codecov enforces an **80% patch coverage target** (0% drop). Dead/untested defensive branches in new code block CI.
  - `cloudbuild/` E2E scripts (`cloudbuild/create_buckets.sh`, `cloudbuild/run_tests.sh`, `cloudbuild/cleanup.sh`, `cloudbuild/e2e-tests-cloudbuild.yaml`) must keep worker counts in lockstep. Quote shell variables (`"$VAR"`) and pass comma/bracket Helm values with `--set-literal`.
  - Dependency policy: no upper bounds on runtime dependencies in `pyproject.toml`; keep `environment_gcsfs.yaml` synced with `pyproject.toml` (preferring conda packages); follow `CONTRIBUTING.md`'s `fsspec` lower-bound policy before bumping `fsspec>=...`.
