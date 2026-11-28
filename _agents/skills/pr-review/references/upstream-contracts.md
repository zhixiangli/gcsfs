# Upstream Library Contracts for `gcsfs` Reviews

Pre-verified contracts for `gcsfs`'s upstream stack. In Step 5 fact-checking:
1. **Tier 1 (preferred)**: inspect the live installed environment (or a local sibling checkout if present) using the commands below.
2. **Tier 2**: use version-tagged entries here as verified authority when `inspect` cannot reach compiled C/C++/Cython extensions (`grpcio` C-core) or when auditing version skew against `pyproject.toml` floors (e.g., `fsspec==2026.9.0` vs CI HEAD). Entries marked **HEAD-only note** must be verified against the target version before flagging.
3. **Tier 3**: anything unconfirmed by Tier 1 or Tier 2 goes under `Questions` or `Not verified`, never `Must fix`.

## Live Source Verification Commands

```bash
# Locate and print any pure-Python upstream class/method in the active environment
python -c "import inspect, fsspec.asyn; print(inspect.getfile(fsspec.asyn.AsyncFileSystem)); print(inspect.getsource(fsspec.asyn.AsyncFileSystem._expand_path))"
python -c "import inspect, google.cloud.storage.asyncio.async_multi_range_downloader as m; print(inspect.getsource(m.AsyncMultiRangeDownloader.download_ranges))"
python -c "import inspect, google.cloud.storage.asyncio.async_appendable_object_writer as m; print(inspect.getsource(m.AsyncAppendableObjectWriter.append))"
python -c "import inspect, google.cloud.storage.asyncio.async_grpc_client as m; print(inspect.getsource(m.AsyncGrpcClient))"
python -c "import inspect, google.cloud.storage_control_v2.services.storage_control.async_client as m; print(inspect.getfile(m))"
python -c "import importlib.metadata as im; [print(p, im.version(p)) for p in ('fsspec','google-cloud-storage','google-cloud-storage-control','google-api-core','grpcio','aiohttp')]"
```

## 1. fsspec (`filesystem_spec`)

**Verified against:** `fsspec==2026.9.0` (`pyproject.toml` floor) vs `filesystem_spec` git HEAD (`9557b405`, installed by `.github/workflows/ci.yml`).

### Floor-Bound Contracts (`fsspec>=2026.9.0`)

- **`AsyncFileSystem` (`fsspec.asyn`)**:
  - `sync()` polls on a 1-second loop and drops the concurrent `Future` on `KeyboardInterrupt` or timeout **without cancelling** the coroutine on the background IO loop.
  - `_runner` catches `Exception` (not `BaseException`), so `asyncio.CancelledError` in a coroutine is swallowed and returns `None` instead of propagating.
  - Constructing with `asynchronous=True` sets `self._loop = None`; code assuming `self.loop` is always non-`None` crashes under native async callers.
  - `_run_coros_in_chunks` does not call `.close()` on unscheduled coroutines when an earlier chunk fails (emitting `RuntimeWarning: coroutine was never awaited`).
  - `batch_size=0` is inconsistent across base methods (`_get` raises `ValueError` while others treat `0` as `None`).
  - `fsspec.asyn.sync_teardown` **already exists** in `fsspec==2026.9.0` (avoid adding duplicate teardown helpers when the base helper fits).
- **`_Cached` & serialization (`fsspec.spec`)**:
  - `_tokenize` hashes `md5(str(args + (kwargs,)))` — sensitive to dict key insertion order; `__dask_tokenize__` includes `os.getpid()`.
  - `skip_instance_cache=True` is popped **after** tokenization and dropped from `storage_options`.
  - `to_dict(include_password=False)` does **not** redact `token` (only `password` keys) — `gcsfs` must guard credentials itself.
  - Event loops, `aiohttp.TCPConnector`, and gRPC channels are unpicklable; keep them out of `storage_options` and `__reduce__` state.
- **`AbstractBufferedFile` & caches (`fsspec.spec`, `fsspec.caching`)**:
  - `AbstractBufferedFile.__exit__` ignores exception arguments and calls `self.close()`, which **commits partial `GCSFile` writes on error** unless a subclass overrides `__exit__` (like `ZonalFile`).
  - `fsspec.open()` (`OpenFile`) and `io.TextIOWrapper` bypass subclass `__exit__` and invoke `close()` directly.
  - `cache.size` is frozen at file open time (causes stale-EOF bugs if an object grows via Zonal append).
  - `check_contained` uses `os.path.abspath` (not `os.path.realpath`), so pre-existing local symlinks can bypass containment unless guarded.
- **Bulk methods, `DirFileSystem` & `FSMap` (`fsspec.implementations.dirfs`, `fsspec.mapping`)**:
  - `GCSFileSystem._strip_protocol` keeps trailing `"/"`, which breaks `DirFileSystem` assertions, `FSMap.dirfs`, `_parent("b/d/")`, and `glob("b/*/")`.
  - `FSMap` delegates to `cat`/`rm`, which interpret glob metacharacters (`*`, `?`, `[`) in object keys unless escaped or bypassed.
  - Base `_du` performs N+1 `_info` calls unless overridden; base `_mv_file(path1, path2)` does not accept `**kwargs`.

### CI-on-HEAD vs `fsspec==2026.9.0` Skew Table

GitHub Actions installs `filesystem_spec` from git `main`, while PyPI users install `fsspec>=2026.9.0`. A PR relying on any post-`2026.9.0` behavior below **passes CI on HEAD but crashes or misbehaves on `fsspec==2026.9.0`** unless `gcsfs` handles `2026.9.0` or maintainers approve bumping the `pyproject.toml` floor:

| Upstream PR(s) | Post-`2026.9.0` behavior (absent on `2026.9.0`) | Impact on `gcsfs` PRs |
|---|---|---|
| `#2151` | `AsyncFileSystem._expand_path(..., withdirs=)` kwarg; `cat(dir, recursive=True)` skips dirs | Passing `withdirs=` to `super()._expand_path` raises `TypeError` on `2026.9.0` |
| `#2292`, `#2340` | `AdaptiveReadaheadCache` binds `BackgroundPrefetcher` to `fs.loop` and drains tasks after EOF | Removing `gcsfs`'s loop-rebind workaround breaks `2026.9.0` |
| `#2186` | `_Cached._TokenLock` concurrent-init deduplication | Assuming deduplicated concurrent `GCSFileSystem()` init fails on `2026.9.0` |
| `#2242`, `#2210` | `atexit` `_stop_io_loop` & dead `iothread` check in `sync()` | Assuming clean loop shutdown or dead-thread detection fails on `2026.9.0` |
| `#2073` | Eager `_lock` init in `get_loop()` | Lazy lock race still exists on `2026.9.0` |
| `#2295` | Breaking exception traceback ref-cycles in `sync()` | Keep `_abort_open` local ref clearing on `2026.9.0` |
| `#2376`, `#2477` | Async `get` relative-path `os.makedirs("")` fix & `stringify_path` | Relative local destinations still fail in base `2026.9.0` |
| `#2205`, `#2441` | `mv` self-move and move-into-self guards | Base `mv` lacks self-move guards on `2026.9.0` |
| `#2487`, `#2213`, `#2212`, `#2110` | `DirFileSystem` `_resolves_dotdot`, `async_impl`/`local_file` mirroring, `_join` trailing-`/` preservation | `DirFileSystem` tests passing on CI HEAD may fail on `2026.9.0` |
| `#2319`, `#2407`, `#2301`, `#2245`, `#2260`, `#2344` | `BlockCache` EOF clamp, `FirstChunkCache` `start >= size`, `MMapCache.close()`, multi-byte `readuntil`, `readinto` closed-file check, `AbstractBufferedFile.len` | Relying on these cache/file fixes breaks `2026.9.0` users |
| `#2375` | `to_dict` nested-dict `password` redaction (still never redacts `token`) | Nested secrets still exposed on `2026.9.0`; `token` unredacted on both |
| `#2287` | `FSMap.setitems` calling `mkdirs` per unique parent | `FSMap` batch writes behave differently on `2026.9.0` |

## 2. google-cloud-storage

**Verified against:** `google-cloud-storage==3.15.0..3.17.1` (`pyproject.toml` floor `>=3.15.0` where applicable) and git HEAD (`842cb11c27f`).

### `AsyncGrpcClient` (`google.cloud.storage.asyncio.async_grpc_client`)

- **`AnonymousCredentials` plaintext trap**: passing `AnonymousCredentials()` unconditionally creates a **plaintext `grpc.aio.insecure_channel`**, breaking TLS public buckets and dropping `client_info` and channel options.
- **`attempt_direct_path=True` default**: `_modify_target_for_direct_path()` strips `:port` and rewrites the target to `google-c2p:///host`. Always require `attempt_direct_path=False` for custom endpoints, emulators, or explicit host:port targets.
- **Pre-built `channel=` 4 MiB cap**: passing a pre-built `channel` bypasses GAPIC's default `max_send_message_length=-1` / `max_receive_message_length=-1`, silently reverting gRPC to its **4 MiB** message limit.
- **Lifecycle & options gaps**: `AsyncGrpcClient` exposes no `close()` method, no `interceptors=` parameter, and no default keepalive options; it mutates `DEFAULT_CLIENT_INFO` and drops custom `client_info` on `x-goog-api-client`; it ignores `universe_domain` when `api_endpoint` is unset; `BucketMetadataCache` retains a reference cycle.

### `AsyncMultiRangeDownloader` (`google.cloud.storage.asyncio.async_multi_range_downloader`)

- **Range caps & EOF semantics**: hard cap of 1,000 ranges per `download_ranges()` call (chunked 100/request internally); `read_length == 0` means **read to EOF** (never 0 bytes).
- **`persisted_size` staleness**: populated **only** on initial open when no `read_handle` is supplied, and never refreshed on subsequent reads. Redirected opens with `read_handle` leave `persisted_size is None`.
- **`DataCorruption` closes the MRD**: on checksum mismatch (`DataCorruption`), the MRD closes its underlying stream. Any pool recycling an MRD without checking `mrd.is_stream_open` poisons future callers.
- **`_StreamMultiplexer.reopen_stream` hazards**: reopening broadcasts a non-retryable `Exception("Stream reopening")` to concurrent sharers (breaking round-robin callers sharing one MRD) and drops incoming chunks after a 20-second queue stall.
- **Zero-copy wire parser & CRC32C**: `BidiReadObject` zero-copy parser returns read-only `memoryview` slices aliasing immutable gRPC `bytes` payloads (callers must respect view lifetime); integrity verification is CRC32C-only (requires `google-crc32c` C extension; offloaded to a 2-thread executor).
- **HEAD-only note (`028ad000c78`..`842cb11c27f`, verify in installed version before flagging)**: recent HEAD commits fixed `CancelledError` swallowing and `_broken_generation` stream recovery; a subtle close leak can still occur if `multiplexer.close()` cancels before `read_obj_str.close()` finishes.

### `AsyncAppendableObjectWriter` (`google.cloud.storage.asyncio.async_appendable_object_writer`)

- **Chunking & flush geometry**: writes in 2 MiB chunks with 16 MiB default flush (`FLUSH_INTERVAL_BYTES` must be a multiple of 2 MiB).
- **Immediate overwrite & metadata drop**: `open()` without preconditions immediately overwrites an existing object. Opening with `generation > 0` sends `AppendObjectSpec`, which **ignores** `Blob` metadata and `kms_key_name`.
- **Retry gaps & data-loss bug**: `flush()`, `simple_flush()`, and `state_lookup()` have **no retry**. In `append()` retry (`3.15.0..3.17.1`), `BytesIO(data).seek(persisted_size)` seeks the per-call buffer by the **cumulative** object `persisted_size` (upstream data-loss bug on mid-stream retry after >1 chunk — belongs upstream, do not rely on AAOW `append()` retry for correctness).
- **Public stream state**: public property `.is_stream_open` exists (`gcsfs` legacy code reads private `._is_stream_open`; prefer public `.is_stream_open` in new code).
- **Stream leak**: `_AsyncWriteObjectStream.open()` / `.close()` can leak `AsyncBidiRpc` on error or cancellation.

### `async_transfer_manager` vs sync `transfer_manager`

- **HEAD-only note (`842cb11c27f`, verify in installed version before flagging)**: `google.cloud.storage._experimental.asyncio.async_transfer_manager` is a stub raising `NotImplementedError` on HEAD and lacks `preconditions`/`metadata` parameters.
- Sync `transfer_manager.upload_chunks_concurrently` (XML MPU) ignores `timeout` on workers and blocks in `ThreadPoolExecutor.__exit__` past `deadline`.

## 3. google-cloud-storage-control & google-api-core

**Verified against:** `google-cloud-storage-control==1.14.0..1.17.0` and `google-api-core`.

- **`StorageControlAsyncClient` (`google.cloud.storage_control_v2`)**:
  - `get_storage_layout` returns `StorageLayout` with `location_type`, `hierarchical_namespace.enabled`, and `rapid_cache_info.cache_type` (`"RAPID_CACHE_ULTRA"` / `"RAPID_CACHE"`).
  - `folder_id` and non-empty `list_folders` `prefix` **must end with `"/"`**.
  - GAPIC default retry configuration **omits `delete_folder`** (`gcsfs` supplies its own retry config via `get_storage_control_retry_config`).
  - Pre-building `channel=` drops GAPIC's `max_send/receive_message_length=-1` (re-introducing the 4 MiB cap) and drops `client.api_endpoint`.
  - `ControlPlaneTelemetryInterceptor` runs before GAPIC's `_LoggingClientAIOInterceptor`, which UTF-8 decodes `bytes` metadata values and crashes on `-bin` keys under `DEBUG` logging.
- **`AsyncOperation` (`google.api_core.operation_async`)**:
  - `_set_result_from_operation` raises untyped base `GoogleAPICallError` rather than typed subclasses (`NotFound`, `FailedPrecondition`, etc.) — `gcsfs`'s `unwrap_operation_result` must map it via `from_grpc_status`.
  - `OperationsAsyncClient` defaults to a 600-second retry/timeout window, and `operation.cancel()` awaits `done()` first (requiring an explicit `asyncio.wait_for` cap on cancellation).

## 4. grpcio / grpc.aio & C-Core

**Verified against:** `grpcio==1.85.3` / `grpc` master (`676a469ac7`, C-core & Cython source).

- **Event-loop binding & retention (`grpc.aio._channel`, `_cygrpc.PollerCompletionQueue`)**:
  - `get_working_loop()` on Python 3.12+/3.14 when no loop is running creates and binds the channel to a brand-new, **never-started** event loop.
  - `PollerCompletionQueue._loops` holds **strong references** to bound loops until global channel `refcount == 0`.
  - `call_soon_threadsafe` targeting a closed loop raises uncaught inside `_handle_events`, leaking `CallbackWrapper` and stalling callbacks for other loops.
  - Starting RPCs and `channel.close(grace > 0)` call non-threadsafe `self._loop.create_task(...)` — channels must be used and closed on their bound loop.
- **`fork()` mechanics (Cython `PollerCompletionQueue.shutdown`)**:
  - `grpc.aio` bypasses Python `fork()` hooks and runs an unmanaged Cython `_poller_thread`. Deallocating the last `AioChannel` in a forked child enters `while not self._shutdown: self._poller_thread.join(...)` — an **infinite busy-loop hang** even when `GRPC_ENABLE_FORK_SUPPORT=1`.
  - This is why `gcsfs`'s `bind_gcs_sync_methods` preserves an instance ref-cycle for `fork()`+`exec()` (e.g. `ray.init()`), and why `fork()` without `exec()` after creating gRPC channels is fundamentally unsafe.
- **Keepalive & HTTP/2 (`src/core/ext/transport/chttp2/transport/writing.cc`)**:
  - **C-core bug (`writing.cc`, `grpc>=1.60`)**: the keepalive ping timer check (`!= TaskHandle::kInvalid`) fails to arm the HTTP/2 ping ACK timeout from `grpc.keepalive_timeout_ms`. Detecting dead connections requires the undocumented `grpc.http2.ping_timeout_ms` channel arg (which `gcsfs` sets in `_CONTROL_PLANE_KEEPALIVE_OPTIONS`).
  - Setting `grpc.keepalive_permit_without_calls=1` or an interval below server minimums triggers `GOAWAY(ENHANCE_YOUR_CALM, "too_many_pings")` and permanent client-side ping backoff.
  - `grpc.http2.max_pings_without_data=2` throttles idle bidi stream pings to 1/min; `grpc.http2.min_time_between_pings_ms` is a deprecated no-op.
- **Targets, TLS, DirectPath & Auth**:
  - Never pass `http://` or `https://` scheme prefixes to a gRPC target string.
  - Default gRPC port is **always 443**, even on `insecure_channel` (explicit `:port` is required for local emulators).
  - `google-c2p:///` DirectPath strips `:port`, checks GCP DMI product name, and conflicts with mTLS client certificates.
  - `AuthMetadataPlugin` spawns a thread per RPC call and maps credential refresh failures to `StatusCode.UNAVAILABLE`.
  - `_validate_universe_domain` is disabled in several GAPIC transport paths; verify explicit `universe_domain` forwarding.
- **Interceptors & Metadata (`grpc.aio._channel`, `_metadata`)**:
  - `Channel.__init__` dispatches interceptors with **`if`/`elif`** across the 4 interceptor ABCs (`UnaryUnary`, `UnaryStream`, `StreamUnary`, `StreamStream`). A single class inheriting multiple ABCs is silently registered for **only the first matching RPC type** — separate interceptor instances are required.
  - Metadata keys must match `[a-z0-9-_.]` (strictly lowercase); `-bin` keys require `bytes` values. Invalid metadata raises `ExecuteBatchError` (leaking `CallbackWrapper`).
  - Mutable `Metadata[k]` and `del Metadata[k]` operate **only on the first value** for repeated keys.
- **Status codes, cancellation & Cython copies**:
  - `google.api_core.exceptions.ResourceExhausted` subclasses `TooManyRequests` — oversized (>4 MiB) messages trigger `RESOURCE_EXHAUSTED` and get pointlessly retried by 429 retry policies.
  - Local RPC cancellation raises `asyncio.CancelledError` (`BaseException`), whereas remote server cancellation raises `AioRpcError(StatusCode.CANCELLED)` (`Exception`); `call.cancelled()` returns `True` for **both**. Catching only `Exception` misses local cancellation.
  - Cython `ReceiveMessageOperation` copies payload bytes twice (`grpc_slice` -> chunks list -> `b"".join`), and `_common.deserialize` swallows deserialization exceptions by returning `None`.

## 5. Upstream vs `gcsfs` Bug Ownership Matrix

Use this table during the Step 2 Necessity & Design Gate when deciding whether a PR fixes the right layer:

| Problem / Bug Class | Right Owner | Legitimate Temporary `gcsfs` Workaround? |
|---|---|---|
| `fsspec` base-class bug or post-`2026.9.0` feature (`_expand_path(withdirs=)`, `AdaptiveReadaheadCache` loop binding, `DirFileSystem`) | `filesystem_spec` | Only to preserve `fsspec==2026.9.0` compatibility without breaking floor users; must document the `fsspec` PR/issue and delete once floor bumps |
| `AsyncGrpcClient` `AnonymousCredentials` plaintext channel, `client_info` mutation, missing `close()`/`interceptors=`, `universe_domain` drop | `google-cloud-storage` | Only via public channel/endpoint kwargs (e.g., `attempt_direct_path=False`); **reject** private `_grpc_client` monkey-patches without a `TODO` + upstream issue link + proof upstream fix isn't cheaper |
| `AAOW.append()` retry `BytesIO.seek(persisted_size)` offset corruption, `_StreamMultiplexer.reopen_stream` sharer error | `google-cloud-storage` | Upstream fix required; `gcsfs` should guard pool health (`mrd.is_stream_open`) rather than monkey-patching SDK internals |
| `AsyncOperation` returning untyped `GoogleAPICallError` or 600s `cancel()` | `google-api-core` / `gcsfs` (`poller.py`) | Already owned cleanly by `gcsfs/poller.py` (`unwrap_operation_result`, `wait_for(..., 15)`) — reuse `poll_lro` instead of duplicating |
| `grpc.aio` post-`fork()` poller hang or `writing.cc` `keepalive_timeout_ms` bug | `grpcio` | Preserve existing narrow guards (`bind_gcs_sync_methods`, `_CONTROL_PLANE_KEEPALIVE_OPTIONS` with `grpc.http2.ping_timeout_ms` + comment/link); never expand fork hacks |
| `gcsfs` dual-class routing, `BucketLayout` fallback/RCU recovery, HNS file/folder coexistence, `DirectMemmoveBuffer` zero-copy safety, `dircache`, telemetry schema | `gcsfs` | Core `gcsfs` ownership — fix symmetrically across all affected layouts, file classes, and sync/async entry points |
