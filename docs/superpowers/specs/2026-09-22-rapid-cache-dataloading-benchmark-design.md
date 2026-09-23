# Rapid Cache coverage for the dataloading subsystem benchmarks

Date: 2026-09-22
Status: Implemented

## Context

Review feedback on the WebDataset subsystem benchmarks asked for coverage of
GCS + Rapid Cache:

> I was wondering if there is any test planned with GCS+Rapid Cache. This is how
> our customers are using for training today. [...] I am hoping to have
> benchmarks both on cache hits and cache misses. In multi epoch training or
> multi run training, this becomes invaluable. So scenario of cache hit is more
> pertinent.

Rapid Cache (formerly Anywhere Cache) is an SSD-backed zonal read cache for
Cloud Storage buckets. It is transparent to `gcsfs`: no client code path
changes. It is a **provisioning dimension** of the benchmark, not a loader
parameter — exactly like `zonal` and `hns` are today.

The two customer situations the feedback names map onto the state of the cache
when a training job *starts*:

| Customer situation | Cache state at job start |
| :--- | :--- |
| First training run on a fresh dataset | cold |
| Second run, or a re-run on the same dataset | warm |

## Goals

- Measure dataloading read performance with a **cold** Rapid Cache and with a
  **warm** Rapid Cache, across the existing sweep.
- Keep the existing 3-epoch shape (`rounds: 3`) in both modes, because
  multi-epoch training is the real customer workload. Cold therefore reports a
  genuine first-epoch miss followed by two warmed epochs; warm reports three hit
  epochs.
- Keep everything inside shared `dataloading/` infrastructure so `ray_data`,
  `huggingface_datasets`, and `checkpointing/` inherit it without rework.
- Require no new BigQuery columns.

## Non-goals

- Partial-hit / eviction scenarios where the dataset exceeds cache capacity.
- `PreloadRapidCache` (post-GA per the internal RCU design doc).
- Rapid Cache Ultra (RCU). Different product, different provisioning.
- Changing `gcsfs` itself. Nothing in the client is affected by this work.

## Design

### Two new bucket types

`--bucket-type` gains two values alongside `regional`, `zonal`, and `hns`:

| Bucket type | Underlying bucket | `ingestOnWrite` | Warm read pass before timing | Epoch 1 | Epochs 2-3 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `rapid_cache_cold` | regional | `false` | no | miss | hit |
| `rapid_cache_warm` | regional | `true` | yes | hit | hit |

`bucket_type` is run-level (`RUN_LEVEL_KEYS` in `dataloading/configurator.py`
already enforces one type per run), so full coverage is two invocations of
`run.py`, each sweeping the same cases.

Both new types require `--zone`, like `zonal` does: a Rapid Cache is zonal and
must be created in the benchmark VM's zone or it will not serve the reads.

Benchmark ID tokens, added to `_BUCKET` in `dataloading/configurator.py` and
`checkpointing/configurator.py`: `rccold` and `rcwarm`.

### Per-case cache lifecycle

Every case keeps its own freshly created bucket. This is what makes read
amplification attributable per case (`dataloading/amplification.py` partitions
by bucket), and it is also what guarantees the cold arm is genuinely cold: a
cache created on a bucket that has never been read cannot hold anything.

New module `dataloading/rapid_cache.py` wraps the JSON API through the existing
`gcsfs` client (`fs.call`), so no new dependency is introduced:

| Helper | Call |
| :--- | :--- |
| `create(fs, bucket, zone, ingest_on_write)` | `POST b/{bucket}/anywhereCaches` |
| `wait_running(fs, bucket, zone, timeout, poll)` | `GET b/{bucket}/anywhereCaches/{zone}` until `state == "RUNNING"` |
| `disable(fs, bucket, zone)` | `POST b/{bucket}/anywhereCaches/{zone}/disable` |
| `warm_if_needed(prefix, bucket_type, fs=None)` | streams every object under `prefix` in 16 MiB chunks across up to 16 worker threads using `skip_instance_cache=True` and invalidates client metadata caches in `finally` |

`dataloading/bucket.py` owns create / wait / disable; it already owns bucket
creation and teardown and it already holds the `BucketSpec` that carries the
zone. Per case:

```text
case_bucket:  mkdir regional bucket
              create cache (ingestOnWrite per arm)
              wait until RUNNING
   read_case: params.ingest(prefix)                  # corpus upload (or driver.setup in checkpoint_case)
              rapid_cache.warm_if_needed(...)        # untimed, warm arm only
              -- timing window opens --
              driver.run_read(...)  x3 epochs
              -- timing window closes --
case_bucket:  disable cache
              rm objects
              best-effort rmdir
```

`warm_if_needed` lives at the `read_case` and `checkpoint_case` call sites
rather than inside `case_bucket` because it must run *after* ingestion/setup,
and `case_bucket` yields before the corpus exists. It is a no-op unless
`bucket_type` is `rapid_cache_warm` and the prefix is a `gs://` URL, so
local-directory runs and the other bucket types are unaffected.

The warm pass is belt-and-braces on top of `ingestOnWrite`: ingestion is
asynchronous, so reading every object from the VM in the cache's zone is what
actually guarantees residency before the clock starts.

### Reporting

No schema change. `bucket_type` already carries the arm, and
`publish_round_stats` already exports every round in `runs` alongside `min`,
`max`, `mean`, `median`, and `stddev` — so the cold arm's first-epoch miss and
its warmed epochs 2-3 are both recoverable from a single row.

Only the `bucket_type` *description* in `subsystembenchmarks_schema.json` needs
updating to name the new values.

### Read amplification

`amplification.py` queries `storage.googleapis.com/network/sent_bytes_count`
and `storage.googleapis.com/api/request_count` against
`resource.type = "gcs_bucket"`. When Rapid Cache serves 100% of reads from the
zonal SSD cache during the timed measurement window, zero bytes and zero origin
read requests hit the `gcs_bucket` resource, so Cloud Monitoring returns either
empty time series (`None`) or `0.0` delta points. `amplification.py:enrich_csv`
coalesces `egress in (None, 0.0) and reqs in (None, 0.0)` to `0.0` for
`rapid_cache_cold` and `rapid_cache_warm` so `run.py --require-amplification`
records `0` origin egress (`0.0` read amplification ratio) instead of treating
100% cache hits as missing metrics, while still retrying if one metric is `> 0`
and the other is `None` due to Cloud Monitoring ingestion lag.

## Constraints and the risks they create

These come from the Rapid Cache documentation. They do not block the design, but
they are the parts most likely to bite on the first real run.

| Constraint | Consequence | Mitigation |
| :--- | :--- | :--- |
| Cache creation is an async LRO, bounded by zonal SSD capacity | A case blocks until its cache is `RUNNING`; a capacity-starved zone can stall or fail it | `--rapid-cache-timeout` (default 1800s), fail fast naming bucket, zone, and last observed state; runbook covers retrying in another zone |
| 1 op/sec across create/disable/resume/update | 27 cases x 2 ops is well inside the limit, but bursts are not | Operations are strictly sequential, one case at a time; retry with backoff on 429 |
| Minimum TTL 24 hours | Each case's cache is billable for 24h even though the case runs for minutes | Disable immediately at teardown; record actual spend on the first run before scheduling recurring runs |
| Disable has a 1-hour grace period, and a bucket cannot be deleted while a cache exists | `rmdir` will fail; empty buckets linger about an hour | Objects are deleted immediately (the real cost), `rmdir` stays best-effort, and an age-based sweep of leftover benchmark buckets runs at the start of each build |
| Project cache-storage quota is per project per zone | A long sweep accumulates disabled-but-not-yet-deleted caches | Start with a `--filter`ed subset; monitor `storage.googleapis.com/anywhere_cache_storage_size` |
| Ingestion granularity is 2 MB chunks | Interacts with the `read_buffer` axis | Note when interpreting that axis; no code change |

> [!CAUTION]
> Per-case cache creation is the expensive part of this design. Before enabling
> the full 27-case sweep on a schedule, run a filtered subset and record both
> wall-clock cost (creation latency per case) and cache storage spend.

## Failure modes

| Failure | Handling |
| :--- | :--- |
| Cache never reaches `RUNNING` within the timeout | Fail the case with bucket, zone, elapsed time, and last state |
| Cache creation rejected (capacity, quota, permission) | Fail fast; surface the API error verbatim |
| Disable fails at teardown | Log and continue; objects are still deleted and the bucket sweep is the safety net |
| `rmdir` fails because the cache is in its grace period | Expected; suppressed, swept later |
| Warm pass fails midway | Fail the case rather than silently report a partially warm run as a hit |
| Partial read (rounds disagree on row count) | Existing behaviour: fail the case |

## Testing

All runnable without live GCS under
`pytest gcsfs/tests/perf/subsystembenchmarks --run-benchmarks-infra`, using the
existing `_FakeFS` pattern extended with a `call` method.

- `BucketSpec` validation: `--zone` required for both new types; both accepted
  by `run.py` argument parsing and by the cloudbuild `_BUCKET_TYPE` guard.
- `bucket_kwargs` returns a plain regional body for both new types.
- `case_bucket` creates the cache with the arm's `ingestOnWrite`, polls until
  `RUNNING`, and disables it on teardown — including when the case raises.
- `wait_running` times out with an actionable message and does not hang.
- `warm_if_needed` reads every object for `rapid_cache_warm`, and does nothing
  for `rapid_cache_cold`, for the other bucket types, and for local prefixes.
- Benchmark IDs carry `rccold` / `rcwarm`.
- Both arms still execute the configured three rounds.

## Known limitations to state alongside results

The baseline corpus fits entirely within the cache, so the warm arm sees a
near-total hit rate. Real workloads run datasets larger than their cache and see
partial hits under LRU eviction. These numbers are an optimistic upper bound and
must be labelled as such so they are not quoted as expected customer speedup.

## Verification checklist for the first run

1. Confirm whether cache-served bytes appear under the `gcs_bucket` resource in
   Cloud Monitoring, and null the amplification columns for these arms if they
   do.
2. Confirm the cold arm's epoch 1 is measurably slower than its epochs 2-3. If
   not, the cold miss is not being achieved.
3. Confirm the warm arm's epoch 1 is comparable to its epochs 2-3. If epoch 1 is
   slower, the warm pass is not landing in the cache.
4. Record cache creation latency per case and actual cache storage cost before
   scheduling recurring runs.
