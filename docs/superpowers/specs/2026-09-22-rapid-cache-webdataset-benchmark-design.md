# Rapid Cache coverage for the WebDataset subsystem benchmark

Date: 2026-09-22
Status: Design, pending review

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
changes. It is therefore a **provisioning dimension** of the benchmark, not a
loader parameter.

The suite today runs every case against a freshly created, per-case bucket and
reports the mean of three rounds. Neither of those is compatible with measuring
cache behaviour, for reasons set out below.

## Goals

- Measure WebDataset read performance on Rapid Cache **cache hits** and **cache
  misses**, across the full existing sweep.
- Report first-epoch and steady-state performance separately, so a hit is never
  averaged together with a miss.
- Keep the change inside shared `dataloading/` infrastructure so other groups
  (`ray_data`, `huggingface_datasets`) can adopt it without rework.

## Non-goals

- Modelling cache behaviour across CI runs or across days. Each run is
  self-contained.
- Partial-hit / eviction scenarios where the dataset exceeds cache capacity.
- `PreloadRapidCache`. It is a post-GA feature per the internal RCU design doc.
- Changing `gcsfs` itself. Nothing in the client is affected by this work.

## Constraints that shape the design

These come from the Rapid Cache documentation and are the reason the design
looks the way it does.

| Constraint | Value | Consequence |
| :--- | :--- | :--- |
| Cache creation | Async long-running operation, up to 48h, blocks on zonal SSD capacity | Cannot be created on a CI critical path |
| Minimum TTL | 24 hours (max 7 days, default 24h) | No such thing as a short-lived cache |
| Disable | 1-hour grace period, then deleted | Teardown cannot be synchronous |
| Bucket deletion | All associated caches must be deleted first | Current `rmdir` teardown would fail |
| Operation rate | 1 op/sec across create/disable/resume/update | Per-case cache churn would be throttled |
| Max caches | One cache per bucket per zone | Arms need separate buckets |
| Ingestion granularity | 2 MB chunks; objects under 2 MB ingested whole | Interacts with the `read_buffer` axis |

> [!IMPORTANT]
> The combination of a 24-hour minimum TTL and an asynchronous, capacity-bound
> create means a Rapid Cache instance **cannot live inside a per-case ephemeral
> bucket**. The bucket and cache must be provisioned before the benchmark runs.
> This is a product constraint, not a design preference.

## Design

### Two arms

Two pre-provisioned buckets, each with its own Rapid Cache instance in the
runner's zone. The arms differ only in `ingestOnWrite`:

| Arm | `ingestOnWrite` | Epoch 1 | Epochs 2-3 |
| :--- | :--- | :--- | :--- |
| `rapid_cache_iow` | `true` | Hit — warmed by the corpus upload | Hit |
| `rapid_cache_noiow` | `false` | Miss — ingests while serving | Hit |

Three epochs come from the existing `rounds: 3`; no config change is needed.

The `noiow` epoch-1 miss doubles as the practical no-cache baseline. It is not
identical to reading an uncached bucket — the miss path also ingests — so
headline comparisons against plain GCS should use an existing regional-bucket
run rather than this row.

### Provisioning

A setup script under `cloudbuild/subsystembenchmarks/scripts/` creates the two
buckets and their caches, and a runbook section documents when to run it.
Provisioning is deliberately **outside** the benchmark and outside Cloud Build.

At run start the benchmark asserts both caches are in `RUNNING` state and fails
fast with an actionable message if not. It never calls create, update, disable,
or resume.

### Bucket reuse and prefix isolation

`dataloading/bucket.py` gains a pre-provisioned mode. For the two Rapid Cache
arms, `case_bucket` yields a prefix inside the fixed bucket instead of creating
one:

```text
gs://<fixed-bucket>/<case-id>-<uuid8>/data/
```

Teardown deletes the prefix, not the bucket. The existing `uuid` suffix is
load-bearing and must be retained: it guarantees the `noiow` arm sees a genuine
cold miss on every rerun, since no prior run will have read those object names.

`BucketSpec` grows a bucket name field for these arms, exported by `run.py` from
a new `--cache-bucket` argument, with the arm selected through `--bucket-type`.

`bucket_type` is a run-level setting — one type per run, as the configurator's
`RUN_LEVEL_KEYS` already enforces. Each invocation therefore executes exactly
one arm, and full coverage is two invocations, each pointed at its own
pre-provisioned bucket.

### Reporting

`publish_round_stats` already records every round in `runs`; the split is a
reporting change only. New columns:

| Column | Meaning |
| :--- | :--- |
| `first_epoch_duration_seconds` | Round 1 duration |
| `first_epoch_throughput_bytes_per_second` | Round 1 logical throughput |
| `steady_state_mean_duration_seconds` | Mean of rounds 2..N |
| `steady_state_mean_throughput_bytes_per_second` | Mean logical throughput of rounds 2..N |
| `rapid_cache_enabled` | Whether the arm's bucket has a cache |
| `rapid_cache_ingest_on_write` | The arm's `ingestOnWrite` setting |

Existing mean-across-rounds columns stay, for continuity with historical rows.

`subsystembenchmarks_schema.json` and `ingest.sql` are updated together with
these columns.

### Read amplification

`amplification.py` queries `storage.googleapis.com/network/sent_bytes_count`
against `resource.type = "gcs_bucket"`. Cache-served bytes are expected not to
appear there, which would make amplification collapse toward zero on hits and
serve as a free hit-rate signal.

This expectation is unverified. The first run must check it explicitly. If cache
reads do show up under the bucket resource, the column is meaningless for these
arms and must be nulled, with hit rate sourced from Rapid Cache's own metrics
instead.

Separately, sharing one bucket across cases means amplification is attributed by
time window rather than by bucket. Cases run sequentially, so windows do not
overlap, but Cloud Monitoring's 60-second grid can let adjacent cases bleed into
each other. A short inter-case pad keeps each case on clean grid cells.

## Scale and cost

The full sweep is 26 variants plus the baseline, so 27 cases per arm and 54
cases across the two invocations.

Cached data accumulates for the whole run and is not evicted until the 24-hour
TTL expires. With several hundred GB per arm across the sweep, cache storage
fees are a material part of the run cost and outlast the run itself.

> [!CAUTION]
> Deleting a case prefix removes the objects from the bucket, but whether the
> corresponding cached chunks are released immediately or held until TTL is not
> documented. Treat cache storage as billable for 24 hours after the run and
> monitor actual spend on the first execution before scheduling this regularly.

Cache creation can also fail outright when a zone lacks SSD capacity. The
runbook must cover retrying in a different zone.

## Failure modes

| Failure | Handling |
| :--- | :--- |
| Cache not `RUNNING` at run start | Fail fast with the cache name, zone, and current state |
| Cache missing entirely | Fail fast pointing at the provisioning script |
| Prefix teardown fails | Log and continue; the existing prefix sweep is the safety net |
| Partial read (rounds disagree on row count) | Existing behaviour: fail the case rather than report inflated throughput |
| Amplification unavailable | Existing behaviour: leave the columns empty unless `--require-amplification` |

## Testing

- Unit tests for the pre-provisioned bucket mode: prefix construction, uuid
  suffix presence, teardown deleting the prefix and never the bucket.
- Unit tests for the epoch split: a three-round `runs` list produces the
  expected first-epoch and steady-state values, and a single-round list degrades
  without error.
- Config tests asserting the two new bucket types are accepted and that the
  cache bucket argument is required for them.
- Schema test asserting every new column exists in
  `subsystembenchmarks_schema.json`.
- All runnable without live GCS under the existing
  `pytest gcsfs/tests/perf/subsystembenchmarks --run-benchmarks-infra`.

## Known limitations to state alongside results

The baseline corpus fits entirely within the cache, so steady-state rounds will
show a near-total hit rate. Real workloads run datasets larger than their cache
and see partial hits under LRU eviction. These numbers are an optimistic upper
bound and must be labelled as such so they are not quoted as expected customer
speedup.

## Verification checklist for the first run

1. Confirm whether cache-served bytes appear under the `gcs_bucket` resource in
   Cloud Monitoring, and null the amplification columns for these arms if they
   do.
2. Confirm the `noiow` arm's epoch 1 is measurably slower than its epochs 2-3.
   If not, the cold miss is not being achieved and prefix uniqueness is suspect.
3. Confirm the `iow` arm's epoch 1 is comparable to its epochs 2-3. If epoch 1
   is slower, ingest-on-write is not warming the cache as expected.
4. Record actual cache storage cost and duration before scheduling recurring
   runs.
