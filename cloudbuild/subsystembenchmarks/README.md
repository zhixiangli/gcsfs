# GCSFS Subsystem Benchmarks Automation

## Introduction

This directory contains the Cloud Build automation for running GCSFS subsystem
benchmarks on an ephemeral Compute Engine VM and retaining their results for
historical analysis. It installs the selected benchmark group, runs every
selected case, uploads JSON and CSV artifacts to Cloud Storage, and uses a
separate pipeline to ingest CSV rows into BigQuery.

The workload itself -- what it exercises, its timing boundaries, and how to run
it directly for debugging -- is documented in the
[subsystem benchmark workload guide](../../gcsfs/tests/perf/subsystembenchmarks/README.md).
This README is the operational guide for provisioning and running it in a GCP
project.

There are two pipelines:

- **Run pipeline** (`subsystembenchmarks-cloudbuild.yaml`): provisions the VM,
  runs the selected group, uploads results, and cleans up ephemeral resources.
- **Ingestion pipeline** (`subsystembenchmarks-ingestion-cloudbuild.yaml`):
  exposes uploaded CSVs through a staging table and appends unseen results to a
  partitioned BigQuery history table.

## Cost and cleanup

> **A run creates billable resources in your project.** The default machine is
> one `c4-standard-192` VM with a 50 GB Hyperdisk Balanced boot disk,
> TIER_1 networking, and gVNIC. The pipeline may run for up to four hours. Each
> benchmark case also creates a temporary GCS bucket, uploads a synthetic
> corpus, and reads that corpus for every configured round.
>
> The VM, temporary OS Login key, and case buckets are cleaned up after normal
> success or failure. A best-effort preflight step deletes matching VMs older
> than ten hours, and the final bucket sweep removes case buckets left by a
> failed benchmark. A cancelled build or hard platform failure can still skip
> cleanup, so inspect Compute Engine and Cloud Storage after abnormal runs.

The results bucket, `gs://<_INFRA_PREFIX>-run-results`, is intentionally
persistent. Its objects continue to incur storage charges until removed.

## Prerequisites

Set up the following before creating the triggers.

1. **APIs:** Cloud Build, Compute Engine, Cloud Storage, Cloud Monitoring,
   BigQuery, IAM, and OS Login.

2. **Private worker pool:** both YAML files use a pool named exactly
   `cloud-build-worker-pool` in the build's `LOCATION`:

   ```text
   projects/<PROJECT_ID>/locations/<LOCATION>/workerPools/cloud-build-worker-pool
   ```

3. **Network reachability:** the run pipeline connects to the benchmark VM with
   `gcloud compute ssh --internal-ip`. The worker pool network must be able to
   reach the VM's VPC interface on TCP port 22. The YAML does not select a
   network, so Compute Engine uses the project's default network. The pipeline
   applies the `allow-ssh` network tag but does not create a firewall rule.

4. **A VM service account:** pass its email as `_VM_SERVICE_ACCOUNT`. The VM
   uses it to create, populate, and delete per-case buckets; upload artifacts to
   the persistent results bucket; and query Cloud Monitoring for read metrics.

5. **A trigger source that supplies branch metadata:** `BRANCH_NAME` must be set
   and must not resolve to `unknown`. The uploaded artifact path uses its
   sanitized value.

### Service accounts and IAM

The Cloud Build service account and the benchmark VM service account have
different responsibilities. The following predefined roles are practical
starting points, not a least-privilege prescription; tighten them for your
project.

**Cloud Build service account:**

- `roles/compute.instanceAdmin.v1` to create, inspect, and delete the benchmark
  VM.
- `roles/compute.osAdminLogin` to add the temporary OS Login key, connect to the
  VM, and run the setup script with `sudo`.
- `roles/storage.admin` to create the persistent results bucket and support the
  cleanup pipeline.
- `roles/iam.serviceAccountUser` on the VM service account so the build can
  attach it to the VM.
- `roles/bigquery.admin` for the ingestion pipeline. A narrower combination of
  BigQuery dataset/table write permissions and `roles/bigquery.jobUser` can be
  used instead.
- Read access to objects in the results bucket when the ingestion query reads
  the external table. `roles/storage.admin` already includes this permission;
  separate ingestion identities can use `roles/storage.objectViewer`.

Depending on the shared VPC and worker-pool setup, the build identity may also
need `roles/compute.networkUser` on the selected network or subnetwork.

**Benchmark VM service account:**

- `roles/storage.admin` to create and delete case buckets, read/write their
  objects, and upload run artifacts.
- `roles/monitoring.viewer` to query the bucket-level GCS metrics used for read
  amplification.

## Run pipeline flow

`subsystembenchmarks-cloudbuild.yaml` performs these steps:

| Step | What it does |
| :--- | :--- |
| `generate-ssh-key` | Creates a build-scoped SSH key and registers it with OS Login. |
| `init-variables` | Validates required substitutions and the bucket profile, derives region/resource names, and writes the remote environment file. |
| `cleanup-leaked-resources` | Best-effort deletion of VMs older than ten hours whose names match the infrastructure prefix. |
| `create-buckets` | Ensures the persistent HNS results bucket exists in the run region. |
| `create-vm` | Creates the Ubuntu benchmark VM with the selected machine type, gVNIC, and TIER_1 networking. |
| `run-subsystembenchmarks` | Waits for SSH, copies the source, creates a virtual environment, installs dependencies and overrides, runs the group, and uploads available artifacts. |
| `cleanup-ssh-key` | Removes the temporary OS Login key. |
| `cleanup-vm` | Deletes the benchmark VM. |
| `delete-buckets` | Sweeps case buckets matching this run's unique prefix, disabling their caches first. |
| `check-failure` | Fails the build after cleanup if an earlier step recorded a failure. |

The setup script installs the copied checkout in editable mode, then the
selected group's pinned `requirements.txt`, then `_REQUIREMENTS`. Installing the
override last makes it possible to test an exact pip-installable `gcsfs` build
instead of the copied checkout's version. The resolved environment and the
normalized override are published with every result row.

## Run substitutions

### Required

| Substitution | Description |
| :--- | :--- |
| `_INFRA_PREFIX` | Lowercase resource prefix used for the VM, temporary case-bucket prefix, and persistent `gs://<_INFRA_PREFIX>-run-results` bucket. |
| `_ZONE` | Compute Engine zone for the VM and zonal/RAPID case buckets, for example `us-central1-a`. The bucket region is derived from it. |
| `_VM_SERVICE_ACCOUNT` | Email of the service account attached to the benchmark VM. |

### Optional

| Substitution | Default | Description |
| :--- | :--- | :--- |
| `_GROUP` | `dataloading/huggingface_datasets` | Runnable `<subsystem>/<implementation>` group. Group discovery requires a `requirements.txt` in that directory. |
| `_BUCKET_TYPE` | `regional` | Case-bucket profile: `regional`, `zonal`, `hns`, `rapid_cache_cold`, `rapid_cache_warm`, `standard_rcu_{z,warm,cold}`, or `hns_rcu_{z,warm,cold}`. `zonal` creates RAPID/HNS buckets in `_ZONE`; `hns` creates regional HNS buckets; `rapid_cache_*` add an Anywhere Cache in `_ZONE` to each case bucket, and the RCU types add a Rapid Cache Ultra cache in `_ZONE` to each case bucket (HNS buckets for `hns_rcu_*`). |
| `_MACHINE_TYPE` | `c4-standard-192` | Compute Engine machine type used for the benchmark VM. |
| `_REQUIREMENTS` | empty | Whitespace-separated pip requirement specs installed last. Use this to select the `gcsfs` build under test. |
| `_SWEEP_AXES` | empty | Whitespace-separated configuration-axis names. Empty runs all configured cases; any selection still includes the implicit baseline case. |
| `_MODE` | `nightly` | `nightly` runs `_GROUP` as configured. `ci` runs the fixed per-PR suite in `scripts/run-ci.sh` (CPU only) and ignores `_GROUP`, `_BUCKET_TYPE`, `_FILTER`, `_SWEEP_AXES`, `_MODEL_ID`, and `_REQUIREMENTS`. |
| `_CI_MODEL_ID` | empty | `_MODE=ci` only: `gs://` copy of `gpt2` that `_VM_SERVICE_ACCOUNT` can read. Empty uses `gs://huggingface-model-weights/gpt2`. |
| `_ACCELERATOR` | `cpu` | Execution target: `cpu` (ephemeral VM, unchanged), `gpu`, or `tpu` (shared GKE cluster via Helm JobSets). |
| `_GKE_CLUSTER` | empty | Target GKE cluster name (required when `_ACCELERATOR` is `gpu` or `tpu`). |
| `_GKE_CLUSTER_PROJECT` | empty | Project owning `_GKE_CLUSTER` (required for `gpu`/`tpu`). |
| `_GKE_CLUSTER_LOCATION` | empty | Region or zone of `_GKE_CLUSTER` for `get-credentials` (required for `gpu`/`tpu`; `_ZONE` must be in its region). |
| `_GKE_NAMESPACE` | `default` | Target Kubernetes namespace for Helm releases and pods. |
| `_GKE_ENDPOINT` | `default` | Cluster control-plane connection mode: `default` (public), `internal-ip`, or `dns` (`--dns-endpoint`). |
| `_GKE_QUEUE` | empty | Optional Kueue LocalQueue label (`kueue.x-k8s.io/queue-name`). |
| `_TPU_PRIORITY` | `low` | TPU Kueue priority class (`medium`, `low`, or `very-low`). |
| `_TPU_DURATION` | `180` | TPU `declared-duration-minutes` pod label. |
| `_TPU_PLACEMENT_POLICY` | empty | `cloud.google.com/placement-policy-name` value; required when a multi-node TPU release is launched. |
| `_MAX_CASES_PER_RELEASE` | empty | Split each node count's selected cases into batches of at most N exact case IDs (useful for TPU groups under the Kueue duration limit). |
| `_MAX_CONCURRENT_RELEASES` | `1` | Maximum concurrent GKE Helm releases per build. |
| `_GKE_DEADLINE_SECONDS` | `16200` | Overall queueing + run deadline in seconds (default 270 min, below the 5 h build timeout). |
| `_GCSFS_REPO` | `https://github.com/fsspec/gcsfs.git` | Git repository cloned by GKE pods at `COMMIT_SHA`. |

`PROJECT_ID`, `LOCATION`, `BUILD_ID`, `BRANCH_NAME`, and `COMMIT_SHA` are Cloud
Build built-ins rather than custom substitutions.

The "baseline" above is only the reference configuration for one-factor
experiments within this run. The pipeline does not download prior results,
perform historical comparisons, or fail because current performance is slower
than an earlier run.

## Creating the run trigger

Create a manual trigger and adjust the placeholders for your project and source
connection:

```bash
gcloud builds triggers create manual \
  --name=gcsfs-subsystembench-run \
  --region=<LOCATION> \
  --repo=<YOUR_REPO> \
  --repo-type=<GITHUB|CLOUD_SOURCE_REPOSITORIES> \
  --branch=main \
  --build-config=cloudbuild/subsystembenchmarks/subsystembenchmarks-cloudbuild.yaml \
  --substitutions=_INFRA_PREFIX=gcsfs-subsystembench,_ZONE=us-central1-a,_VM_SERVICE_ACCOUNT=<SA_EMAIL>
```

Trigger overrides can select a different bucket profile, machine, requirement,
or subset of configuration axes. Because `_SWEEP_AXES` contains spaces when
selecting more than one axis, quote the complete `--substitutions` argument in
shell invocations.

## Results and failure behavior

The remote upload script copies the complete local `__run__` directory to:

```text
gs://<_INFRA_PREFIX>-run-results/subsystembenchmarks/
└── branch=<branch>/
    └── <YYYYMMDD>/
        └── <build_id>/
            └── <YYYYMMDD-HHMMSS>/
                ├── results.json
                └── results.csv
```

If benchmark execution returns nonzero but produced artifacts, the pipeline
records the failure and still attempts the upload before propagating the
original status. Cleanup steps run afterward. Upload or infrastructure failures
are also recorded and reported by `check-failure`.

The Cloud Build execution path passes `--require-amplification`, so eligible
rows that still lack GCS read metrics after the wait and retry fail the run. A
nonzero build therefore indicates execution, correctness, infrastructure,
upload, or required-metric failure. A historical performance change by itself
never fails the build.

## Per-PR CI mode

`_MODE=ci` runs a lightweight suite meant for pull requests: 11 small cases
(data loading on Rapid, HNS, and Standard buckets; PyTorch Lightning and Ray
checkpoint save/load on Rapid and Standard buckets) run concurrently on one VM.
The case table lives in [`scripts/run-ci.sh`](scripts/run-ci.sh). Every case uses
its group's `configs.ci.yaml`; checkpoint cases save and load `gpt2` (from
`_CI_MODEL_ID`, default `gs://huggingface-model-weights/gpt2`) with `ddp` on 2
ranks, twice each.

The `gcsfs` under test is the checkout. `fsspec` (`master`) and
`google-cloud-storage` (`main` of `googleapis/google-cloud-python`) are installed
from their upstream default branches, after the framework requirements, so
unreleased upstream changes are caught too. The commits and per-venv versions
used are recorded in `__run__/ci/versions.txt`.

Each case has a 540-second limit. On timeout the case's processes are frozen and
then aborted one at a time, so each Python process writes its own thread-stack
dump to `__run__/ci/<case>/run.log` without interleaving, and the build log
prints that log from the timeout on. The build fails if any case fails or times
out; throughput and read amplification are never gated.

Case buckets are named `<_INFRA_PREFIX>-ci-<build>-<case>-<suffix>` (each case
picks its own bucket type), and artifacts go to
`gs://<_INFRA_PREFIX>-run-results/subsystembenchmarks-ci/`, a prefix BigQuery
ingestion never reads. Still give the CI trigger its own `_INFRA_PREFIX` (for
example `gcsfs-ci-ssb`) so its VMs and buckets stay separate from nightly runs.

### Securing the pull request trigger

A pull request trigger builds from the pull request's branch, so the pull
request supplies the scripts the VM runs and, when the trigger reads its build
config from the repository (`--build-config`), this YAML too. The scripts run as
`_VM_SERVICE_ACCOUNT`; the YAML's steps run as the Cloud Build service account,
which holds the broader roles listed under
[Service accounts and IAM](#service-accounts-and-iam). Checks inside the
repository cannot limit what that code does; only the trigger and IAM can:

- **Gate external pull requests.** Set the trigger's comment control to
  `COMMENTS_ENABLED_FOR_EXTERNAL_CONTRIBUTORS_ONLY`, so a maintainer must comment
  `/gcbrun` before a contributor outside the repository can run the build.
  Before commenting, review the pull request's changes to
  `subsystembenchmarks-cloudbuild.yaml` as well as to the scripts.
- **Bound the Cloud Build service account.** Scoping the VM's account does not
  limit a pull request that changes the YAML. Run pull request triggers in a
  dedicated CI project that holds nothing a build must not reach, such as
  nightly results, or create the trigger with `--inline-config` so that the
  trigger, not the pull request, supplies the build steps (update the trigger
  whenever this YAML changes). Grant the Cloud Build service account
  `roles/iam.serviceAccountUser` on the VM service account only, never
  project-wide, so a build cannot attach any other account to a VM.
- **Give the VM a dedicated, narrowly scoped service account.** Do not reuse a
  VM service account that other pipelines rely on, and do not grant it
  project-wide `roles/storage.admin`. The VM needs that role only on the
  per-case buckets, which in `_MODE=ci` are always named
  `<_INFRA_PREFIX>-ci-...`, so limit it to that prefix with an IAM condition.
  On the two shared buckets, grant bucket-level object roles instead: read on
  the `_CI_MODEL_ID` bucket and read/write on the results bucket. Create the
  results bucket first; the build otherwise creates it only on its first run.
  With `_INFRA_PREFIX=gcsfs-ci-ssb` and
  `_CI_MODEL_ID=gs://gcsfs-ci-ssb-models/gpt2`:

  ```bash
  gcloud iam service-accounts create ssb-ci-vm --project=<PROJECT_ID>
  gcloud iam service-accounts add-iam-policy-binding \
    ssb-ci-vm@<PROJECT_ID>.iam.gserviceaccount.com \
    --project=<PROJECT_ID> \
    --member=serviceAccount:<CLOUD_BUILD_SA_EMAIL> \
    --role=roles/iam.serviceAccountUser
  gcloud projects add-iam-policy-binding <PROJECT_ID> \
    --member=serviceAccount:ssb-ci-vm@<PROJECT_ID>.iam.gserviceaccount.com \
    --role=roles/storage.admin \
    --condition='expression=resource.name.startsWith("projects/_/buckets/gcsfs-ci-ssb-ci-"),title=only-gcsfs-ci-ssb-ci-buckets'
  gcloud projects add-iam-policy-binding <PROJECT_ID> \
    --member=serviceAccount:ssb-ci-vm@<PROJECT_ID>.iam.gserviceaccount.com \
    --role=roles/serviceusage.serviceUsageConsumer --condition=None
  gcloud storage buckets add-iam-policy-binding gs://gcsfs-ci-ssb-models \
    --member=serviceAccount:ssb-ci-vm@<PROJECT_ID>.iam.gserviceaccount.com \
    --role=roles/storage.objectViewer
  gcloud storage buckets create gs://gcsfs-ci-ssb-run-results \
    --project=<PROJECT_ID> --location=<REGION> \
    --enable-hierarchical-namespace --uniform-bucket-level-access
  gcloud storage buckets add-iam-policy-binding gs://gcsfs-ci-ssb-run-results \
    --member=serviceAccount:ssb-ci-vm@<PROJECT_ID>.iam.gserviceaccount.com \
    --role=roles/storage.objectUser
  ```

  Pass that account as the trigger's `_VM_SERVICE_ACCOUNT`. The VM never lists
  the project's buckets, which a bucket-name condition cannot allow; the
  leaked-bucket sweeps run as the Cloud Build service account.

## BigQuery ingestion

The ingestion pipeline requires both substitutions:

| Substitution | Description |
| :--- | :--- |
| `_DATASET_NAME` | BigQuery dataset to create or update in the build `LOCATION`. |
| `_INFRA_PREFIX` | Prefix of the run pipeline's results bucket. It must match the run trigger. |

Create its trigger separately:

```bash
gcloud builds triggers create manual \
  --name=gcsfs-subsystembench-ingest \
  --region=<LOCATION> \
  --repo=<YOUR_REPO> \
  --repo-type=<GITHUB|CLOUD_SOURCE_REPOSITORIES> \
  --branch=main \
  --build-config=cloudbuild/subsystembenchmarks/subsystembenchmarks-ingestion-cloudbuild.yaml \
  --substitutions=_INFRA_PREFIX=gcsfs-subsystembench,_DATASET_NAME=gcsfs_subsystembenchmarks
```

`subsystembenchmarks_schema.json` is the authoritative external-table schema and
artifact URI pattern. On every ingestion run, the pipeline:

1. Ensures `_DATASET_NAME` exists in `LOCATION`.
2. Recreates `staging_subsystem` as an external CSV table over the results
   bucket, so checked-in schema changes take effect.
3. Adds new staging columns to `history_subsystem` when necessary.
4. Inserts rows from previously unseen artifacts, deriving `run_date`,
   `build_id`, `run_timestamp`, `source_uri`, and `branch_name` from each object
   path.

`history_subsystem` is partitioned by `run_date` and deduplicated by
`source_uri`, so rerunning ingestion does not append the same artifact twice.
The complete evolving metric-column inventory lives in
[`subsystembenchmarks_schema.json`](subsystembenchmarks_schema.json).

## Running on GKE GPU and TPU clusters

Every benchmark group can also run directly on GKE GPU (8×H200 per node) and
TPU (4 TPU7x chips / 8 TensorCores per node) clusters using the Helm chart in
[`helm_chart/`](helm_chart/) and the launcher script
[`scripts/run-gke.sh`](scripts/run-gke.sh). All cluster-specific values
(context, project, region, zone, placement policy) are passed as flags rather
than hard-coded in the repository.

### Prerequisites

1. **Tools:** `gcloud`, `kubectl`, `helm` (v3+), and a local Python environment
   that can collect case node counts (`python3 -m gcsfs.tests.perf.subsystembenchmarks.run --list-node-counts`).
2. **Cluster credentials:**

   ```bash
   gcloud container clusters get-credentials <gpu-cluster> \
     --region <gpu-region> --project <gpu-cluster-project>

   gcloud container clusters get-credentials <tpu-cluster> \
     --region <tpu-region> --project <tpu-cluster-project>
   ```

3. **IAM & bucket access:**
   - **GPU pods:** run with `hostNetwork: false` so GKE Workload Identity binds
     the namespace's `default` Kubernetes service account. That identity needs
     bucket create/delete access in `<bucket-project>`, write access to
     `--results-uri`, `roles/monitoring.viewer` in `<bucket-project>` (for
     `--require-amplification`), and `roles/storage.objectViewer` on the model
     weight buckets.
   - **TPU pods:** run under the cluster's GCE node service account and need
     bucket create/delete access in `<bucket-project>`, write access to
     `--results-uri`, `roles/monitoring.viewer` in `<bucket-project>`, and
     `roles/storage.objectViewer` on the model weight buckets.
   - Pods run `run.py --preflight` before executing any benchmark case, failing
     fast with the exact resource if bucket create/delete or model read access
     is missing.

4. **Source revision:** push the branch or commit you want the pods to clone and
   pass `--commit <SHA>` (defaults to local `HEAD`).

### Launching with `run-gke.sh`

`run-gke.sh` queries `--list-node-counts` (or `--list-cases` when
`--max-cases-per-release` is set) for the target group and accelerator, then
installs one Helm release (JobSet + headless Service) per planned batch (for
example, 1-node for `world_size <= 8` cases and 2-node SPMD for the 16-rank
DCP cases). On TPU, releases satisfy Kueue rules automatically: release names
are `<= 20` characters (`ssb-<abbr>-<n>n-<4hex>`), JobSet labels set
`kueue.x-k8s.io/priority-class` (`medium`, `low`, or `very-low`), and pods set
`declared-duration-minutes` (default `180`). Multi-node TPU releases require
`--tpu-placement-policy`.

```bash
# Dry-run: print rendered helm install commands without scheduling pods
cloudbuild/subsystembenchmarks/scripts/run-gke.sh \
  --accelerator gpu \
  --kube-context <gpu-kube-context> \
  --project <bucket-project> \
  --location <gpu-region> \
  --zone <gpu-zone> \
  --group checkpointing/pytorch_save \
  --results-uri gs://<your-results-bucket>/subsystembenchmarks \
  --dry-run

# Run a group on a GPU cluster (regional buckets)
cloudbuild/subsystembenchmarks/scripts/run-gke.sh \
  --accelerator gpu \
  --kube-context <gpu-kube-context> \
  --project <bucket-project> \
  --location <gpu-region> \
  --zone <gpu-zone> \
  --group checkpointing/pytorch_save \
  --bucket-type regional \
  --results-uri gs://<your-results-bucket>/subsystembenchmarks

# Run a batched TPU checkpoint group with a multi-node placement policy
cloudbuild/subsystembenchmarks/scripts/run-gke.sh \
  --accelerator tpu \
  --kube-context <tpu-kube-context> \
  --project <bucket-project> \
  --location <tpu-region> \
  --zone <tpu-zone> \
  --tpu-placement-policy <tpu-placement-policy> \
  --max-cases-per-release 4 \
  --group checkpointing/pytorch_reshard \
  --bucket-type regional \
  --results-uri gs://<your-results-bucket>/subsystembenchmarks
```

Node 0 uploads `/results` (`results.json` and `results.csv`) whenever
`results.csv` exists, using the BigQuery ingestion layout:

```text
<results-uri>/branch=<branch>/<YYYYMMDD>/<run-id>/<release>/<YYYYMMDD-HHMMSS>/
```

### Scheduled GPU and TPU runs via Cloud Build

The same `subsystembenchmarks-cloudbuild.yaml` pipeline runs GPU and TPU groups
when triggered with `_ACCELERATOR=gpu` or `_ACCELERATOR=tpu` (for example from
Terraform triggers and Cloud Scheduler jobs):

- CPU-only VM steps no-op; GKE steps (`init-gke-variables`,
  `cleanup-leaked-gke`, `run-subsystembenchmarks-gke`, `cleanup-gke`) validate
  inputs, install pinned `gcloud`/`kubectl`/`helm`, fetch cluster credentials,
  run `scripts/run-gke.sh`, and clean up releases by build ID.
- Case buckets are created in the build's `PROJECT_ID` with the same
  `BUCKET_PREFIX` so the existing `delete-buckets` and `cleanup-leaked-resources`
  steps cover GPU/TPU runs unchanged.
- Artifacts upload to `gs://<_INFRA_PREFIX>-run-results/subsystembenchmarks/branch=<branch>/<YYYYMMDD>/<BUILD_ID>/<release>/<YYYYMMDD-HHMMSS>/`,
  which the unchanged `subsystembenchmarks-ingestion-cloudbuild.yaml` pipeline
  loads into `history_subsystem` (rows carry `compute_accelerator_type` and
  `machine_type`).
- `_HF_TOKEN` must be empty and `_MODE` must be `nightly`; `COMMIT_SHA` must be
  set so pods clone the exact revision under test.
- **Additional one-time access for Cloud Build GKE triggers:**
  - Trigger service account: `roles/container.developer` on
    `<gpu-cluster-project>` and `<tpu-cluster-project>`, plus network
    reachability from `cloud-build-worker-pool` to the cluster control planes
    (adjust `_GKE_ENDPOINT` to `internal-ip` or `dns` if needed).
  - GPU/TPU pod identities: `roles/storage.admin` conditioned on
    `<_INFRA_PREFIX>-*` buckets in `<build-project>`, object write on
    `gs://<_INFRA_PREFIX>-run-results`, `roles/monitoring.viewer` in
    `<build-project>`, and `roles/storage.objectViewer` on the model weight
    buckets.

```bash
gcloud builds triggers create manual \
  --name=gcsfs-subsystembench-tpu-save \
  --region=<LOCATION> \
  --repo=<YOUR_REPO> \
  --repo-type=<GITHUB|CLOUD_SOURCE_REPOSITORIES> \
  --branch=main \
  --build-config=cloudbuild/subsystembenchmarks/subsystembenchmarks-cloudbuild.yaml \
  --substitutions=_ACCELERATOR=tpu,_INFRA_PREFIX=gcsfs-subsystembench,_ZONE=<tpu-zone>,_GROUP=checkpointing/pytorch_save,_GKE_CLUSTER=<tpu-cluster>,_GKE_CLUSTER_PROJECT=<tpu-cluster-project>,_GKE_CLUSTER_LOCATION=<tpu-region>,_TPU_PLACEMENT_POLICY=<tpu-placement-policy>,_MAX_CASES_PER_RELEASE=4
```

### Cleanup

`run-gke.sh` uninstalls each Helm release (and its JobSet) once it completes,
fails, or exceeds `--deadline-seconds` (default `16200` s = 270 min), and
registers an exit/signal trap for interrupted runs. After a hard interrupt or
pod crash, verify no releases, JobSets, or temporary `ssb-<gpu|tpu>-*` buckets
remain:

```bash
helm list --kube-context <context> -n <namespace>
kubectl --context <context> -n <namespace> get jobsets
gcloud storage buckets list --project <bucket-project> --filter='name:ssb-*'
```
