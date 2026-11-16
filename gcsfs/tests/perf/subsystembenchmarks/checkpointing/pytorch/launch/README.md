# GKE JobSet Launch for `checkpointing/pytorch`

This directory contains the entrypoint, JobSet templates, and submission script
for running `checkpointing/pytorch` save, load, and reshard benchmarks on GKE
GPU (**A3 Ultra**) and TPU (**TPU7x** with TorchTPU) clusters.

## Prerequisites & Access

1. **Cluster credentials:**
   - **A3 Ultra (`imo-a3ultra`):**
     ```bash
     gcloud container clusters get-credentials imo-a3ultra \
       --zone europe-west1-b \
       --project supercomputer-testing
     ```
   - **TPU7x (`bodaborg-tpu7x-nap`):**
     ```bash
     gcloud container clusters get-credentials bodaborg-tpu7x-nap \
       --region us-central1 \
       --project cloud-tpu-shared-capacity \
       --dns-endpoint
     ```

2. **Queue membership (TPU7x):**
   - Submitting to the `default` namespace/queue on `bodaborg-tpu7x-nap`
     requires membership in `mdb/multipod-users` (`%multipod-users.prod`).
   - Only priority classes `medium`, `low`, and `very-low` are accepted by
     Kueue (`jobset-tpu7x.yaml` sets `kueue.x-k8s.io/priority-class: medium`
     and `declared-duration-minutes: "120"`).
   - Completed TPU7x JobSets and pod logs are garbage-collected after 120
     minutes; `entrypoint.sh` forwards `--results-uri` to `run.py` so node 0
     uploads `results.json` and `results.csv` to GCS before exiting.

3. **GCS permissions:**
   - The GKE pod Workload Identity / service account must have permission to:
     - Read the uploaded source tarball (`CODE_URI`) and model configs
       (`MODEL_ID`, default `gs://gcs-aiml-huggingface-model-weights/Llama-3.1-8B`
       and `OLMoE-1B-7B-0924`).
     - Create and delete per-case benchmark buckets in `--project`.
     - Write results to `RESULTS_URI` and (for cross-device handoff) read/write
       `CHECKPOINT_EXPORT_ROOT` / `CHECKPOINT_SOURCE_ROOT`.

## Submitting Runs

`submit.sh` tars the current checkout, uploads it to `CODE_URI`, renders the
target JobSet template via `envsubst`, applies it with `kubectl apply`, and
prints the commands to watch pods and stream leader logs.

### 1. A3 Ultra (1 or 2 nodes, 8x H200 per node)

```bash
CODE_URI=gs://<staging-bucket>/code/gcsfs.tar.gz \
RESULTS_URI=gs://<staging-bucket>/results/a3ultra-2n \
bash gcsfs/tests/perf/subsystembenchmarks/checkpointing/pytorch/launch/submit.sh \
  a3ultra 2 -- \
  --project=<PROJECT_ID> \
  --location=europe-west1 \
  --bucket-prefix=<unique-prefix>
```

### 2. TPU7x (`2x2x1` = 1 host / 8 devices, `2x2x2` = 2 hosts / 16 devices)

```bash
CODE_URI=gs://<staging-bucket>/code/gcsfs.tar.gz \
RESULTS_URI=gs://<staging-bucket>/results/tpu7x-2x2x2 \
bash gcsfs/tests/perf/subsystembenchmarks/checkpointing/pytorch/launch/submit.sh \
  tpu7x 2x2x2 -- \
  --project=<PROJECT_ID> \
  --location=us-central1 \
  --bucket-prefix=<unique-prefix>
```

### 3. GPU → TPU Checkpoint Handoff

1. **Export from A3 Ultra** (`checkpoint_write` scenario, exporting to a bucket
   in `us-central1`):
   ```bash
   CODE_URI=gs://<staging-bucket>/code/gcsfs.tar.gz \
   RESULTS_URI=gs://<staging-bucket>/results/handoff-gpu-save \
   CHECKPOINT_EXPORT_ROOT=gs://<us-central1-bucket>/handoff/run-01 \
   bash gcsfs/tests/perf/subsystembenchmarks/checkpointing/pytorch/launch/submit.sh \
     a3ultra 2 -- \
     --project=<PROJECT_ID> \
     --location=europe-west1 \
     --bucket-prefix=<unique-prefix> \
     --filter=save_
   ```

2. **Import and Reshard on TPU7x** (`checkpoint_read` scenario, importing from
   the exported root):
   ```bash
   CODE_URI=gs://<staging-bucket>/code/gcsfs.tar.gz \
   RESULTS_URI=gs://<staging-bucket>/results/handoff-tpu-load \
   CHECKPOINT_SOURCE_ROOT=gs://<us-central1-bucket>/handoff/run-01 \
   bash gcsfs/tests/perf/subsystembenchmarks/checkpointing/pytorch/launch/submit.sh \
     tpu7x 2x2x2 -- \
     --project=<PROJECT_ID> \
     --location=us-central1 \
     --bucket-prefix=<unique-prefix> \
     --filter=load_
   ```
