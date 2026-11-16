#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <a3ultra|tpu7x> <nnodes|topology> -- <run.py args>" >&2
  exit 1
}

if [ "$#" -lt 3 ]; then
  usage
fi

TARGET="$1"
SIZE_OR_TOPOLOGY="$2"
shift 2

if [ "${1:-}" != "--" ]; then
  usage
fi
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../../../.." && pwd)"

case "${TARGET}" in
  a3ultra)
    export NNODES="${SIZE_OR_TOPOLOGY}"
    export TOPOLOGY=""
    export PLACEMENT_POLICY=""
    TEMPLATE="${SCRIPT_DIR}/jobset-a3ultra.yaml"
    export JOBSET_NAME="${JOBSET_NAME:-dcp-a3ultra}"
    REPLICATED_JOB="worker"
    ;;
  tpu7x)
    case "${SIZE_OR_TOPOLOGY}" in
      2x2x1|1)
        export TOPOLOGY="2x2x1"
        export NNODES="1"
        export PLACEMENT_POLICY=""
        ;;
      2x2x2|2)
        export TOPOLOGY="2x2x2"
        export NNODES="2"
        export PLACEMENT_POLICY="tpu7x-16-2x2x2-placement-policy"
        ;;
      *)
        echo "Unsupported TPU7x topology: ${SIZE_OR_TOPOLOGY} (expected 2x2x1 or 2x2x2)" >&2
        exit 1
        ;;
    esac
    TEMPLATE="${SCRIPT_DIR}/jobset-tpu7x.yaml"
    export JOBSET_NAME="${JOBSET_NAME:-dcp-tpu7x}"
    REPLICATED_JOB="slice-job"
    ;;
  *)
    usage
    ;;
esac

# Quote per-argument boundaries in a YAML-safe single-quoted form when arguments
# contain whitespace or shell metacharacters.
HAS_GROUP_FLAG=0
QUOTED_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --group=*|--group)
      HAS_GROUP_FLAG=1
      ;;
  esac
  if [[ "${arg}" =~ ^[A-Za-z0-9_./:=,@+-]+$ ]]; then
    QUOTED_ARGS+=("${arg}")
  else
    bs='\'
    escaped_arg="${arg//\'/\'${bs}\'\'}"
    QUOTED_ARGS+=("'${escaped_arg}'")
  fi
done

if [ "${HAS_GROUP_FLAG}" -eq 0 ]; then
  QUOTED_ARGS=("--group=checkpointing/pytorch" "${QUOTED_ARGS[@]}")
fi
export GROUP_ARGS="${QUOTED_ARGS[*]}"

RUN_TS="$(date +%Y%m%d-%H%M%S)"
BUCKET_PREFIX=""
for arg in "$@"; do
  case "${arg}" in
    --bucket-prefix=*)
      BUCKET_PREFIX="${arg#--bucket-prefix=}"
      ;;
    --results-uri=*)
      RESULTS_URI="${arg#--results-uri=}"
      ;;
  esac
done

export MASTER_PORT="${MASTER_PORT:-29500}"
export CODE_URI="${CODE_URI:-gs://${BUCKET_PREFIX:-gcsfs-bmk}-launch/code/gcsfs-${RUN_TS}.tar.gz}"
export RESULTS_URI="${RESULTS_URI:-gs://${BUCKET_PREFIX:-gcsfs-bmk}-launch/results/${TARGET}-${RUN_TS}}"
export MODEL_ID="${MODEL_ID:-gs://gcs-aiml-huggingface-model-weights/Llama-3.1-8B,gs://gcs-aiml-huggingface-model-weights/OLMoE-1B-7B-0924}"
export CHECKPOINT_EXPORT_ROOT="${CHECKPOINT_EXPORT_ROOT:-}"
export CHECKPOINT_SOURCE_ROOT="${CHECKPOINT_SOURCE_ROOT:-}"

# 1. Tar the checkout and upload it to CODE_URI.
TARBALL="/tmp/gcsfs-${RUN_TS}.tar.gz"
TAR_EXCLUDES=(
  "--exclude=.git"
  "--exclude=.worktrees"
  "--exclude=.venv"
  "--exclude=venv"
  "--exclude=.pytest_cache"
  "--exclude=__pycache__"
  "--exclude=.superpowers"
  "--exclude=docs/superpowers"
  "--exclude=gcsfs/tests/perf/subsystembenchmarks/__run__"
)
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "tar ${TAR_EXCLUDES[*]} -czf ${TARBALL} -C ${REPO_ROOT} ."
  echo "gcloud storage cp ${TARBALL} ${CODE_URI}"
else
  tar "${TAR_EXCLUDES[@]}" -czf "${TARBALL}" -C "${REPO_ROOT}" .
  gcloud storage cp "${TARBALL}" "${CODE_URI}"
  rm -f "${TARBALL}"
fi

# 2. Render the template with envsubst using an explicit variable allowlist.
ENVSUBST_VARS='${JOBSET_NAME} ${NNODES} ${TOPOLOGY} ${PLACEMENT_POLICY} ${CODE_URI} ${RESULTS_URI} ${MODEL_ID} ${GROUP_ARGS} ${MASTER_PORT} ${CHECKPOINT_EXPORT_ROOT} ${CHECKPOINT_SOURCE_ROOT}'
RENDERED="$(envsubst "${ENVSUBST_VARS}" < "${TEMPLATE}" | sed '/placement-policy-name:[[:space:]]*$/d')"

# 3. Apply the JobSet manifest.
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "${RENDERED}"
  echo "kubectl apply -f -"
else
  kubectl delete jobset "${JOBSET_NAME}" -n default --ignore-not-found
  printf '%s\n' "${RENDERED}" | kubectl apply -f -
fi

# 4. Print the follow and log commands.
echo "Follow JobSet status:"
echo "  kubectl get jobset ${JOBSET_NAME} -n default -w"
echo "  kubectl get pods -n default -l jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME} -w"
echo "Stream leader (node 0) logs:"
echo "  kubectl logs -n default -f -l jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME},batch.kubernetes.io/job-completion-index=0"
