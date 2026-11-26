#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

TOOLS_DIR="${SSB_TOOLS_DIR:-/workspace/.ssb-tools}"
if [[ -d "$TOOLS_DIR" ]]; then
  export PATH="${TOOLS_DIR}:${PATH}"
fi

BUILD_VARS="${SSB_BUILD_VARS:-/workspace/build_vars.env}"
GKE_VARS="${SSB_GKE_VARS:-/workspace/gke_vars.env}"
source "$BUILD_VARS"
source "$GKE_VARS"

if [[ "${SSB_SKIP_VENV_SETUP:-0}" != "1" ]]; then
  VENV_DIR="${SSB_VENV_DIR:-/tmp/ssb-gke-venv}"
  rm -rf "$VENV_DIR"
  python3 -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  pip install --upgrade pip >/dev/null

  cd "$REPO_ROOT"
  if [[ ! -d .git ]]; then
    export SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0"
  fi

  # Install group requirements using CPU PyTorch wheels so pytest --collect-only
  # can import benchmark modules without downloading multi-GB CUDA wheels.
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/install-group.sh"
  PIP_INDEX_URL="https://download.pytorch.org/whl/cpu" \
    PIP_EXTRA_INDEX_URL="https://pypi.org/simple" \
    install_group_requirements
  # Install local editable gcsfs after group requirements so its fsspec floor wins.
  install_editable_gcsfs

  export SSB_PYTHON="${VENV_DIR}/bin/python"
fi

KUBE_CONTEXT=$(bash "${SCRIPT_DIR}/gke-credentials.sh" \
  "$GKE_CLUSTER" "$GKE_CLUSTER_PROJECT" "$GKE_CLUSTER_LOCATION" "$GKE_ENDPOINT")

RUN_GKE="${SSB_RUN_GKE_SCRIPT:-${SCRIPT_DIR}/run-gke.sh}"
ARGS=(
  --accelerator "$ACCELERATOR"
  --kube-context "$KUBE_CONTEXT"
  --namespace "$GKE_NAMESPACE"
  --project "$PROJECT_ID"
  --location "$REGION"
  --zone "$ZONE"
  --group "$GROUP"
  --bucket-type "$BUCKET_TYPE"
  --bucket-prefix "$BUCKET_PREFIX"
  --results-uri "gs://${RESULTS_BUCKET}/subsystembenchmarks"
  --branch "$BRANCH_NAME"
  --run-id "$RUN_ID"
  --repo "$GCSFS_REPO"
  --commit "$COMMIT_SHA"
  --build-id "$RUN_ID"
  --infra-prefix "$INFRA_PREFIX"
  --require-amplification
  --priority "$TPU_PRIORITY"
  --duration "$TPU_DURATION"
  --max-concurrent "$MAX_CONCURRENT_RELEASES"
  --deadline-seconds "$GKE_DEADLINE_SECONDS"
)

if [[ -n "${FILTER:-}" ]]; then
  ARGS+=(--filter "$FILTER")
fi
if [[ -n "${SWEEP_AXES:-}" ]]; then
  ARGS+=(--sweep-axes "$SWEEP_AXES")
fi
if [[ -n "${MODEL_ID:-}" ]]; then
  ARGS+=(--model-id "$MODEL_ID")
fi
if [[ -n "${REQUIREMENTS_OVERRIDE:-}" ]]; then
  ARGS+=(--requirements "$REQUIREMENTS_OVERRIDE")
fi
if [[ -n "${GKE_QUEUE:-}" ]]; then
  ARGS+=(--queue-name "$GKE_QUEUE")
fi
if [[ -n "${TPU_PLACEMENT_POLICY:-}" ]]; then
  ARGS+=(--tpu-placement-policy "$TPU_PLACEMENT_POLICY")
fi
if [[ -n "${MAX_CASES_PER_RELEASE:-}" ]]; then
  ARGS+=(--max-cases-per-release "$MAX_CASES_PER_RELEASE")
fi

exec bash "$RUN_GKE" "${ARGS[@]}"
