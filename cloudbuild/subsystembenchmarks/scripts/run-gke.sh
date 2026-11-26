#!/usr/bin/env bash
set -euo pipefail

# Store Helm release metadata in ConfigMaps by default so shared clusters whose
# namespace RBAC withholds Secrets (e.g. TPU power-users) work out of the box.
export HELM_DRIVER="${HELM_DRIVER:-configmap}"

usage() {
  cat <<'EOF'
Usage: run-gke.sh --accelerator gpu|tpu --kube-context CTX --project PROJECT \
  --location LOCATION --zone ZONE --group GROUP --results-uri URI [options]

Required:
  --accelerator gpu|tpu     Target accelerator type
  --kube-context CTX        kubectl context for the target GKE cluster
  --project PROJECT         GCP project for temporary per-case buckets
  --location LOCATION       GCS region for temporary buckets (e.g. us-central1)
  --zone ZONE               GCE zone for zonal/cache buckets and zone pinning
  --group GROUP             Subsystem benchmark group, e.g. checkpointing/pytorch_save
  --results-uri URI         GCS URI prefix for uploaded results

Optional:
  --namespace NS            Kubernetes namespace (default: default)
  --branch BRANCH           Branch label in upload path (default: local git branch)
  --run-id UUID             Run/build UUID in upload path (default: random uuid4)
  --filter EXPR             pytest -k filter expression
  --sweep-axes AXES         Whitespace-separated sweep axes
  --bucket-type TYPE        Storage tier (default: regional)
  --model-id ID             Override model ID
  --bucket-prefix PREFIX    Override temporary per-case bucket prefix
  --repo URL                Git repo URL cloned by pods (default: origin HTTPS URL)
  --commit SHA              Git commit or branch cloned by pods (default: HEAD)
  --requirements SPECS      Whitespace-separated pip requirements installed last
  --require-amplification   Require Cloud Monitoring read-amplification metrics
  --priority PRIO           TPU Kueue priority class (default: low)
  --duration MINUTES        TPU declared-duration-minutes (default: 180)
  --tpu-placement-policy P  Placement policy name (required for multi-node TPU)
  --queue-name NAME         Optional Kueue LocalQueue label (kueue.x-k8s.io/queue-name)
  --devices-per-node N      Accelerators per node (default: 8)
  --max-cases-per-release N Split each node count into batches of at most N exact case IDs
  --max-concurrent N        Max concurrent Helm releases (default: 1)
  --deadline-seconds S      Overall queueing + execution deadline in seconds (default: 16200)
  --build-id ID             Label value for gcsfs-ssb/build-id
  --infra-prefix PREFIX     Label value for gcsfs-ssb/infra-prefix
  --dry-run                 Print helm install commands without executing
  -h, --help                Show this help
EOF
}

ACCELERATOR=""
KUBE_CONTEXT=""
PROJECT=""
LOCATION=""
ZONE=""
GROUP=""
RESULTS_URI=""
NAMESPACE="default"
BRANCH=""
RUN_ID=""
FILTER=""
SWEEP_AXES=""
BUCKET_TYPE="regional"
MODEL_ID=""
BUCKET_PREFIX=""
GCSFS_REPO=""
GCSFS_COMMIT=""
REQUIREMENTS=""
REQUIRE_AMPLIFICATION="false"
TPU_PRIORITY="low"
TPU_DURATION="180"
TPU_PLACEMENT_POLICY=""
QUEUE_NAME=""
DEVICES_PER_NODE="8"
MAX_CASES_PER_RELEASE=""
MAX_CONCURRENT="1"
DEADLINE_SECONDS="16200"
BUILD_ID_LABEL=""
INFRA_PREFIX_LABEL=""
DRY_RUN=0

require_arg() {
  if [[ "$2" -lt 2 ]]; then
    echo "ERROR: $1 requires a value" >&2
    usage >&2
    exit 2
  fi
}

helm_escape() {
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//,/\\,}"
  s="${s//\{/\\\{}"
  s="${s//\}/\\\}}"
  printf '%s' "$s"
}

validate_positive_int() {
  local flag="$1"
  local val="$2"
  if ! [[ "$val" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ${flag} must be a positive integer, got '${val}'" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --accelerator) require_arg "$1" "$#"; ACCELERATOR="$2"; shift 2 ;;
    --kube-context) require_arg "$1" "$#"; KUBE_CONTEXT="$2"; shift 2 ;;
    --project) require_arg "$1" "$#"; PROJECT="$2"; shift 2 ;;
    --location) require_arg "$1" "$#"; LOCATION="$2"; shift 2 ;;
    --zone) require_arg "$1" "$#"; ZONE="$2"; shift 2 ;;
    --group) require_arg "$1" "$#"; GROUP="$2"; shift 2 ;;
    --results-uri) require_arg "$1" "$#"; RESULTS_URI="$2"; shift 2 ;;
    --namespace) require_arg "$1" "$#"; NAMESPACE="$2"; shift 2 ;;
    --branch) require_arg "$1" "$#"; BRANCH="$2"; shift 2 ;;
    --run-id) require_arg "$1" "$#"; RUN_ID="$2"; shift 2 ;;
    --filter) require_arg "$1" "$#"; FILTER="$2"; shift 2 ;;
    --sweep-axes) require_arg "$1" "$#"; SWEEP_AXES="$2"; shift 2 ;;
    --bucket-type) require_arg "$1" "$#"; BUCKET_TYPE="$2"; shift 2 ;;
    --model-id) require_arg "$1" "$#"; MODEL_ID="$2"; shift 2 ;;
    --bucket-prefix) require_arg "$1" "$#"; BUCKET_PREFIX="$2"; shift 2 ;;
    --repo) require_arg "$1" "$#"; GCSFS_REPO="$2"; shift 2 ;;
    --commit) require_arg "$1" "$#"; GCSFS_COMMIT="$2"; shift 2 ;;
    --requirements) require_arg "$1" "$#"; REQUIREMENTS="$2"; shift 2 ;;
    --require-amplification) REQUIRE_AMPLIFICATION="true"; shift ;;
    --priority) require_arg "$1" "$#"; TPU_PRIORITY="$2"; shift 2 ;;
    --duration) require_arg "$1" "$#"; TPU_DURATION="$2"; shift 2 ;;
    --tpu-placement-policy) require_arg "$1" "$#"; TPU_PLACEMENT_POLICY="$2"; shift 2 ;;
    --queue-name) require_arg "$1" "$#"; QUEUE_NAME="$2"; shift 2 ;;
    --devices-per-node) require_arg "$1" "$#"; DEVICES_PER_NODE="$2"; shift 2 ;;
    --max-cases-per-release) require_arg "$1" "$#"; MAX_CASES_PER_RELEASE="$2"; shift 2 ;;
    --max-concurrent) require_arg "$1" "$#"; MAX_CONCURRENT="$2"; shift 2 ;;
    --deadline-seconds) require_arg "$1" "$#"; DEADLINE_SECONDS="$2"; shift 2 ;;
    --build-id) require_arg "$1" "$#"; BUILD_ID_LABEL="$2"; shift 2 ;;
    --infra-prefix) require_arg "$1" "$#"; INFRA_PREFIX_LABEL="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

check_required() {
  local flag="$1"
  local val="$2"
  if [[ -z "$val" ]]; then
    echo "ERROR: ${flag} is required" >&2
    usage >&2
    exit 2
  fi
}

check_required "--accelerator" "$ACCELERATOR"
check_required "--kube-context" "$KUBE_CONTEXT"
check_required "--project" "$PROJECT"
check_required "--location" "$LOCATION"
check_required "--zone" "$ZONE"
check_required "--group" "$GROUP"
check_required "--results-uri" "$RESULTS_URI"

case "$ACCELERATOR" in
  gpu) DEVICE_FLAG="cuda" ;;
  tpu) DEVICE_FLAG="tpu" ;;
  *)
    echo "ERROR: --accelerator must be 'gpu' or 'tpu', got '$ACCELERATOR'" >&2
    exit 2
    ;;
esac

validate_positive_int "--duration" "$TPU_DURATION"
validate_positive_int "--devices-per-node" "$DEVICES_PER_NODE"
validate_positive_int "--max-concurrent" "$MAX_CONCURRENT"
validate_positive_int "--deadline-seconds" "$DEADLINE_SECONDS"
if [[ -n "$MAX_CASES_PER_RELEASE" ]]; then
  validate_positive_int "--max-cases-per-release" "$MAX_CASES_PER_RELEASE"
fi

abbreviate_group() {
  local grp="$1"
  local n="$2"
  case "$grp" in
    dataloading/huggingface_datasets) printf 'hf' ;;
    dataloading/ray_data) printf 'raydata' ;;
    dataloading/webdataset) printf 'wds' ;;
    checkpointing/pytorch_save) printf 'save' ;;
    checkpointing/pytorch_load) printf 'load' ;;
    checkpointing/pytorch_reshard) printf 'rshd' ;;
    checkpointing/pytorch_lightning) printf 'pl' ;;
    checkpointing/ray_pytorch) printf 'raypt' ;;
    *)
      # Cap slug so ssb-<abbr>-<n>n-<4hex> stays <= 20 chars (TPU Kueue limit).
      local max_len=$((9 - ${#n}))
      (( max_len < 1 )) && max_len=1
      local slug
      slug=$(printf '%s' "${grp##*/}" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9')
      slug="${slug:0:$max_len}"
      printf '%s' "${slug:-grp}"
      ;;
  esac
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
CHART_DIR="${REPO_ROOT}/cloudbuild/subsystembenchmarks/helm_chart"

if [[ -z "$GCSFS_REPO" ]]; then
  raw_url=$(git -C "$REPO_ROOT" remote get-url origin 2>/dev/null || echo "https://github.com/fsspec/gcsfs.git")
  if [[ "$raw_url" == git@github.com:* ]]; then
    GCSFS_REPO="https://github.com/${raw_url#git@github.com:}"
  else
    GCSFS_REPO="$raw_url"
  fi
fi
if [[ -z "$GCSFS_COMMIT" ]]; then
  GCSFS_COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)
fi
if [[ -z "$BRANCH" ]]; then
  BRANCH=$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "detached")
fi
BRANCH=$(printf '%s' "$BRANCH" | tr -c 'a-zA-Z0-9_.-' '-')
if [[ -z "$RUN_ID" ]]; then
  RUN_ID=$(python3 -c 'import uuid; print(uuid.uuid4())')
fi
EFFECTIVE_BUILD_ID="${BUILD_ID_LABEL:-${RUN_ID}}"

PY="${SSB_PYTHON:-python3}"
LIST_ARGS=(
  -m gcsfs.tests.perf.subsystembenchmarks.run
  --group "$GROUP"
  --devices-per-node "$DEVICES_PER_NODE"
  --device "$DEVICE_FLAG"
  --bucket-type "$BUCKET_TYPE"
)
if [[ -n "$MAX_CASES_PER_RELEASE" ]]; then
  LIST_ARGS+=(--list-cases)
else
  LIST_ARGS+=(--list-node-counts)
fi
if [[ -n "$FILTER" ]]; then
  LIST_ARGS+=(--filter "$FILTER")
fi
if [[ -n "$SWEEP_AXES" ]]; then
  LIST_ARGS+=(--sweep-axes "$SWEEP_AXES")
fi
if [[ -n "$MODEL_ID" ]]; then
  LIST_ARGS+=(--model-id "$MODEL_ID")
fi

LIST_JSON=$(cd "$REPO_ROOT" && PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "$PY" "${LIST_ARGS[@]}")

# Build parallel arrays PLAN_NODES[i] and PLAN_CASES[i] (space-separated exact IDs, or empty).
PLAN_NODES=()
PLAN_CASES=()
while IFS=$'\t' read -r p_nodes p_cases; do
  [[ -n "$p_nodes" ]] || continue
  PLAN_NODES+=("$p_nodes")
  PLAN_CASES+=("$p_cases")
done < <(python3 - "$LIST_JSON" "${MAX_CASES_PER_RELEASE:-0}" <<'PY'
import json
import sys
from collections import defaultdict

data = json.loads(sys.argv[1])
max_per = int(sys.argv[2])
if max_per > 0:
    by_nodes = defaultdict(list)
    for entry in data:
        by_nodes[int(entry["nodes"])].append(str(entry["id"]))
    for n in sorted(by_nodes):
        ids = by_nodes[n]
        for i in range(0, len(ids), max_per):
            print(f"{n}\t{' '.join(ids[i : i + max_per])}")
else:
    for n in data:
        print(f"{int(n)}\t")
PY
)

if [[ ${#PLAN_NODES[@]} -eq 0 ]]; then
  echo "No runnable cases for group=${GROUP} on accelerator=${ACCELERATOR}; nothing to launch."
  exit 0
fi

if [[ "$ACCELERATOR" == "tpu" && -z "$TPU_PLACEMENT_POLICY" ]]; then
  for n in "${PLAN_NODES[@]}"; do
    if [[ "$n" -gt 1 ]]; then
      echo "ERROR: --tpu-placement-policy is required when launching multi-node TPU releases" >&2
      exit 2
    fi
  done
fi

rand_hex4() {
  if [[ -r /proc/sys/kernel/random/uuid ]]; then
    local u
    IFS= read -r u < /proc/sys/kernel/random/uuid
    printf '%s\n' "${u:0:4}"
  else
    python3 -c 'import secrets; print(secrets.token_hex(2))'
  fi
}

build_helm_cmd() {
  local release="$1"
  local n="$2"
  local cases_str="$3"
  local bprefix="$4"
  HELM_CMD=(
    helm install "$release" "$CHART_DIR"
    --kube-context "$KUBE_CONTEXT"
    --namespace "$NAMESPACE"
    --set-string "accelerator=$(helm_escape "${ACCELERATOR}")"
    --set "nodes=${n}"
    --set "devicesPerNode=${DEVICES_PER_NODE}"
    --set-string "group=$(helm_escape "${GROUP}")"
    --set-string "filter=$(helm_escape "${FILTER}")"
    --set-string "cases=$(helm_escape "${cases_str}")"
    --set-string "sweepAxes=$(helm_escape "${SWEEP_AXES}")"
    --set-string "requirements=$(helm_escape "${REQUIREMENTS}")"
    --set-string "bucketPrefix=$(helm_escape "${bprefix}")"
    --set-string "project=$(helm_escape "${PROJECT}")"
    --set-string "location=$(helm_escape "${LOCATION}")"
    --set-string "bucketType=$(helm_escape "${BUCKET_TYPE}")"
    --set-string "zone=$(helm_escape "${ZONE}")"
    --set-string "modelId=$(helm_escape "${MODEL_ID}")"
    --set-string "resultsUri=$(helm_escape "${RESULTS_URI}")"
    --set-string "resultsBranch=$(helm_escape "${BRANCH}")"
    --set-string "runId=$(helm_escape "${RUN_ID}")"
    --set "requireAmplification=${REQUIRE_AMPLIFICATION}"
    --set-string "buildId=$(helm_escape "${BUILD_ID_LABEL}")"
    --set-string "infraPrefix=$(helm_escape "${INFRA_PREFIX_LABEL}")"
    --set-string "gcsfsRepo=$(helm_escape "${GCSFS_REPO}")"
    --set-string "gcsfsCommit=$(helm_escape "${GCSFS_COMMIT}")"
  )
  if [[ "$ACCELERATOR" == "tpu" ]]; then
    HELM_CMD+=(
      --set-string "tpu.priority=$(helm_escape "${TPU_PRIORITY}")"
      --set "tpu.durationMinutes=${TPU_DURATION}"
    )
    if [[ "$n" -gt 1 ]]; then
      HELM_CMD+=(--set-string "tpu.multiNodePlacementPolicy=$(helm_escape "${TPU_PLACEMENT_POLICY}")")
    fi
  fi
  if [[ -n "$QUEUE_NAME" ]]; then
    HELM_CMD+=(--set-string "kueueQueue=$(helm_escape "${QUEUE_NAME}")")
  fi
}

if [[ "$DRY_RUN" -eq 1 ]]; then
  for idx in "${!PLAN_NODES[@]}"; do
    n="${PLAN_NODES[$idx]}"
    cases_str="${PLAN_CASES[$idx]}"
    HEX4=$(rand_hex4)
    ABBR=$(abbreviate_group "$GROUP" "$n")
    RELEASE="ssb-${ABBR}-${n}n-${HEX4}"
    BPREFIX="${BUCKET_PREFIX:-ssb-${ACCELERATOR}-${HEX4}}"
    build_helm_cmd "$RELEASE" "$n" "$cases_str" "$BPREFIX"
    printf '%q ' "${HELM_CMD[@]}"
    printf '\n'
  done
  exit 0
fi

declare -a ACTIVE_RELEASES=()
declare -A ACTIVE_NODES=()

uninstall_release() {
  local rel="$1"
  echo "Cleaning up Helm release ${rel}..."
  helm uninstall "$rel" --kube-context "$KUBE_CONTEXT" --namespace "$NAMESPACE" >/dev/null 2>&1 || true
}

cleanup_all() {
  local rel
  for rel in "${ACTIVE_RELEASES[@]}"; do
    uninstall_release "$rel" &
  done
  wait
  ACTIVE_RELEASES=()
}
trap cleanup_all EXIT INT TERM

print_release_logs() {
  local rel="$1"
  local n="$2"
  local failed="$3"
  local max_idx=0
  if [[ "$failed" -eq 1 ]]; then
    max_idx=$((n - 1))
  fi
  local pod_idx
  for ((pod_idx = 0; pod_idx <= max_idx; pod_idx++)); do
    echo "=== Logs from ${rel} pod ${pod_idx} ==="
    kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE" logs \
      -l "jobset.sigs.k8s.io/jobset-name=${rel},batch.kubernetes.io/job-completion-index=${pod_idx}" \
      --tail=-1 || true
  done
}

POLL_SECONDS="${SSB_POLL_SECONDS:-10}"
DEADLINE_EPOCH=$(( $(date +%s) + DEADLINE_SECONDS ))
NEXT_IDX=0
TOTAL_PLANNED=${#PLAN_NODES[@]}
OVERALL_RC=0

while (( NEXT_IDX < TOTAL_PLANNED || ${#ACTIVE_RELEASES[@]} > 0 )); do
  while (( NEXT_IDX < TOTAL_PLANNED && ${#ACTIVE_RELEASES[@]} < MAX_CONCURRENT )); do
    if (( $(date +%s) >= DEADLINE_EPOCH )); then
      echo "ERROR: Deadline (${DEADLINE_SECONDS}s) exceeded before launching remaining releases (${NEXT_IDX}/${TOTAL_PLANNED} launched)." >&2
      cleanup_all
      exit 1
    fi
    n="${PLAN_NODES[$NEXT_IDX]}"
    cases_str="${PLAN_CASES[$NEXT_IDX]}"
    NEXT_IDX=$((NEXT_IDX + 1))
    HEX4=$(rand_hex4)
    ABBR=$(abbreviate_group "$GROUP" "$n")
    RELEASE="ssb-${ABBR}-${n}n-${HEX4}"
    BPREFIX="${BUCKET_PREFIX:-ssb-${ACCELERATOR}-${HEX4}}"
    build_helm_cmd "$RELEASE" "$n" "$cases_str" "$BPREFIX"
    ACTIVE_RELEASES+=("$RELEASE")
    ACTIVE_NODES["$RELEASE"]="$n"
    if ! "${HELM_CMD[@]}"; then
      echo "ERROR: helm install failed for ${RELEASE}" >&2
      uninstall_release "$RELEASE"
      unset 'ACTIVE_NODES[$RELEASE]'
      ACTIVE_RELEASES=("${ACTIVE_RELEASES[@]:0:${#ACTIVE_RELEASES[@]}-1}")
      OVERALL_RC=1
      continue
    fi
    echo "Waiting for JobSet ${RELEASE} (${KUBE_CONTEXT}, ns=${NAMESPACE})..."
  done

  if (( ${#ACTIVE_RELEASES[@]} == 0 )); then
    break
  fi

  declare -A COND_BY_REL=()
  while IFS=$'\t' read -r js_name js_cond; do
    [[ -n "$js_name" ]] || continue
    COND_BY_REL["$js_name"]="$js_cond"
  done < <(kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE" get jobsets \
    -l "gcsfs-ssb/build-id=${EFFECTIVE_BUILD_ID}" \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{range .status.conditions[*]}{.type}={.status}{" "}{end}{"\n"}{end}' \
    2>/dev/null || true)

  STILL_ACTIVE=()
  for rel in "${ACTIVE_RELEASES[@]}"; do
    cond="${COND_BY_REL[$rel]:-}"
    if [[ "$cond" == *"Completed=True"* ]]; then
      echo "JobSet ${rel} completed."
      print_release_logs "$rel" "${ACTIVE_NODES[$rel]}" 0
      uninstall_release "$rel"
      unset 'ACTIVE_NODES[$rel]'
    elif [[ "$cond" == *"Failed=True"* ]]; then
      echo "JobSet ${rel} failed." >&2
      print_release_logs "$rel" "${ACTIVE_NODES[$rel]}" 1
      uninstall_release "$rel"
      unset 'ACTIVE_NODES[$rel]'
      OVERALL_RC=1
    else
      STILL_ACTIVE+=("$rel")
    fi
  done
  unset COND_BY_REL
  ACTIVE_RELEASES=("${STILL_ACTIVE[@]}")

  if (( ${#ACTIVE_RELEASES[@]} > 0 )); then
    if (( $(date +%s) >= DEADLINE_EPOCH )); then
      echo "ERROR: Deadline (${DEADLINE_SECONDS}s) exceeded with active releases: ${ACTIVE_RELEASES[*]}" >&2
      kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE" get workloads 2>/dev/null || true
      kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE" get pods,events 2>/dev/null || true
      cleanup_all
      exit 1
    fi
    if [[ "$POLL_SECONDS" == "0" ]]; then
      sleep 0.05
    else
      sleep "$POLL_SECONDS"
    fi
  fi
done

exit "$OVERALL_RC"
