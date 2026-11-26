#!/usr/bin/env bash
set -euo pipefail

OUT_FILE="${1:-/workspace/gke_vars.env}"

ACC="${_ACCELERATOR-cpu}"
MODE="${_MODE-nightly}"
ENDPOINT="${_GKE_ENDPOINT:-default}"
NAMESPACE="${_GKE_NAMESPACE:-default}"
TPU_PRIO="${_TPU_PRIORITY:-low}"
TPU_DUR="${_TPU_DURATION:-180}"
MAX_CONC="${_MAX_CONCURRENT_RELEASES:-1}"
DEADLINE="${_GKE_DEADLINE_SECONDS:-16200}"
REPO="${_GCSFS_REPO:-https://github.com/fsspec/gcsfs.git}"
MAX_CASES="${_MAX_CASES_PER_RELEASE:-}"

case "$ACC" in
  cpu|gpu|tpu) ;;
  *)
    echo "ERROR: unknown _ACCELERATOR '${ACC}' (expected cpu, gpu, or tpu)" >&2
    exit 1
    ;;
esac

case "$ENDPOINT" in
  default|internal-ip|dns) ;;
  *)
    echo "ERROR: unknown _GKE_ENDPOINT '${ENDPOINT}' (expected default, internal-ip, or dns)" >&2
    exit 1
    ;;
esac

validate_pos_int() {
  local name="$1"
  local val="$2"
  if ! [[ "$val" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ${name} must be a positive integer, got '${val}'" >&2
    exit 1
  fi
}

if [[ "$ACC" != "cpu" ]]; then
  if [[ "$MODE" != "nightly" ]]; then
    echo "ERROR: _ACCELERATOR=${ACC} only supports _MODE=nightly (got '${MODE}')" >&2
    exit 1
  fi
  if [[ -n "${_HF_TOKEN:-}" ]]; then
    echo "ERROR: _HF_TOKEN must be empty for _ACCELERATOR=${ACC} (secrets are never forwarded to pods)" >&2
    exit 1
  fi
  if [[ -z "${COMMIT_SHA:-}" ]]; then
    echo "ERROR: COMMIT_SHA must be non-empty for _ACCELERATOR=${ACC} so pods know what revision to clone" >&2
    exit 1
  fi
  for req_var in _GKE_CLUSTER _GKE_CLUSTER_PROJECT _GKE_CLUSTER_LOCATION; do
    if [[ -z "${!req_var:-}" ]]; then
      echo "ERROR: ${req_var} is required when _ACCELERATOR=${ACC}" >&2
      exit 1
    fi
  done

  if [[ -n "${_ZONE:-}" ]]; then
    zone_region="${_ZONE%-*}"
    loc="${_GKE_CLUSTER_LOCATION}"
    if [[ "$loc" =~ ^[a-z]+-[a-z0-9]+-[a-z]$ ]]; then
      cluster_region="${loc%-*}"
    else
      cluster_region="$loc"
    fi
    if [[ "$zone_region" != "$cluster_region" ]]; then
      echo "ERROR: _ZONE '${_ZONE}' (region '${zone_region}') does not match _GKE_CLUSTER_LOCATION '${loc}' (region '${cluster_region}')" >&2
      exit 1
    fi
  fi

  validate_pos_int "_GKE_DEADLINE_SECONDS" "$DEADLINE"
  validate_pos_int "_MAX_CONCURRENT_RELEASES" "$MAX_CONC"
  validate_pos_int "_TPU_DURATION" "$TPU_DUR"
  if [[ -n "$MAX_CASES" ]]; then
    validate_pos_int "_MAX_CASES_PER_RELEASE" "$MAX_CASES"
  fi
  if [[ "$ACC" == "tpu" ]]; then
    case "$TPU_PRIO" in
      medium|low|very-low) ;;
      *)
        echo "ERROR: _TPU_PRIORITY must be medium, low, or very-low (got '${TPU_PRIO}')" >&2
        exit 1
        ;;
    esac
  fi
fi

mkdir -p "$(dirname "$OUT_FILE")"
{
  printf 'export ACCELERATOR=%q\n' "$ACC"
  printf 'export GKE_CLUSTER=%q\n' "${_GKE_CLUSTER:-}"
  printf 'export GKE_CLUSTER_PROJECT=%q\n' "${_GKE_CLUSTER_PROJECT:-}"
  printf 'export GKE_CLUSTER_LOCATION=%q\n' "${_GKE_CLUSTER_LOCATION:-}"
  printf 'export GKE_NAMESPACE=%q\n' "$NAMESPACE"
  printf 'export GKE_ENDPOINT=%q\n' "$ENDPOINT"
  printf 'export GKE_QUEUE=%q\n' "${_GKE_QUEUE:-}"
  printf 'export TPU_PRIORITY=%q\n' "$TPU_PRIO"
  printf 'export TPU_DURATION=%q\n' "$TPU_DUR"
  printf 'export TPU_PLACEMENT_POLICY=%q\n' "${_TPU_PLACEMENT_POLICY:-}"
  printf 'export MAX_CASES_PER_RELEASE=%q\n' "$MAX_CASES"
  printf 'export MAX_CONCURRENT_RELEASES=%q\n' "$MAX_CONC"
  printf 'export GKE_DEADLINE_SECONDS=%q\n' "$DEADLINE"
  printf 'export GCSFS_REPO=%q\n' "$REPO"
  printf 'export INFRA_PREFIX=%q\n' "${_INFRA_PREFIX:-}"
} > "$OUT_FILE"
