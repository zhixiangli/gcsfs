#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: gke-credentials.sh CLUSTER PROJECT LOCATION [default|internal-ip|dns]" >&2
  exit 2
fi

CLUSTER="$1"
PROJECT="$2"
LOCATION="$3"
ENDPOINT="${4:-default}"

EXTRA_ARGS=()
case "$ENDPOINT" in
  default) ;;
  internal-ip) EXTRA_ARGS+=(--internal-ip) ;;
  dns) EXTRA_ARGS+=(--dns-endpoint) ;;
  *)
    echo "ERROR: unsupported endpoint mode '${ENDPOINT}' (expected default, internal-ip, or dns)" >&2
    exit 2
    ;;
esac

CTX="gke_${PROJECT}_${LOCATION}_${CLUSTER}"

if [[ -n "${KUBECONFIG:-}" && -s "${KUBECONFIG}" ]] \
  && command -v kubectl >/dev/null 2>&1 \
  && kubectl config get-contexts -o name 2>/dev/null | grep -Fxq "$CTX"; then
  printf '%s\n' "$CTX"
  exit 0
fi

gcloud container clusters get-credentials "$CLUSTER" \
  --project "$PROJECT" \
  --location "$LOCATION" \
  "${EXTRA_ARGS[@]}" >&2

printf '%s\n' "$CTX"
