#!/usr/bin/env bash
set -euo pipefail

# Match run-gke.sh: read/delete Helm release records stored as ConfigMaps by default.
export HELM_DRIVER="${HELM_DRIVER:-configmap}"

KUBE_CONTEXT=""
NAMESPACE="default"
BUILD_ID=""
INFRA_PREFIX=""
OLDER_THAN_HOURS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kube-context) KUBE_CONTEXT="$2"; shift 2 ;;
    --namespace) NAMESPACE="$2"; shift 2 ;;
    --build-id) BUILD_ID="$2"; shift 2 ;;
    --infra-prefix) INFRA_PREFIX="$2"; shift 2 ;;
    --older-than-hours) OLDER_THAN_HOURS="$2"; shift 2 ;;
    *)
      echo "ERROR: gke-cleanup.sh unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$KUBE_CONTEXT" ]]; then
  echo "WARNING: gke-cleanup.sh missing --kube-context; skipping" >&2
  exit 0
fi

SELECTOR=""
THRESHOLD_SECONDS=0
if [[ -n "$BUILD_ID" ]]; then
  SELECTOR="gcsfs-ssb/build-id=${BUILD_ID}"
elif [[ -n "$INFRA_PREFIX" ]]; then
  SELECTOR="gcsfs-ssb/infra-prefix=${INFRA_PREFIX}"
  if [[ -n "$OLDER_THAN_HOURS" && "$OLDER_THAN_HOURS" =~ ^[0-9]+$ ]]; then
    THRESHOLD_SECONDS=$((OLDER_THAN_HOURS * 3600))
  fi
else
  echo "WARNING: gke-cleanup.sh requires --build-id or --infra-prefix; skipping" >&2
  exit 0
fi

NOW_EPOCH=$(date +%s)
ROWS=$(kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE" get jobsets \
  -l "$SELECTOR" \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.creationTimestamp}{"\n"}{end}' \
  2>/dev/null || true)

TO_DELETE=()
while IFS=$'\t' read -r name created; do
  [[ -n "$name" ]] || continue
  if (( THRESHOLD_SECONDS > 0 )); then
    created_epoch=$(date -d "$created" +%s 2>/dev/null || true)
    if [[ -z "$created_epoch" ]]; then
      continue
    fi
    age=$((NOW_EPOCH - created_epoch))
    if (( age <= THRESHOLD_SECONDS )); then
      echo "Skipping active GKE release ${name} (age=${age}s <= ${THRESHOLD_SECONDS}s)"
      continue
    fi
  fi
  TO_DELETE+=("$name")
done <<< "$ROWS"

for name in "${TO_DELETE[@]}"; do
  echo "Uninstalling GKE release ${name} (ns=${NAMESPACE})..."
  helm uninstall "$name" --kube-context "$KUBE_CONTEXT" --namespace "$NAMESPACE" >/dev/null 2>&1 || true &
done
wait

exit 0
