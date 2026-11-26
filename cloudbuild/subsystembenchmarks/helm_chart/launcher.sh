#!/usr/bin/env bash
set -euo pipefail

collect_tpu_slice_env() {
  echo "Collecting multi-host TorchTPU slice configuration..."
  local slice_topo_hint=""
  if [[ -n "${SSB_TPU_TOPOLOGY:-}" ]]; then
    slice_topo_hint=$(python3 - <<'PY'
import math, os, re

raw = os.environ["SSB_TPU_TOPOLOGY"].strip()
dims = [int(p) for p in re.split(r"[,xX]", raw) if p]
nodes = int(os.environ.get("GCSFS_SUBSYSTEM_NODES", "1"))
per_node = int(os.environ.get("GCSFS_SUBSYSTEM_DEVICES_PER_NODE", "8"))
if dims:
    chips = math.prod(dims)
    total_devices = nodes * per_node
    if chips > 0 and total_devices > chips and total_devices % chips == 0:
        dims.append(total_devices // chips)
    print(",".join(str(d) for d in dims))
PY
)
  fi
  local raw_tpu_env="" parsed="" slice_topo="" slice_addrs="" tpu_node_rank="" attempt
  for attempt in $(seq 1 60); do
    if [[ -n "$slice_topo_hint" ]]; then
      raw_tpu_env=$(TORCH_TPU_TOPOLOGY="$slice_topo_hint" NODE_RANK="${GCSFS_SUBSYSTEM_NODE_RANK:-0}" python3 -m torch.tpu.distributed 2>/tmp/tpu_dist.err) || { sleep 2; continue; }
    else
      raw_tpu_env=$(python3 -m torch.tpu.distributed 2>/tmp/tpu_dist.err) || { sleep 2; continue; }
    fi
    parsed=$(RAW_TPU_ENV="$raw_tpu_env" python3 - <<'PY'
import os, re, shlex

vals = {}
for line in os.environ["RAW_TPU_ENV"].splitlines():
    m = re.match(r"^(?:export\s+)?([A-Z0-9_]+)=(.*)$", line.strip())
    if not m:
        continue
    key, raw_val = m.group(1), m.group(2).strip()
    try:
        parts = shlex.split(raw_val)
        val = parts[0] if len(parts) == 1 else raw_val
    except ValueError:
        val = raw_val.strip("\"'")
    vals[key] = val

print(vals.get("TORCH_TPU_TOPOLOGY", ""))
print(vals.get("TORCH_TPU_SLICEBUILDER_ADDRESSES", ""))
print(vals.get("NODE_RANK", ""))
PY
)
    IFS=$'\n' read -r -d '' slice_topo slice_addrs tpu_node_rank < <(printf '%s\0' "$parsed") || true
    if [[ -n "$slice_topo" ]] && [[ -n "$slice_addrs" ]]; then
      break
    fi
    sleep 2
  done
  if [[ -z "$slice_topo" ]] || [[ -z "$slice_addrs" ]]; then
    echo "ERROR: python3 -m torch.tpu.distributed failed to emit slice topology/addresses:" >&2
    printf '%s\n' "${raw_tpu_env:-}" >&2
    cat /tmp/tpu_dist.err >&2 || true
    exit 1
  fi
  export GCSFS_SUBSYSTEM_TPU_SLICE_TOPOLOGY="$slice_topo"
  export GCSFS_SUBSYSTEM_TPU_SLICE_ADDRESSES="$slice_addrs"
  unset TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES
  echo "  slice_topology=${GCSFS_SUBSYSTEM_TPU_SLICE_TOPOLOGY} tpu_node_rank=${tpu_node_rank:-unknown}"
}

checkout_gcsfs_repo() {
  local repo="$1"
  local commit="$2"
  local dest="$3"
  rm -rf "$dest"
  mkdir -p "$(dirname "$dest")"
  git init "$dest" >/dev/null
  git -C "$dest" remote add origin "$repo"
  if git -C "$dest" fetch --depth 1 origin "$commit" >/dev/null 2>&1; then
    git -C "$dest" checkout --detach FETCH_HEAD
  else
    git -C "$dest" fetch origin >/dev/null
    git -C "$dest" checkout --detach "$commit" 2>/dev/null \
      || git -C "$dest" checkout --detach "origin/${commit}"
  fi
  echo "Checked out gcsfs commit $(git -C "$dest" rev-parse HEAD)"
}

results_dest_uri() {
  local root="$1"
  local branch="$2"
  local run_id="$3"
  local release="$4"
  local date_dir="$5"
  local ts="$6"
  printf '%s/branch=%s/%s/%s/%s/%s\n' "${root%/}" "$branch" "$date_dir" "$run_id" "$release" "$ts"
}

resolve_machine_type() {
  local detected=""
  detected=$(python3 - <<'PY' 2>/dev/null || true
import urllib.request

req = urllib.request.Request(
    "http://metadata.google.internal/computeMetadata/v1/instance/machine-type",
    headers={"Metadata-Flavor": "Google"},
)
with urllib.request.urlopen(req, timeout=3) as resp:
    raw = resp.read().decode("utf-8").strip()
if raw:
    print(raw.rsplit("/", 1)[-1])
PY
)
  export MACHINE_TYPE="${detected:-${SSB_MACHINE_TYPE_DEFAULT:-}}"
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  return 0
fi

export PYTHONUNBUFFERED=1
export GCSFS_SUBSYSTEM_NODE_RANK="${GCSFS_SUBSYSTEM_NODE_RANK:-${JOB_COMPLETION_INDEX:-0}}"

echo "=== GCSFS Subsystem Benchmark Pod ==="
echo "  release=${SSB_RELEASE_NAME:-}"
echo "  accelerator=${GCSFS_SUBSYSTEM_ACCELERATOR:-}"
echo "  group=${SSB_GROUP:-}"
echo "  nodes=${GCSFS_SUBSYSTEM_NODES:-1} node_rank=${GCSFS_SUBSYSTEM_NODE_RANK} devices_per_node=${GCSFS_SUBSYSTEM_DEVICES_PER_NODE:-8}"
echo "  leader=${GCSFS_SUBSYSTEM_LEADER_ADDR:-}"

if [[ -n "${SSB_RESULTS_URI:-}" ]] && [[ -z "${SSB_RESULTS_BRANCH:-}" || -z "${SSB_RUN_ID:-}" ]]; then
  echo "ERROR: SSB_RESULTS_BRANCH and SSB_RUN_ID must be set when SSB_RESULTS_URI is set" >&2
  exit 1
fi
if [[ "${GCSFS_SUBSYSTEM_NODE_RANK}" == "0" ]]; then
  resolve_machine_type
fi

if ! command -v git >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y --no-install-recommends git ca-certificates >/dev/null
fi

if [[ "${GCSFS_SUBSYSTEM_NODES:-1}" -gt 1 ]] && [[ -n "${GCSFS_SUBSYSTEM_LEADER_ADDR:-}" ]]; then
  echo "Waiting for leader DNS (${GCSFS_SUBSYSTEM_LEADER_ADDR})..."
  attempts=0
  until getent hosts "${GCSFS_SUBSYSTEM_LEADER_ADDR}" >/dev/null 2>&1; do
    attempts=$((attempts + 1))
    if [[ "$attempts" -ge 150 ]]; then
      echo "ERROR: Timed out waiting for ${GCSFS_SUBSYSTEM_LEADER_ADDR}" >&2
      exit 1
    fi
    sleep 2
  done
  echo "Leader DNS resolved."
fi

export PIP_CONFIG_FILE=/dev/null
unset PIP_EXTRA_INDEX_URL PIP_CONSTRAINT

# Isolate pip installs in a system-site-packages venv when available so prebuilt
# container torch/torch_tpu/libtpu are inherited while Debian packages without
# RECORD files (e.g. PyYAML in NGC images) can be cleanly shadowed.
rm -rf /tmp/ssb-venv
if python3 -m venv --system-site-packages /tmp/ssb-venv 2>/dev/null; then
  export VIRTUAL_ENV=/tmp/ssb-venv
  export PATH="/tmp/ssb-venv/bin:${PATH}"
elif python3 -m venv --system-site-packages --without-pip /tmp/ssb-venv 2>/dev/null && /tmp/ssb-venv/bin/python -m pip --version >/dev/null 2>&1; then
  cat > /tmp/ssb-venv/bin/pip <<'EOF'
#!/usr/bin/env bash
exec /tmp/ssb-venv/bin/python -m pip "$@"
EOF
  chmod +x /tmp/ssb-venv/bin/pip
  export VIRTUAL_ENV=/tmp/ssb-venv
  export PATH="/tmp/ssb-venv/bin:${PATH}"
fi

checkout_gcsfs_repo "${GCSFS_REPO}" "${GCSFS_COMMIT}" /workspace/gcsfs
cd /workspace/gcsfs

export GROUP="${SSB_GROUP}"
export MODEL_ID="${SSB_MODEL_ID:-}"
# shellcheck disable=SC1091
source cloudbuild/subsystembenchmarks/scripts/install-group.sh

# Preserve pre-installed container torch/torch_tpu/libtpu while installing group deps + gcsfs.
TORCH_CONSTRAINT=/tmp/ssb_torch_constraint.txt
write_accelerator_constraints "$TORCH_CONSTRAINT"
install_group_requirements
install_accelerator_packages
# Install local editable gcsfs after group requirements so its fsspec floor wins.
install_editable_gcsfs "$TORCH_CONSTRAINT"
install_requirements_override "${SSB_REQUIREMENTS:-}" "$TORCH_CONSTRAINT"

export COMMIT_SHA="$(git -C /workspace/gcsfs rev-parse HEAD)"
if [[ "${GCSFS_SUBSYSTEM_NODE_RANK}" == "0" ]]; then
  export GCSFS_SUBSYSTEM_REQUIREMENTS_RESOLVED="$(pip list --format=json)"
fi

if [[ "${GCSFS_SUBSYSTEM_ACCELERATOR:-}" == "tpu" ]] && [[ "${GCSFS_SUBSYSTEM_NODES:-1}" -gt 1 ]]; then
  collect_tpu_slice_env
fi

DEVICE_FLAG="auto"
case "${GCSFS_SUBSYSTEM_ACCELERATOR:-}" in
  gpu) DEVICE_FLAG="cuda" ;;
  tpu) DEVICE_FLAG="tpu" ;;
esac

RUN_ARGS=(
  --preflight
  --group "${SSB_GROUP}"
  --nodes "${GCSFS_SUBSYSTEM_NODES:-1}"
  --devices-per-node "${GCSFS_SUBSYSTEM_DEVICES_PER_NODE:-8}"
  --device "${DEVICE_FLAG}"
  --bucket-prefix "${SSB_BUCKET_PREFIX}"
  --project "${SSB_PROJECT}"
  --location "${SSB_LOCATION}"
  --bucket-type "${SSB_BUCKET_TYPE:-regional}"
  --results-dir /results
)
if [[ -n "${SSB_ZONE:-}" ]]; then
  RUN_ARGS+=(--zone "${SSB_ZONE}")
fi
if [[ -n "${SSB_FILTER:-}" ]]; then
  RUN_ARGS+=(--filter "${SSB_FILTER}")
fi
if [[ -n "${SSB_CASES:-}" ]]; then
  RUN_ARGS+=(--cases "${SSB_CASES}")
fi
if [[ -n "${SSB_SWEEP_AXES:-}" ]]; then
  RUN_ARGS+=(--sweep-axes "${SSB_SWEEP_AXES}")
fi
if [[ -n "${SSB_MODEL_ID:-}" ]]; then
  RUN_ARGS+=(--model-id "${SSB_MODEL_ID}")
fi
if [[ "${SSB_REQUIRE_AMPLIFICATION:-0}" == "1" ]]; then
  RUN_ARGS+=(--require-amplification)
fi

RUN_RC=0
python3 -m gcsfs.tests.perf.subsystembenchmarks.run "${RUN_ARGS[@]}" || RUN_RC=$?

if [[ "${GCSFS_SUBSYSTEM_NODE_RANK}" == "0" ]] && [[ -n "${SSB_RESULTS_URI:-}" ]] && [[ -f /results/results.csv ]]; then
  RUN_DATE=$(date -u +%Y%m%d)
  RUN_TS=$(date -u +%Y%m%d-%H%M%S)
  DEST_URI=$(results_dest_uri "${SSB_RESULTS_URI}" "${SSB_RESULTS_BRANCH}" "${SSB_RUN_ID}" "${SSB_RELEASE_NAME}" "${RUN_DATE}" "${RUN_TS}")
  echo "Uploading /results to ${DEST_URI}/ ..."
  python3 - "$DEST_URI" <<'PY'
import os
import sys

import gcsfs

dest = sys.argv[1].rstrip("/")
fs = gcsfs.GCSFileSystem()
for name in sorted(os.listdir("/results")):
    local_path = os.path.join("/results", name)
    if os.path.isfile(local_path):
        fs.put(local_path, f"{dest}/{name}")
print(f"Uploaded results to {dest}/")
PY
fi

exit "$RUN_RC"
