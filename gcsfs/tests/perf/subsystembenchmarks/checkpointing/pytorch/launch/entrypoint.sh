#!/usr/bin/env bash
set -euo pipefail

run_cmd() {
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "$*"
  else
    eval "$*"
  fi
}

DEVICE="${DEVICE:-cpu}"
NNODES="${NNODES:-1}"
JOB_COMPLETION_INDEX="${JOB_COMPLETION_INDEX:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
MODEL_ID="${MODEL_ID:-gs://gcs-aiml-huggingface-model-weights/Llama-3.1-8B,gs://gcs-aiml-huggingface-model-weights/OLMoE-1B-7B-0924}"
CODE_URI="${CODE_URI:-}"
GROUP_ARGS="${GROUP_ARGS:---group=checkpointing/pytorch}"
RESULTS_URI="${RESULTS_URI:-}"
WORKDIR="${WORKDIR:-/tmp/gcsfs-src}"
VENV_DIR="${VENV_DIR:-/tmp/gcsfs-venv}"

# 1. Fetch and extract CODE_URI.
if [ -n "${CODE_URI}" ]; then
  run_cmd "mkdir -p ${WORKDIR}"
  run_cmd "gcloud storage cp ${CODE_URI} /tmp/gcsfs-src.tar.gz"
  run_cmd "tar -xzf /tmp/gcsfs-src.tar.gz -C ${WORKDIR}"
  run_cmd "cd ${WORKDIR}"
fi

# 2. Create a venv and install requirements-$DEVICE.txt (requirements.txt for cpu) and -e .
if [ "${DEVICE}" = "cpu" ]; then
  REQ_FILE="gcsfs/tests/perf/subsystembenchmarks/checkpointing/pytorch/requirements.txt"
else
  REQ_FILE="gcsfs/tests/perf/subsystembenchmarks/checkpointing/pytorch/requirements-${DEVICE}.txt"
fi
run_cmd "python3 -m venv --system-site-packages ${VENV_DIR}"
run_cmd ". ${VENV_DIR}/bin/activate"
run_cmd "pip install -r ${REQ_FILE}"
run_cmd "pip install -e ."

# 3. Copy model config.json to /tmp/<name>/.
IFS=',' read -ra _MODELS <<< "${MODEL_ID}"
for raw_model in "${_MODELS[@]}"; do
  model_uri="${raw_model%/}"
  if [ -n "${model_uri}" ]; then
    model_name="$(basename "${model_uri}")"
    run_cmd "mkdir -p /tmp/${model_name}"
    run_cmd "gcloud storage cp -r ${model_uri}/config.json /tmp/${model_name}/"
  fi
done

# 4. On tpu, initialize TorchTPU distributed slice environment.
if [ "${DEVICE}" = "tpu" ]; then
  run_cmd "eval \$(python -m torch.tpu.distributed | sed 's/^/export /')"
fi

# 5. Export node environment and run leader (index 0) or worker (index > 0).
export GCSFS_SUBSYSTEM_DEVICE="${DEVICE}"
export GCSFS_SUBSYSTEM_NNODES="${NNODES}"
export GCSFS_SUBSYSTEM_NODE_RANK="${JOB_COMPLETION_INDEX}"
export GCSFS_SUBSYSTEM_MASTER_ADDR="${MASTER_ADDR}"
export GCSFS_SUBSYSTEM_MASTER_PORT="${MASTER_PORT}"
export GCSFS_SUBSYSTEM_PORT_BASE="${MASTER_PORT}"
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "export GCSFS_SUBSYSTEM_DEVICE=${DEVICE}"
  echo "export GCSFS_SUBSYSTEM_NNODES=${NNODES}"
  echo "export GCSFS_SUBSYSTEM_NODE_RANK=${JOB_COMPLETION_INDEX}"
  echo "export GCSFS_SUBSYSTEM_MASTER_ADDR=${MASTER_ADDR}"
  echo "export GCSFS_SUBSYSTEM_MASTER_PORT=${MASTER_PORT}"
fi

if [ "${JOB_COMPLETION_INDEX}" = "0" ]; then
  EXTRA_RUN_ARGS=""
  if [ -n "${CHECKPOINT_EXPORT_ROOT:-}" ]; then
    EXTRA_RUN_ARGS="${EXTRA_RUN_ARGS} --checkpoint-export-root=${CHECKPOINT_EXPORT_ROOT}"
  fi
  if [ -n "${CHECKPOINT_SOURCE_ROOT:-}" ]; then
    EXTRA_RUN_ARGS="${EXTRA_RUN_ARGS} --checkpoint-source-root=${CHECKPOINT_SOURCE_ROOT}"
  fi
  run_cmd "python -m gcsfs.tests.perf.subsystembenchmarks.run --device=${DEVICE} --nnodes=${NNODES} --node-rank=0 --master-addr=${MASTER_ADDR} ${GROUP_ARGS} --results-uri=${RESULTS_URI}${EXTRA_RUN_ARGS}"
else
  run_cmd "python -m gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.worker --device=${DEVICE} --nnodes=${NNODES} --node-rank=${JOB_COMPLETION_INDEX} --master-addr=${MASTER_ADDR} --master-port=${MASTER_PORT}"
fi
