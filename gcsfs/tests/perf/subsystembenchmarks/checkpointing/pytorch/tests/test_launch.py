from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import yaml

LAUNCH_DIR = Path(__file__).resolve().parents[1] / "launch"
ENTRYPOINT = LAUNCH_DIR / "entrypoint.sh"
SUBMIT = LAUNCH_DIR / "submit.sh"
A3ULTRA_TEMPLATE = LAUNCH_DIR / "jobset-a3ultra.yaml"
TPU7X_TEMPLATE = LAUNCH_DIR / "jobset-tpu7x.yaml"


def _render_with_envsubst(template_path: Path, extra_env: dict[str, str]) -> dict:
    env = os.environ.copy()
    env.update(extra_env)
    proc = subprocess.run(
        ["envsubst"],
        input=template_path.read_text(encoding="utf-8"),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    doc = yaml.safe_load(proc.stdout)
    assert isinstance(doc, dict)
    return doc


def test_templates_render():
    assert A3ULTRA_TEMPLATE.is_file(), f"Missing {A3ULTRA_TEMPLATE}"
    assert TPU7X_TEMPLATE.is_file(), f"Missing {TPU7X_TEMPLATE}"

    base_env = {
        "CODE_URI": "gs://test-bucket/code/gcsfs.tar.gz",
        "MODEL_ID": "gs://gcs-aiml-huggingface-model-weights/Llama-3.1-8B",
        "GROUP_ARGS": "--group=checkpointing/pytorch --project=test-proj --location=europe-west1 --bucket-prefix=bmk",
        "RESULTS_URI": "gs://test-bucket/results/run-01",
        "CHECKPOINT_EXPORT_ROOT": "gs://test-bucket/exports/run-01",
        "CHECKPOINT_SOURCE_ROOT": "",
    }

    for nnodes in (1, 2):
        a3_doc = _render_with_envsubst(
            A3ULTRA_TEMPLATE,
            {**base_env, "NNODES": str(nnodes)},
        )
        assert a3_doc["kind"] == "JobSet"
        rjob = a3_doc["spec"]["replicatedJobs"][0]
        job_spec = rjob["template"]["spec"]
        assert job_spec["parallelism"] == nnodes
        assert job_spec["completions"] == nnodes
        container = job_spec["template"]["spec"]["containers"][0]
        assert "nvidia" in container["image"].lower() and "pytorch" in container["image"].lower()
        assert container["resources"]["limits"]["nvidia.com/gpu"] == 8
        cmd_str = " ".join(container["command"])
        assert "google-cloud-cli" in cmd_str

    for topology, nnodes, placement_policy in (
        ("2x2x1", 1, ""),
        ("2x2x2", 2, "tpu7x-16-2x2x2-placement-policy"),
    ):
        tpu_doc = _render_with_envsubst(
            TPU7X_TEMPLATE,
            {
                **base_env,
                "NNODES": str(nnodes),
                "TOPOLOGY": topology,
                "PLACEMENT_POLICY": placement_policy,
            },
        )
        assert tpu_doc["kind"] == "JobSet"
        metadata = tpu_doc["metadata"]
        assert metadata["namespace"] == "default"
        assert len(metadata["name"]) <= 12
        assert metadata["labels"]["kueue.x-k8s.io/priority-class"] == "medium"

        rjob = tpu_doc["spec"]["replicatedJobs"][0]
        job_spec = rjob["template"]["spec"]
        assert job_spec["parallelism"] == nnodes
        assert job_spec["completions"] == nnodes

        pod_tmpl = job_spec["template"]
        pod_labels = pod_tmpl["metadata"]["labels"]
        assert "declared-duration-minutes" in pod_labels
        assert int(pod_labels["declared-duration-minutes"]) > 0

        node_selector = pod_tmpl["spec"]["nodeSelector"]
        assert node_selector["cloud.google.com/gke-tpu-accelerator"] == "tpu7x"
        assert node_selector["cloud.google.com/gke-tpu-topology"] == topology
        if topology == "2x2x2":
            assert (
                node_selector["cloud.google.com/placement-policy-name"]
                == "tpu7x-16-2x2x2-placement-policy"
            )

        container = pod_tmpl["spec"]["containers"][0]
        assert container["image"] == "python:3.14"
        assert container["resources"]["limits"]["google.com/tpu"] == 4
        cmd_str = " ".join(container["command"])
        assert "google-cloud-cli" in cmd_str


def test_entrypoint_branches():
    assert ENTRYPOINT.is_file(), f"Missing {ENTRYPOINT}"

    common_env = {
        **os.environ,
        "DRY_RUN": "1",
        "CODE_URI": "gs://test-bucket/code/gcsfs.tar.gz",
        "MODEL_ID": "gs://gcs-aiml-huggingface-model-weights/Llama-3.1-8B",
        "GROUP_ARGS": "--group=checkpointing/pytorch --project=test-proj --location=us-central1 --bucket-prefix=bmk",
        "RESULTS_URI": "gs://test-bucket/results/run-01",
        "NNODES": "2",
        "MASTER_ADDR": "dcp-a3ultra-worker-0-0.dcp-a3ultra",
        "MASTER_PORT": "29500",
        "CHECKPOINT_EXPORT_ROOT": "",
        "CHECKPOINT_SOURCE_ROOT": "",
    }

    # Branch 1: Leader (JOB_COMPLETION_INDEX=0) on CUDA
    leader_proc = subprocess.run(
        ["bash", str(ENTRYPOINT)],
        env={
            **common_env,
            "DEVICE": "cuda",
            "JOB_COMPLETION_INDEX": "0",
            "LD_LIBRARY_PATH": "/usr/local/lib/python3.12/dist-packages/torch/lib",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    leader_out = leader_proc.stdout
    assert "gcloud storage cp gs://test-bucket/code/gcsfs.tar.gz" in leader_out
    assert "requirements-cuda.txt" in leader_out
    assert "SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 pip install -e ." in leader_out
    assert (
        "gcloud storage cp -r gs://gcs-aiml-huggingface-model-weights/Llama-3.1-8B/config.json /tmp/Llama-3.1-8B/"
        in leader_out
    )
    assert "torch.tpu.distributed" not in leader_out
    assert (
        "python -m gcsfs.tests.perf.subsystembenchmarks.run "
        "--device=cuda --nnodes=2 --node-rank=0 "
        "--master-addr=dcp-a3ultra-worker-0-0.dcp-a3ultra "
        "--master-port=29500 "
        "--group=checkpointing/pytorch --project=test-proj --location=us-central1 --bucket-prefix=bmk "
        "--results-uri=gs://test-bucket/results/run-01"
    ) in leader_out
    assert "checkpointing.pytorch.worker" not in leader_out

    # Branch 2: Worker (JOB_COMPLETION_INDEX=1) on TPU with multiple models
    worker_proc = subprocess.run(
        ["bash", str(ENTRYPOINT)],
        env={
            **common_env,
            "DEVICE": "tpu",
            "JOB_COMPLETION_INDEX": "1",
            "MASTER_ADDR": "dcp-tpu7x-slice-job-0-0.dcp-tpu7x",
            "MODEL_ID": (
                "gs://gcs-aiml-huggingface-model-weights/Llama-3.1-8B,"
                "gs://gcs-aiml-huggingface-model-weights/OLMoE-1B-7B-0924"
            ),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    worker_out = worker_proc.stdout
    assert "requirements-tpu.txt" in worker_out
    assert (
        "gcloud storage cp -r gs://gcs-aiml-huggingface-model-weights/Llama-3.1-8B/config.json /tmp/Llama-3.1-8B/"
        in worker_out
    )
    assert (
        "gcloud storage cp -r gs://gcs-aiml-huggingface-model-weights/OLMoE-1B-7B-0924/config.json /tmp/OLMoE-1B-7B-0924/"
        in worker_out
    )
    assert (
        "eval $(python -m torch.tpu.distributed | grep -E '^TORCH_TPU_[A-Z0-9_]+=' | sed 's/^/export /')"
        in worker_out
    )
    assert "export GCSFS_SUBSYSTEM_DEVICE=tpu" in worker_out
    assert "export GCSFS_SUBSYSTEM_NNODES=2" in worker_out
    assert "export GCSFS_SUBSYSTEM_NODE_RANK=1" in worker_out
    assert "export GCSFS_SUBSYSTEM_MASTER_ADDR=dcp-tpu7x-slice-job-0-0.dcp-tpu7x" in worker_out
    assert (
        "python -m gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.worker"
        in worker_out
    )
    assert "gcsfs.tests.perf.subsystembenchmarks.run" not in worker_out

    # Branch 3: CPU requirements.txt and handoff env flags on leader
    cpu_proc = subprocess.run(
        ["bash", str(ENTRYPOINT)],
        env={
            **common_env,
            "DEVICE": "cpu",
            "JOB_COMPLETION_INDEX": "0",
            "CHECKPOINT_EXPORT_ROOT": "gs://test-bucket/exports",
            "CHECKPOINT_SOURCE_ROOT": "gs://test-bucket/sources",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    cpu_out = cpu_proc.stdout
    assert "checkpointing/pytorch/requirements.txt" in cpu_out
    assert "--checkpoint-export-root=gs://test-bucket/exports" in cpu_out
    assert "--checkpoint-source-root=gs://test-bucket/sources" in cpu_out


def test_submit_script_dry_run():
    assert SUBMIT.is_file(), f"Missing {SUBMIT}"

    for target, size_arg in (("a3ultra", "2"), ("tpu7x", "2x2x1"), ("tpu7x", "2x2x2")):
        proc = subprocess.run(
            [
                "bash",
                str(SUBMIT),
                target,
                size_arg,
                "--",
                "--project=test-proj",
                "--location=us-central1",
                "--bucket-prefix=bmk",
                "--sweep-axes=strategy model reshard",
                "--filter=save or load",
            ],
            env={
                **os.environ,
                "DRY_RUN": "1",
                "CODE_URI": "gs://test-bucket/code.tar.gz",
                "RESULTS_URI": "gs://test-bucket/results/run-01",
            },
            capture_output=True,
            text=True,
            check=True,
        )
        out = proc.stdout
        assert "--exclude=.worktrees" in out
        assert "--exclude=.superpowers" in out
        assert "--exclude=docs/superpowers" in out
        tar_line = [
            line for line in out.splitlines() if line.startswith("tar ")
        ][0]
        tar_tokens = shlex.split(tar_line)
        repo_root = Path(tar_tokens[tar_tokens.index("-C") + 1])
        assert (repo_root / "pyproject.toml").is_file()
        assert (
            repo_root
            / "gcsfs/tests/perf/subsystembenchmarks/checkpointing/pytorch/launch/entrypoint.sh"
        ).is_file()
        assert "kubectl apply" in out
        assert "kubectl logs" in out

        # Extract rendered YAML between gcloud upload and kubectl apply
        yaml_start = out.index("apiVersion: jobset.x-k8s.io/v1alpha2")
        yaml_end = out.index("kubectl apply -f -")
        rendered_doc = yaml.safe_load(out[yaml_start:yaml_end])
        container_env = {
            item["name"]: item["value"]
            for item in rendered_doc["spec"]["replicatedJobs"][0]["template"]["spec"][
                "template"
            ]["spec"]["containers"][0]["env"]
        }
        group_args_val = container_env["GROUP_ARGS"]
        assert "'--sweep-axes=strategy model reshard'" in group_args_val
        assert "'--filter=save or load'" in group_args_val

        # Verify entrypoint.sh preserves the multi-word arguments when passed GROUP_ARGS
        ep_proc = subprocess.run(
            ["bash", str(ENTRYPOINT)],
            env={
                **os.environ,
                "DRY_RUN": "1",
                "DEVICE": "cuda",
                "NNODES": "2",
                "JOB_COMPLETION_INDEX": "0",
                "MASTER_ADDR": "dcp-a3ultra-worker-0-0.dcp-a3ultra",
                "CODE_URI": "gs://test-bucket/code.tar.gz",
                "MODEL_ID": "gs://gcs-aiml-huggingface-model-weights/Llama-3.1-8B",
                "GROUP_ARGS": group_args_val,
                "RESULTS_URI": "gs://test-bucket/results/run-01",
            },
            capture_output=True,
            text=True,
            check=True,
        )
        run_line = [
            line
            for line in ep_proc.stdout.splitlines()
            if "gcsfs.tests.perf.subsystembenchmarks.run" in line
        ][0]
        parsed_tokens = shlex.split(run_line)
        assert "--sweep-axes=strategy model reshard" in parsed_tokens
        assert "--filter=save or load" in parsed_tokens
