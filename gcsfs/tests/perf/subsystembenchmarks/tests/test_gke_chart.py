import os
import re
import shutil
import subprocess
import sys

import pytest
import yaml

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
_CHART_DIR = os.path.join(_REPO_ROOT, "cloudbuild", "subsystembenchmarks", "helm_chart")
_RUN_GKE = os.path.join(
    _REPO_ROOT, "cloudbuild", "subsystembenchmarks", "scripts", "run-gke.sh"
)

_HAS_HELM = shutil.which("helm") is not None


def _render_chart(release_name: str, *extra_args: str) -> subprocess.CompletedProcess:
    cmd = ["helm", "template", release_name, _CHART_DIR, *extra_args]
    return subprocess.run(cmd, capture_output=True, text=True)


def _docs_by_kind(rendered_yaml: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for doc in yaml.safe_load_all(rendered_yaml):
        if not doc:
            continue
        out.setdefault(doc["kind"], []).append(doc)
    return out


@pytest.mark.skipif(not _HAS_HELM, reason="helm CLI not installed")
def test_helm_lint_passes():
    proc = subprocess.run(["helm", "lint", _CHART_DIR], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "0 chart(s) failed" in proc.stdout


@pytest.mark.skipif(not _HAS_HELM, reason="helm CLI not installed")
def test_chart_gpu_one_node():
    proc = _render_chart(
        "ssb-save-1n-abcd",
        "--set",
        "accelerator=gpu",
        "--set",
        "nodes=1",
        "--set",
        "devicesPerNode=8",
        "--set",
        "group=checkpointing/pytorch_save",
        "--set",
        "bucketPrefix=ssb-test",
        "--set",
        "project=my-project",
        "--set",
        "location=europe-west1",
        "--set",
        "zone=europe-west1-b",
        "--set",
        "resultsUri=gs://results-bkt/ssb",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    docs = _docs_by_kind(proc.stdout)

    assert len(docs["JobSet"]) == 1
    assert len(docs["Service"]) == 1
    assert len(docs["ConfigMap"]) == 1

    svc = docs["Service"][0]
    assert svc["metadata"]["name"] == "ssb-save-1n-abcd"
    assert svc["spec"]["clusterIP"] == "None"

    js = docs["JobSet"][0]
    job_spec = js["spec"]["replicatedJobs"][0]["template"]["spec"]
    assert job_spec["parallelism"] == 1
    assert job_spec["completions"] == 1
    assert job_spec["completionMode"] == "Indexed"

    pod_spec = job_spec["template"]["spec"]
    assert pod_spec.get("hostNetwork") is not True
    assert any(
        v.get("emptyDir", {}).get("medium") == "Memory"
        for v in pod_spec.get("volumes", [])
    )
    assert any(
        t.get("key") == "nvidia.com/gpu" for t in pod_spec.get("tolerations", [])
    )

    container = pod_spec["containers"][0]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 8
    assert container["resources"]["requests"]["nvidia.com/gpu"] == 8
    env = {e["name"]: e for e in container["env"]}
    assert env["GCSFS_SUBSYSTEM_NODES"]["value"] == "1"
    assert env["GCSFS_SUBSYSTEM_DEVICES_PER_NODE"]["value"] == "8"
    assert "GCSFS_SUBSYSTEM_NODE_RANK" in env
    assert env["GCSFS_SUBSYSTEM_LEADER_ADDR"]["value"].startswith(
        "ssb-save-1n-abcd-workload-0-0.ssb-save-1n-abcd."
    )


@pytest.mark.skipif(not _HAS_HELM, reason="helm CLI not installed")
def test_chart_tpu_two_nodes():
    proc = _render_chart(
        "ssb-save-2n-ef01",
        "--set",
        "accelerator=tpu",
        "--set",
        "nodes=2",
        "--set",
        "devicesPerNode=8",
        "--set",
        "group=checkpointing/pytorch_save",
        "--set",
        "bucketPrefix=ssb-test",
        "--set",
        "project=my-project",
        "--set",
        "location=us-central1",
        "--set",
        "zone=us-central1-c",
        "--set",
        "resultsUri=gs://results-bkt/ssb",
        "--set",
        "tpu.multiNodePlacementPolicy=my-placement-policy",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    docs = _docs_by_kind(proc.stdout)
    js = docs["JobSet"][0]
    assert js["metadata"]["labels"]["kueue.x-k8s.io/priority-class"] == "low"

    job_spec = js["spec"]["replicatedJobs"][0]["template"]["spec"]
    assert job_spec["parallelism"] == 2
    assert job_spec["completions"] == 2
    assert job_spec["completionMode"] == "Indexed"

    pod_meta = job_spec["template"]["metadata"]
    assert pod_meta["labels"]["declared-duration-minutes"] == "180"

    pod_spec = job_spec["template"]["spec"]
    ns = pod_spec["nodeSelector"]
    assert ns["cloud.google.com/gke-tpu-accelerator"] == "tpu7x"
    assert ns["cloud.google.com/gke-tpu-topology"] == "2x2x2"
    assert ns["cloud.google.com/placement-policy-name"] == "my-placement-policy"
    assert any(
        t.get("key") == "google.com/tpu" for t in pod_spec.get("tolerations", [])
    )

    container = pod_spec["containers"][0]
    assert container["resources"]["limits"]["google.com/tpu"] == 4
    assert container["resources"]["requests"]["google.com/tpu"] == 4
    env = {e["name"]: e["value"] for e in container["env"] if "value" in e}
    assert env["SSB_TPU_TOPOLOGY"] == "2x2x2"


@pytest.mark.skipif(not _HAS_HELM, reason="helm CLI not installed")
def test_chart_tpu_rejects_long_release_name():
    long_name = "a" * 21
    proc = _render_chart(long_name, "--set", "accelerator=tpu", "--set", "nodes=1")
    assert proc.returncode != 0
    assert "20" in (proc.stdout + proc.stderr)


_BASE_GPU_ARGS = [
    "--accelerator",
    "gpu",
    "--kube-context",
    "gke_my-project_europe-west1_my-gpu-cluster",
    "--project",
    "my-project",
    "--location",
    "europe-west1",
    "--zone",
    "europe-west1-b",
]
_BASE_TPU_ARGS = [
    "--accelerator",
    "tpu",
    "--kube-context",
    "gke_my-tpu-project_us-central1_my-tpu-cluster",
    "--project",
    "my-bucket-project",
    "--location",
    "us-central1",
    "--zone",
    "us-central1-c",
]


def test_run_gke_dry_run_one_release_per_node_count(tmp_path):
    assert os.path.isfile(_RUN_GKE), f"missing {_RUN_GKE}"
    subprocess.run(["bash", "-n", _RUN_GKE], check=True)

    # Stub python binary that returns [1, 2] for --list-node-counts.
    stub_py = tmp_path / "stub_python.sh"
    stub_py.write_text("#!/usr/bin/env bash\necho '[1, 2]'\n")
    stub_py.chmod(0o755)

    env = dict(os.environ, SSB_PYTHON=str(stub_py))
    proc = subprocess.run(
        [
            "bash",
            _RUN_GKE,
            *_BASE_TPU_ARGS,
            "--tpu-placement-policy",
            "my-placement-policy",
            "--group",
            "checkpointing/pytorch_save",
            "--results-uri",
            "gs://test-results/ssb",
            "--commit",
            "deadbeef",
            "--dry-run",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    install_lines = [
        line.strip() for line in proc.stdout.splitlines() if "helm install" in line
    ]
    assert len(install_lines) == 2, proc.stdout
    for line, expected_nodes in zip(install_lines, ("1", "2")):
        match = re.search(r"helm install\s+(\S+)", line)
        assert match is not None, line
        release = match.group(1)
        assert len(release) <= 20, f"release name too long: {release!r}"
        assert f"-{expected_nodes}n-" in release, release
        assert f"nodes={expected_nodes}" in line
        assert "accelerator=tpu" in line


@pytest.mark.skipif(not _HAS_HELM, reason="helm CLI not installed")
def test_chart_optional_kueue_queue_label():
    proc = _render_chart(
        "ssb-wds-1n-1234",
        "--set",
        "accelerator=gpu",
        "--set",
        "nodes=1",
        "--set",
        "kueueQueue=default",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    docs = _docs_by_kind(proc.stdout)
    js = docs["JobSet"][0]
    assert js["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] == "default"


def test_run_gke_dry_run_accepts_kube_context_and_queue_name(tmp_path):
    stub_py = tmp_path / "stub_python.sh"
    stub_py.write_text("#!/usr/bin/env bash\necho '[1]'\n")
    stub_py.chmod(0o755)

    env = dict(os.environ, SSB_PYTHON=str(stub_py))
    proc = subprocess.run(
        [
            "bash",
            _RUN_GKE,
            *_BASE_GPU_ARGS,
            "--queue-name",
            "default",
            "--group",
            "dataloading/webdataset",
            "--results-uri",
            "gs://test-results/ssb",
            "--commit",
            "deadbeef",
            "--dry-run",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--kube-context gke_my-project_europe-west1_my-gpu-cluster" in proc.stdout
    assert "kueueQueue=default" in proc.stdout


def test_run_gke_resolves_repo_root_from_other_cwd(tmp_path):
    # Create a fake cwd containing a broken gcsfs/ package that would shadow REPO_ROOT
    # if run-gke.sh ran python -m from the caller's cwd instead of REPO_ROOT.
    other_cwd = tmp_path / "other"
    (other_cwd / "gcsfs").mkdir(parents=True)
    (other_cwd / "gcsfs" / "__init__.py").write_text(
        "raise RuntimeError('imported shadowed gcsfs from caller cwd')\n"
    )
    proc = subprocess.run(
        [
            "bash",
            _RUN_GKE,
            *_BASE_GPU_ARGS,
            "--group",
            "dataloading/webdataset",
            "--results-uri",
            "gs://test-results/ssb",
            "--commit",
            "deadbeef",
            "--dry-run",
        ],
        cwd=str(other_cwd),
        env=dict(os.environ, SSB_PYTHON=sys.executable),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "helm install" in proc.stdout


def test_run_gke_fetches_logs_by_jobset_label_selector(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    kubectl_log = tmp_path / "kubectl.log"
    rel_file = tmp_path / "release.txt"
    (bin_dir / "helm").write_text(
        f"#!/usr/bin/env bash\n"
        f'if [[ "$1" == "install" ]]; then echo "$2" > "{rel_file}"; fi\n'
        f"exit 0\n"
    )
    (bin_dir / "helm").chmod(0o755)
    (bin_dir / "kubectl").write_text(
        f"#!/usr/bin/env bash\n"
        f'echo "$@" >> "{kubectl_log}"\n'
        f'if [[ "$*" == *"get jobset"* ]]; then\n'
        f'  printf "%s\\tCompleted=True \\n" "$(cat "{rel_file}")"\n'
        f"fi\n"
        f"exit 0\n"
    )
    (bin_dir / "kubectl").chmod(0o755)
    stub_py = tmp_path / "stub_python.sh"
    stub_py.write_text("#!/usr/bin/env bash\necho '[1]'\n")
    stub_py.chmod(0o755)

    env = dict(
        os.environ,
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        SSB_PYTHON=str(stub_py),
        SSB_POLL_SECONDS="0",
    )
    subprocess.run(
        [
            "bash",
            _RUN_GKE,
            *_BASE_GPU_ARGS,
            "--group",
            "dataloading/webdataset",
            "--results-uri",
            "gs://test-results/ssb",
            "--commit",
            "deadbeef",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    log_calls = [
        line for line in kubectl_log.read_text().splitlines() if " logs " in line
    ]
    assert len(log_calls) == 1, log_calls
    assert "jobset.sigs.k8s.io/jobset-name=" in log_calls[0], log_calls[0]
    assert "batch.kubernetes.io/job-completion-index=0" in log_calls[0], log_calls[0]


def test_launcher_collects_tpu_slice_env_with_autoload_and_banner(tmp_path):
    launcher = os.path.join(_CHART_DIR, "launcher.sh")
    subprocess.run(["bash", "-n", launcher], check=True)
    launcher_text = open(launcher).read()
    assert (
        "TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m torch.tpu.distributed"
        not in launcher_text
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_py = bin_dir / "python3"
    fake_py.write_text(
        f"""#!/usr/bin/env bash
if [[ "$1" == "-m" && "$2" == "torch.tpu.distributed" ]]; then
  if [[ "${{TORCH_DEVICE_BACKEND_AUTOLOAD:-}}" == "0" ]]; then
    echo "ModuleNotFoundError: No module named 'torch.tpu'" >&2
    exit 1
  fi
  cat <<'OUT'
Successfully renamed PrivateUse1 backend to 'tpu'. Device: device(type='tpu')
Registered Python module for 'tpu'.
TORCH_TPU_TOPOLOGY="2,2,2,2"
TORCH_TPU_SLICEBUILDER_ADDRESSES="10.0.0.1:8476,10.0.0.2:8476"
NODE_RANK="1"
OUT
  exit 0
fi
exec "{sys.executable}" "$@"
"""
    )
    fake_py.chmod(0o755)

    env = dict(
        os.environ,
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        GCSFS_SUBSYSTEM_ACCELERATOR="tpu",
        GCSFS_SUBSYSTEM_NODES="2",
    )
    proc = subprocess.run(
        [
            "bash",
            "-c",
            (
                f'source "{launcher}" && collect_tpu_slice_env && '
                'printf "TOPO=%s\\nADDRS=%s\\nRAW_TOPO=%s\\n" '
                '"${GCSFS_SUBSYSTEM_TPU_SLICE_TOPOLOGY:-}" '
                '"${GCSFS_SUBSYSTEM_TPU_SLICE_ADDRESSES:-}" '
                '"${TORCH_TPU_TOPOLOGY:-}"'
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "TOPO=2,2,2,2" in proc.stdout, proc.stdout
    assert "ADDRS=10.0.0.1:8476,10.0.0.2:8476" in proc.stdout, proc.stdout
    assert "RAW_TOPO=\n" in proc.stdout, proc.stdout


def test_launcher_multihost_tpu_passes_full_slice_topology_and_node_rank(tmp_path):
    launcher = os.path.join(_CHART_DIR, "launcher.sh")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_py = bin_dir / "python3"
    # Reproduces torch.tpu.distributed: if TORCH_TPU_TOPOLOGY is unset it falls back
    # to the single-host topology ("2,2,1,2") even while emitting 16 slicebuilder addrs.
    fake_py.write_text(
        f"""#!/usr/bin/env bash
if [[ "$1" == "-m" && "$2" == "torch.tpu.distributed" ]]; then
  topo="${{TORCH_TPU_TOPOLOGY:-2,2,1,2}}"
  nrank="${{NODE_RANK:-unset}}"
  echo "TORCH_TPU_TOPOLOGY=\\"${{topo}}\\""
  echo "TORCH_TPU_SLICEBUILDER_ADDRESSES=\\"h0:8070,h0:8071,h1:8070,h1:8071\\""
  echo "NODE_RANK=\\"${{nrank}}\\""
  exit 0
fi
exec "{sys.executable}" "$@"
"""
    )
    fake_py.chmod(0o755)

    env = dict(
        os.environ,
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        GCSFS_SUBSYSTEM_ACCELERATOR="tpu",
        GCSFS_SUBSYSTEM_NODES="2",
        GCSFS_SUBSYSTEM_DEVICES_PER_NODE="8",
        GCSFS_SUBSYSTEM_NODE_RANK="1",
        SSB_TPU_TOPOLOGY="2x2x2",
    )
    proc = subprocess.run(
        [
            "bash",
            "-c",
            (
                f'source "{launcher}" && collect_tpu_slice_env && '
                'printf "TOPO=%s\\nADDRS=%s\\n" '
                '"${GCSFS_SUBSYSTEM_TPU_SLICE_TOPOLOGY:-}" '
                '"${GCSFS_SUBSYSTEM_TPU_SLICE_ADDRESSES:-}"'
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "TOPO=2,2,2,2" in proc.stdout, proc.stdout
    assert "tpu_node_rank=1" in proc.stdout, proc.stdout


def test_launcher_uploads_when_results_csv_exists_even_on_failure():
    launcher_text = open(os.path.join(_CHART_DIR, "launcher.sh")).read()
    assert '[[ "$RUN_RC" -eq 0 ]]' not in launcher_text
    assert "[[ -f /results/results.csv ]]" in launcher_text


def test_results_dest_uri_matches_ingest_sql_regexes():
    launcher = os.path.join(_CHART_DIR, "launcher.sh")
    run_uuid = "0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0"
    proc = subprocess.run(
        [
            "bash",
            "-c",
            (
                f'source "{launcher}" && '
                f'results_dest_uri "gs://p-run-results/subsystembenchmarks" '
                f'"feat-x" "{run_uuid}" "ssb-save-1n-abcd" "20261126" "20261126-101500"'
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    dest_csv = proc.stdout.strip() + "/results.csv"
    expected_csv = (
        f"gs://p-run-results/subsystembenchmarks/branch=feat-x/20261126/"
        f"{run_uuid}/ssb-save-1n-abcd/20261126-101500/results.csv"
    )
    assert dest_csv == expected_csv

    ingest_sql = open(
        os.path.join(_REPO_ROOT, "cloudbuild", "subsystembenchmarks", "ingest.sql")
    ).read()
    raw_patterns = re.findall(r"REGEXP_EXTRACT\(_FILE_NAME,\s*r'([^']+)'\)", ingest_sql)
    assert len(raw_patterns) == 4, raw_patterns
    extracted = []
    for raw in raw_patterns:
        pattern = raw.replace("\\\\", "\\")
        m = re.search(pattern, dest_csv)
        assert m is not None, f"pattern {pattern!r} failed to match {dest_csv!r}"
        extracted.append(m.group(1))
    assert extracted == ["20261126", run_uuid, "20261126-101500", "feat-x"]


@pytest.mark.skipif(not _HAS_HELM, reason="helm CLI not installed")
def test_chart_tpu_two_nodes_requires_placement_policy():
    proc = _render_chart(
        "ssb-save-2n-ef01",
        "--set",
        "accelerator=tpu",
        "--set",
        "nodes=2",
    )
    assert proc.returncode != 0
    assert "tpu.multiNodePlacementPolicy" in (proc.stdout + proc.stderr)


@pytest.mark.skipif(not _HAS_HELM, reason="helm CLI not installed")
def test_chart_labels_zone_pin_and_new_env():
    proc = _render_chart(
        "ssb-wds-1n-1234",
        "--set",
        "accelerator=gpu",
        "--set",
        "nodes=1",
        "--set",
        "zone=us-central1-a",
        "--set",
        "pinZone=true",
        "--set-string",
        "buildId=build-1234",
        "--set-string",
        "infraPrefix=my-prefix",
        "--set-string",
        "cases=case-a case-b",
        "--set-string",
        "requirements=gcsfs==2026.9.0",
        "--set-string",
        "resultsBranch=main",
        "--set-string",
        "runId=0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0",
        "--set",
        "requireAmplification=true",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    docs = _docs_by_kind(proc.stdout)
    js = docs["JobSet"][0]
    assert js["metadata"]["labels"]["gcsfs-ssb/build-id"] == "build-1234"
    assert js["metadata"]["labels"]["gcsfs-ssb/infra-prefix"] == "my-prefix"

    pod_tmpl = js["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]
    assert pod_tmpl["metadata"]["labels"]["gcsfs-ssb/build-id"] == "build-1234"
    assert pod_tmpl["metadata"]["labels"]["gcsfs-ssb/infra-prefix"] == "my-prefix"
    assert (
        pod_tmpl["spec"]["nodeSelector"]["topology.kubernetes.io/zone"]
        == "us-central1-a"
    )

    env = {
        e["name"]: e["value"]
        for e in pod_tmpl["spec"]["containers"][0]["env"]
        if "value" in e
    }
    assert env["SSB_CASES"] == "case-a case-b"
    assert env["SSB_REQUIREMENTS"] == "gcsfs==2026.9.0"
    assert env["SSB_RESULTS_BRANCH"] == "main"
    assert env["SSB_RUN_ID"] == "0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0"
    assert env["SSB_REQUIRE_AMPLIFICATION"] == "1"
    assert env["SSB_MACHINE_TYPE_DEFAULT"] == "a3-ultragpu-8g"


def test_launcher_resolve_machine_type_falls_back_to_default(tmp_path):
    launcher = os.path.join(_CHART_DIR, "launcher.sh")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_py = bin_dir / "python3"
    fake_py.write_text("#!/usr/bin/env bash\nexit 1\n")
    fake_py.chmod(0o755)
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{launcher}" && resolve_machine_type && echo "MT=$MACHINE_TYPE"',
        ],
        env=dict(
            os.environ,
            PATH=f"{bin_dir}:{os.environ['PATH']}",
            SSB_MACHINE_TYPE_DEFAULT="tpu7x-standard-4t",
        ),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "MT=tpu7x-standard-4t" in proc.stdout, proc.stdout


def test_launcher_requirements_override_exports_normalized_override(tmp_path):
    install_group = os.path.join(
        _REPO_ROOT, "cloudbuild", "subsystembenchmarks", "scripts", "install-group.sh"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pip_log = tmp_path / "pip.log"
    fake_pip = bin_dir / "pip"
    fake_pip.write_text(f'#!/usr/bin/env bash\necho "$@" >> "{pip_log}"\n')
    fake_pip.chmod(0o755)
    proc = subprocess.run(
        [
            "bash",
            "-c",
            (
                f'source "{install_group}" && '
                'TORCH_CONSTRAINT="" && '
                'install_requirements_override "  a==1   b==2  " && '
                'echo "OVR=$GCSFS_SUBSYSTEM_REQUIREMENTS_OVERRIDE"'
            ),
        ],
        env=dict(
            os.environ,
            PATH=f"{bin_dir}:{os.environ['PATH']}",
        ),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OVR=a==1 b==2" in proc.stdout, proc.stdout
    calls = pip_log.read_text().splitlines()
    assert len(calls) == 2, calls
    assert calls[0] == "install -- a==1 b==2"
    assert calls[1] == "install --no-deps --force-reinstall -- a==1 b==2"


def test_launcher_checkout_handles_short_sha_and_branch(tmp_path):
    launcher = os.path.join(_CHART_DIR, "launcher.sh")
    bare = tmp_path / "remote.git"
    work = tmp_path / "work"
    subprocess.run(
        ["git", "init", "--bare", "-b", "main", str(bare)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "init", "-b", "main", str(work)], check=True, capture_output=True
    )
    git_env = dict(
        os.environ,
        GIT_AUTHOR_NAME="Test",
        GIT_AUTHOR_EMAIL="test@example.com",
        GIT_COMMITTER_NAME="Test",
        GIT_COMMITTER_EMAIL="test@example.com",
    )
    (work / "file.txt").write_text("v1\n")
    subprocess.run(["git", "-C", str(work), "add", "file.txt"], check=True, env=git_env)
    subprocess.run(
        ["git", "-C", str(work), "commit", "-m", "v1"],
        check=True,
        env=git_env,
        capture_output=True,
    )
    sha1 = subprocess.run(
        ["git", "-C", str(work), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (work / "file.txt").write_text("v2\n")
    subprocess.run(["git", "-C", str(work), "add", "file.txt"], check=True, env=git_env)
    subprocess.run(
        ["git", "-C", str(work), "commit", "-m", "v2"],
        check=True,
        env=git_env,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(work), "push", str(bare), "HEAD:refs/heads/main"],
        check=True,
        env=git_env,
        capture_output=True,
    )

    dest = tmp_path / "checkout"
    short_sha1 = sha1[:7]
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{launcher}" && checkout_gcsfs_repo "{bare}" "{short_sha1}" "{dest}"',
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    checked_out = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert checked_out == sha1, f"expected {sha1}, got {checked_out}"


@pytest.mark.skipif(not _HAS_HELM, reason="helm CLI not installed")
def test_run_gke_escapes_helm_set_string_and_checks_missing_arg(tmp_path):
    import shlex

    stub_py = tmp_path / "stub_python.sh"
    stub_py.write_text("#!/usr/bin/env bash\necho '[1]'\n")
    stub_py.chmod(0o755)
    env = dict(os.environ, SSB_PYTHON=str(stub_py))

    # Missing option value must exit non-zero cleanly rather than unbound $2.
    bad = subprocess.run(
        [
            "bash",
            _RUN_GKE,
            *_BASE_GPU_ARGS,
            "--group",
            "dataloading/webdataset",
            "--results-uri",
            "gs://b/r",
            "--filter",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 2
    assert "requires a value" in bad.stderr

    # Comma in --filter must not split into an injected Helm value override.
    injected_filter = "a,gcsfsRepo=https://evil.example/repo.git"
    proc = subprocess.run(
        [
            "bash",
            _RUN_GKE,
            *_BASE_GPU_ARGS,
            "--group",
            "dataloading/webdataset",
            "--results-uri",
            "gs://test-results/ssb",
            "--repo",
            "https://github.com/fsspec/gcsfs.git",
            "--commit",
            "deadbeef",
            "--filter",
            injected_filter,
            "--dry-run",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    cmd_line = next(l for l in proc.stdout.splitlines() if "helm install" in l)
    tokens = shlex.split(cmd_line)
    # Turn `helm install <release> <chart> ...` into `helm template <release> <chart> ...`
    assert tokens[:2] == ["helm", "install"]
    rest = list(tokens[2:])
    ctx_idx = rest.index("--kube-context")
    del rest[ctx_idx : ctx_idx + 2]
    template_cmd = ["helm", "template", *rest]
    rendered = subprocess.run(template_cmd, capture_output=True, text=True)
    assert rendered.returncode == 0, rendered.stdout + rendered.stderr
    docs = _docs_by_kind(rendered.stdout)
    pod_spec = docs["JobSet"][0]["spec"]["replicatedJobs"][0]["template"]["spec"][
        "template"
    ]["spec"]
    c_env = {
        e["name"]: e["value"] for e in pod_spec["containers"][0]["env"] if "value" in e
    }
    assert c_env["SSB_FILTER"] == injected_filter
    assert c_env["GCSFS_REPO"] == "https://github.com/fsspec/gcsfs.git"


def test_run_gke_list_node_counts_passes_bucket_type(tmp_path):
    args_log = tmp_path / "list_args.txt"
    stub_py = tmp_path / "stub_python.sh"
    stub_py.write_text(f'#!/usr/bin/env bash\necho "$@" > "{args_log}"\necho "[1]"\n')
    stub_py.chmod(0o755)
    env = dict(os.environ, SSB_PYTHON=str(stub_py))
    subprocess.run(
        [
            "bash",
            _RUN_GKE,
            *_BASE_GPU_ARGS,
            "--group",
            "dataloading/webdataset",
            "--bucket-type",
            "zonal",
            "--results-uri",
            "gs://test-results/ssb",
            "--commit",
            "deadbeef",
            "--dry-run",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    logged = args_log.read_text()
    assert "--bucket-type zonal" in logged, logged


def test_launcher_exports_commit_sha_and_requirements_resolved():
    launcher_text = open(os.path.join(_CHART_DIR, "launcher.sh")).read()
    assert "export COMMIT_SHA=" in launcher_text
    assert "export GCSFS_SUBSYSTEM_REQUIREMENTS_RESOLVED=" in launcher_text


def test_chart_defaults_to_upstream_gcsfs_repo():
    values = yaml.safe_load(open(os.path.join(_CHART_DIR, "values.yaml")))
    assert values["gcsfsRepo"] == "https://github.com/fsspec/gcsfs.git"
    run_gke_text = open(_RUN_GKE).read()
    assert "zhixiangli/gcsfs" not in run_gke_text


def test_run_gke_requires_cluster_params():
    proc = subprocess.run(
        [
            "bash",
            _RUN_GKE,
            "--accelerator",
            "gpu",
            "--project",
            "my-project",
            "--location",
            "europe-west1",
            "--zone",
            "europe-west1-b",
            "--group",
            "dataloading/webdataset",
            "--results-uri",
            "gs://test-results/ssb",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "--kube-context is required" in proc.stderr


def test_run_gke_dry_run_batches_by_exact_case_ids(tmp_path):
    import json
    import shlex

    cases = [
        {"id": "c1", "nodes": 1},
        {"id": "c2", "nodes": 1},
        {"id": "c3", "nodes": 1},
        {"id": "c4", "nodes": 1},
        {"id": "c5", "nodes": 1},
        {"id": "c6", "nodes": 2},
    ]
    stub_py = tmp_path / "stub_python.sh"
    stub_py.write_text(f"#!/usr/bin/env bash\necho '{json.dumps(cases)}'\n")
    stub_py.chmod(0o755)
    proc = subprocess.run(
        [
            "bash",
            _RUN_GKE,
            *_BASE_TPU_ARGS,
            "--tpu-placement-policy",
            "my-placement-policy",
            "--group",
            "checkpointing/pytorch_save",
            "--results-uri",
            "gs://test-results/ssb",
            "--commit",
            "deadbeef",
            "--max-cases-per-release",
            "2",
            "--dry-run",
        ],
        env=dict(os.environ, SSB_PYTHON=str(stub_py)),
        capture_output=True,
        text=True,
        check=True,
    )
    install_lines = [l for l in proc.stdout.splitlines() if "helm install" in l]
    assert len(install_lines) == 4, install_lines
    seen_chunks = []
    for line in install_lines:
        tokens = shlex.split(line)
        kv = {}
        for i, tok in enumerate(tokens):
            if tok in ("--set", "--set-string") and i + 1 < len(tokens):
                k, v = tokens[i + 1].split("=", 1)
                kv[k] = v
        seen_chunks.append((kv["nodes"], kv["cases"].split()))
    assert seen_chunks == [
        ("1", ["c1", "c2"]),
        ("1", ["c3", "c4"]),
        ("1", ["c5"]),
        ("2", ["c6"]),
    ]


def test_run_gke_dry_run_sets_labels_layout_and_pin(tmp_path):
    stub_py = tmp_path / "stub_python.sh"
    stub_py.write_text("#!/usr/bin/env bash\necho '[1]'\n")
    stub_py.chmod(0o755)
    proc = subprocess.run(
        [
            "bash",
            _RUN_GKE,
            *_BASE_GPU_ARGS,
            "--namespace",
            "custom-ns",
            "--group",
            "dataloading/webdataset",
            "--bucket-type",
            "zonal",
            "--results-uri",
            "gs://test-results/ssb",
            "--branch",
            "feat-x",
            "--run-id",
            "0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0",
            "--build-id",
            "0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0",
            "--infra-prefix",
            "my-prefix",
            "--requirements",
            "gcsfs==2026.9.0",
            "--require-amplification",
            "--commit",
            "deadbeef",
            "--dry-run",
        ],
        env=dict(os.environ, SSB_PYTHON=str(stub_py)),
        capture_output=True,
        text=True,
        check=True,
    )
    out = proc.stdout
    assert "--namespace custom-ns" in out
    assert "namespace=custom-ns" not in out
    assert "bucketType=zonal" in out
    assert "buildId=0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0" in out
    assert "infraPrefix=my-prefix" in out
    assert "resultsBranch=feat-x" in out
    assert "runId=0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0" in out
    assert "requireAmplification=true" in out
    assert "requirements=gcsfs==2026.9.0" in out


def test_run_gke_tpu_two_node_without_placement_policy_fails_early(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    helm_log = tmp_path / "helm.log"
    (bin_dir / "helm").write_text(f'#!/usr/bin/env bash\necho "$@" >> "{helm_log}"\n')
    (bin_dir / "helm").chmod(0o755)
    stub_py = tmp_path / "stub_python.sh"
    stub_py.write_text("#!/usr/bin/env bash\necho '[1, 2]'\n")
    stub_py.chmod(0o755)
    proc = subprocess.run(
        [
            "bash",
            _RUN_GKE,
            *_BASE_TPU_ARGS,
            "--group",
            "checkpointing/pytorch_save",
            "--results-uri",
            "gs://test-results/ssb",
            "--commit",
            "deadbeef",
        ],
        env=dict(
            os.environ,
            PATH=f"{bin_dir}:{os.environ['PATH']}",
            SSB_PYTHON=str(stub_py),
        ),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "--tpu-placement-policy" in proc.stderr
    assert not helm_log.exists()


def test_run_gke_deadline_uninstalls_all_active_releases(tmp_path):
    import json

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    helm_log = tmp_path / "helm.log"
    (bin_dir / "helm").write_text(
        f'#!/usr/bin/env bash\necho "$@" >> "{helm_log}"\nexit 0\n'
    )
    (bin_dir / "helm").chmod(0o755)
    (bin_dir / "kubectl").write_text("#!/usr/bin/env bash\nexit 0\n")
    (bin_dir / "kubectl").chmod(0o755)

    cases = [{"id": "c1", "nodes": 1}, {"id": "c2", "nodes": 1}]
    stub_py = tmp_path / "stub_python.sh"
    stub_py.write_text(f"#!/usr/bin/env bash\necho '{json.dumps(cases)}'\n")
    stub_py.chmod(0o755)

    proc = subprocess.run(
        [
            "bash",
            _RUN_GKE,
            *_BASE_GPU_ARGS,
            "--group",
            "dataloading/webdataset",
            "--results-uri",
            "gs://test-results/ssb",
            "--commit",
            "deadbeef",
            "--max-cases-per-release",
            "1",
            "--max-concurrent",
            "2",
            "--deadline-seconds",
            "1",
        ],
        env=dict(
            os.environ,
            PATH=f"{bin_dir}:{os.environ['PATH']}",
            SSB_PYTHON=str(stub_py),
            SSB_POLL_SECONDS="0",
        ),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    lines = helm_log.read_text().splitlines()
    installed = [l.split()[1] for l in lines if l.startswith("install ")]
    uninstalled = [l.split()[1] for l in lines if l.startswith("uninstall ")]
    assert len(installed) == 2, lines
    assert set(installed) == set(uninstalled), (installed, uninstalled)


def test_run_gke_partial_batch_failure_exits_nonzero_after_all_finish(tmp_path):
    import json

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    helm_log = tmp_path / "helm.log"
    releases_file = tmp_path / "releases.txt"
    (bin_dir / "helm").write_text(
        f"#!/usr/bin/env bash\n"
        f'echo "$@" >> "{helm_log}"\n'
        f'if [[ "$1" == "install" ]]; then echo "$2" >> "{releases_file}"; fi\n'
        f"exit 0\n"
    )
    (bin_dir / "helm").chmod(0o755)
    (bin_dir / "kubectl").write_text(
        f"#!/usr/bin/env bash\n"
        f'if [[ "$*" == *"get jobset"* ]]; then\n'
        f"  i=0\n"
        f"  while IFS= read -r rel; do\n"
        f'    [[ -n "$rel" ]] || continue\n'
        f"    if (( i == 0 )); then\n"
        f'      printf "%s\\tFailed=True \\n" "$rel"\n'
        f"    else\n"
        f'      printf "%s\\tCompleted=True \\n" "$rel"\n'
        f"    fi\n"
        f"    i=$((i + 1))\n"
        f'  done < "{releases_file}"\n'
        f"fi\n"
        f"exit 0\n"
    )
    (bin_dir / "kubectl").chmod(0o755)

    cases = [{"id": "c1", "nodes": 1}, {"id": "c2", "nodes": 1}]
    stub_py = tmp_path / "stub_python.sh"
    stub_py.write_text(f"#!/usr/bin/env bash\necho '{json.dumps(cases)}'\n")
    stub_py.chmod(0o755)

    proc = subprocess.run(
        [
            "bash",
            _RUN_GKE,
            *_BASE_GPU_ARGS,
            "--group",
            "dataloading/webdataset",
            "--results-uri",
            "gs://test-results/ssb",
            "--commit",
            "deadbeef",
            "--max-cases-per-release",
            "1",
            "--max-concurrent",
            "2",
        ],
        env=dict(
            os.environ,
            PATH=f"{bin_dir}:{os.environ['PATH']}",
            SSB_PYTHON=str(stub_py),
            SSB_POLL_SECONDS="0",
        ),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    lines = helm_log.read_text().splitlines()
    installed = [l.split()[1] for l in lines if l.startswith("install ")]
    uninstalled = [l.split()[1] for l in lines if l.startswith("uninstall ")]
    assert len(installed) == 2, lines
    assert set(installed) == set(uninstalled), (installed, uninstalled)


def test_run_gke_defaults_helm_driver_to_configmap_and_allows_override(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    driver_log = tmp_path / "driver.log"
    rel_file = tmp_path / "release.txt"
    (bin_dir / "helm").write_text(
        f"#!/usr/bin/env bash\n"
        f'echo "$1:${{HELM_DRIVER:-<unset>}}" >> "{driver_log}"\n'
        f'if [[ "$1" == "install" ]]; then echo "$2" > "{rel_file}"; fi\n'
        f"exit 0\n"
    )
    (bin_dir / "helm").chmod(0o755)
    (bin_dir / "kubectl").write_text(
        f"#!/usr/bin/env bash\n"
        f'if [[ "$*" == *"get jobset"* ]]; then\n'
        f'  printf "%s\\tCompleted=True \\n" "$(cat "{rel_file}")"\n'
        f"fi\n"
        f"exit 0\n"
    )
    (bin_dir / "kubectl").chmod(0o755)
    stub_py = tmp_path / "stub_python.sh"
    stub_py.write_text("#!/usr/bin/env bash\necho '[1]'\n")
    stub_py.chmod(0o755)

    base_cmd = [
        "bash",
        _RUN_GKE,
        *_BASE_GPU_ARGS,
        "--group",
        "dataloading/webdataset",
        "--results-uri",
        "gs://test-results/ssb",
        "--commit",
        "deadbeef",
    ]
    base_env = {k: v for k, v in os.environ.items() if k != "HELM_DRIVER"}
    base_env.update(
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        SSB_PYTHON=str(stub_py),
        SSB_POLL_SECONDS="0",
    )

    subprocess.run(base_cmd, env=base_env, capture_output=True, text=True, check=True)
    assert driver_log.read_text().splitlines() == [
        "install:configmap",
        "uninstall:configmap",
    ]

    driver_log.unlink()
    override_env = dict(base_env, HELM_DRIVER="secret")
    subprocess.run(
        base_cmd, env=override_env, capture_output=True, text=True, check=True
    )
    assert driver_log.read_text().splitlines() == [
        "install:secret",
        "uninstall:secret",
    ]


def test_run_gke_stale_filter_exits_nonzero_while_all_skipped_exits_zero(tmp_path):
    bad = subprocess.run(
        [
            "bash",
            _RUN_GKE,
            *_BASE_GPU_ARGS,
            "--group",
            "dataloading/webdataset",
            "--results-uri",
            "gs://test-results/ssb",
            "--commit",
            "deadbeef",
            "--filter",
            "no_such_case_xyz",
            "--dry-run",
        ],
        env=dict(os.environ, SSB_PYTHON=sys.executable),
        capture_output=True,
        text=True,
    )
    assert bad.returncode != 0, bad.stdout + bad.stderr

    stub_py = tmp_path / "stub_python.sh"
    stub_py.write_text("#!/usr/bin/env bash\necho '[]'\n")
    stub_py.chmod(0o755)
    ok = subprocess.run(
        [
            "bash",
            _RUN_GKE,
            *_BASE_TPU_ARGS,
            "--group",
            "checkpointing/pytorch_lightning",
            "--results-uri",
            "gs://test-results/ssb",
            "--commit",
            "deadbeef",
            "--dry-run",
        ],
        env=dict(os.environ, SSB_PYTHON=str(stub_py)),
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "No runnable cases" in ok.stdout


def test_run_gke_deadline_prevents_subsequent_batch_launch(tmp_path):
    import json

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    helm_log = tmp_path / "helm.log"
    rel_file = tmp_path / "release.txt"
    (bin_dir / "helm").write_text(
        f"#!/usr/bin/env bash\n"
        f'echo "$@" >> "{helm_log}"\n'
        f'if [[ "$1" == "install" ]]; then echo "$2" > "{rel_file}"; fi\n'
        f"exit 0\n"
    )
    (bin_dir / "helm").chmod(0o755)
    (bin_dir / "kubectl").write_text(
        f"#!/usr/bin/env bash\n"
        f'if [[ "$*" == *"get jobset"* ]]; then\n'
        f"  sleep 1.2\n"
        f'  printf "%s\\tCompleted=True \\n" "$(cat "{rel_file}")"\n'
        f"fi\n"
        f"exit 0\n"
    )
    (bin_dir / "kubectl").chmod(0o755)

    cases = [{"id": "c1", "nodes": 1}, {"id": "c2", "nodes": 1}]
    stub_py = tmp_path / "stub_python.sh"
    stub_py.write_text(f"#!/usr/bin/env bash\necho '{json.dumps(cases)}'\n")
    stub_py.chmod(0o755)

    proc = subprocess.run(
        [
            "bash",
            _RUN_GKE,
            *_BASE_GPU_ARGS,
            "--group",
            "dataloading/webdataset",
            "--results-uri",
            "gs://test-results/ssb",
            "--commit",
            "deadbeef",
            "--max-cases-per-release",
            "1",
            "--max-concurrent",
            "1",
            "--deadline-seconds",
            "1",
        ],
        env=dict(
            os.environ,
            PATH=f"{bin_dir}:{os.environ['PATH']}",
            SSB_PYTHON=str(stub_py),
            SSB_POLL_SECONDS="0",
        ),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    lines = helm_log.read_text().splitlines()
    installed = [l.split()[1] for l in lines if l.startswith("install ")]
    uninstalled = [l.split()[1] for l in lines if l.startswith("uninstall ")]
    assert len(installed) == 1, lines
    assert set(installed) == set(uninstalled), (installed, uninstalled)


def test_run_gke_deadline_diagnostics_queries_workloads_and_pods_events_separately(
    tmp_path,
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    kubectl_log = tmp_path / "kubectl.log"
    (bin_dir / "helm").write_text("#!/usr/bin/env bash\nexit 0\n")
    (bin_dir / "helm").chmod(0o755)
    (bin_dir / "kubectl").write_text(
        f"#!/usr/bin/env bash\n"
        f'echo "$@" >> "{kubectl_log}"\n'
        f'if [[ "$*" == *"workloads"* ]]; then\n'
        f'  echo "error: the server doesn\'t have a resource type \\"workloads\\"" >&2\n'
        f"  exit 1\n"
        f"fi\n"
        f"exit 0\n"
    )
    (bin_dir / "kubectl").chmod(0o755)

    stub_py = tmp_path / "stub_python.sh"
    stub_py.write_text("#!/usr/bin/env bash\necho '[1]'\n")
    stub_py.chmod(0o755)

    proc = subprocess.run(
        [
            "bash",
            _RUN_GKE,
            *_BASE_GPU_ARGS,
            "--group",
            "dataloading/webdataset",
            "--results-uri",
            "gs://test-results/ssb",
            "--commit",
            "deadbeef",
            "--deadline-seconds",
            "1",
        ],
        env=dict(
            os.environ,
            PATH=f"{bin_dir}:{os.environ['PATH']}",
            SSB_PYTHON=str(stub_py),
            SSB_POLL_SECONDS="0",
        ),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    k_lines = kubectl_log.read_text().splitlines()
    assert any(" get workloads" in l and "events" not in l for l in k_lines), k_lines
    assert any(" get pods,events" in l for l in k_lines), k_lines


@pytest.mark.skipif(not _HAS_HELM, reason="helm CLI not installed")
def test_chart_release_namespace_build_id_default_and_bucket_type_zone_pin():
    proc = _render_chart(
        "ssb-wds-1n-1234",
        "--namespace",
        "team-ns",
        "--set",
        "accelerator=gpu",
        "--set",
        "nodes=1",
        "--set",
        "bucketType=zonal",
        "--set",
        "zone=us-central1-a",
        "--set-string",
        "runId=run-uuid-42",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    docs = _docs_by_kind(proc.stdout)
    for kind in ("JobSet", "Service", "ConfigMap"):
        assert docs[kind][0]["metadata"]["namespace"] == "team-ns"
    js = docs["JobSet"][0]
    assert js["metadata"]["labels"]["gcsfs-ssb/build-id"] == "run-uuid-42"
    pod_spec = js["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]
    assert pod_spec["nodeSelector"]["topology.kubernetes.io/zone"] == "us-central1-a"
    env = {
        e["name"]: e["value"] for e in pod_spec["containers"][0]["env"] if "value" in e
    }
    assert env["GCSFS_SUBSYSTEM_LEADER_ADDR"].endswith(".team-ns.svc.cluster.local")


@pytest.mark.skipif(not _HAS_HELM, reason="helm CLI not installed")
def test_chart_parameterizes_tpu_hardware_fields():
    proc = _render_chart(
        "ssb-save-1n-abcd",
        "--set",
        "accelerator=tpu",
        "--set",
        "nodes=1",
        "--set",
        "tpu.accelerator=tpu-v6e-slice",
        "--set",
        "tpu.chipsPerNode=8",
        "--set",
        "tpu.shmSize=400Gi",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    docs = _docs_by_kind(proc.stdout)
    pod_spec = docs["JobSet"][0]["spec"]["replicatedJobs"][0]["template"]["spec"][
        "template"
    ]["spec"]
    assert (
        pod_spec["nodeSelector"]["cloud.google.com/gke-tpu-accelerator"]
        == "tpu-v6e-slice"
    )
    assert pod_spec["containers"][0]["resources"]["limits"]["google.com/tpu"] == 8
    shm = next(v for v in pod_spec["volumes"] if v["name"] == "shared-memory")
    assert shm["emptyDir"]["sizeLimit"] == "400Gi"

    bad = _render_chart(
        "ssb-save-1n-abcd",
        "--set",
        "accelerator=tpu",
        "--set",
        "nodes=1",
        "--set",
        "tpu.chipsPerNode=0",
    )
    assert bad.returncode != 0
    assert "tpu.chipsPerNode" in (bad.stdout + bad.stderr)


def test_run_gke_dry_run_accepts_future_group_with_bounded_release_name(tmp_path):
    stub_py = tmp_path / "stub_python.sh"
    stub_py.write_text("#!/usr/bin/env bash\necho '[16]'\n")
    stub_py.chmod(0o755)
    proc = subprocess.run(
        [
            "bash",
            _RUN_GKE,
            *_BASE_GPU_ARGS,
            "--group",
            "checkpointing/megatron_very_long_name",
            "--results-uri",
            "gs://test-results/ssb",
            "--commit",
            "deadbeef",
            "--dry-run",
        ],
        env=dict(os.environ, SSB_PYTHON=str(stub_py)),
        capture_output=True,
        text=True,
        check=True,
    )
    match = re.search(r"helm install\s+(\S+)", proc.stdout)
    assert match is not None, proc.stdout
    release = match.group(1)
    assert len(release) <= 20, release
    assert release.startswith("ssb-megatro-16n-"), release
