import os
import subprocess
from datetime import datetime, timedelta, timezone

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, "cloudbuild", "subsystembenchmarks", "scripts")


def test_init_gke_variables_writes_quoted_env(tmp_path):
    script = os.path.join(_SCRIPTS_DIR, "init-gke-variables.sh")
    out_file = tmp_path / "gke_vars.env"
    env = dict(
        os.environ,
        _ACCELERATOR="gpu",
        _MODE="nightly",
        _HF_TOKEN="",
        _ZONE="europe-west1-b",
        COMMIT_SHA="abcdef1234567890",
        BUILD_ID="0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0",
        _INFRA_PREFIX="my-prefix",
        _GKE_CLUSTER="my-gpu-cluster",
        _GKE_CLUSTER_PROJECT="my-cluster-project",
        _GKE_CLUSTER_LOCATION="europe-west1",
        _GKE_NAMESPACE="",
        _GKE_ENDPOINT="default",
        _GKE_QUEUE="default",
        _TPU_PRIORITY="low",
        _TPU_DURATION="180",
        _TPU_PLACEMENT_POLICY="",
        _MAX_CASES_PER_RELEASE="",
        _MAX_CONCURRENT_RELEASES="1",
        _GKE_DEADLINE_SECONDS="16200",
        _GCSFS_REPO="https://github.com/fsspec/gcsfs.git",
    )
    proc = subprocess.run(
        ["bash", script, str(out_file)], env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out_file.is_file()

    check = subprocess.run(
        [
            "bash",
            "-c",
            (
                f'source "{out_file}" && '
                'printf "ACC=%s\\nNS=%s\\nCL=%s\\nEP=%s\\nDL=%s\\nPREF=%s\\n" '
                '"$ACCELERATOR" "$GKE_NAMESPACE" "$GKE_CLUSTER" '
                '"$GKE_ENDPOINT" "$GKE_DEADLINE_SECONDS" "$INFRA_PREFIX"'
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "ACC=gpu\n" in check.stdout
    assert "NS=default\n" in check.stdout
    assert "CL=my-gpu-cluster\n" in check.stdout
    assert "EP=default\n" in check.stdout
    assert "DL=16200\n" in check.stdout
    assert "PREF=my-prefix\n" in check.stdout


def test_gke_credentials_endpoint_flags(tmp_path):
    script = os.path.join(_SCRIPTS_DIR, "gke-credentials.sh")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gcloud_log = tmp_path / "gcloud.log"
    (bin_dir / "gcloud").write_text(
        f'#!/usr/bin/env bash\necho "$@" > "{gcloud_log}"\nexit 0\n'
    )
    (bin_dir / "gcloud").chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")

    base = "container clusters get-credentials my-cls --project my-proj --location us-central1"
    expected = {
        "default": base,
        "internal-ip": f"{base} --internal-ip",
        "dns": f"{base} --dns-endpoint",
    }
    for mode, want_args in expected.items():
        proc = subprocess.run(
            ["bash", script, "my-cls", "my-proj", "us-central1", mode],
            env=env,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert proc.stdout.strip() == "gke_my-proj_us-central1_my-cls"
        assert gcloud_log.read_text().strip() == want_args


def test_gke_credentials_skips_gcloud_when_context_cached_in_kubeconfig(tmp_path):
    script = os.path.join(_SCRIPTS_DIR, "gke-credentials.sh")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gcloud_log = tmp_path / "gcloud.log"
    (bin_dir / "gcloud").write_text(
        f'#!/usr/bin/env bash\necho "$@" >> "{gcloud_log}"\nexit 0\n'
    )
    (bin_dir / "gcloud").chmod(0o755)
    (bin_dir / "kubectl").write_text(
        "#!/usr/bin/env bash\necho gke_my-proj_us-central1_my-cls\n"
    )
    (bin_dir / "kubectl").chmod(0o755)
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n")

    env = dict(
        os.environ,
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        KUBECONFIG=str(kubeconfig),
    )
    proc = subprocess.run(
        ["bash", script, "my-cls", "my-proj", "us-central1", "dns"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip() == "gke_my-proj_us-central1_my-cls"
    assert not gcloud_log.exists()


def test_gke_tools_verifies_helm_checksum(tmp_path):
    script = os.path.join(_SCRIPTS_DIR, "gke-tools.sh")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    install_dir = tmp_path / "tools"
    # Stub gcloud/kubectl so apt-get is skipped, and stub curl to emit corrupt tarball.
    for cmd in ("gcloud", "kubectl", "gke-gcloud-auth-plugin"):
        p = bin_dir / cmd
        p.write_text("#!/usr/bin/env bash\nexit 0\n")
        p.chmod(0o755)
    fake_curl = bin_dir / "curl"
    fake_curl.write_text(
        "#!/usr/bin/env bash\n"
        'out=""\n'
        "while [[ $# -gt 0 ]]; do\n"
        '  if [[ "$1" == "-o" ]]; then out="$2"; shift 2; else shift; fi\n'
        "done\n"
        'echo "corrupt-not-helm" > "$out"\n'
    )
    fake_curl.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{bin_dir}:/usr/bin:/bin",
        SSB_TOOLS_DIR=str(install_dir),
    )
    proc = subprocess.run(["bash", script], env=env, capture_output=True, text=True)
    assert proc.returncode != 0
    assert not (install_dir / "helm").exists()
    assert (
        "checksum" in (proc.stdout + proc.stderr).lower()
        or "sha" in (proc.stdout + proc.stderr).lower()
    )


def test_gke_cleanup_uninstalls_by_build_id_and_age(tmp_path):
    script = os.path.join(_SCRIPTS_DIR, "gke-cleanup.sh")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    kubectl_log = tmp_path / "kubectl.log"
    helm_log = tmp_path / "helm.log"

    now = datetime.now(timezone.utc)
    old_ts = (now - timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ")
    fresh_ts = (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    (bin_dir / "kubectl").write_text(
        f"#!/usr/bin/env bash\n"
        f'echo "$@" >> "{kubectl_log}"\n'
        f'if [[ "$*" == *"gcsfs-ssb/build-id=b-123"* ]]; then\n'
        f'  printf "ssb-wds-1n-aaaa\\t{fresh_ts}\\n"\n'
        f'elif [[ "$*" == *"gcsfs-ssb/infra-prefix=my-pref"* ]]; then\n'
        f'  printf "ssb-save-1n-old1\\t{old_ts}\\nssb-save-1n-new1\\t{fresh_ts}\\n"\n'
        f"fi\n"
        f"exit 0\n"
    )
    (bin_dir / "kubectl").chmod(0o755)
    (bin_dir / "helm").write_text(
        f'#!/usr/bin/env bash\necho "$@" >> "{helm_log}"\nexit 0\n'
    )
    (bin_dir / "helm").chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")

    p1 = subprocess.run(
        [
            "bash",
            script,
            "--kube-context",
            "ctx-1",
            "--namespace",
            "ns-1",
            "--build-id",
            "b-123",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert p1.returncode == 0, p1.stdout + p1.stderr
    assert "gcsfs-ssb/build-id=b-123" in kubectl_log.read_text()
    assert (
        "uninstall ssb-wds-1n-aaaa --kube-context ctx-1 --namespace ns-1"
        in helm_log.read_text()
    )

    helm_log.unlink()
    p2 = subprocess.run(
        [
            "bash",
            script,
            "--kube-context",
            "ctx-1",
            "--namespace",
            "ns-1",
            "--infra-prefix",
            "my-pref",
            "--older-than-hours",
            "10",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert p2.returncode == 0, p2.stdout + p2.stderr
    helm_calls = helm_log.read_text()
    assert "ssb-save-1n-old1" in helm_calls
    assert "ssb-save-1n-new1" not in helm_calls


def test_gke_run_maps_build_vars_to_run_gke(tmp_path):
    script = os.path.join(_SCRIPTS_DIR, "gke-run.sh")
    build_vars = tmp_path / "build_vars.env"
    gke_vars = tmp_path / "gke_vars.env"
    build_vars.write_text(
        "\n".join(
            [
                "export BRANCH_NAME=main",
                "export RUN_ID=0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0",
                "export COMMIT_SHA=deadbeef1234",
                "export GROUP=checkpointing/pytorch_save",
                "export BUCKET_TYPE=regional",
                "export PROJECT_ID=my-bucket-proj",
                "export REGION=us-central1",
                "export ZONE=us-central1-c",
                "export BUCKET_PREFIX=my-pref-regional-0f1e2d3c",
                "export RESULTS_BUCKET=my-pref-run-results",
                "export REQUIREMENTS_OVERRIDE='gcsfs==2026.9.0'",
                "export SWEEP_AXES=''",
                "export FILTER='fsdp'",
                "export MODEL_ID=''",
            ]
        )
        + "\n"
    )
    gke_vars.write_text(
        "\n".join(
            [
                "export ACCELERATOR=tpu",
                "export GKE_CLUSTER=my-tpu-cls",
                "export GKE_CLUSTER_PROJECT=my-tpu-proj",
                "export GKE_CLUSTER_LOCATION=us-central1",
                "export GKE_NAMESPACE=custom-ns",
                "export GKE_ENDPOINT=dns",
                "export GKE_QUEUE=''",
                "export TPU_PRIORITY=low",
                "export TPU_DURATION=180",
                "export TPU_PLACEMENT_POLICY=my-pp",
                "export MAX_CASES_PER_RELEASE=4",
                "export MAX_CONCURRENT_RELEASES=2",
                "export GKE_DEADLINE_SECONDS=16200",
                "export GCSFS_REPO=https://github.com/fsspec/gcsfs.git",
                "export INFRA_PREFIX=my-pref",
            ]
        )
        + "\n"
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gcloud").write_text("#!/usr/bin/env bash\nexit 0\n")
    (bin_dir / "gcloud").chmod(0o755)

    recorded = tmp_path / "run_gke_argv.txt"
    fake_run_gke = tmp_path / "fake_run_gke.sh"
    fake_run_gke.write_text(
        f"#!/usr/bin/env bash\n"
        f'for a in "$@"; do printf "%s\\n" "$a" >> "{recorded}"; done\n'
        f"exit 7\n"
    )
    fake_run_gke.chmod(0o755)

    env = dict(
        os.environ,
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        SSB_BUILD_VARS=str(build_vars),
        SSB_GKE_VARS=str(gke_vars),
        SSB_RUN_GKE_SCRIPT=str(fake_run_gke),
        SSB_SKIP_VENV_SETUP="1",
    )
    proc = subprocess.run(["bash", script], env=env, capture_output=True, text=True)
    assert proc.returncode == 7, proc.stdout + proc.stderr
    argv = recorded.read_text().splitlines()
    assert argv == [
        "--accelerator",
        "tpu",
        "--kube-context",
        "gke_my-tpu-proj_us-central1_my-tpu-cls",
        "--namespace",
        "custom-ns",
        "--project",
        "my-bucket-proj",
        "--location",
        "us-central1",
        "--zone",
        "us-central1-c",
        "--group",
        "checkpointing/pytorch_save",
        "--bucket-type",
        "regional",
        "--bucket-prefix",
        "my-pref-regional-0f1e2d3c",
        "--results-uri",
        "gs://my-pref-run-results/subsystembenchmarks",
        "--branch",
        "main",
        "--run-id",
        "0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0",
        "--repo",
        "https://github.com/fsspec/gcsfs.git",
        "--commit",
        "deadbeef1234",
        "--build-id",
        "0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0",
        "--infra-prefix",
        "my-pref",
        "--require-amplification",
        "--priority",
        "low",
        "--duration",
        "180",
        "--max-concurrent",
        "2",
        "--deadline-seconds",
        "16200",
        "--filter",
        "fsdp",
        "--requirements",
        "gcsfs==2026.9.0",
        "--tpu-placement-policy",
        "my-pp",
        "--max-cases-per-release",
        "4",
    ]
    # Verify empty optional flags were omitted.
    for omitted in ("--sweep-axes", "--model-id", "--queue-name"):
        assert omitted not in argv


_CLOUDBUILD_YAML = os.path.join(
    _REPO_ROOT,
    "cloudbuild",
    "subsystembenchmarks",
    "subsystembenchmarks-cloudbuild.yaml",
)
_GOLDEN_YAML = os.path.join(
    os.path.dirname(__file__), "fixtures", "cloudbuild_cpu_golden.yaml"
)
_CPU_GUARD = (
    'if [ "${_ACCELERATOR}" != "cpu" ]; then '
    'echo "Skipping: _ACCELERATOR=${_ACCELERATOR}"; exit 0; fi'
)
_GKE_GUARD = (
    'if [ "${_ACCELERATOR}" = "cpu" ]; then echo "Skipping: CPU run"; exit 0; fi'
)


def _load_yaml(path):
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def test_cpu_steps_match_golden_after_guard():
    golden = _load_yaml(_GOLDEN_YAML)
    current = _load_yaml(_CLOUDBUILD_YAML)

    # All existing substitutions preserved with identical defaults.
    for k, v in golden["substitutions"].items():
        assert current["substitutions"][k] == v, f"substitution {k} changed"

    golden_steps = {s["id"]: s for s in golden["steps"]}
    current_steps = {s["id"]: s for s in current["steps"]}

    cpu_only = {
        "generate-ssh-key",
        "create-vm",
        "run-subsystembenchmarks",
        "cleanup-ssh-key",
        "cleanup-vm",
    }
    unchanged = {
        "init-variables",
        "cleanup-leaked-resources",
        "create-buckets",
        "delete-buckets",
        "check-failure",
    }
    assert set(golden_steps) == cpu_only | unchanged

    for step_id, g_step in golden_steps.items():
        assert step_id in current_steps, f"missing golden step {step_id}"
        c_body = current_steps[step_id]["args"][-1]
        g_body = g_step["args"][-1]
        if step_id in cpu_only:
            assert c_body == f"{_CPU_GUARD}\n{g_body}", f"step {step_id} body drifted"
        else:
            assert c_body == g_body, f"unchanged step {step_id} body drifted"

    # build_vars.env printf block in init-variables is untouched.
    assert (
        current_steps["init-variables"]["args"][-1]
        == golden_steps["init-variables"]["args"][-1]
    )


def test_new_substitution_defaults():
    subs = _load_yaml(_CLOUDBUILD_YAML)["substitutions"]
    expected = {
        "_ACCELERATOR": "cpu",
        "_GKE_CLUSTER": "",
        "_GKE_CLUSTER_PROJECT": "",
        "_GKE_CLUSTER_LOCATION": "",
        "_GKE_NAMESPACE": "default",
        "_GKE_ENDPOINT": "default",
        "_GKE_QUEUE": "",
        "_TPU_PRIORITY": "low",
        "_TPU_DURATION": "180",
        "_TPU_PLACEMENT_POLICY": "",
        "_MAX_CASES_PER_RELEASE": "",
        "_MAX_CONCURRENT_RELEASES": "1",
        "_GKE_DEADLINE_SECONDS": "16200",
        "_GCSFS_REPO": "https://github.com/fsspec/gcsfs.git",
    }
    for key, val in expected.items():
        assert subs.get(key) == val, f"{key}: expected {val!r}, got {subs.get(key)!r}"


def test_gke_steps_guarded_and_wired():
    cfg = _load_yaml(_CLOUDBUILD_YAML)
    steps = {s["id"]: s for s in cfg["steps"]}

    assert "init-gke-variables" in steps
    assert steps["init-gke-variables"]["waitFor"] == ["-"]
    assert not steps["init-gke-variables"].get("allowFailure")

    for gid in ("cleanup-leaked-gke", "run-subsystembenchmarks-gke", "cleanup-gke"):
        assert gid in steps, f"missing GKE step {gid}"
        body = steps[gid]["args"][-1]
        assert body.startswith(_GKE_GUARD + "\n"), f"{gid} missing GKE guard"
        assert "export KUBECONFIG=/workspace/.kube/config" in body

    assert (
        steps["cleanup-leaked-gke"]["name"]
        == "gcr.io/google.com/cloudsdktool/cloud-sdk"
    )
    assert steps["cleanup-gke"]["name"] == "gcr.io/google.com/cloudsdktool/cloud-sdk"

    assert set(steps["cleanup-leaked-gke"]["waitFor"]) == {
        "init-variables",
        "init-gke-variables",
    }
    assert steps["cleanup-leaked-gke"].get("allowFailure") is True
    assert set(steps["run-subsystembenchmarks-gke"]["waitFor"]) == {
        "create-buckets",
        "cleanup-leaked-gke",
    }
    assert steps["run-subsystembenchmarks-gke"].get("allowFailure") is True
    assert "cleanup-gke" in steps["check-failure"]["waitFor"]
    assert "run-subsystembenchmarks-gke" in steps["cleanup-gke"]["waitFor"]
    assert "init-gke-variables" in steps["cleanup-leaked-resources"]["waitFor"]
    assert "init-gke-variables" in steps["create-buckets"]["waitFor"]
    assert "init-gke-variables" in steps["create-vm"]["waitFor"]
    assert "cleanup-gke" in steps["delete-buckets"]["waitFor"]


def test_init_gke_variables_rejects_misconfiguration(tmp_path):
    cfg = _load_yaml(_CLOUDBUILD_YAML)
    steps = {s["id"]: s for s in cfg["steps"]}
    step = steps["init-gke-variables"]
    raw_script = step["args"][-1]
    raw_env = step.get("env", [])

    def run_step(overrides):
        subs = dict(cfg["substitutions"])
        subs.update(
            {
                "COMMIT_SHA": "abcdef1234567890",
                "BUILD_ID": "0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0",
                "_INFRA_PREFIX": "my-prefix",
                "_ZONE": "us-central1-c",
                "_GKE_CLUSTER": "my-cluster",
                "_GKE_CLUSTER_PROJECT": "my-cluster-proj",
                "_GKE_CLUSTER_LOCATION": "us-central1",
            }
        )
        subs.update(overrides)

        def render(text):
            for k, v in subs.items():
                text = text.replace(f"${{{k}}}", str(v))
            return text.replace("$$", "$")

        env = dict(os.environ)
        for item in raw_env:
            k, v = item.split("=", 1)
            env[k] = render(v)
        out_path = (
            tmp_path / f"gke_vars_{abs(hash(tuple(sorted(overrides.items()))))}.env"
        )
        rendered = render(raw_script).replace("/workspace/gke_vars.env", str(out_path))
        return subprocess.run(
            ["bash", "-c", rendered],
            cwd=_REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )

    # CPU with empty GKE vars succeeds.
    ok_cpu = run_step(
        {
            "_ACCELERATOR": "cpu",
            "_GKE_CLUSTER": "",
            "_GKE_CLUSTER_PROJECT": "",
            "_GKE_CLUSTER_LOCATION": "",
            "COMMIT_SHA": "",
        }
    )
    assert ok_cpu.returncode == 0, ok_cpu.stdout + ok_cpu.stderr

    # Invalid configurations must fail.
    bad_cases = [
        {"_ACCELERATOR": ""},
        {"_ACCELERATOR": "xpu"},
        {"_ACCELERATOR": "gpu", "_GKE_CLUSTER": ""},
        {"_ACCELERATOR": "gpu", "_GKE_CLUSTER_PROJECT": ""},
        {"_ACCELERATOR": "gpu", "_GKE_CLUSTER_LOCATION": ""},
        {"_ACCELERATOR": "gpu", "_MODE": ""},
        {"_ACCELERATOR": "tpu", "_MODE": "ci"},
        {"_ACCELERATOR": "gpu", "_HF_TOKEN": "hf_secret"},
        {"_ACCELERATOR": "gpu", "COMMIT_SHA": ""},
        {"_ACCELERATOR": "gpu", "_GKE_ENDPOINT": "bogus"},
        {
            "_ACCELERATOR": "gpu",
            "_ZONE": "europe-west1-b",
            "_GKE_CLUSTER_LOCATION": "us-central1",
        },
    ]
    for case in bad_cases:
        res = run_step(case)
        assert res.returncode != 0, f"expected failure for {case}, got rc=0"


def test_no_internal_identifiers_in_yaml_and_readme():
    subs = _load_yaml(_CLOUDBUILD_YAML)["substitutions"]
    for key in (
        "_GKE_CLUSTER",
        "_GKE_CLUSTER_PROJECT",
        "_GKE_CLUSTER_LOCATION",
        "_TPU_PLACEMENT_POLICY",
    ):
        assert subs[key] == "", f"{key} must default to empty string"
    readme = open(
        os.path.join(_REPO_ROOT, "cloudbuild", "subsystembenchmarks", "README.md")
    ).read()
    assert "_ACCELERATOR" in readme
    assert "_GKE_CLUSTER" in readme
    assert "_TPU_PLACEMENT_POLICY" in readme


def test_gke_cleanup_defaults_helm_driver_to_configmap_and_allows_override(tmp_path):
    script = os.path.join(_SCRIPTS_DIR, "gke-cleanup.sh")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    driver_log = tmp_path / "helm_driver.log"
    (bin_dir / "kubectl").write_text(
        '#!/usr/bin/env bash\nprintf "ssb-wds-1n-aaaa\\t2026-04-20T00:00:00Z\\n"\n'
    )
    (bin_dir / "kubectl").chmod(0o755)
    (bin_dir / "helm").write_text(
        f"#!/usr/bin/env bash\n"
        f'echo "$1:${{HELM_DRIVER:-<unset>}}" >> "{driver_log}"\n'
        f"exit 0\n"
    )
    (bin_dir / "helm").chmod(0o755)

    base_env = {k: v for k, v in os.environ.items() if k != "HELM_DRIVER"}
    base_env["PATH"] = f"{bin_dir}:{os.environ['PATH']}"
    cmd = [
        "bash",
        script,
        "--kube-context",
        "ctx-1",
        "--namespace",
        "ns-1",
        "--build-id",
        "b-123",
    ]

    p1 = subprocess.run(cmd, env=base_env, capture_output=True, text=True)
    assert p1.returncode == 0, p1.stdout + p1.stderr
    assert driver_log.read_text().splitlines() == ["uninstall:configmap"]

    driver_log.unlink()
    p2 = subprocess.run(
        cmd, env=dict(base_env, HELM_DRIVER="secret"), capture_output=True, text=True
    )
    assert p2.returncode == 0, p2.stdout + p2.stderr
    assert driver_log.read_text().splitlines() == ["uninstall:secret"]


def test_gke_run_reinstalls_local_gcsfs_after_group_requirements(tmp_path):
    script = os.path.join(_SCRIPTS_DIR, "gke-run.sh")
    build_vars = tmp_path / "build_vars.env"
    gke_vars = tmp_path / "gke_vars.env"
    build_vars.write_text(
        "\n".join(
            [
                "export BRANCH_NAME=main",
                "export RUN_ID=0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0",
                "export COMMIT_SHA=deadbeef1234",
                "export GROUP=dataloading/webdataset",
                "export BUCKET_TYPE=regional",
                "export PROJECT_ID=my-bucket-proj",
                "export REGION=us-central1",
                "export ZONE=us-central1-c",
                "export BUCKET_PREFIX=my-pref-regional-0f1e2d3c",
                "export RESULTS_BUCKET=my-pref-run-results",
                "export REQUIREMENTS_OVERRIDE=''",
                "export SWEEP_AXES=''",
                "export FILTER=''",
                "export MODEL_ID=''",
            ]
        )
        + "\n"
    )
    gke_vars.write_text(
        "\n".join(
            [
                "export ACCELERATOR=gpu",
                "export GKE_CLUSTER=my-gpu-cls",
                "export GKE_CLUSTER_PROJECT=my-gpu-proj",
                "export GKE_CLUSTER_LOCATION=us-central1",
                "export GKE_NAMESPACE=default",
                "export GKE_ENDPOINT=default",
                "export GKE_QUEUE=''",
                "export TPU_PRIORITY=low",
                "export TPU_DURATION=180",
                "export TPU_PLACEMENT_POLICY=''",
                "export MAX_CASES_PER_RELEASE=''",
                "export MAX_CONCURRENT_RELEASES=1",
                "export GKE_DEADLINE_SECONDS=16200",
                "export GCSFS_REPO=https://github.com/fsspec/gcsfs.git",
                "export INFRA_PREFIX=my-pref",
            ]
        )
        + "\n"
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pip_log = tmp_path / "pip.log"
    (bin_dir / "gcloud").write_text("#!/usr/bin/env bash\nexit 0\n")
    (bin_dir / "gcloud").chmod(0o755)
    (bin_dir / "pip").write_text(
        f'#!/usr/bin/env bash\necho "$@" >> "{pip_log}"\nexit 0\n'
    )
    (bin_dir / "pip").chmod(0o755)
    fake_py = bin_dir / "python3"
    fake_py.write_text(
        f"#!/usr/bin/env bash\n"
        f'if [[ "$1" == "-m" && "$2" == "venv" ]]; then\n'
        f'  mkdir -p "$3/bin"\n'
        f'  printf \'export PATH="%s:$PATH"\\n\' "{bin_dir}" > "$3/bin/activate"\n'
        f"  printf '#!/usr/bin/env bash\\nexit 0\\n' > \"$3/bin/python\"\n"
        f'  chmod +x "$3/bin/python"\n'
        f"  exit 0\n"
        f"fi\n"
        f"exit 0\n"
    )
    fake_py.chmod(0o755)
    (bin_dir / "python").write_text(f'#!/usr/bin/env bash\nexec "{fake_py}" "$@"\n')
    (bin_dir / "python").chmod(0o755)

    fake_run_gke = tmp_path / "fake_run_gke.sh"
    fake_run_gke.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_run_gke.chmod(0o755)

    env = {k: v for k, v in os.environ.items() if k != "SSB_SKIP_VENV_SETUP"}
    env.update(
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        SSB_BUILD_VARS=str(build_vars),
        SSB_GKE_VARS=str(gke_vars),
        SSB_VENV_DIR=str(tmp_path / "venv"),
        SSB_RUN_GKE_SCRIPT=str(fake_run_gke),
    )
    proc = subprocess.run(["bash", script], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    calls = pip_log.read_text().splitlines()
    editable_indices = [i for i, c in enumerate(calls) if c == "install -e ."]
    req_indices = [i for i, c in enumerate(calls) if c.startswith("install -r ")]
    assert req_indices, calls
    assert editable_indices == [editable_indices[-1]], calls
    assert editable_indices[0] > req_indices[-1], calls


def test_gke_cleanup_rejects_unknown_option():
    script = os.path.join(_SCRIPTS_DIR, "gke-cleanup.sh")
    proc = subprocess.run(
        ["bash", script, "--unknown-flag"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "unknown option" in proc.stderr
