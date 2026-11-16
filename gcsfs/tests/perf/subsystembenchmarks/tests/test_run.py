import os

import pytest

from gcsfs.tests.perf.subsystembenchmarks import run

_REQUIRED = [
    "--bucket-prefix=p",
    "--project=pr",
    "--location=us-central1",
]

_NEW_ENV_KEYS = (
    "GCSFS_SUBSYSTEM_DEVICE",
    "GCSFS_SUBSYSTEM_NNODES",
    "GCSFS_SUBSYSTEM_NODE_RANK",
    "GCSFS_SUBSYSTEM_MASTER_ADDR",
    "GCSFS_SUBSYSTEM_MASTER_PORT",
    "GCSFS_SUBSYSTEM_NPROC_PER_NODE",
    "GCSFS_SUBSYSTEM_CHECKPOINT_EXPORT_ROOT",
    "GCSFS_SUBSYSTEM_CHECKPOINT_SOURCE_ROOT",
    "GCSFS_SUBSYSTEM_RESULTS_URI",
)


_ALL_ENV_KEYS = (
    "GCSFS_SUBSYSTEM_BUCKET_PREFIX",
    "GCSFS_SUBSYSTEM_BUCKET_TYPE",
    "GCSFS_SUBSYSTEM_PROJECT",
    "GCSFS_SUBSYSTEM_LOCATION",
    "GCSFS_SUBSYSTEM_ZONE",
    "GCSFS_SUBSYSTEM_SWEEP_AXES",
    "GCSFS_SUBSYSTEM_MODEL_ID",
    "GCSFS_SUBSYSTEM_GROUP",
    "GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT",
    "STORAGE_EMULATOR_HOST",
    "FSSPEC_GCS_RCU_RAPID_WRITE",
    *_NEW_ENV_KEYS,
)


def test_device_flags_export_env(monkeypatch):
    for key in _ALL_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    # Defaults when optional flags are omitted.
    default_args = run.parse_args(
        ["--group=checkpointing/pytorch"] + _REQUIRED
    )
    run._setup_environment(default_args)
    assert os.environ["GCSFS_SUBSYSTEM_DEVICE"] == "cpu"
    assert os.environ["GCSFS_SUBSYSTEM_NNODES"] == "1"
    assert os.environ["GCSFS_SUBSYSTEM_NODE_RANK"] == "0"
    assert os.environ["GCSFS_SUBSYSTEM_MASTER_ADDR"] == "localhost"
    assert "GCSFS_SUBSYSTEM_MASTER_PORT" not in os.environ
    assert "GCSFS_SUBSYSTEM_NPROC_PER_NODE" not in os.environ
    assert "GCSFS_SUBSYSTEM_CHECKPOINT_EXPORT_ROOT" not in os.environ
    assert "GCSFS_SUBSYSTEM_CHECKPOINT_SOURCE_ROOT" not in os.environ
    assert "GCSFS_SUBSYSTEM_RESULTS_URI" not in os.environ

    # Explicit values export all env variables.
    args = run.parse_args(
        [
            "--group=checkpointing/pytorch",
            "--device=cuda",
            "--nnodes=2",
            "--node-rank=0",
            "--master-addr=leader-0.svc",
            "--master-port=29501",
            "--nproc-per-node=8",
            "--checkpoint-export-root=gs://export-bucket/root",
            "--checkpoint-source-root=gs://source-bucket/root",
            "--results-uri=gs://results-bucket/run-1",
        ]
        + _REQUIRED
    )
    run._setup_environment(args)
    assert os.environ["GCSFS_SUBSYSTEM_DEVICE"] == "cuda"
    assert os.environ["GCSFS_SUBSYSTEM_NNODES"] == "2"
    assert os.environ["GCSFS_SUBSYSTEM_NODE_RANK"] == "0"
    assert os.environ["GCSFS_SUBSYSTEM_MASTER_ADDR"] == "leader-0.svc"
    assert os.environ["GCSFS_SUBSYSTEM_MASTER_PORT"] == "29501"
    assert os.environ["GCSFS_SUBSYSTEM_NPROC_PER_NODE"] == "8"
    assert (
        os.environ["GCSFS_SUBSYSTEM_CHECKPOINT_EXPORT_ROOT"]
        == "gs://export-bucket/root"
    )
    assert (
        os.environ["GCSFS_SUBSYSTEM_CHECKPOINT_SOURCE_ROOT"]
        == "gs://source-bucket/root"
    )
    assert (
        os.environ["GCSFS_SUBSYSTEM_RESULTS_URI"]
        == "gs://results-bucket/run-1"
    )

    # TPU choice is also accepted.
    tpu_args = run.parse_args(
        ["--group=checkpointing/pytorch", "--device=tpu"] + _REQUIRED
    )
    run._setup_environment(tpu_args)
    assert os.environ["GCSFS_SUBSYSTEM_DEVICE"] == "tpu"


@pytest.mark.parametrize("node_rank", ["1", "2", "-1"])
def test_node_rank_nonzero_rejected(capsys, node_rank):
    with pytest.raises(SystemExit):
        run.parse_args(
            [
                "--group=checkpointing/pytorch",
                f"--node-rank={node_rank}",
            ]
            + _REQUIRED
        )
    err = capsys.readouterr().err
    assert "--node-rank" in err
    assert "worker" in err


def test_results_uri_copies_results(monkeypatch, tmp_path):
    for key in _ALL_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    results_dir = tmp_path / "local_results"
    dest_dir = tmp_path / "copied_results"
    results_uri = f"file://{dest_dir}"

    def fake_run_suite(suite_dir, out_dir, filter_expr=None):
        json_path = os.path.join(out_dir, "results.json")
        csv_path = os.path.join(out_dir, "results.csv")
        with open(json_path, "w") as f:
            f.write('{"benchmarks": []}')
        with open(csv_path, "w") as f:
            f.write("benchmark_case_id,score\ncase-1,10\n")
        return 0, csv_path

    def fake_scrape_amplification(csv_path, args):
        # Mutate the CSV during amplification scrape to verify the copy happens after scrape.
        with open(csv_path, "w") as f:
            f.write("benchmark_case_id,score,amplification\ncase-1,10,1.25\n")

    monkeypatch.setattr(run.cli, "run_suite", fake_run_suite)
    monkeypatch.setattr(run, "_scrape_amplification", fake_scrape_amplification)
    monkeypatch.setattr(run.report, "print_csv_to_shell", lambda csv_path: None)

    with pytest.raises(SystemExit) as exc_info:
        run.main(
            [
                "--group=checkpointing/pytorch",
                f"--results-dir={results_dir}",
                f"--results-uri={results_uri}",
            ]
            + _REQUIRED
        )
    assert exc_info.value.code == 0

    copied_csv = dest_dir / "results.csv"
    copied_json = dest_dir / "results.json"
    assert copied_csv.is_file()
    assert copied_json.is_file()
    assert (
        copied_csv.read_text()
        == "benchmark_case_id,score,amplification\ncase-1,10,1.25\n"
    )
    assert copied_json.read_text() == '{"benchmarks": []}'
