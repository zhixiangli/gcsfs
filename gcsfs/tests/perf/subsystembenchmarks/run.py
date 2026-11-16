import argparse
import glob
import logging
import os
from datetime import datetime

from gcsfs.tests.perf.subsystembenchmarks._common import cli, report
from gcsfs.tests.perf.subsystembenchmarks.dataloading.bucket import BUCKET_TYPES
from gcsfs.tests.perf.subsystembenchmarks.dataloading.rapid_cache import (
    RAPID_CACHE_BUCKET_TYPES,
)
from gcsfs.tests.perf.subsystembenchmarks.dataloading.rcu import (
    RCU_BUCKET_TYPES,
    rcu_state,
)


def discover_groups():
    """Return runnable ``<subsystem>/<engine>`` benchmark groups."""
    root = os.path.dirname(os.path.abspath(__file__))
    return sorted(
        os.path.relpath(os.path.dirname(path), root)
        for path in glob.glob(os.path.join(root, "*", "*", "requirements.txt"))
    )


def _setup_environment(args):
    """Export per-case bucket, device, multi-host, checkpoint handoff, and results URI settings."""
    os.environ["GCSFS_SUBSYSTEM_BUCKET_PREFIX"] = args.bucket_prefix
    os.environ["GCSFS_SUBSYSTEM_BUCKET_TYPE"] = args.bucket_type
    os.environ["GCSFS_SUBSYSTEM_PROJECT"] = args.project
    os.environ["GCSFS_SUBSYSTEM_LOCATION"] = args.location
    os.environ["GCSFS_SUBSYSTEM_ZONE"] = args.zone or ""
    os.environ["GCSFS_SUBSYSTEM_SWEEP_AXES"] = args.sweep_axes
    if args.model_id:
        os.environ["GCSFS_SUBSYSTEM_MODEL_ID"] = args.model_id
    else:
        os.environ.pop("GCSFS_SUBSYSTEM_MODEL_ID", None)
    os.environ["GCSFS_SUBSYSTEM_DEVICE"] = args.device
    os.environ["GCSFS_SUBSYSTEM_NNODES"] = str(args.nnodes)
    os.environ["GCSFS_SUBSYSTEM_NODE_RANK"] = str(args.node_rank)
    os.environ["GCSFS_SUBSYSTEM_MASTER_ADDR"] = args.master_addr
    if args.master_port is not None:
        os.environ["GCSFS_SUBSYSTEM_MASTER_PORT"] = str(args.master_port)
    else:
        os.environ.pop("GCSFS_SUBSYSTEM_MASTER_PORT", None)
    if args.nproc_per_node is not None:
        os.environ["GCSFS_SUBSYSTEM_NPROC_PER_NODE"] = str(args.nproc_per_node)
    else:
        os.environ.pop("GCSFS_SUBSYSTEM_NPROC_PER_NODE", None)
    if args.checkpoint_export_root:
        os.environ["GCSFS_SUBSYSTEM_CHECKPOINT_EXPORT_ROOT"] = (
            args.checkpoint_export_root
        )
    else:
        os.environ.pop("GCSFS_SUBSYSTEM_CHECKPOINT_EXPORT_ROOT", None)
    if args.checkpoint_source_root:
        os.environ["GCSFS_SUBSYSTEM_CHECKPOINT_SOURCE_ROOT"] = (
            args.checkpoint_source_root
        )
    else:
        os.environ.pop("GCSFS_SUBSYSTEM_CHECKPOINT_SOURCE_ROOT", None)
    if args.results_uri:
        os.environ["GCSFS_SUBSYSTEM_RESULTS_URI"] = args.results_uri
    else:
        os.environ.pop("GCSFS_SUBSYSTEM_RESULTS_URI", None)
    os.environ["GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT"] = "true"
    os.environ["STORAGE_EMULATOR_HOST"] = "https://storage.googleapis.com"
    if rcu_state(args.bucket_type) == "z":
        # *_rcu_z stages and saves RAPID objects, which gcsfs writes to RCU buckets
        # only with rcu_rapid_write. The suite's processes read it from the env.
        os.environ["FSSPEC_GCS_RCU_RAPID_WRITE"] = "true"
    else:
        os.environ.pop("FSSPEC_GCS_RCU_RAPID_WRITE", None)


def _build_parser():
    parser = argparse.ArgumentParser(description="Run gcsfs subsystem benchmarks.")
    parser.add_argument(
        "--group", required=True, help="e.g. dataloading/huggingface_datasets"
    )
    parser.add_argument(
        "--sweep-axes",
        default="",
        help="whitespace-separated config sweep axes; baseline is always included",
    )
    parser.add_argument(
        "--filter",
        default="",
        help="pytest -k filter expression to limit case execution",
    )
    parser.add_argument(
        "--bucket-prefix",
        required=True,
        help="name prefix for the per-case buckets this run creates",
    )
    parser.add_argument(
        "--project", required=True, help="GCP project owning buckets and metrics"
    )
    parser.add_argument(
        "--location", required=True, help="bucket region, e.g. us-central1"
    )
    parser.add_argument(
        "--bucket-type",
        choices=BUCKET_TYPES,
        default="regional",
        help="storage tier used by every case in the run",
    )
    parser.add_argument(
        "--zone",
        help="placement zone; required when --bucket-type is zonal, rapid_cache_* or "
        "*_rcu_*",
    )
    parser.add_argument(
        "--model-id",
        default="",
        help="model id override (e.g. gs://path/to/model or hf-repo-id)",
    )
    parser.add_argument(
        "--amplification-wait",
        type=int,
        default=300,
        help="seconds to wait before the read-amplification scrape",
    )
    parser.add_argument(
        "--amplification-retry-wait",
        type=int,
        default=60,
        help="seconds to wait before retrying missing Monitoring metrics",
    )
    parser.add_argument(
        "--require-amplification",
        action="store_true",
        help="fail when eligible rows still lack amplification metrics",
    )
    parser.add_argument(
        "--results-dir",
        default="",
        help="directory for results.json/results.csv (default: __run__/<timestamp>); "
        "concurrent runs need distinct directories",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "tpu"),
        default="cpu",
        help="compute device for the run",
    )
    parser.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="number of hosts in the run",
    )
    parser.add_argument(
        "--node-rank",
        type=int,
        default=0,
        help="rank of this host (0 is the leader; workers use the worker module)",
    )
    parser.add_argument(
        "--master-addr",
        default="localhost",
        help="leader host address for multi-host rendezvous",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=None,
        help="leader TCPStore port for multi-host rendezvous",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=None,
        help="processes per host (defaults to per-device count)",
    )
    parser.add_argument(
        "--checkpoint-export-root",
        default="",
        help="root URI to export saved checkpoints and manifests",
    )
    parser.add_argument(
        "--checkpoint-source-root",
        default="",
        help="root URI to import source checkpoints from",
    )
    parser.add_argument(
        "--results-uri",
        default="",
        help="URI to copy results_dir to after amplification scrape",
    )
    return parser


def parse_args(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    zone_types = ("zonal", *RAPID_CACHE_BUCKET_TYPES, *RCU_BUCKET_TYPES)
    if args.bucket_type in zone_types and not args.zone:
        parser.error(f"--zone is required when --bucket-type={args.bucket_type}")
    if args.amplification_wait < 0:
        parser.error("--amplification-wait must be >= 0")
    if args.amplification_retry_wait < 0:
        parser.error("--amplification-retry-wait must be >= 0")
    if args.node_rank != 0:
        parser.error("--node-rank must be 0 for run.py; workers use the worker module")
    if args.nnodes < 1:
        parser.error("--nnodes must be >= 1")
    if args.nproc_per_node is not None and args.nproc_per_node < 1:
        parser.error("--nproc-per-node must be >= 1")
    if args.master_port is not None and args.master_port < 1:
        parser.error("--master-port must be >= 1")
    groups = discover_groups()
    if args.group not in groups:
        parser.error(f"unknown --group {args.group!r}; available: {', '.join(groups)}")
    return args


_AMPLIFICATION_COLS = (
    "gcs_bucket_name",
    "measurement_window_start_unix_seconds",
    "measurement_window_end_unix_seconds",
)

_AMPLIFICATION_COLS_DATASET = ("dataset_size_bytes",)
_AMPLIFICATION_COLS_CHECKPOINT = ("checkpoint_physical_size_bytes",)


def _csv_has_amplification_inputs(csv_path):
    import csv

    try:
        with open(csv_path, newline="") as file:
            header = next(csv.reader(file), [])
        if not all(column in header for column in _AMPLIFICATION_COLS):
            return False
        return any(column in header for column in _AMPLIFICATION_COLS_DATASET) or any(
            column in header for column in _AMPLIFICATION_COLS_CHECKPOINT
        )
    except Exception:
        return False


def enrich_amplification_with_retry(csv_path, project, client, *, retry_wait, sleep):
    from gcsfs.tests.perf.subsystembenchmarks.dataloading import amplification

    result = amplification.enrich_csv(csv_path, project, client=client)
    if result.missing_buckets:
        if retry_wait:
            sleep(retry_wait)
        result = amplification.enrich_csv(csv_path, project, client=client)
    return result


def require_complete_amplification(result):
    if result.missing_buckets:
        raise RuntimeError(
            "read-amplification metrics missing for buckets: "
            + ", ".join(result.missing_buckets)
        )


def _scrape_amplification(csv_path, args):
    if not _csv_has_amplification_inputs(csv_path):
        return None
    try:
        import time

        from google.cloud import monitoring_v3

        if args.amplification_wait:
            time.sleep(args.amplification_wait)
        client = monitoring_v3.MetricServiceClient()
        result = enrich_amplification_with_retry(
            csv_path,
            args.project,
            client,
            retry_wait=args.amplification_retry_wait,
            sleep=time.sleep,
        )
        logging.info(
            "amplification enriched %s/%s rows in %s",
            result.enriched,
            result.eligible,
            csv_path,
        )
        if result.missing_buckets:
            logging.warning(
                "amplification still missing for buckets: %s",
                ", ".join(result.missing_buckets),
            )
        if args.require_amplification:
            require_complete_amplification(result)
        return result
    except Exception as exc:
        if args.require_amplification:
            raise RuntimeError(f"required amplification scrape failed: {exc}") from exc
        logging.warning("amplification scrape skipped: %s", exc)
        return None


def _copy_results(results_dir, results_uri):
    if not results_uri:
        return
    import fsspec

    fs, fs_path = fsspec.core.url_to_fs(results_uri)
    fs.put(
        os.path.join(results_dir, ""),
        fs_path.rstrip("/") + "/",
        recursive=True,
    )


def main(argv=None):
    args = parse_args(argv)
    _setup_environment(args)
    os.environ["GCSFS_SUBSYSTEM_GROUP"] = args.group
    suite_dir = os.path.join(os.path.dirname(__file__), args.group)
    if args.results_dir:
        results_dir = args.results_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_dir = os.path.join(os.path.dirname(__file__), "__run__", timestamp)
    os.makedirs(results_dir, exist_ok=True)

    rc, csv_path = cli.run_suite(suite_dir, results_dir, filter_expr=args.filter)
    if csv_path is None:
        logging.error("no benchmark results produced by group %s", args.group)
        raise SystemExit(rc or 1)
    _scrape_amplification(csv_path, args)
    _copy_results(results_dir, args.results_uri)
    report.print_csv_to_shell(csv_path)
    logging.info("subsystembenchmarks run rc=%s csv=%s", rc, csv_path)
    raise SystemExit(rc)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
