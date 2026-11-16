from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import logging
import posixpath
import shutil
import sys
from collections.abc import Mapping
from typing import Any, Optional

import fsspec

from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.common import (
    SETUP_SEED,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.hosts import HostConfig
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.layout import Layout
from gcsfs.tests.perf.subsystembenchmarks.dataloading.rcu import rcu_state

logger = logging.getLogger(__name__)

_CHECKPOINT_SUFFIXES = (".distcp", ".metadata")
_STREAM_CHUNK_BYTES = 16 * 1024 * 1024


def _torch_tpu_version() -> Optional[str]:
    try:
        return importlib.metadata.version("torch_tpu")
    except importlib.metadata.PackageNotFoundError:
        pass
    mod = sys.modules.get("torch_tpu")
    if mod is None:
        try:
            import torch_tpu as mod  # type: ignore[no-redef]
        except ImportError:
            return None
    ver = getattr(mod, "__version__", None)
    return str(ver) if ver is not None else None


def current_versions() -> dict[str, Optional[str]]:
    import gcsfs
    import torch
    import transformers

    return {
        "torch": str(torch.__version__),
        "torch_tpu": _torch_tpu_version(),
        "transformers": str(transformers.__version__),
        "gcsfs": str(gcsfs.__version__),
    }


@dataclasses.dataclass
class Manifest:
    model_id: str
    layout: dict[str, Any]
    seed: int
    device: str
    nnodes: int
    torch: str
    torch_tpu: Optional[str]
    transformers: str
    gcsfs: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Manifest:
        return cls(
            model_id=str(data["model_id"]),
            layout=dict(data["layout"]),
            seed=int(data["seed"]),
            device=str(data["device"]),
            nnodes=int(data["nnodes"]),
            torch=str(data["torch"]),
            torch_tpu=(
                str(data["torch_tpu"])
                if data.get("torch_tpu") is not None
                else None
            ),
            transformers=str(data["transformers"]),
            gcsfs=str(data["gcsfs"]),
        )

    @staticmethod
    def current_versions() -> dict[str, Optional[str]]:
        return current_versions()


def _copy_file(
    src_fs,
    src_file: str,
    dst_fs,
    dst_file: str,
    *,
    src_uri: str,
    dst_uri: str,
    allow_server_copy: bool = True,
) -> None:
    if (
        allow_server_copy
        and src_uri.startswith("gs://")
        and dst_uri.startswith("gs://")
    ):
        try:
            src_fs.copy(src_file, dst_file)
            return
        except Exception:
            pass
    parent = posixpath.dirname(dst_file)
    if parent:
        dst_fs.makedirs(parent, exist_ok=True)
    with src_fs.open(src_file, "rb") as rf, dst_fs.open(dst_file, "wb") as wf:
        shutil.copyfileobj(rf, wf, length=_STREAM_CHUNK_BYTES)


def read_manifest(root: str, key: str) -> Optional[Manifest]:
    src_uri = f"{root.rstrip('/')}/{key}"
    try:
        src_fs, src_root = fsspec.core.url_to_fs(src_uri)
        src_fs.invalidate_cache()
        manifest_path = f"{src_root.rstrip('/')}/manifest.json"
        if not src_fs.exists(manifest_path):
            return None
        with src_fs.open(manifest_path, "rb") as f:
            return Manifest.from_dict(json.loads(f.read().decode("utf-8")))
    except Exception:
        return None


def export_checkpoint(
    prefix: str,
    root: str,
    params,
    host: Optional[HostConfig] = None,
) -> str:
    resolved_host = host or HostConfig.from_env()
    key = params.source_key()
    dest_uri = f"{root.rstrip('/')}/{key}"

    src_fs, src_root = fsspec.core.url_to_fs(prefix)
    dst_fs, dst_root = fsspec.core.url_to_fs(dest_uri)

    src_fs.invalidate_cache()
    dst_fs.invalidate_cache()
    if dst_fs.exists(dst_root):
        dst_fs.rm(dst_root, recursive=True)
    dst_fs.makedirs(dst_root, exist_ok=True)

    ckpt_files = [
        f for f in src_fs.find(src_root) if f.endswith(_CHECKPOINT_SUFFIXES)
    ]
    for src_file in ckpt_files:
        rel = posixpath.relpath(src_file, src_root)
        dst_file = f"{dst_root.rstrip('/')}/{rel}"
        _copy_file(
            src_fs,
            src_file,
            dst_fs,
            dst_file,
            src_uri=prefix,
            dst_uri=dest_uri,
            allow_server_copy=True,
        )

    versions = current_versions()
    manifest = Manifest(
        model_id=params.model_id,
        layout=dataclasses.asdict(params.target_layout()),
        seed=SETUP_SEED,
        device=resolved_host.device,
        nnodes=resolved_host.nnodes,
        torch=versions["torch"] or "",
        torch_tpu=versions["torch_tpu"],
        transformers=versions["transformers"] or "",
        gcsfs=versions["gcsfs"] or "",
    )
    manifest_path = f"{dst_root.rstrip('/')}/manifest.json"
    with dst_fs.open(manifest_path, "wb") as f:
        f.write(json.dumps(manifest.to_dict(), indent=2, sort_keys=True).encode("utf-8"))
    dst_fs.invalidate_cache()
    return dest_uri


def import_checkpoint(root: str, prefix: str, params) -> Manifest:
    key = params.source_key()
    src_uri = f"{root.rstrip('/')}/{key}"
    manifest_uri = f"{src_uri}/manifest.json"

    src_fs, src_root = fsspec.core.url_to_fs(src_uri)
    src_fs.invalidate_cache()
    manifest_path = f"{src_root.rstrip('/')}/manifest.json"
    if not src_fs.exists(manifest_path):
        raise ValueError(
            f"Checkpoint key {key!r} not found under {root!r} (missing {manifest_uri})"
        )

    with src_fs.open(manifest_path, "rb") as f:
        manifest = Manifest.from_dict(json.loads(f.read().decode("utf-8")))

    if manifest.model_id != params.model_id:
        raise ValueError(
            f"Manifest mismatch for {key!r} on field 'model_id': "
            f"expected {params.model_id!r}, got {manifest.model_id!r}"
        )

    manifest_layout = Layout.from_mapping(manifest.layout)
    expected_layout = params.source_layout()
    if manifest_layout != expected_layout:
        raise ValueError(
            f"Manifest mismatch for {key!r} on field 'layout': "
            f"expected {expected_layout!r}, got {manifest_layout!r}"
        )

    if manifest.seed != SETUP_SEED:
        raise ValueError(
            f"Manifest mismatch for {key!r} on field 'seed': "
            f"expected {SETUP_SEED!r}, got {manifest.seed!r}"
        )

    current = current_versions()
    diffs = []
    for pkg in ("torch", "torch_tpu", "transformers", "gcsfs"):
        saved_ver = getattr(manifest, pkg)
        curr_ver = current.get(pkg)
        if saved_ver != curr_ver:
            diffs.append(f"{pkg}: manifest={saved_ver!r} vs current={curr_ver!r}")
    if diffs:
        logger.warning(
            "Checkpoint %r was exported with different package version(s): %s",
            key,
            ", ".join(diffs),
        )

    dst_fs, dst_root = fsspec.core.url_to_fs(prefix)
    dst_fs.invalidate_cache()
    if dst_fs.exists(dst_root):
        dst_fs.rm(dst_root, recursive=True)
    dst_fs.makedirs(dst_root, exist_ok=True)

    # *_rcu_z requires RAPID objects written through gcsfs's appendable writer
    # rather than GCS server-side rewrite (which preserves STANDARD storageClass).
    allow_server_copy = rcu_state(getattr(params, "bucket_type", None)) != "z"

    ckpt_files = [
        f for f in src_fs.find(src_root) if f.endswith(_CHECKPOINT_SUFFIXES)
    ]
    for src_file in ckpt_files:
        rel = posixpath.relpath(src_file, src_root)
        dst_file = f"{dst_root.rstrip('/')}/{rel}"
        _copy_file(
            src_fs,
            src_file,
            dst_fs,
            dst_file,
            src_uri=src_uri,
            dst_uri=prefix,
            allow_server_copy=allow_server_copy,
        )
    dst_fs.invalidate_cache()

    params.source_device = manifest.device
    return manifest
