import dataclasses
import json
import logging

import fsspec
import pytest

pytest.importorskip("transformers")

from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.handoff import (
    Manifest,
    export_checkpoint,
    import_checkpoint,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.hosts import HostConfig
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.read.driver import (
    PyTorchCheckpointReadDriver,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.tests.test_parallelize import (
    _create_tiny_llama_config_dir,
    _params,
)
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.write.driver import (
    PyTorchCheckpointWriteDriver,
)


def _write_params(config_dir, **kwargs):
    return dataclasses.replace(
        _params(config_dir, **kwargs), scenario="checkpoint_write"
    )


def _seed_fake_checkpoint(prefix_dir):
    prefix_dir.mkdir(parents=True, exist_ok=True)
    (prefix_dir / ".metadata").write_bytes(b"metadata-bytes")
    (prefix_dir / "__0_0.distcp").write_bytes(b"rank0-bytes")
    (prefix_dir / "__1_0.distcp").write_bytes(b"rank1-bytes")


def test_export_then_import_roundtrip(tmp_path, monkeypatch):
    config_dir = _create_tiny_llama_config_dir(tmp_path, num_layers=8)
    world_size = 4

    export_root_dir = tmp_path / "export_root"
    export_root = export_root_dir.as_uri()
    monkeypatch.setenv("GCSFS_SUBSYSTEM_CHECKPOINT_EXPORT_ROOT", export_root)

    save_params = _write_params(
        config_dir,
        strategy="fsdp",
        dp=world_size,
        world_size=world_size,
        rounds=1,
    )
    save_prefix = (tmp_path / "save_ckpt").as_uri()
    write_driver = PyTorchCheckpointWriteDriver()
    save_result = write_driver.run(save_prefix, save_params)
    save_result.verify()

    exported_dir = export_root_dir / save_params.source_key()
    assert (exported_dir / "manifest.json").is_file()
    assert (exported_dir / ".metadata").is_file()
    assert len(list(exported_dir.glob("*.distcp"))) == world_size

    manifest = Manifest.from_dict(
        json.loads((exported_dir / "manifest.json").read_text(encoding="utf-8"))
    )
    assert manifest.model_id == config_dir
    assert manifest.layout == dataclasses.asdict(save_params.target_layout())
    assert manifest.device == "cpu"
    assert manifest.nnodes == 1

    # Now run a reshard load case (source=fsdp, target=tp) importing from export_root.
    monkeypatch.delenv("GCSFS_SUBSYSTEM_CHECKPOINT_EXPORT_ROOT", raising=False)
    monkeypatch.setenv("GCSFS_SUBSYSTEM_CHECKPOINT_SOURCE_ROOT", export_root)

    load_params = _params(
        config_dir,
        strategy="tp",
        tp=world_size,
        world_size=world_size,
        rounds=1,
    )
    load_params.source = {
        "strategy": "fsdp",
        "data_parallel_size": world_size,
        "world_size": world_size,
    }
    assert load_params.source_key() == save_params.source_key()

    load_prefix = (tmp_path / "load_ckpt").as_uri()
    read_driver = PyTorchCheckpointReadDriver()
    read_driver.setup(load_prefix, load_params)
    load_result = read_driver.run(load_prefix, load_params)
    load_result.verify()


def test_import_missing_key_errors(tmp_path):
    config_dir = _create_tiny_llama_config_dir(tmp_path)
    params = _params(config_dir, strategy="fsdp", dp=4, world_size=4)
    empty_root = (tmp_path / "empty_root").as_uri()
    dst_prefix = (tmp_path / "dst").as_uri()

    with pytest.raises(ValueError) as exc_info:
        import_checkpoint(empty_root, dst_prefix, params)
    assert params.source_key() in str(exc_info.value)


def test_import_layout_mismatch_errors(tmp_path):
    config_dir = _create_tiny_llama_config_dir(tmp_path)
    params = _write_params(config_dir, strategy="fsdp", dp=4, world_size=4)
    src_dir = tmp_path / "src_ckpt"
    _seed_fake_checkpoint(src_dir)

    export_root_dir = tmp_path / "export_root"
    export_root = export_root_dir.as_uri()
    export_checkpoint(src_dir.as_uri(), export_root, params, HostConfig())

    manifest_file = export_root_dir / params.source_key() / "manifest.json"
    orig_data = json.loads(manifest_file.read_text(encoding="utf-8"))

    # 1. Layout mismatch
    bad_layout_data = dict(orig_data)
    bad_layout_data["layout"] = {
        **orig_data["layout"],
        "strategy": "tp",
        "data_parallel_size": 1,
        "tensor_parallel_size": 4,
    }
    manifest_file.write_text(json.dumps(bad_layout_data), encoding="utf-8")

    read_params = _params(config_dir, strategy="fsdp", dp=4, world_size=4)
    with pytest.raises(ValueError) as exc_info:
        import_checkpoint(export_root, (tmp_path / "dst1").as_uri(), read_params)
    msg = str(exc_info.value)
    assert read_params.source_key() in msg
    assert "layout" in msg

    # 2. model_id mismatch
    bad_model_data = dict(orig_data)
    bad_model_data["model_id"] = "gs://wrong-bucket/wrong-model"
    manifest_file.write_text(json.dumps(bad_model_data), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        import_checkpoint(export_root, (tmp_path / "dst2").as_uri(), read_params)
    msg = str(exc_info.value)
    assert read_params.source_key() in msg
    assert "model_id" in msg

    # 3. seed mismatch
    bad_seed_data = dict(orig_data)
    bad_seed_data["seed"] = 999
    manifest_file.write_text(json.dumps(bad_seed_data), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        import_checkpoint(export_root, (tmp_path / "dst3").as_uri(), read_params)
    msg = str(exc_info.value)
    assert read_params.source_key() in msg
    assert "seed" in msg


def test_import_version_mismatch_warns(tmp_path, caplog):
    config_dir = _create_tiny_llama_config_dir(tmp_path)
    params = _write_params(config_dir, strategy="fsdp", dp=4, world_size=4)
    src_dir = tmp_path / "src_ckpt"
    _seed_fake_checkpoint(src_dir)

    export_root_dir = tmp_path / "export_root"
    export_root = export_root_dir.as_uri()
    export_checkpoint(src_dir.as_uri(), export_root, params, HostConfig())

    manifest_file = export_root_dir / params.source_key() / "manifest.json"
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    data["torch"] = "0.0.0-mismatch"
    manifest_file.write_text(json.dumps(data), encoding="utf-8")

    read_params = _params(config_dir, strategy="fsdp", dp=4, world_size=4)
    dst_dir = tmp_path / "dst_ckpt"
    with caplog.at_level(logging.WARNING):
        manifest = import_checkpoint(export_root, dst_dir.as_uri(), read_params)

    assert manifest.torch == "0.0.0-mismatch"
    assert "torch" in caplog.text
    assert "0.0.0-mismatch" in caplog.text
    assert (dst_dir / ".metadata").read_bytes() == b"metadata-bytes"
    assert (dst_dir / "__0_0.distcp").read_bytes() == b"rank0-bytes"


def test_source_device_column_from_manifest(tmp_path, monkeypatch):
    config_dir = _create_tiny_llama_config_dir(tmp_path)
    save_params = _write_params(config_dir, strategy="fsdp", dp=4, world_size=4)
    src_dir = tmp_path / "src_ckpt"
    _seed_fake_checkpoint(src_dir)

    export_root_dir = tmp_path / "export_root"
    export_root = export_root_dir.as_uri()
    gpu_host = HostConfig(device="cuda", nnodes=2, nproc_per_node=8)
    export_checkpoint(src_dir.as_uri(), export_root, save_params, gpu_host)

    monkeypatch.setenv("GCSFS_SUBSYSTEM_DEVICE", "cpu")
    monkeypatch.setenv("GCSFS_SUBSYSTEM_NNODES", "1")
    monkeypatch.delenv("GCSFS_SUBSYSTEM_CHECKPOINT_SOURCE_ROOT", raising=False)

    # Generate mode: source_device defaults to current host device ("cpu").
    gen_params = _params(config_dir, strategy="fsdp", dp=4, world_size=4)
    gen_cols = gen_params.extra_columns()
    assert gen_cols["device"] == "cpu"
    assert gen_cols["nnodes"] == 1
    assert gen_cols["source_device"] == "cpu"

    # Import mode via env var (even when setup() ran in a separate child process):
    monkeypatch.setenv("GCSFS_SUBSYSTEM_CHECKPOINT_SOURCE_ROOT", export_root)
    env_params = _params(config_dir, strategy="fsdp", dp=4, world_size=4)
    env_cols = env_params.extra_columns()
    assert env_cols["device"] == "cpu"
    assert env_cols["nnodes"] == 1
    assert env_cols["source_device"] == "cuda"

    # Direct in-process import_checkpoint also populates params.source_device:
    monkeypatch.delenv("GCSFS_SUBSYSTEM_CHECKPOINT_SOURCE_ROOT", raising=False)
    direct_params = _params(config_dir, strategy="fsdp", dp=4, world_size=4)
    import_checkpoint(export_root, (tmp_path / "dst").as_uri(), direct_params)
    assert direct_params.source_device == "cuda"
    direct_cols = direct_params.extra_columns()
    assert direct_cols["device"] == "cpu"
    assert direct_cols["source_device"] == "cuda"


def test_import_gs_server_side_copy_and_streaming_fallback(tmp_path, monkeypatch):
    config_dir = _create_tiny_llama_config_dir(tmp_path)
    save_params = _write_params(config_dir, strategy="fsdp", dp=4, world_size=4)
    src_dir = tmp_path / "src_ckpt"
    _seed_fake_checkpoint(src_dir)

    export_root_dir = tmp_path / "export_root"
    export_checkpoint(src_dir.as_uri(), export_root_dir.as_uri(), save_params, HostConfig())

    orig_url_to_fs = fsspec.core.url_to_fs
    local_fs = fsspec.filesystem("file")
    copy_calls = []
    fail_copy = False

    class _ProxyFS:
        def __getattr__(self, name):
            return getattr(local_fs, name)

        def copy(self, src, dst, *args, **kwargs):
            copy_calls.append((src, dst))
            if fail_copy:
                raise RuntimeError("server-side rewrite not supported")
            return local_fs.copy(src, dst, *args, **kwargs)

    proxy_fs = _ProxyFS()

    def fake_url_to_fs(url, **kwargs):
        if url.startswith("gs://export-bucket"):
            rel = url[len("gs://export-bucket") :].lstrip("/")
            path = str(export_root_dir / rel) if rel else str(export_root_dir)
            return proxy_fs, path
        if url.startswith("gs://case-bucket/"):
            rel = url[len("gs://case-bucket/") :]
            return proxy_fs, str(tmp_path / rel)
        return orig_url_to_fs(url, **kwargs)

    monkeypatch.setattr(fsspec.core, "url_to_fs", fake_url_to_fs)

    # 1. Server-side copy succeeds on regional gs:// -> gs://
    reg_params = _params(config_dir, strategy="fsdp", dp=4, world_size=4)
    reg_params.bucket_type = "regional"
    import_checkpoint("gs://export-bucket", "gs://case-bucket/dst_reg", reg_params)
    assert len(copy_calls) == 3
    assert (tmp_path / "dst_reg" / ".metadata").read_bytes() == b"metadata-bytes"

    # 2. Server-side copy raises -> falls back to streaming
    copy_calls.clear()
    fail_copy = True
    import_checkpoint("gs://export-bucket", "gs://case-bucket/dst_fallback", reg_params)
    assert len(copy_calls) == 3
    assert (tmp_path / "dst_fallback" / "__0_0.distcp").read_bytes() == b"rank0-bytes"

    # 3. *_rcu_z skips server-side copy and streams so gcsfs stages RAPID objects
    copy_calls.clear()
    fail_copy = False
    rcu_z_params = _params(config_dir, strategy="fsdp", dp=4, world_size=4)
    rcu_z_params.bucket_type = "standard_rcu_z"
    import_checkpoint("gs://export-bucket", "gs://case-bucket/dst_rcu_z", rcu_z_params)
    assert copy_calls == []
    assert (tmp_path / "dst_rcu_z" / "__1_0.distcp").read_bytes() == b"rank1-bytes"

