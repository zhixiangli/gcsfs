import dataclasses

import fsspec
import pytest
from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.multiprocessing.spawn import ProcessRaisedException

pytest.importorskip("transformers")

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


def _total_tensor_bytes(prefix: str) -> int:
    md = FsspecReader(prefix).read_metadata().state_dict_metadata
    return sum(
        c.sizes.numel() * item_md.properties.dtype.itemsize
        for item_md in md.values()
        if isinstance(item_md, TensorStorageMetadata)
        for c in item_md.chunks
    )


def test_save_rounds_leave_one_checkpoint(tmp_path):
    config_dir = _create_tiny_llama_config_dir(tmp_path)
    params = _write_params(
        config_dir,
        strategy="fsdp",
        dp=4,
        world_size=4,
        rounds=3,
    )
    ckpt_dir = tmp_path / "ckpt"
    prefix = ckpt_dir.as_uri()
    driver = PyTorchCheckpointWriteDriver()

    # setup() is a no-op for write cases.
    driver.setup(prefix, params)
    assert not ckpt_dir.exists()

    # Seed a stale file before run() to verify round cleanup removes existing files.
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "__stale_0.distcp").write_bytes(b"stale")

    result = driver.run(prefix, params)
    result.verify()

    assert len(result.durations) == 3
    assert all(d > 0 for d in result.durations)

    fs, root = fsspec.core.url_to_fs(prefix)
    files = fs.find(root)
    ckpt_files = [f for f in files if driver.is_checkpoint_file(f)]
    assert len(ckpt_files) == len(files) == 5
    assert sum(f.endswith(".distcp") for f in ckpt_files) == 4
    assert sum(f.endswith(".metadata") for f in ckpt_files) == 1
    assert not any("__stale_0.distcp" in f for f in files)


def test_save_verify_detects_corruption(tmp_path):
    config_dir = _create_tiny_llama_config_dir(tmp_path)
    params = _write_params(
        config_dir,
        strategy="fsdp",
        dp=4,
        world_size=4,
        rounds=1,
    )
    ckpt_dir = tmp_path / "ckpt"
    prefix = ckpt_dir.as_uri()
    driver = PyTorchCheckpointWriteDriver()

    result = driver.run(prefix, params)

    # Corrupt the middle of the largest saved storage item after run() finishes.
    storage = FsspecReader(prefix).read_metadata().storage_data
    info = max(storage.values(), key=lambda s: s.length)
    path = ckpt_dir / info.relative_path
    data = bytearray(path.read_bytes())
    mid = info.offset + info.length // 2
    data[mid : mid + 16] = bytes(b ^ 0xFF for b in data[mid : mid + 16])
    path.write_bytes(bytes(data))

    with pytest.raises((ProcessRaisedException, RuntimeError), match="value mismatch"):
        result.verify()


def test_save_reports_write_skew(tmp_path):
    config_dir = _create_tiny_llama_config_dir(tmp_path)
    world_size = 4
    driver = PyTorchCheckpointWriteDriver()

    fsdp_params = _write_params(
        config_dir,
        strategy="fsdp",
        dp=world_size,
        world_size=world_size,
        rounds=1,
    )
    fsdp_prefix = (tmp_path / "ckpt_fsdp").as_uri()
    fsdp_result = driver.run(fsdp_prefix, fsdp_params)
    fsdp_result.verify()

    tp_params = _write_params(
        config_dir,
        strategy="tp",
        tp=world_size,
        world_size=world_size,
        rounds=1,
    )
    tp_prefix = (tmp_path / "ckpt_tp").as_uri()
    tp_result = driver.run(tp_prefix, tp_params)
    tp_result.verify()

    assert set(fsdp_result.extra_columns.keys()) == {"dcp_write_bytes_per_rank_max"}
    assert set(tp_result.extra_columns.keys()) == {"dcp_write_bytes_per_rank_max"}

    fsdp_total = _total_tensor_bytes(fsdp_prefix)
    tp_total = _total_tensor_bytes(tp_prefix)
    assert fsdp_total == tp_total

    fsdp_max = fsdp_result.extra_columns["dcp_write_bytes_per_rank_max"]
    tp_max = tp_result.extra_columns["dcp_write_bytes_per_rank_max"]

    # FSDP shards every parameter and optimizer state tensor evenly across ranks
    # (only tiny scalar step tensors are replicated and deduplicated).
    assert fsdp_max >= fsdp_total / world_size
    # TP leaves layernorm tensors replicated across all ranks; DCP deduplicates
    # them onto a subset of ranks rather than sharding them 4 ways, creating
    # strictly higher per-rank write skew than FSDP.
    assert tp_max > tp_total / world_size
    assert tp_max > fsdp_max
