from __future__ import annotations

import hashlib

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from transformers import AutoModelForCausalLM

from gcsfs.tests.perf.subsystembenchmarks._common.spawn import ensure_open_file_limit
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.model_staging import (  # noqa: F401
    _has_safetensors,
    model_dir,
    stage_model,
)


def load_weights(model_id: str) -> dict[str, torch.Tensor]:
    ensure_open_file_limit()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir(model_id), dtype=torch.bfloat16, local_files_only=True
    )
    state = model.state_dict()
    del model
    for tensor in state.values():
        tensor.share_memory_()
    return state


def _random_slice(
    seed: int,
    key: str,
    global_shape: tuple[int, ...],
    offsets: tuple[int, ...],
    local_shape: tuple[int, ...],
    device: torch.device | str,
) -> torch.Tensor:
    dev = torch.device(device)
    digest = hashlib.sha256(f"{seed}|{key}".encode()).digest()
    gen = torch.Generator(device="cpu").manual_seed(
        int.from_bytes(digest[:8], "little")
    )
    ndim = len(global_shape)
    if ndim == 0:
        return torch.randn((), generator=gen, dtype=torch.float32, device="cpu").to(dev)
    scale = ndim**-0.5
    out: torch.Tensor | None = None
    for d, (full_n, off, local_n) in enumerate(zip(global_shape, offsets, local_shape)):
        axis = (
            torch.randn((full_n,), generator=gen, dtype=torch.float32, device="cpu")
            .mul_(scale)[off : off + local_n]
            .to(dev)
        )
        view_shape = [1] * ndim
        view_shape[d] = local_n
        term = axis.reshape(view_shape)
        out = term.expand(local_shape).clone() if out is None else out.add_(term)
    assert out is not None
    return out


def random_local(
    seed: int,
    key: str,
    target: torch.Tensor,
    device: torch.device | str,
) -> torch.Tensor:
    if isinstance(target, DTensor):
        local_shape, offsets = compute_local_shape_and_global_offset(
            target.shape, target.device_mesh, target.placements
        )
    else:
        local_shape = tuple(target.shape)
        offsets = (0,) * target.ndim
    return _random_slice(
        seed, key, tuple(target.shape), tuple(offsets), tuple(local_shape), device
    )


def random_full(
    seed: int,
    key: str,
    shape: torch.Size | tuple[int, ...],
    device: torch.device | str,
) -> torch.Tensor:
    shape_t = tuple(shape)
    return _random_slice(seed, key, shape_t, (0,) * len(shape_t), shape_t, device)


def local_slice(full: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if not isinstance(t, DTensor):
        return full
    shape, offset = compute_local_shape_and_global_offset(
        t.shape, t.device_mesh, t.placements
    )
    slices = tuple(slice(o, o + s) for s, o in zip(shape, offset))
    return full[slices]
