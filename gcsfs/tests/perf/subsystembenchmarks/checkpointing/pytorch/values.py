import ctypes
import hashlib
import math

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

_C1 = ctypes.c_int64(0x9E3779B97F4A7C15).value
_C2 = ctypes.c_int64(0xBF58476D1CE4E5B9).value
_C3 = ctypes.c_int64(0x94D049BB133111EB).value
_MASK34 = (1 << 34) - 1
_MASK37 = (1 << 37) - 1
_MASK33 = (1 << 33) - 1
_CHUNK_ELEMENTS = 1 << 26


def fqn_key(seed: int, fqn: str) -> int:
    digest = hashlib.sha256(f"{seed}|{fqn}".encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=True)


def local_region(t: torch.Tensor) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if isinstance(t, DTensor):
        _, offset = compute_local_shape_and_global_offset(
            t.shape, t.device_mesh, t.placements
        )
        return tuple(t.shape), tuple(offset)
    return tuple(t.shape), (0,) * t.ndim


def _splitmix64_bf16(z: torch.Tensor, key: int, dtype: torch.dtype) -> torch.Tensor:
    z.bitwise_xor_(key)
    z.add_(_C1)
    z.bitwise_xor_((z >> 30) & _MASK34).mul_(_C2)
    z.bitwise_xor_((z >> 27) & _MASK37).mul_(_C3)
    z.bitwise_xor_((z >> 31) & _MASK33)
    bits = ((z >> 48) & -0x8000) | 0x3C00 | ((z >> 56) & 0x7F)
    bf16 = bits.to(torch.int16).view(torch.bfloat16)
    return bf16 if dtype == torch.bfloat16 else bf16.to(dtype)


def _fill_region(
    out: torch.Tensor,
    local_shape: tuple[int, ...],
    offset: tuple[int, ...],
    strides: list[int],
    key: int,
    dtype: torch.dtype,
    chunk_elements: int,
) -> None:
    numel = math.prod(local_shape)
    if numel <= chunk_elements:
        ndim = len(local_shape)
        if ndim == 0:
            idx = torch.zeros(1, dtype=torch.int64)
            out.copy_(_splitmix64_bf16(idx, key, dtype).reshape(()))
            return
        idx = torch.zeros(local_shape, dtype=torch.int64)
        for d, (n, off, stride) in enumerate(zip(local_shape, offset, strides)):
            shape = [1] * ndim
            shape[d] = n
            axis = torch.arange(off, off + n, dtype=torch.int64) * stride
            idx.add_(axis.reshape(shape))
        out.copy_(_splitmix64_bf16(idx.reshape(-1), key, dtype).reshape(local_shape))
        return

    d = next(i for i, size in enumerate(local_shape) if size > 1)
    slice_numel = numel // local_shape[d]
    step = max(1, chunk_elements // slice_numel)
    for s0 in range(0, local_shape[d], step):
        s1 = min(s0 + step, local_shape[d])
        sub_shape = (*local_shape[:d], s1 - s0, *local_shape[d + 1 :])
        sub_offset = (*offset[:d], offset[d] + s0, *offset[d + 1 :])
        _fill_region(
            out.narrow(d, s0, s1 - s0),
            sub_shape,
            sub_offset,
            strides,
            key,
            dtype,
            chunk_elements,
        )


def expected(
    local_shape: tuple[int, ...],
    global_shape: tuple[int, ...],
    offset: tuple[int, ...],
    seed: int,
    fqn: str,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    local_shape = tuple(local_shape)
    global_shape = tuple(global_shape)
    offset = tuple(offset)
    ndim = len(local_shape)
    if len(global_shape) != ndim or len(offset) != ndim:
        raise ValueError(
            f"Rank mismatch: local_shape={local_shape}, "
            f"global_shape={global_shape}, offset={offset}"
        )

    out = torch.empty(local_shape, dtype=dtype, device="cpu")
    if out.numel() == 0:
        return out

    strides = [1] * ndim
    for d in range(ndim - 2, -1, -1):
        strides[d] = strides[d + 1] * global_shape[d + 1]

    _fill_region(
        out,
        local_shape,
        offset,
        strides,
        fqn_key(seed, fqn),
        dtype,
        _CHUNK_ELEMENTS,
    )
    return out


def fill_(t: torch.Tensor, seed: int, fqn: str) -> None:
    global_shape, offset = local_region(t)
    local = t.to_local() if isinstance(t, DTensor) else t
    vals = expected(
        tuple(local.shape), global_shape, offset, seed, fqn, dtype=local.dtype
    )
    with torch.no_grad():
        local.copy_(vals)
