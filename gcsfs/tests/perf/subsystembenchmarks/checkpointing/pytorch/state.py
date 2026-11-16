import google_crc32c
import torch
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
from torch.distributed.tensor import DTensor

from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.values import (
    expected,
    local_region,
)

# Keys param_groups by FQN. Unflattened, it is one list per rank that differs across
# PP stages, and DCP saves only one rank's copy of it.
STATE_DICT_OPTIONS = StateDictOptions(flatten_optimizer_state_dict=True)


def get_fqn_state_dict(model, optim) -> dict:
    model_state, optim_state = get_state_dict(model, optim, options=STATE_DICT_OPTIONS)
    return {"model": model_state, "optim": optim_state}


def _tensors(state, prefix=""):
    items = state.items() if isinstance(state, dict) else enumerate(state)
    for key, value in items:
        name = f"{prefix}{key}"
        if isinstance(value, (DTensor, torch.Tensor)):
            yield name, value
        elif isinstance(value, (dict, list, tuple)):
            yield from _tensors(value, f"{name}.")


def _local_tensors(state, prefix=""):
    for name, value in _tensors(state, prefix):
        yield name, value.to_local() if isinstance(value, DTensor) else value


def _state_name_to_fqn(state: dict, name: str) -> str:
    if isinstance(state, dict):
        if name.startswith("model.") and isinstance(
            state.get("model"), (dict, list, tuple)
        ):
            return name[len("model.") :]
        if isinstance(state.get("optim"), (dict, list, tuple)):
            if name.startswith("optim.state."):
                return name[len("optim.state.") :]
            if name.startswith("optim."):
                return name[len("optim.") :]
    if name.startswith("state."):
        return name[len("state.") :]
    return name


def verify_state(state: dict, seed: int) -> list[str]:
    bad: list[str] = []
    for name, t in _tensors(state):
        fqn = _state_name_to_fqn(state, name)
        local_cpu = (t.to_local() if isinstance(t, DTensor) else t).detach().cpu()
        if fqn == "step" or fqn.endswith(".step"):
            if not torch.equal(local_cpu, torch.tensor(seed, dtype=local_cpu.dtype)):
                bad.append(name)
            continue
        global_shape, offset = local_region(t)
        exp = expected(
            tuple(local_cpu.shape),
            global_shape,
            offset,
            seed,
            fqn,
            dtype=local_cpu.dtype,
        )
        if not torch.equal(local_cpu, exp):
            bad.append(name)
    return sorted(bad)


def expected_bytes_requested(state: dict) -> int:
    return sum(t.numel() * t.element_size() for _, t in _local_tensors(state))


def state_checksum(state: dict) -> dict[str, str]:
    # Hardware CRC32C runs at several GB/s per thread, so checking a rank's
    # tensors takes about a second where SHA-256 took tens of seconds.
    return {
        name: f"{google_crc32c.value(_raw_bytes(t)):08x}"
        for name, t in _local_tensors(state)
    }


def _raw_bytes(t: torch.Tensor):
    return t.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
