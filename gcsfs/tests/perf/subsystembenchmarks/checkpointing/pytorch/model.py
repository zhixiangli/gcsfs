import os
import tempfile

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

import gcsfs
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.values import fill_


def _resolve_model_config(model_id: str) -> AutoConfig:
    if model_id.startswith("gs://"):
        dir_name = os.path.basename(model_id.rstrip("/"))
        staged_dir = os.path.join("/tmp", dir_name)
        if os.path.exists(os.path.join(staged_dir, "config.json")):
            return AutoConfig.from_pretrained(staged_dir)

        fs = gcsfs.GCSFileSystem()
        config_path = f"{model_id.rstrip('/')}/config.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_config_file = os.path.join(tmp_dir, "config.json")
            fs.get_file(config_path, local_config_file)
            return AutoConfig.from_pretrained(tmp_dir)

    return AutoConfig.from_pretrained(model_id)


def build_model(params) -> nn.Module:
    cfg = _resolve_model_config(params.model_id)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    return model


def materialize(
    model: nn.Module, seed: int, device: torch.device | str = torch.device("cpu")
) -> None:
    model.to_empty(device=device)
    for fqn, p in model.named_parameters():
        p.requires_grad_(True)
        fill_(p, seed, fqn)


def _adamw(params, seed: int) -> torch.optim.AdamW:
    # A seed-dependent lr makes the saved param_groups differ between seeds,
    # so a reader that skips the BYTE_IO items fails verify_state too.
    return torch.optim.AdamW(params, lr=1e-3 * (seed + 1))


def expected_param_group(seed: int) -> dict:
    """The non-tensor param_groups values build_optimizer(..., seed) saves per FQN."""
    group = _adamw([torch.empty(0)], seed).param_groups[0]
    return {k: v for k, v in group.items() if k != "params"}


def build_optimizer(model: nn.Module, seed: int) -> torch.optim.AdamW:
    optimizer = _adamw(model.parameters(), seed)
    for fqn, p in model.named_parameters():
        state = optimizer.state[p]
        state["step"] = torch.tensor(seed, dtype=torch.int64)
        state["exp_avg"] = torch.empty_like(p)
        state["exp_avg_sq"] = torch.empty_like(p)
        fill_(state["exp_avg"], seed, f"{fqn}.exp_avg")
        fill_(state["exp_avg_sq"], seed, f"{fqn}.exp_avg_sq")
    return optimizer
