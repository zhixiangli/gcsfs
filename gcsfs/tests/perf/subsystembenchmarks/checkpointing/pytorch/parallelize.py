import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Shard, distribute_tensor

from gcsfs.tests.perf.subsystembenchmarks.checkpointing._llama_tp import apply_llama_tp
from gcsfs.tests.perf.subsystembenchmarks.checkpointing.pytorch.layout import (
    STRATEGY_MESH_DIMS,
    Layout,
)


def build_mesh(layout: Layout, device_type: str = "cpu") -> DeviceMesh:
    dims = layout.mesh_dims()
    return init_device_mesh(
        device_type,
        tuple(size for _, size in dims),
        mesh_dim_names=tuple(name for name, _ in dims),
    )


def _keep(modules: nn.ModuleList, begin: int, count: int) -> nn.ModuleDict:
    # Keyed by the original index so FQNs stay canonical, e.g. model.layers.16.
    return nn.ModuleDict({str(i): modules[i] for i in range(begin, begin + count)})


def split_stage(model: nn.Module, stage: int, num_stages: int) -> None:
    """Keeps only this stage's modules. State-dict layout only: the ModuleDict
    below breaks HF forward, which iterates layers as a list."""
    if getattr(model.config, "tie_word_embeddings", False):
        raise ValueError("split_stage does not support tie_word_embeddings")
    inner = getattr(model, "model", model)
    layers = inner.layers
    if len(layers) % num_stages:
        raise ValueError(f"{len(layers)} layers do not split into {num_stages} stages")
    per_stage = len(layers) // num_stages
    inner.layers = _keep(layers, stage * per_stage, per_stage)
    if stage != 0:
        inner.embed_tokens = None
    if stage != num_stages - 1:
        inner.norm = None
        if hasattr(model, "lm_head"):
            model.lm_head = None


def shard_experts(model: nn.Module, mesh: DeviceMesh) -> set[nn.Parameter]:
    """Shards fused 3D expert parameters along dim 0 (expert dim) over ``mesh``.

    In transformers 5, ``OlmoeSparseMoeBlock.experts`` (``OlmoeExperts``) stores
    fused 3D parameters with ``num_experts`` at dim 0 rather than a per-expert
    ``nn.ModuleList``: ``mlp.experts.gate_up_proj`` has shape
    ``(num_experts, 2 * intermediate_size, hidden_size)`` and
    ``mlp.experts.down_proj`` has shape ``(num_experts, hidden_size, intermediate_size)``.
    """
    num_ranks = mesh.size()
    sharded: set[nn.Parameter] = set()
    for layer in getattr(model, "model", model).layers.children():
        experts = getattr(layer.mlp, "experts", None)
        if experts is None:
            continue
        for name, param in list(experts.named_parameters(recurse=False)):
            num_experts = param.shape[0]
            if num_experts % num_ranks:
                raise ValueError(
                    f"{num_experts} experts do not split into {num_ranks} ranks"
                )
            sharded_param = nn.Parameter(
                distribute_tensor(param, mesh, [Shard(0)]),
                requires_grad=param.requires_grad,
            )
            setattr(experts, name, sharded_param)
            sharded.add(sharded_param)
    return sharded


def parallelize(model: nn.Module, mesh: DeviceMesh, layout: Layout) -> nn.Module:
    if layout.strategy not in STRATEGY_MESH_DIMS:
        raise ValueError(f"Unsupported strategy {layout.strategy!r}")
    dims = STRATEGY_MESH_DIMS[layout.strategy]
    if "pp" in dims:
        split_stage(model, mesh["pp"].get_local_rank(), mesh["pp"].size())
    if "tp" in dims:
        apply_llama_tp(model, mesh["tp"])
    ignored: set[nn.Parameter] = set()
    if layout.strategy == "ep":
        # EP reuses the dp_shard ranks, sharding fused 3D experts along dim 0.
        ignored = shard_experts(model, mesh["dp_shard"])

    dp_dims = tuple(d for d in dims if d.startswith("dp_"))
    if dp_dims:
        dp_mesh = mesh[dp_dims]
        for layer in getattr(model, "model", model).layers.children():
            placement_fn = None
            if layout.strategy != "ep" and hasattr(layer.mlp, "experts"):
                placement_fn = (
                    lambda param, _exp=set(layer.mlp.experts.parameters()): Shard(1)
                    if param in _exp
                    else Shard(0)
                )
            fully_shard(
                layer,
                mesh=dp_mesh,
                shard_placement_fn=placement_fn,
                ignored_params=ignored,
            )
        fully_shard(model, mesh=dp_mesh, ignored_params=ignored)
    return model
