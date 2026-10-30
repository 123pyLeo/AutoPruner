"""
Boundary adapter (LLM-Streamline-lite) for layer-pruned VLA recovery.

Ported from openvla-oft/prismatic/training/replacement_adapter.py and adapted
for Qwen2.5-VL backbone used inside starVLA frameworks.

For each contiguous removed-layer segment in the LLM stack, insert a tiny gated
MLP at the position of the *kept layer immediately following* the segment. The
adapter acts on the residual stream input to that kept layer:

    h' = h + up( silu(down(rmsnorm(h))) )

up.weight is zero-initialized so the adapter starts as identity (= the
no-recovery pruned student). Trainable params per adapter
~= 2 * hidden_size * bottleneck (default bottleneck=1024 -> 8M on a 4096-d
model, 7M on a 3584-d model, 4M on a 2048-d model, etc.).

Usage:
    from starVLA.model.modules.adapter import (
        install_adapters, adapter_parameters, save_adapters, load_adapters_into_layers,
    )

    layers = framework.qwen_vl_interface.model.model.language_model.layers
    adapters = install_adapters(
        layers, remove_layers=[8, 9, ..., 23], total_layers=28,
        hidden_size=3584, bottleneck=1024, dtype=torch.bfloat16,
    )
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps).to(x.dtype)
        return x * self.weight


class Adapter(nn.Module):
    """Tiny gated MLP with residual + (optionally) zero-init up_proj.

    With ``up_init_std == 0`` (default), the adapter is exactly identity at
    init. That preserves the pretrained pruned-only baseline at step 0, but
    creates a gradient deadlock: because ``∂out/∂{down.weight, norm.weight}``
    flows through ``up.weight``, those two tensors receive zero gradient until
    ``up.weight`` itself becomes non-zero — which can be very slow.

    Setting a small ``up_init_std`` breaks the deadlock by giving every
    adapter parameter non-zero gradient from step 1. Step-0 output will no
    longer be exactly identity, but distillation losses recover fast.
    """

    def __init__(self, hidden_size: int, bottleneck: int = 1024,
                 up_init_std: float = 0.0, residual: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.bottleneck = bottleneck
        self.residual = residual
        self.norm = _RMSNorm(hidden_size)
        self.down = nn.Linear(hidden_size, bottleneck, bias=False)
        self.act = nn.SiLU()
        self.up = nn.Linear(bottleneck, hidden_size, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        if up_init_std > 0:
            nn.init.normal_(self.up.weight, std=up_init_std)
        else:
            nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.up(self.act(self.down(self.norm(x))))
        if self.residual:
            return x + out
        return out


class _LayerWithAdapter(nn.Module):
    """Wraps a kept transformer decoder layer with a pre-applied adapter."""

    def __init__(self, adapter: Adapter, inner_layer: nn.Module):
        super().__init__()
        self.adapter = adapter
        self.inner = inner_layer
        # Forward-compat: expose attributes the surrounding model touches
        # (Qwen2.5-VL reads `self_attn` on each layer for layer_idx mutation).
        if hasattr(inner_layer, "self_attn"):
            self.self_attn = inner_layer.self_attn

    def forward(self, hidden_states, *args, **kwargs):
        adapted = self.adapter(hidden_states)
        return self.inner(adapted, *args, **kwargs)


def find_boundary_indices(remove_layers: List[int], total_layers: int) -> Dict[int, int]:
    """
    Map student-layer index -> original-layer index for each kept layer that
    immediately follows a removed segment.

    Example: total_layers=32, remove_layers=[8..23]
        kept = [0..7, 24..31] (16 layers in the student)
        boundary at student idx 8 (= original 24): previous original index was 7
        -> returns {8: 24}
    """
    remove_set = set(remove_layers)
    kept = [i for i in range(total_layers) if i not in remove_set]
    boundaries: Dict[int, int] = {}
    for j, orig in enumerate(kept):
        prev_orig = kept[j - 1] if j > 0 else -1
        if (orig - prev_orig) > 1:
            boundaries[j] = orig
    return boundaries


def install_adapters(
    layers: nn.ModuleList,
    remove_layers: List[int],
    total_layers: int,
    hidden_size: int,
    bottleneck: int = 1024,
    dtype: torch.dtype = torch.bfloat16,
    up_init_std: float = 0.0,
    residual: bool = True,
) -> Dict[int, Adapter]:
    """
    After `layers` has already been pruned (i.e. it contains only the kept
    layers, length == total_layers - len(remove_layers)), wrap each boundary
    kept-layer with an Adapter. Returns student_index -> Adapter dict.

    Pass the resulting dict to `adapter_parameters` for the optimizer and to
    `save_adapters` / `load_adapters_into_layers` for checkpointing.
    """
    expected = total_layers - len(set(remove_layers))
    assert len(layers) == expected, (
        f"layers={len(layers)} but kept count={expected}; "
        f"did you prune before calling install_adapters?"
    )
    boundaries = find_boundary_indices(remove_layers, total_layers=total_layers)
    adapters: Dict[int, Adapter] = {}
    for j in sorted(boundaries.keys()):
        a = Adapter(
            hidden_size=hidden_size,
            bottleneck=bottleneck,
            up_init_std=up_init_std,
            residual=residual,
        ).to(dtype=dtype)
        wrapped = _LayerWithAdapter(a, layers[j]).to(dtype=dtype)
        layers[j] = wrapped
        adapters[j] = a
    return adapters


def adapter_parameters(adapters: Dict[int, Adapter]) -> Iterable[nn.Parameter]:
    for j in sorted(adapters.keys()):
        for p in adapters[j].parameters():
            yield p


def save_adapters(adapters: Dict[int, Adapter], path: str, meta: dict | None = None) -> None:
    state = {
        "boundaries": {int(j): adapters[j].state_dict() for j in sorted(adapters.keys())},
        "meta": meta or {},
    }
    torch.save(state, path)


def load_adapters_into_layers(
    layers: nn.ModuleList,
    remove_layers: List[int],
    adapter_path: str,
    total_layers: int,
    hidden_size: int,
    bottleneck: int = 1024,
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[int, Adapter]:
    """Convenience for eval: install fresh adapters into already-pruned `layers`
    then load saved weights."""
    adapters = install_adapters(
        layers, remove_layers,
        total_layers=total_layers,
        hidden_size=hidden_size, bottleneck=bottleneck,
        dtype=dtype,
    )
    state = torch.load(adapter_path, map_location="cpu")
    for j_str, sd in state["boundaries"].items():
        j = int(j_str)
        if j not in adapters:
            raise KeyError(f"adapter for student layer {j} not present in installed set")
        adapters[j].load_state_dict(sd)
    device = next(layers[0].parameters()).device
    for a in adapters.values():
        a.to(device=device, dtype=dtype)
    return adapters
