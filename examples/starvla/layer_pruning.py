"""
Layer pruning utilities for the Qwen2.5-VL backbone used by starVLA frameworks.

`apply_layer_mask` physically removes the given decoder-layer indices from the
language model stack and re-numbers the surviving `self_attn.layer_idx`
attributes so HuggingFace's KV-cache and attention masking keep working.

This mirrors openvla-oft/vla-scripts/finetune_phase2_only.py::apply_layer_mask
but is rewritten for Qwen2.5-VL's module layout. The set of accepted layout
paths is intentionally small but extensible: try the path that exists.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch.nn as nn


def _locate_layers(model: nn.Module) -> nn.ModuleList:
    """
    Return the nn.ModuleList that holds the LLM decoder layers inside a
    Qwen2.5-VL-style backbone, regardless of HF transformers version.

    Layout for Qwen2_5_VLForConditionalGeneration:
        * older HF:  model.model.layers
        * newer HF:  model.model.language_model.layers
    """
    candidates = [
        # (module-path-as-tuple, label-for-error)
        (("model", "language_model", "layers"), "model.language_model.layers"),
        (("model", "layers"), "model.layers"),
    ]
    for path, _label in candidates:
        obj = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and isinstance(obj, nn.ModuleList):
            return obj
    raise AttributeError(
        "Could not locate decoder layers on the given Qwen2.5-VL model. "
        f"Tried: {[c[1] for c in candidates]}"
    )


def parse_mask(mask: str | Sequence[int] | None) -> List[int]:
    """Accept '8,9,10' or [8,9,10] or '8-23' (inclusive) and return sorted unique list."""
    if mask is None or mask == "":
        return []
    if isinstance(mask, (list, tuple)):
        return sorted({int(x) for x in mask})
    out = set()
    for chunk in str(mask).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(chunk))
    return sorted(out)


def apply_layer_mask(qwen_vl_model: nn.Module, remove_layers: Sequence[int]) -> Dict:
    """
    Physically delete the given LLM layer indices from a Qwen2.5-VL backbone.

    `qwen_vl_model` is the `Qwen2_5_VLForConditionalGeneration` instance (i.e.
    `framework.qwen_vl_interface.model`). After this call the model has fewer
    layers and the remaining layers' `self_attn.layer_idx` are renumbered
    contiguously starting from 0 so HF's KV-cache logic continues to work.

    Returns a dict with:
        - kept_layers:    list of ORIGINAL indices that survived
        - removed_layers: list of ORIGINAL indices that were removed
        - total_original: total layer count before pruning
    """
    layers = _locate_layers(qwen_vl_model)
    total_original = len(layers)

    remove = sorted({int(i) for i in remove_layers})
    if remove and (remove[0] < 0 or remove[-1] >= total_original):
        raise ValueError(
            f"remove_layers {remove} out of range for a {total_original}-layer stack"
        )

    kept_indices = [i for i in range(total_original) if i not in set(remove)]
    new_layers = nn.ModuleList([layers[i] for i in kept_indices])

    # Renumber layer_idx on each surviving layer so KV-cache lookups still index
    # into the right slot. Qwen2.5-VL's decoder layer exposes self_attn.layer_idx.
    for new_idx, layer in enumerate(new_layers):
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = new_idx

    # Swap the ModuleList in place at its original parent so all references stay valid.
    parent, attr = _resolve_parent(qwen_vl_model)
    setattr(parent, attr, new_layers)

    # Update num_hidden_layers in the config so anything that reads it sees the new count.
    cfg = getattr(qwen_vl_model, "config", None)
    if cfg is not None:
        text_cfg = getattr(cfg, "text_config", cfg)
        if hasattr(text_cfg, "num_hidden_layers"):
            text_cfg.num_hidden_layers = len(new_layers)

    return {
        "kept_layers": kept_indices,
        "removed_layers": remove,
        "total_original": total_original,
    }


def _resolve_parent(model: nn.Module):
    """Return (parent_module, attribute_name) such that getattr(parent, attr)
    is the decoder-layer ModuleList. Mirrors `_locate_layers` lookup."""
    candidates = [
        ("model", "language_model", "layers"),
        ("model", "layers"),
    ]
    for path in candidates:
        obj = model
        ok = True
        for attr in path[:-1]:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and hasattr(obj, path[-1]) and isinstance(getattr(obj, path[-1]), nn.ModuleList):
            return obj, path[-1]
    raise AttributeError("Could not resolve the layers' parent module.")
