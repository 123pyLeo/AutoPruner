"""ProbeQueryService — agent-invoked probe tool API.

The agent does NOT receive a pre-filled probe table. Instead it calls
    service.query(probe_name, layer_ids) -> {layer_id: value}

There are two backends:
  - "cached": lookup from experiments/cache/probes_<model>_<suite>.json
              (used for the libero_object demo where probes were pre-computed)
  - "real":   actually run the probe on the model+suite (NOT implemented yet;
              would hook into vla-scripts/ probes and compute on demand)

This separation is the honest path for cross-model demonstration: for the demo
suite we accept cached probes (cheap), but for new models the same API must
run the real computation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Catalog of probes the agent can call. Each entry advertises what the probe
# measures and an estimated cost (real-mode) so the agent can plan budget-wise.
PROBE_CATALOG: dict[str, dict] = {
    "isometry": {
        "description": "Distance-preservation score of the layer's mapping. Low at input-shaping layers, high at redundant refinement layers, sharp dips mark role transitions.",
        "value_range": "[0, 1] typical",
        "real_cost_estimate": "~5 min/model (one forward sweep)",
    },
    "effective_rank": {
        "description": "Approximate intrinsic dimensionality of the layer's hidden state distribution. Ramps from ~1 to large in early layers; can DIP at format-aligner.",
        "value_range": "[1, hidden_dim]",
        "real_cost_estimate": "~5 min/model (SVD on activations)",
    },
    "linear_probe_l1": {
        "description": "L1 distance of a linear action probe trained on this layer's hidden vs true action. LOW = action-ready (linearly decodable). HIGH alone does NOT imply removable (W7).",
        "value_range": "[0, 1] typical",
        "real_cost_estimate": "~30 min/model (32 linear probes)",
    },
    "adjacent_cka_to_next": {
        "description": "Linear CKA similarity between this layer's hidden and the next layer's hidden. HIGH = redundant neighbors; sharp drop = role transition.",
        "value_range": "[0, 1]",
        "real_cost_estimate": "~10 min/model",
    },
    "fim_avg": {
        "description": "Average Fisher Information across layer parameters. MISLEADING (W1) — format-aligner anchors often have LOWEST FIM.",
        "value_range": "[0, ~1]",
        "real_cost_estimate": "~20 min/model (one backward sweep on validation batch)",
    },
    "residual_l2_mean": {
        "description": "Mean L2 norm of the layer's residual write (output - input). A SPIKE (3-5x neighbors) marks the final-calibration anchor.",
        "value_range": "[0, ~500]",
        "real_cost_estimate": "~5 min/model",
    },
    "residual_cos_mean": {
        "description": "Mean cosine of the layer's residual write direction vs its input. Low residual_cos_mean indicates the layer is writing in a NEW direction (important rewrite).",
        "value_range": "[-1, 1]",
        "real_cost_estimate": "~5 min/model",
    },
}


@dataclass
class ProbeQueryService:
    mode: str = "cached"             # "cached" or "real"
    cache_path: Optional[Path] = None
    model_id: Optional[str] = None
    suite: Optional[str] = None
    _cache: dict = field(default_factory=dict)
    _calls: list[dict] = field(default_factory=list)   # call log for audit

    def __post_init__(self) -> None:
        if self.mode == "cached":
            if self.cache_path is None or not Path(self.cache_path).exists():
                raise FileNotFoundError(f"cached mode requires cache_path; got {self.cache_path}")
            self._cache = json.load(open(self.cache_path))
        elif self.mode == "real":
            # Real-mode backend would hook into vla-scripts/<probe>.py and run them
            # on the fly. Not implemented in this version — raise on .query() instead.
            pass
        else:
            raise ValueError(f"unknown mode: {self.mode}")

    def list_available(self) -> list[dict]:
        """Return catalog of probes that can be queried."""
        out = []
        for name, meta in PROBE_CATALOG.items():
            avail = True
            if self.mode == "cached":
                avail = name in self._cache.get("data", {})
            out.append({
                "name": name,
                "available": avail,
                **meta,
            })
        return out

    def query(self, probe: str, layer_ids: list[int]) -> dict[int, float | None]:
        """Return {layer_id: value, ...}. Missing values are None."""
        if probe not in PROBE_CATALOG:
            raise KeyError(f"unknown probe '{probe}'; available: {list(PROBE_CATALOG.keys())}")
        result: dict[int, float | None] = {}
        if self.mode == "cached":
            probe_data = self._cache.get("data", {}).get(probe, {})
            for L in layer_ids:
                v = probe_data.get(str(L))
                result[int(L)] = float(v) if v is not None else None
        else:  # real
            raise NotImplementedError(
                f"real-mode probe execution not implemented yet — for new models, "
                f"hook this to vla-scripts/<probe_pipeline>.py"
            )
        self._calls.append({"probe": probe, "layer_ids": list(layer_ids), "result": result})
        return result

    def query_log(self) -> list[dict]:
        return list(self._calls)

    def reset_log(self) -> None:
        self._calls = []
