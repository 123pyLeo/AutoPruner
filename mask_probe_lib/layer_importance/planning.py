"""
PlanningInfoProbe -- per-layer multi-step planning information (future-xyz R²).
==============================================================================

Source
------
* openvla : ``scripts/run_subgoal_probe.py`` (P1..P5), ``scripts/probe_deeper_designs.py``

Idea
----
For each layer, ridge-regress the hidden state to *future* targets and report R²:

* **P2 future-xyz** : ``actions[:, -1, :3]``  -- 8-step-ahead end-effector xyz.
* **P5 chunk-delta** : ``actions[:, -1, :3] - actions[:, 0, :3]`` -- net motion.

``plan_score = (max(0,P2) + max(0,P5)) / 2``.

Verified finding
----------------
Planning information is **accumulated, not localised**: P2 R² rises monotonically
with depth (openvla L0=0.00 -> L8=0.81 -> L13=0.90 -> L28=0.95 peak).  This
overturned an earlier "single sub-goal-critical layer" hypothesis.

This per-layer R² curve is also the input to the verified **planning-gap rule**
in ``mask_recoverability/planning_gap.py``: a contiguous mask is recoverable iff
the layer feeding the post-deletion boundary still has P2 R² >= ~0.85.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.backend import ModelBackend
from ..core.kernels import ridge_probe
from ..core.registry import register
from ..core.types import ActivationBundle, LayerImportanceReport, LayerScore
from .base import LayerImportanceProbe


@register("layer_importance", "planning")
class PlanningInfoProbe(LayerImportanceProbe):
    """Per-layer planning-information content (future-xyz / chunk-delta R²)."""

    name = "planning"

    def __init__(
        self,
        n_batches: int = 200,
        protect_quantile: float = 0.80,
        soft_quantile: float = 0.60,
        seed: int = 0,
    ):
        self.n_batches = n_batches
        self.protect_quantile = protect_quantile
        self.soft_quantile = soft_quantile
        self.seed = seed

    def score_layers(
        self, backend: ModelBackend, bundle: Optional[ActivationBundle] = None
    ) -> LayerImportanceReport:
        if bundle is None:
            bundle = backend.collect_activations(n_batches=self.n_batches, seed=self.seed)

        future = bundle.action_target("future")
        delta = bundle.action_target("delta")

        scores = []
        for L in bundle.layer_indices:
            p2 = ridge_probe(bundle.hidden[L], future, seed=self.seed)["r2"]
            p5 = ridge_probe(bundle.hidden[L], delta, seed=self.seed)["r2"]
            plan = (max(0.0, p2) + max(0.0, p5)) / 2.0
            scores.append(
                LayerScore(layer=L, score=float(plan), raw={"p2_r2": float(p2), "p5_r2": float(p5)})
            )

        self._tag_by_quantile(scores, self.protect_quantile, self.soft_quantile)
        return LayerImportanceReport(
            method=self.name,
            scores=scores,
            higher_is_more_important=True,
            notes=(
                "ridge R² to future-xyz (P2) and chunk-delta (P5). Planning info is "
                "accumulated with depth (monotonic). Feeds the planning-gap "
                "recoverability rule (post-deletion input layer needs P2 R² >= 0.85)."
            ),
        )
