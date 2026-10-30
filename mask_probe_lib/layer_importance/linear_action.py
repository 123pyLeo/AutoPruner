"""
LinearActionProbe -- per-layer linear action decodability (ridge L1 / R²).
==========================================================================

Source
------
* openvla : ``scripts/linear_probe_action.py``, ``scripts/probe_action_anchor.py``
* starVLA : ``scripts/probe/probe_linear_action.py`` ("Ported from openvla-oft")

Idea
----
For each residual-stream point, fit a closed-form ridge regression from the
action-token hidden state to the ground-truth action chunk; report the held-out
L1 (lower == more linearly action-decodable) and R².

Two verified roles fall out of the curve:

* **action-ready** layers  -- lowest L1 (e.g. openvla L7-L10; QwenOFT L26-L35).
* **format-aligner** anchor -- a layer with *high* L1 (not linearly decodable)
  that is nonetheless essential because it reformats reps for the head
  (openvla L30).  This is the W7 trap: high L1 != removable.

Verdict: VERIFIED as a ranking / protection signal (every successful mask in
both repos protects the most action-decodable late stack).  Caveat: open-loop
L1 != closed-loop SR -- never use as a standalone SR predictor.

Importance direction
--------------------
Action-readiness alone is ambiguous (a format-aligner is important *because* it
is hard to decode).  We therefore define importance as **distance of L1 from the
median** -- both the most-decodable and the least-decodable layers score high --
and expose the raw ``l1`` / ``r2`` for callers that want the plain curve.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.backend import ModelBackend
from ..core.kernels import ridge_probe
from ..core.registry import register
from ..core.types import ActivationBundle, LayerImportanceReport, LayerScore
from .base import LayerImportanceProbe


@register("layer_importance", "linear_action")
class LinearActionProbe(LayerImportanceProbe):
    """Per-layer ridge decodability of the action chunk."""

    name = "linear_action"

    def __init__(
        self,
        target: str = "chunk",            # see ActivationBundle.action_target
        n_batches: int = 200,
        protect_quantile: float = 0.80,
        soft_quantile: float = 0.60,
        seed: int = 0,
    ):
        self.target = target
        self.n_batches = n_batches
        self.protect_quantile = protect_quantile
        self.soft_quantile = soft_quantile
        self.seed = seed

    def score_layers(
        self, backend: ModelBackend, bundle: Optional[ActivationBundle] = None
    ) -> LayerImportanceReport:
        if bundle is None:
            bundle = backend.collect_activations(n_batches=self.n_batches, seed=self.seed)
        Y = bundle.action_target(self.target)

        l1_by_layer = {}
        r2_by_layer = {}
        for L in bundle.layer_indices:
            res = ridge_probe(bundle.hidden[L], Y, seed=self.seed)
            # report L1 relative to the predict-the-mean baseline (scale-free)
            l1_by_layer[L] = res["val_l1"] / max(res["baseline_l1"], 1e-9)
            r2_by_layer[L] = res["r2"]

        l1_vals = np.array(list(l1_by_layer.values()))
        median_l1 = float(np.median(l1_vals))

        scores = []
        for L in bundle.layer_indices:
            l1 = l1_by_layer[L]
            # importance = deviation from median (catches both action-ready and
            # format-aligner anchors); higher == protect.
            importance = abs(l1 - median_l1)
            scores.append(
                LayerScore(
                    layer=L,
                    score=float(importance),
                    raw={
                        "l1_rel": float(l1),
                        "r2": float(r2_by_layer[L]),
                        "action_ready": float(l1 < median_l1),
                    },
                )
            )

        self._tag_by_quantile(scores, self.protect_quantile, self.soft_quantile)
        return LayerImportanceReport(
            method=self.name,
            scores=scores,
            higher_is_more_important=True,
            notes=(
                "ridge(hidden -> action chunk); raw['l1_rel'] lower == action-ready. "
                "Importance = |l1 - median| so format-aligner anchors (high l1, W7 "
                "trap) and action-ready layers both score high. Ranking signal only."
            ),
        )
