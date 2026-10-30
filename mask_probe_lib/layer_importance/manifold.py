"""
ManifoldGeometryProbe -- effective rank + anisotropy + isometry (combined).
===========================================================================

Source
------
* openvla : ``scripts/per_layer_manifold.py``
* starVLA : ``scripts/probe/probe_layer_redundancy.py``, ``_layerprobe.py``

Idea
----
Three representation-geometry signals per layer, read from the action-token
activations:

* **effective rank**  (Roy-Vetterli, exp-entropy of singular values) -- a sharp
  drop near the tail marks a projection / format-aligner layer.
* **anisotropy**       (mean random-pair cosine) -- a jump marks a compression
  layer collapsing the representation.
* **isometry**         (Spearman of pairwise distances in vs out) -- low after a
  plateau marks a distortion / role-transition layer.

Verdict: VERIFIED **only in combination**.  Each one alone is misleading
(isometry of the essential L30 was 0.985; CKA marked it redundant).  Used
together they correctly flag the late format-aligner anchor where FIM/CKA fail.

Importance signal
-----------------
A layer is flagged important when it stands out on the *change* between its input
and output geometry: ``|Δ effective_rank|`` (input vs output) + anisotropy jump.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.backend import ModelBackend
from ..core.kernels import anisotropy, effective_rank, isometry
from ..core.registry import register
from ..core.types import ActivationBundle, LayerImportanceReport, LayerScore
from .base import LayerImportanceProbe


def peak_relative_effrank_importance(er_by_layer, n_layers):
    """Read effective_rank by its PEAK, not by magnitude.

    effective_rank rises as the network *builds* its representation, reaches a peak
    (the informative core), then falls as later layers *refine / re-compress*.  Two
    regions have LOW eff_rank for opposite reasons:

      * ascending side (before the peak): low because the representation is still
        being ENCODED -- foundational, must be PROTECTED (removing it starves every
        downstream layer, incl. a separate action head; verified: pruning CogACT's
        early {2-9} -> 0.08 even with retraining).
      * descending side (after the peak): low because the layer is REFINING an
        already-built representation -- genuinely prunable (verified: CogACT late
        {24-31} -> 1.00 with no retraining).

    Ranking by eff_rank *magnitude* conflates these and mis-labels the foundational
    early layers as removable.  This function instead returns a per-layer importance
    (higher == protect) in which EVERY ascending-side/peak layer outranks EVERY
    descending-side layer, so a lowest-importance contiguous window lands entirely on
    the post-peak refinement tail.

    Parameters
    ----------
    er_by_layer : dict {layer_index: effective_rank}
    n_layers    : decoder layer count (report covers 0..n_layers-1)
    """
    if not er_by_layer:
        return {L: 0.0 for L in range(n_layers)}
    peak = max(er_by_layer, key=er_by_layer.get)
    er_max = max(er_by_layer.values())
    importance = {}
    for L in range(n_layers):
        base = float(er_by_layer.get(L, 0.0))
        # a bonus > any eff_rank guarantees pre-peak/peak layers all outrank post-peak ones
        importance[L] = base + (er_max + 1.0 if L <= peak else 0.0)
    return importance


@register("layer_importance", "manifold")
class ManifoldGeometryProbe(LayerImportanceProbe):
    """Effective-rank / anisotropy / isometry geometry, combined per layer."""

    name = "manifold"

    def __init__(
        self,
        n_batches: int = 200,
        n_pairs: int = 2000,
        protect_quantile: float = 0.80,
        soft_quantile: float = 0.60,
        seed: int = 0,
    ):
        self.n_batches = n_batches
        self.n_pairs = n_pairs
        self.protect_quantile = protect_quantile
        self.soft_quantile = soft_quantile
        self.seed = seed

    def score_layers(
        self, backend: ModelBackend, bundle: Optional[ActivationBundle] = None
    ) -> LayerImportanceReport:
        if bundle is None:
            bundle = backend.collect_activations(n_batches=self.n_batches, seed=self.seed)

        idx = bundle.layer_indices
        er = {L: effective_rank(bundle.hidden[L]) for L in idx}
        ani = {L: anisotropy(bundle.hidden[L], self.n_pairs, self.seed) for L in idx}

        scores = []
        for L in idx:
            nxt = L + 1
            if nxt in bundle.hidden:
                d_er = abs(er[nxt] - er[L])
                d_ani = abs(ani[nxt] - ani[L])
                iso = isometry(bundle.hidden[L], bundle.hidden[nxt], self.n_pairs, self.seed)
            else:                                   # final residual point
                d_er = d_ani = 0.0
                iso = 1.0
            # normalise the two change signals to [0,1] later via quantile tag;
            # raw combination: rank-change + anisotropy-jump + distortion (1-iso)
            importance = d_er / max(max(er.values()), 1e-9) + d_ani + (1.0 - iso)
            scores.append(
                LayerScore(
                    layer=L,
                    score=float(importance),
                    raw={
                        "effective_rank": float(er[L]),
                        "anisotropy": float(ani[L]),
                        "d_effective_rank": float(d_er),
                        "isometry": float(iso),
                    },
                )
            )

        self._tag_by_quantile(scores, self.protect_quantile, self.soft_quantile)
        return LayerImportanceReport(
            method=self.name,
            scores=scores,
            higher_is_more_important=True,
            notes=(
                "effective_rank + anisotropy + isometry, combined. VERIFIED only "
                "as a combination -- any single one alone is misleading. Detects "
                "compression / format-aligner anchor layers (e.g. openvla L30)."
            ),
        )

    def effrank_protect_report(
        self, backend: ModelBackend, bundle: Optional[ActivationBundle] = None
    ) -> LayerImportanceReport:
        """Peak-relative effective_rank report -- the correct flat-knockout fallback.

        Unlike :meth:`score_layers` (which combines the *change* signals |Δrank| etc.
        and, like magnitude, peaks at the wrong place for a strong-separate-head model),
        this reads eff_rank by its PEAK: protect the ascending build-up + peak, prune
        the descending refinement tail.  See :func:`peak_relative_effrank_importance`.
        """
        if bundle is None:
            bundle = backend.collect_activations(n_batches=self.n_batches, seed=self.seed)
        er = {L: effective_rank(bundle.hidden[L]) for L in bundle.layer_indices}
        peak = max(er, key=er.get)
        imp = peak_relative_effrank_importance(er, backend.n_layers)
        scores = [
            LayerScore(
                layer=L, score=float(imp[L]),
                raw={"effective_rank": float(er.get(L, 0.0)), "peak": peak,
                     "side": "protect" if L <= peak else "prunable"},
            )
            for L in range(backend.n_layers)
        ]
        return LayerImportanceReport(
            method="manifold_effrank_peak",
            scores=scores,
            higher_is_more_important=True,
            notes=(
                f"eff_rank read by PEAK (L{peak}): ascending build-up + peak protected, "
                "descending refinement tail prunable. The correct fallback when single-layer "
                "knockout is flat (strong separate action head, e.g. CogACT)."
            ),
        )
