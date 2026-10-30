"""
AblationKnockoutProbe -- THE verified, decisive layer-selection method.
=======================================================================

Source
------
* starVLA  : ``scripts/recovery/anchor_knockout_probe.py``  (identity-skip -> dAll)
* openvla  : the closed-loop single-layer-drop SR ledger (``b1_tf_layer_X``) is the
             expensive ground-truth version of the same idea.

Idea
----
For each layer L, **identity-skip** it on the frozen teacher (the layer's output
is replaced by its input) and measure how far the predicted action chunk moves
from the unskipped prediction:

    d_all(L)   = mean | a_knockout(L) - a_dense |              (overall)
    d_trans(L) = mean | a_knockout(L) - a_dense |  on transition frames

High movement  -> the layer is load-bearing -> **ANCHOR (protect)**.
Low  movement  -> the layer is tolerant      -> safe to prune here.

Verified evidence
-----------------
With prune-ratio and recovery recipe held fixed (28 %, matched capacity), the
mask's ``sum(d_all over removed layers)`` monotonically predicts closed-loop SR:

    anti-anchor (Σ≈0.307) 0.484  <  contiguous 0.680
        <  distributed 0.758  <  anchor (Σ≈0.061) 0.788

This is the single most load-bearing selection signal in either repo, and it is
cheap (open-loop forward passes only -- no training, no closed-loop rollouts).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.backend import ModelBackend
from ..core.kernels import action_perturbation
from ..core.registry import register
from ..core.types import ActivationBundle, LayerImportanceReport, LayerScore
from .base import LayerImportanceProbe


@register("layer_importance", "knockout")
class AblationKnockoutProbe(LayerImportanceProbe):
    """Identity-skip each layer; rank by action-chunk perturbation.

    Parameters
    ----------
    n_batches:
        eval-set size for each forward pass (must be fixed so the reference and
        knockout predictions are element-wise comparable).
    transition_weight:
        ``d_all`` is combined with ``d_trans`` as
        ``score = (1-w)*d_all + w*d_trans``; the transition term up-weights
        sub-goal / gripper-flip frames where collapse is most damaging.  Set to 0
        to reproduce the plain ``d_all`` ranking.
    protect_quantile:
        layers above this quantile of the score are tagged ``anchor``.
    """

    name = "knockout"

    def __init__(
        self,
        n_batches: int = 50,
        transition_weight: float = 0.0,
        protect_quantile: float = 0.80,
        soft_quantile: float = 0.60,
        seed: int = 0,
        metric: str = "paper_l2_v1",
    ):
        self.n_batches = n_batches
        self.transition_weight = transition_weight
        self.protect_quantile = protect_quantile
        self.soft_quantile = soft_quantile
        self.seed = seed
        self.metric = metric

    def score_layers(
        self, backend: ModelBackend, bundle: Optional[ActivationBundle] = None
    ) -> LayerImportanceReport:
        # reference (dense) prediction on the fixed eval set
        a_dense = backend.predict_action(n_batches=self.n_batches, seed=self.seed)

        # a transition-frame mask, if the backend / bundle supplies one
        trans = None
        if bundle is not None and bundle.transition_mask is not None:
            # bundle may have more samples than the eval set; align lengths
            trans = bundle.transition_mask[: a_dense.shape[0]]

        scores = []
        for L in range(backend.n_layers):
            a_ko = backend.predict_action(
                knockout_layer=L, n_batches=self.n_batches, seed=self.seed
            )
            d_all = action_perturbation(a_dense, a_ko, metric=self.metric)
            d_trans = (
                action_perturbation(a_dense, a_ko, mask=trans, metric=self.metric)
                if trans is not None and trans.any()
                else d_all
            )
            score = (1 - self.transition_weight) * d_all + self.transition_weight * d_trans
            scores.append(
                LayerScore(layer=L, score=float(score), raw={"d_all": d_all, "d_trans": d_trans})
            )

        self._tag_by_quantile(scores, self.protect_quantile, self.soft_quantile)
        return LayerImportanceReport(
            method=self.name,
            scores=scores,
            higher_is_more_important=True,
            notes=(
                "identity-skip per layer; score = action-chunk perturbation. "
                "Verified to monotonically predict closed-loop SR (starVLA). "
                "Protect anchors (high score), prune tolerant (low score)."
            ),
        )
