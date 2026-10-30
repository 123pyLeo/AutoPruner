"""
WindowedKnockoutProbe -- block-level functional knockout for strong-separate-head models.
=========================================================================================

Why
---
The single-layer :class:`AblationKnockoutProbe` is THE verified selection signal for models
whose language backbone produces the action directly (openvla, starVLA, FastWAM): identity-
skipping one layer moves the action chunk sharply and cleanly ranks layers.

It goes FLAT for models with a powerful, SEPARATE action head conditioned on a single
backbone vector -- CogACT (Prismatic LLaMA-2 + a Diffusion Transformer head). Removing any
ONE layer barely changes the cognition vector *enough for the DiT to still absorb it*, so
d_all is ~equal for every layer (measured max/median 1.17) and cannot rank anything.

Fix
---
Remove a contiguous WINDOW of `window` layers at a time instead of one. A whole block is too
large for the head to absorb, so the perturbation again depends on WHERE the block is -- the
functional signal returns, now at block granularity. Early (foundational build-up) windows
perturb the output far more than late (refinement) windows.

Per-layer score = mean d_all over all size-`window` blocks containing L (higher == load-
bearing == protect), so it drops straight into the pipeline as the ``importance_probe``.
Reads only open-loop forward passes with ``predict_action(mask=block)`` -- no training.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.backend import ModelBackend
from ..core.kernels import action_perturbation
from ..core.registry import register
from ..core.types import ActivationBundle, LayerImportanceReport, LayerScore
from .base import LayerImportanceProbe


@register("layer_importance", "windowed_knockout")
class WindowedKnockoutProbe(LayerImportanceProbe):
    """Block-knockout importance -- use when single-layer knockout is flat.

    Parameters
    ----------
    window:
        number of contiguous layers removed per probe (block size). Large enough that a
        separate action head cannot absorb it (4-8 for a 32-layer backbone).
    n_batches:
        fixed eval-set size (reference and knockout predictions must be comparable).
    protect_quantile / soft_quantile:
        quantiles above which a layer is tagged ``anchor`` / ``soft-anchor``.
    """

    name = "windowed_knockout"

    def __init__(
        self,
        window: int = 4,
        n_batches: int = 50,
        protect_quantile: float = 0.80,
        soft_quantile: float = 0.60,
        seed: int = 0,
    ):
        self.window = window
        self.n_batches = n_batches
        self.protect_quantile = protect_quantile
        self.soft_quantile = soft_quantile
        self.seed = seed

    def score_layers(
        self, backend: ModelBackend, bundle: Optional[ActivationBundle] = None
    ) -> LayerImportanceReport:
        n = backend.n_layers
        w = max(1, min(self.window, n))
        a_dense = backend.predict_action(n_batches=self.n_batches, seed=self.seed)

        # d_all for every contiguous block [s, s+w)
        block_d = {}
        for s in range(0, n - w + 1):
            block = list(range(s, s + w))
            a_ko = backend.predict_action(mask=block, n_batches=self.n_batches, seed=self.seed)
            block_d[s] = float(action_perturbation(a_dense, a_ko))

        # per-layer score = mean d_all over blocks containing L
        scores = []
        for L in range(n):
            hits = [block_d[s] for s in block_d if s <= L < s + w]
            score = float(np.mean(hits)) if hits else 0.0
            scores.append(LayerScore(layer=L, score=score, raw={"n_blocks": len(hits)}))

        self._tag_by_quantile(scores, self.protect_quantile, self.soft_quantile)
        return LayerImportanceReport(
            method=self.name,
            scores=scores,
            higher_is_more_important=True,
            notes=(
                f"block-knockout (window={w}); score = mean action perturbation over blocks "
                "containing the layer. Use when single-layer knockout is flat (strong separate "
                "action head, e.g. CogACT). Protect high (foundational), prune low (refinement)."
            ),
        )
