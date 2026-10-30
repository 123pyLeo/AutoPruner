"""
Layer 1 -- Per-layer importance probes (base interface).

A layer-importance probe answers: *how load-bearing is each individual layer?*
Its output (:class:`LayerImportanceReport`) ranks layers so the selection layer
(``mask_selection``) can protect anchors and prune tolerant layers.

All probes share one contract::

    report = probe.score_layers(backend, bundle=None)
    report.anchors()    # layers to protect
    report.tolerant()   # layers safe to prune

``score`` is always normalised so **higher == more important / protect**.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from ..core.backend import ModelBackend
from ..core.types import ActivationBundle, LayerImportanceReport, LayerScore


class LayerImportanceProbe(ABC):
    """Base class for all per-layer importance probes."""

    #: human-readable method name (also the registry key)
    name: str = "base"

    @abstractmethod
    def score_layers(
        self, backend: ModelBackend, bundle: Optional[ActivationBundle] = None
    ) -> LayerImportanceReport:
        """Return a per-layer importance table.

        ``bundle`` lets several probes share one expensive forward sweep
        (``backend.collect_activations``); pass ``None`` to capture internally.
        """

    # -- shared helper: split layers into anchor / soft / tolerant by quantile #
    @staticmethod
    def _tag_by_quantile(
        scores: List[LayerScore],
        anchor_q: float = 0.80,
        soft_q: float = 0.60,
    ) -> None:
        """Tag layers in place: top ``anchor_q`` -> anchor, next band -> soft."""
        vals = np.array([s.score for s in scores])
        if vals.size == 0:
            return
        # degenerate guard: if scores have almost no spread, nothing is an anchor
        # (avoids flagging a flat cluster of tied tolerant layers as anchors).
        if float(vals.std()) < 1e-9:
            for s in scores:
                s.tag = "tolerant"
            return
        hi = np.quantile(vals, anchor_q)
        mid = np.quantile(vals, soft_q)
        # use strict '>' at the anchor boundary so a tie *at* the quantile value
        # (a flat tolerant cluster sitting on the threshold) is not all promoted.
        for s in scores:
            if s.score > hi:
                s.tag = "anchor"
            elif s.score >= mid:
                s.tag = "soft-anchor"
            else:
                s.tag = "tolerant"
