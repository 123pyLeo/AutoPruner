"""
PlanningGapRule -- the verified contiguous-mask recoverability gate.
====================================================================

Source
------
* openvla : the P2 future-xyz planning probe (``run_subgoal_probe.py``) +
            summary.md §6 gap rule.

Rule
----
The boundary adapter can "fast-forward" the accumulated layers it replaces, but it
**cannot fabricate information absent from its input**.  So a contiguous mask is
recoverable iff the layer whose output *feeds* the post-deletion boundary still
carries the planning information:

    input_layer = min(mask) - 1          # last kept layer before the gap
    recoverable  <=>  P2_R2[input_layer] >= ~0.85

Verified evidence
-----------------
* ``[22-29]`` input L21 P2 R²=0.92 -> SR 92 %  (gap 0.03, recoverable)
* k=26 ``[2-27]`` input L1 R²=0.10 -> SR 50 %  (gap 0.84, NOT recoverable)
* head-cut ``[0-7]``               -> SR 0 %   (no input layer at all)

This predicts which masks fail with **one 30-min probe and zero adapter training**.

The per-layer P2 R² curve comes from
``mask_probe_lib.layer_importance.PlanningInfoProbe`` -- so this class consumes a
layer-1 report and turns it into a per-mask gate.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..core.backend import ModelBackend
from ..core.registry import register
from ..core.types import LayerImportanceReport, MaskScore
from ..layer_importance.planning import PlanningInfoProbe
from .base import MaskRecoverabilityProbe


@register("mask_recoverability", "planning_gap")
class PlanningGapRule(MaskRecoverabilityProbe):
    """Gate a contiguous mask on the planning R² of its post-deletion input layer."""

    name = "planning_gap"

    def __init__(self, threshold: float = 0.85, planning_report: Optional[LayerImportanceReport] = None,
                 n_batches: int = 200, seed: int = 0):
        self.threshold = threshold
        self._report = planning_report          # reuse a precomputed P2 curve if given
        self.n_batches = n_batches
        self.seed = seed

    def _p2_curve(self, backend: ModelBackend):
        if self._report is None:
            self._report = PlanningInfoProbe(n_batches=self.n_batches, seed=self.seed).score_layers(backend)
        return {s.layer: s.raw.get("p2_r2", s.score) for s in self._report.scores}

    def score_mask(self, backend: ModelBackend, mask: Sequence[int]) -> MaskScore:
        mask = tuple(sorted(int(m) for m in mask))
        p2 = self._p2_curve(backend)

        if len(mask) == 0:
            return MaskScore(mask=mask, method=self.name, value=1.0, recoverable=True,
                             raw={"input_layer": -1, "note": "empty mask"})

        input_layer = mask[0] - 1
        if input_layer < 0:
            # head-cut: no kept layer feeds the boundary -> unrecoverable
            return MaskScore(mask=mask, method=self.name, value=0.0, recoverable=False,
                             raw={"input_layer": -1, "note": "head-cut, no input layer"})

        r2 = float(p2.get(input_layer, 0.0))
        return MaskScore(
            mask=mask,
            method=self.name,
            value=r2,
            recoverable=bool(r2 >= self.threshold),
            raw={"input_layer": float(input_layer), "input_p2_r2": r2,
                 "gap": float(self.threshold - r2)},
        )
