"""
RecoverabilityFunnel -- the F-series staged screening protocol.
===============================================================

Source
------
* openvla : ``f_series_recoverability_mask_search.md`` (the pre-registered
            falsification funnel) + the agentic evidence-gated loop.

Idea
----
Avoid expensive training on bad masks by screening candidates through a funnel of
increasing cost, stopping early if the cheap stages already disqualify a mask:

    Stage 0  component sanity   -- verify ckpt component identities (md5 / mask def)
    Stage 1  static screening   -- per-layer importance + per-mask recoverability
                                   (layers 1-2 of this library); drop obviously bad
    Stage 2  SR surrogate        -- DriftPredictor predicted SR (layer 3); rank
    Stage 3  cheap closed-loop   -- few-trial SR on a handful of tasks
    Stage 4  50-trial confirm    -- only the top 1-2 candidates

The verified decision signal is the **mini-recovery SR slope**, NOT offline loss
(a verified failure mode: offline loss dropping while closed-loop SR did not).

This class orchestrates the cheap stages and hands back a ranked shortlist; the
expensive closed-loop stages call ``backend.closed_loop_sr`` when available.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

from ..core.backend import ModelBackend
from ..core.registry import register
from ..core.types import MaskCandidate
from ..mask_recoverability.boundary_lowrank import BoundaryLowRankProbe
from ..mask_recoverability.planning_gap import PlanningGapRule


@register("validation", "recoverability_funnel")
class RecoverabilityFunnel:
    """Staged, increasing-cost screening of candidate masks."""

    name = "recoverability_funnel"

    def __init__(
        self,
        backend: ModelBackend,
        recoverability_probe: Optional[BoundaryLowRankProbe] = None,
        gap_rule: Optional[PlanningGapRule] = None,
        surrogate=None,
    ):
        self.backend = backend
        self.recoverability_probe = recoverability_probe or BoundaryLowRankProbe()
        self.gap_rule = gap_rule
        self.surrogate = surrogate

    # -- Stage 1: static screening ------------------------------------------ #
    def static_screen(self, candidates: Sequence[MaskCandidate]) -> List[MaskCandidate]:
        """Attach recoverability (+ optional planning-gap) scores; drop failures."""
        kept = []
        for c in candidates:
            rec = self.recoverability_probe.score_mask(self.backend, c.mask)
            c.recoverability = rec.value
            c.scores["recoverability"] = rec.value
            ok = rec.recoverable is not False
            if self.gap_rule is not None and c.is_contiguous():
                gap = self.gap_rule.score_mask(self.backend, c.mask)
                c.scores["planning_gap_r2"] = gap.value
                ok = ok and (gap.recoverable is not False)
            if ok:
                kept.append(c)
        return kept

    # -- Stage 2: SR surrogate ranking -------------------------------------- #
    def surrogate_rank(self, candidates: Sequence[MaskCandidate]) -> List[MaskCandidate]:
        if self.surrogate is None:
            return list(candidates)
        for c in candidates:
            feats = self.surrogate.features(self.backend, c.mask)
            X = [[feats[n] for n in self.surrogate.feature_names]]
            try:
                c.predicted_sr = float(self.surrogate.predict(X)[0])
                c.scores["predicted_sr"] = c.predicted_sr
            except RuntimeError:
                pass                       # surrogate not yet fit -> skip ranking
        return sorted(candidates, key=lambda c: (c.predicted_sr or -1), reverse=True)

    # -- Stages 3-4: closed-loop confirmation ------------------------------- #
    def confirm(self, candidates: Sequence[MaskCandidate], top_n: int = 2,
                recovery: str = None) -> List[MaskCandidate]:
        for c in list(candidates)[:top_n]:
            c.sr = self.backend.closed_loop_sr(list(c.mask), recovery=recovery)
            c.scores["closed_loop_sr"] = c.sr
        return list(candidates)

    # -- full funnel -------------------------------------------------------- #
    def run(self, candidates: Sequence[MaskCandidate], top_n: int = 2,
            confirm: bool = False, recovery: str = None) -> Dict[str, List[MaskCandidate]]:
        s1 = self.static_screen(candidates)
        s2 = self.surrogate_rank(s1)
        out = {"static_passed": s1, "surrogate_ranked": s2}
        if confirm:
            out["confirmed"] = self.confirm(s2, top_n=top_n, recovery=recovery)
        return out
