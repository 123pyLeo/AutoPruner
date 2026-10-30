"""
AntiAnchorControl -- the causal negative control for any importance metric.
===========================================================================

Source
------
* starVLA : the ``k10antianchor`` config -- "10 highest-importance, distributed,
            non-adjacent" layers removed -> SR crashed to 0.484 vs the anchor
            mask's 0.788 (everything else, including capacity, held identical).

Idea
----
An importance metric is only *causal* if deliberately removing the layers it calls
most important destroys performance.  Build the "anti-anchor" mask (remove the
top-importance layers), at the **same prune ratio and recovery recipe** as the
real (anchor-protecting) mask, and check that:

    SR(anti_anchor)  <<  SR(anchor)

If the gap does not open, the importance ranking is not actually load-bearing and
must not be used to select masks.  This is the cheapest, most decisive guard
against a probe that merely *correlates* without being *causal*.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..core.registry import register
from ..core.types import LayerImportanceReport, MaskCandidate, ValidationResult
from ..mask_selection.anchor_policy import AnchorProtectionPolicy


@register("validation", "anti_anchor")
class AntiAnchorControl:
    """Falsification control: remove the highest-importance layers and expect collapse."""

    name = "anti_anchor"

    def __init__(self, n_layers: int, min_gap_pts: float = 0.15):
        #: required SR drop (anchor - anti_anchor) for the metric to be 'causal'
        self.n_layers = n_layers
        self.min_gap_pts = min_gap_pts

    def build_anti_anchor(self, report: LayerImportanceReport, k: int,
                          distributed: bool = True) -> MaskCandidate:
        """Remove the ``k`` HIGHEST-importance layers (distributed, non-adjacent)."""
        ranked = report.ranked(most_important_first=True)
        chosen, picked = [], []
        for s in ranked:
            if len(chosen) >= k:
                break
            if not distributed or all(abs(s.layer - c) >= 2 for c in chosen):
                chosen.append(s.layer)
                picked.append(s.layer)
        # top up if spacing blocked us
        for s in ranked:
            if len(chosen) >= k:
                break
            if s.layer not in chosen:
                chosen.append(s.layer)
        return MaskCandidate(mask=tuple(sorted(chosen[:k])),
                             rationale="ANTI-ANCHOR control: remove highest-importance layers")

    def run(self, backend, report: LayerImportanceReport, anchor_mask: Sequence[int],
            recovery: str = None) -> ValidationResult:
        """Compare SR of the anchor mask vs an equal-size anti-anchor mask."""
        k = len(list(anchor_mask))
        anti = self.build_anti_anchor(report, k)
        sr_anchor = backend.closed_loop_sr(list(anchor_mask), recovery=recovery)
        sr_anti = backend.closed_loop_sr(list(anti.mask), recovery=recovery)
        gap = (sr_anchor or 0.0) - (sr_anti or 0.0)
        passed = gap >= self.min_gap_pts
        return ValidationResult(
            name=self.name,
            passed=bool(passed),
            statistic={"sr_anchor": float(sr_anchor or float("nan")),
                       "sr_anti_anchor": float(sr_anti or float("nan")),
                       "gap": float(gap)},
            detail=(
                f"anchor SR {sr_anchor} vs anti-anchor SR {sr_anti} (gap {gap:+.3f}); "
                f"importance metric is {'CAUSAL' if passed else 'NOT causal -- do not use'} "
                f"@ min gap {self.min_gap_pts}. anti-anchor mask={anti.mask}"
            ),
        )
