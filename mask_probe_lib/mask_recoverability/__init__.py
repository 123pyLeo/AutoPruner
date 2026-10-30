"""
Layer 2 -- Mask-recoverability probes.

Score a *whole candidate mask* by whether a small adapter can restore the deleted
block ("recoverability, not redundancy").  All implement
:class:`MaskRecoverabilityProbe`.

Verified methods (see ``METHODS.md``):
    boundary_lowrank -- BoundaryLowRankProbe  (Δ eff-rank + ridge/MLP recovery cos)
    planning_gap     -- PlanningGapRule        (post-deletion input-layer P2 R²>=0.85)
    multi_anchor     -- MultiAnchorProbe        (non-contiguous anchors raise ceiling)
"""

from .base import MaskRecoverabilityProbe
from .boundary_lowrank import BoundaryLowRankProbe
from .planning_gap import PlanningGapRule
from .multi_anchor import MultiAnchorProbe

__all__ = [
    "MaskRecoverabilityProbe",
    "BoundaryLowRankProbe",
    "PlanningGapRule",
    "MultiAnchorProbe",
]
