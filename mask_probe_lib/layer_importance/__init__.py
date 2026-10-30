"""
Layer 1 -- Per-layer importance probes.

Rank individual layers so the selection layer can protect anchors and prune
tolerant layers.  All probes implement :class:`LayerImportanceProbe`.

Verified methods (see ``METHODS.md``):
    knockout       -- AblationKnockoutProbe   (decisive; identity-skip perturbation)
    windowed_knockout -- WindowedKnockoutProbe (block-knockout; use when single-layer is flat)
    linear_action  -- LinearActionProbe       (ridge L1/R²; action-ready vs aligner)
    manifold       -- ManifoldGeometryProbe   (eff-rank + anisotropy + isometry)
    planning       -- PlanningInfoProbe        (future-xyz R²; feeds the gap rule)
"""

from .base import LayerImportanceProbe
from .knockout import AblationKnockoutProbe
from .windowed_knockout import WindowedKnockoutProbe
from .linear_action import LinearActionProbe
from .manifold import ManifoldGeometryProbe
from .planning import PlanningInfoProbe

__all__ = [
    "LayerImportanceProbe",
    "AblationKnockoutProbe",
    "WindowedKnockoutProbe",
    "LinearActionProbe",
    "ManifoldGeometryProbe",
    "PlanningInfoProbe",
]
