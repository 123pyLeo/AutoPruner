"""
Layer 4 -- Mask selection policy.

Turn per-layer importance scores (layer 1) and recoverability/SR signals
(layers 2-3) into concrete masks.

Verified components (see ``METHODS.md``):
    anchor_policy -- AnchorProtectionPolicy (keep early+late anchors, prune tolerant)
    ratio         -- PruneRatioSweep        (28 % sweet spot; cliff above ~44 %)
    qd_archive    -- QualityDiversityArchive (elitist champion per behavioural cell)
"""

from .anchor_policy import AnchorProtectionPolicy
from .ratio import PruneRatioSweep
from .qd_archive import QualityDiversityArchive, mask_descriptors

__all__ = [
    "AnchorProtectionPolicy",
    "PruneRatioSweep",
    "QualityDiversityArchive",
    "mask_descriptors",
]
