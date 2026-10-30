"""
[2] Recovery objectives -- verified loss recipes (compose core.losses kernels).

    distill_bc            -- DistillBC            (action L1 + frozen-teacher distill; Phase-2)
    trust_region_distill  -- TrustRegionDistill   (+ multi-hidden anchor + trust-region; on-policy ★)
"""

from .base import RecoveryObjective
from .distill_bc import DistillBC
from .trust_region import TrustRegionDistill

__all__ = ["RecoveryObjective", "DistillBC", "TrustRegionDistill"]
