"""
[3] Recovery strategies -- the end-to-end verified recipes.

    phase2_joint        -- PhaseTwoJointRecovery       (foundational; k16 98%, k10anchor 0.788)
    onpolicy_distill    -- OnPolicyDistillTrustRegion   (★ the win: 0.788→0.828, breaks 0.80 ceiling)
    per_task_specialist -- PerTaskSpecialist            (verified-conditional; t8 +20pt; feeds routing)
"""

from .base import RecoveryStrategy
from .phase2_joint import PhaseTwoJointRecovery
from .onpolicy_distill import OnPolicyDistillTrustRegion
from .per_task_specialist import PerTaskSpecialist

__all__ = [
    "RecoveryStrategy",
    "PhaseTwoJointRecovery",
    "OnPolicyDistillTrustRegion",
    "PerTaskSpecialist",
]
