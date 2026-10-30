"""
[4] Supervision sources.

Recovery data is one verified axis: WHERE the supervised states come from.

    OfflineDemoSource    -- streamed offline demos (the Phase-2 default)
    OnPolicyRolloutSource-- DAgger: the student's OWN visited states, teacher-relabelled
                            (the covariate-shift fix that unlocks the 0.828 win)
    TeacherCache         -- interface for cached frozen-teacher logits/features

Verified contrast: on-policy states are what let recovery break the frozen-backbone
ceiling; the same anchor loss on offline states HURTS. So the *data source* is not a
detail -- it is the lever.
"""

from .offline_demos import OfflineDemoSource
from .onpolicy_rollouts import OnPolicyRolloutSource
from .teacher_cache import TeacherCache

__all__ = ["OfflineDemoSource", "OnPolicyRolloutSource", "TeacherCache"]
