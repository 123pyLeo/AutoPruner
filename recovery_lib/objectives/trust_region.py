"""
TrustRegionDistill -- on-policy distillation + anti-collapse anchor.  ★ the win
===============================================================================

Source
------
* starVLA : ``OnPolicyDistillTrainer._train_step`` -- the recipe that reached 0.828,
            the first non-oracle method to break the frozen-backbone 0.80 ceiling.

Objective
---------
On top of :class:`DistillBC`, two verified additions:

* ``multi_hidden_mse`` (weight ~10, relative/scale-free, skip 2 absorber layers per
  boundary) -- anchor every kept layer's hidden to the teacher;
* ``trust_region`` (weight ~0.5, smooth-L1 to a FROZEN snapshot of the
  pre-continuation student) -- the load-bearing anti-basin-collapse guardrail that
  stops self-imitation drift.

CRUCIAL: this objective only helps when paired with **on-policy** batches (the
student's own visited states). The exact same multi-hidden anchor applied to
*offline* states HURT (multitraj 0.708 < 0.788). The strategy supplies on-policy
data; this objective supplies the matching loss.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

from ..core import losses
from ..core.registry import register
from ..core.types import Batch, StudentOutput, TeacherOutput
from .base import RecoveryObjective
from .distill_bc import DistillBC


@register("recovery_objective", "trust_region_distill")
class TrustRegionDistill(RecoveryObjective):
    """DistillBC + relative multi-hidden anchor + trust-region anchor."""

    name = "trust_region_distill"

    def __init__(self, multi_hidden_weight: float = 10.0, trust_region_weight: float = 0.5,
                 skip_post: int = 2, removed_after: Sequence[int] = (), **distill_kwargs):
        self.base = DistillBC(**distill_kwargs)
        self.multi_hidden_weight = multi_hidden_weight
        self.trust_region_weight = trust_region_weight
        self.skip_post = skip_post
        self.removed_after = tuple(removed_after)

    def compute(self, student: StudentOutput, teacher: TeacherOutput, batch: Batch,
                anchor: Optional[StudentOutput] = None) -> Tuple[object, Dict[str, float]]:
        loss, parts = self.base.compute(student, teacher, batch)

        # relative multi-layer hidden anchor (scale-free), on-policy states
        if (self.multi_hidden_weight > 0 and student.hidden is not None
                and teacher.hidden is not None):
            mh = losses.multi_hidden_mse(student.hidden, teacher.hidden,
                                         skip_post=self.skip_post, relative=True,
                                         removed_after=self.removed_after)
            if mh is not None:
                loss = loss + self.multi_hidden_weight * mh
                parts["multi_hidden_mse"] = float(mh)

        # trust-region anchor to the frozen snapshot (the anti-collapse term)
        if anchor is not None and self.trust_region_weight > 0:
            tr = losses.trust_region_anchor(student.action, anchor.action,
                                            weight=self.trust_region_weight)
            loss = loss + tr
            parts["trust_region"] = float(tr)

        if not batch.on_policy:
            parts["_warning"] = 1.0   # this objective is verified only on on-policy data
        parts["total"] = float(loss)
        return loss, parts
