"""
DistillBC -- action BC + frozen-teacher distillation (the Phase-2 objective).
=============================================================================

Source
------
* openvla : ``run_phase2_forward_pass`` -- ``1.0*L1(action) + 1.0*KL(action-range logits)``
* starVLA : ``Phase2AdapterTrainer`` -- ``action_l1 + a*distill_action_l1 + b*hidden_mse``

The verified foundational objective. Combines:
* ``bc`` -- L1 of predicted vs ground-truth action chunk;
* ``distill`` -- match the frozen dense teacher, via KL on action-range logits
  (openvla, when logits available) OR action-chunk L1 + action-token hidden MSE
  (starVLA, when only actions/hidden available).

Weights default to the verified 1:1 settings.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from ..core import losses
from ..core.registry import register
from ..core.types import Batch, StudentOutput, TeacherOutput
from .base import RecoveryObjective


@register("recovery_objective", "distill_bc")
class DistillBC(RecoveryObjective):
    """L1 behaviour cloning + frozen-teacher distillation (KL or action+hidden)."""

    name = "distill_bc"

    def __init__(self, bc_weight: float = 1.0, kl_weight: float = 1.0,
                 distill_action_weight: float = 1.0, hidden_weight: float = 1.0,
                 temperature: float = 1.0):
        self.bc_weight = bc_weight
        self.kl_weight = kl_weight
        self.distill_action_weight = distill_action_weight
        self.hidden_weight = hidden_weight
        self.temperature = temperature

    def compute(self, student: StudentOutput, teacher: TeacherOutput, batch: Batch,
                anchor: Optional[StudentOutput] = None) -> Tuple[object, Dict[str, float]]:
        parts: Dict[str, float] = {}

        # core supervised term
        bc = losses.bc_action_l1(student.action, batch.gt_action)
        loss = self.bc_weight * bc
        parts["bc"] = float(bc)

        # distillation: prefer logit-KL (openvla); else action+hidden (starVLA)
        if student.logits is not None and teacher.logits is not None and self.kl_weight > 0:
            kl = losses.distill_kl(student.logits, teacher.logits, self.temperature)
            loss = loss + self.kl_weight * kl
            parts["kl"] = float(kl)
        else:
            d_act = losses.action_distill_l1(student.action, teacher.action)
            loss = loss + self.distill_action_weight * d_act
            parts["distill_action"] = float(d_act)
            if student.hidden is not None and teacher.hidden is not None and self.hidden_weight > 0:
                hm = losses.feature_mse(
                    student.hidden[max(student.hidden)], teacher.hidden[max(teacher.hidden)]
                )
                loss = loss + self.hidden_weight * hm
                parts["hidden_mse"] = float(hm)

        parts["total"] = float(loss)
        return loss, parts
