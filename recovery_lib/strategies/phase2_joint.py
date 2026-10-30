"""
PhaseTwoJointRecovery -- the verified foundational recovery recipe.
===================================================================

Source
------
* openvla : ``finetune_phase2_adapter.py`` (C2v2) -- the recipe the agentic framework
            hardcodes as the only choice.
* starVLA : ``Phase2AdapterTrainer`` -- the 0.788 baseline everything builds on.

Recipe
------
* Train: boundary adapter + action_head + proprio (+ gently-unfrozen kept layers).
* Loss : action L1 + frozen-teacher distillation (KL on action-range logits, or
         action-L1 + hidden-MSE).  (objective = ``DistillBC``)
* Data : offline demos.

Verified evidence
-----------------
openvla: k16 → 98.0 %, k28 keep{L0,L1,L30,L31} → 96.5 % = dense, k8 libero_10 → 92 %.
starVLA: k10anchor → 0.788 (394/500). Joint beats adapter-only (frozen sidecars):
k16 85.5 % vs 98.0 % -- the +12.5 pt comes from co-adapting head+proprio.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..core.backend import RecoveryBackend
from ..core.registry import register
from ..core.types import RecoveryResult, TrainableSpec
from ..data.offline_demos import OfflineDemoSource
from ..objectives.distill_bc import DistillBC
from .base import RecoveryStrategy


@register("recovery_strategy", "phase2_joint")
class PhaseTwoJointRecovery(RecoveryStrategy):
    """Boundary adapter + head + proprio, distilled on offline demos."""

    name = "phase2_joint"

    def __init__(self, steps: int = 10_000, bottleneck: int = 1024, eval_every: int = 2000,
                 eval_trials: int = 50, trainable: Optional[TrainableSpec] = None,
                 objective: Optional[DistillBC] = None, data: Optional[OfflineDemoSource] = None,
                 task_filter: Optional[str] = None):
        self.steps = steps
        self.bottleneck = bottleneck
        self.eval_every = eval_every
        self.eval_trials = eval_trials
        self.trainable = trainable or TrainableSpec.joint()
        self.objective = objective or DistillBC()
        self.data = data or OfflineDemoSource(task_filter=task_filter)

    def recover(self, backend: RecoveryBackend, mask: Sequence[int]) -> RecoveryResult:
        backend.build_student(mask, self.trainable, adapter_bottleneck=self.bottleneck)
        sr_curve, loss_curve = self._train_loop(
            backend, self.data, self.objective, self.steps, self.eval_every,
            eval_trials=self.eval_trials,
        )
        result = RecoveryResult(
            recipe=self.name, mask=tuple(sorted(mask)),
            final_sr=sr_curve[-1][1], sr_curve=sr_curve, loss_curve=loss_curve,
            trainable=self.trainable,
            notes="boundary adapter + head + proprio, frozen-teacher distill on offline demos",
        )
        result.converged = result.slope >= 0
        return result
