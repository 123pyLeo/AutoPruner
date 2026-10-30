"""
PerTaskSpecialist -- verified-conditional single-task recovery (feeds routing).
===============================================================================

Source
------
* openvla : ``finetune_phase9_per_task.py`` / E30 (dense-OFT distill + full unfreeze +
            per-task data filter).

Recipe
------
* Train: full sidecar unfreeze (adapter + head + proprio), init from the general
  recovered student.  The adapter delta a single task needs is HIGH-RANK, so LoRA /
  ORAR is rank-insufficient (E29 null) -- full unfreeze is required.
* Loss : dense-teacher KL + action L1 (``DistillBC``).
* Data : offline demos FILTERED to one task.

Verified evidence / scope
-------------------------
t8 56 → 76/78 (+20 pt, reproducible). **Verified ONLY as a single-task specialist
that feeds task routing** (see ``deploy/routing.py``); as a standalone deployable
ckpt it MIXES (mean SR < baseline due to collateral damage on other tasks). Use it
to build the per-task ckpt set, then route.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..core.backend import RecoveryBackend
from ..core.registry import register
from ..core.types import RecoveryResult, TrainableSpec
from ..data.offline_demos import OfflineDemoSource
from ..objectives.distill_bc import DistillBC
from .base import RecoveryStrategy


@register("recovery_strategy", "per_task_specialist")
class PerTaskSpecialist(RecoveryStrategy):
    """Per-task full-unfreeze distillation; build one specialist per hard task."""

    name = "per_task_specialist"

    def __init__(self, task_filter: str, steps: int = 3000, eval_every: int = 500,
                 eval_trials: int = 50, warm_start: Optional[object] = None):
        if not task_filter:
            raise ValueError("PerTaskSpecialist requires a task_filter (the single target task)")
        self.task_filter = task_filter
        self.steps = steps
        self.eval_every = eval_every
        self.eval_trials = eval_trials
        self.warm_start = warm_start

    def recover(self, backend: RecoveryBackend, mask: Sequence[int]) -> RecoveryResult:
        backend.build_student(mask, TrainableSpec.per_task_specialist(),
                              init_from=self.warm_start or "current")
        data = OfflineDemoSource(task_filter=self.task_filter)
        sr_curve, loss_curve = self._train_loop(
            backend, data, DistillBC(), self.steps, self.eval_every, eval_trials=self.eval_trials,
        )
        result = RecoveryResult(
            recipe=self.name, mask=tuple(sorted(mask)), final_sr=sr_curve[-1][1],
            sr_curve=sr_curve, loss_curve=loss_curve, trainable=TrainableSpec.per_task_specialist(),
            notes=(f"per-task specialist for '{self.task_filter}'; full unfreeze (LoRA is "
                   f"rank-insufficient). Feed into TaskRouter -- not a standalone deployable ckpt."),
        )
        result.converged = result.slope >= 0
        return result
