"""
[3] Recovery strategies (base interface) -- the end-to-end verified recipes.

A strategy orchestrates: build the student (adapter + trainable set) → loop over a
data source computing the objective and applying gradients → periodically evaluate
SR → return a :class:`RecoveryResult` whose **slope** is the verified decision
signal (judge recovery by the SR slope, not offline loss -- F-series rule).

Contract::

    result = strategy.recover(backend, mask)
    result.final_sr, result.slope, result.sr_curve
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

from ..core.backend import RecoveryBackend
from ..core.types import Batch, RecoveryResult, StudentOutput, TrainableSpec


class RecoveryStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def recover(self, backend: RecoveryBackend, mask: Sequence[int]) -> RecoveryResult:
        """Train a recovery for ``mask`` and return the result (with SR curve)."""

    # -- shared training loop (used by the concrete strategies) ------------- #
    def _train_loop(
        self,
        backend: RecoveryBackend,
        data_source,
        objective,
        steps: int,
        eval_every: int,
        anchor_snapshot: Optional[object] = None,
        eval_trials: int = 50,
    ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        sr_curve: List[Tuple[int, float]] = [(0, backend.evaluate_sr(eval_trials))]
        loss_curve: List[Tuple[int, float]] = []
        for step, batch in enumerate(data_source.batches(backend, steps), start=1):
            student = backend.student_forward(batch)
            teacher = backend.teacher_forward(batch)
            anchor = (backend.anchor_forward(anchor_snapshot, batch)
                      if anchor_snapshot is not None else None)
            loss, parts = objective.compute(student, teacher, batch, anchor=anchor)
            backend.apply_gradients(loss)
            loss_curve.append((step, parts.get("total", float(loss))))
            if step % eval_every == 0 or step == steps:
                sr_curve.append((step, backend.evaluate_sr(eval_trials)))
        return sr_curve, loss_curve
