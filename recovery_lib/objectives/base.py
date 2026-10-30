"""
[2] Recovery objectives (base interface).

An objective turns (student forward, teacher forward, batch[, anchor forward]) into
a scalar loss + a breakdown dict. It is pure: it calls the verified loss kernels in
``core.losses`` and works on whatever array type the backend passes (numpy for the
mock, torch tensors for real backends -- graph preserved for backprop).

Contract::

    loss, parts = objective.compute(student_out, teacher_out, batch, anchor_out=None)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from ..core.types import Batch, StudentOutput, TeacherOutput


class RecoveryObjective(ABC):
    name: str = "base"

    @abstractmethod
    def compute(
        self,
        student: StudentOutput,
        teacher: TeacherOutput,
        batch: Batch,
        anchor: Optional[StudentOutput] = None,
    ) -> Tuple[object, Dict[str, float]]:
        """Return ``(loss, parts)``. ``loss`` is a torch tensor (real backend) or a
        float (numpy backend); ``parts`` is a name→float breakdown for logging."""
