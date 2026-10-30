"""
recovery_lib.core.backend
=========================

``RecoveryBackend`` extends ``mask_probe_lib.core.ModelBackend`` with the training
and rollout hooks recovery needs. Everything above this seam (objectives,
strategies, validation, deploy) is model-agnostic; a concrete VLA implements one
``RecoveryBackend`` subclass and every verified recipe runs unchanged.

The design keeps the **verified loss formulas inside the library** (the objectives
compute them from student/teacher outputs) and leaves only the mechanics to the
backend: build the student, run forwards, apply gradients, snapshot an anchor,
roll out on-policy, and evaluate SR.

Reference implementations:
    * backends/mock.py        -- synthetic, fully runnable (simulates training→SR)
    * backends/openvla_oft.py -- skeleton citing finetune_phase2_adapter.py etc.
    * backends/qwenoft.py     -- skeleton citing finetune_onpolicy_distill.py etc.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Iterable, Optional, Sequence

from mask_probe_lib.core.backend import ModelBackend

from .types import Batch, RecoveryRecipe, StudentOutput, TeacherOutput, TrainableSpec


class RecoveryBackend(ModelBackend):
    """A VLA that can be pruned, recovered (trained), rolled out, and evaluated."""

    # --- build the student to recover -------------------------------------- #
    @abstractmethod
    def build_student(self, mask: Sequence[int], trainable: TrainableSpec,
                      adapter_bottleneck: int = 1024, init_from: Optional[object] = None) -> None:
        """Physically prune ``mask``, install a zero-init residual boundary adapter,
        and set ``requires_grad`` per ``trainable`` (build the per-group optimizer).

        ``init_from`` warm-starts from a prior recovered student (used by the
        on-policy continuation). VERIFIED architecture constraints: the adapter must
        be nonlinear + (attention or MLP) + SGD-trained -- linear/closed-form/LoRA-
        only adapters were verified to give 0 % SR.
        """

    # --- forwards the objectives consume ----------------------------------- #
    @abstractmethod
    def student_forward(self, batch: Batch) -> StudentOutput:
        """Forward the current (masked + adapter) student at action-token positions."""

    @abstractmethod
    def teacher_forward(self, batch: Batch) -> TeacherOutput:
        """Frozen dense-teacher targets for the batch (cached or live).

        For on-policy/DAgger batches this is the covariate-shift fix: the teacher is
        queried on the student's OWN visited states (``batch.on_policy`` True).
        """

    # --- optimisation ------------------------------------------------------ #
    @abstractmethod
    def apply_gradients(self, loss) -> float:
        """Backward + per-group optimizer step on a backend-native loss tensor.
        Returns the scalar loss value. (For numpy backends, ``loss`` is a float.)"""

    # --- trust-region anchor support --------------------------------------- #
    def snapshot_anchor(self) -> object:
        """Return a frozen copy of the current student (for the trust-region term).
        Default: not supported -- override for the on-policy recipe."""
        raise NotImplementedError("snapshot_anchor is needed for trust-region recipes")

    def anchor_forward(self, snapshot: object, batch: Batch) -> StudentOutput:
        """Forward a frozen anchor snapshot on the batch."""
        raise NotImplementedError("anchor_forward is needed for trust-region recipes")

    # --- data sources ------------------------------------------------------ #
    @abstractmethod
    def iter_offline_batches(self, steps: int, batch_size: int = 4,
                             task_filter: Optional[str] = None) -> Iterable[Batch]:
        """Stream offline demo batches (optionally filtered to one task)."""

    def collect_onpolicy_rollouts(self, n_rollouts: int = 500,
                                  success_only: bool = False) -> Iterable[Batch]:
        """Roll out the current student and return batches of the states it VISITED
        (DAgger). Default: not supported -- override for the on-policy recipe."""
        raise NotImplementedError("collect_onpolicy_rollouts is needed for on-policy recipes")

    # --- evaluation -------------------------------------------------------- #
    def evaluate_sr(self, n_trials: int = 50) -> float:
        """Closed-loop SR of the CURRENT recovered student. Default routes to the
        inherited oracle for the built mask; override for a live eval."""
        mask = getattr(self, "_current_mask", None)
        if mask is None:
            raise RuntimeError("build_student must be called before evaluate_sr")
        return self.closed_loop_sr(mask, recovery=getattr(self, "_current_recovery", "joint"),
                                   n_trials=n_trials)
