"""
recovery_lib.core.types
=======================

Data structures for the recovery half of the prune→recover→eval loop.

Recovery sits downstream of ``mask_probe_lib``: a mask was chosen there; here we
restore the success rate (SR) the pruning cost. The verified recipes converge on
ONE architecture (a zero-init residual boundary adapter) and differ in *what else*
is co-adapted, *what loss* trains it, and *what data* supervises it -- so those
three axes are first-class objects below.

Naming stays consistent with ``mask_probe_lib``: ``mask`` = layers removed; ``sr`` =
closed-loop success rate (the only arbiter, W5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
#  What is trained (the verified "joint co-adaptation" trainable set)          #
# --------------------------------------------------------------------------- #
@dataclass
class TrainableSpec:
    """Which module groups to unfreeze, and the LR for each.

    The single most load-bearing recovery finding (both repos): recovery is
    **joint co-adaptation**, not adapter-only reconstruction. Freezing the
    downstream collapses SR (starVLA frozen-downstream → 0.005; openvla
    adapter-only k16 85.5 % vs joint 98.0 %). So the verified default unfreezes the
    boundary adapter + action head + proprio (+ gently the kept LLM layers), each
    at its own LR.
    """

    lrs: Dict[str, float] = field(default_factory=dict)   # group name -> learning rate
    note: str = ""

    @property
    def groups(self) -> List[str]:
        return list(self.lrs)

    # -- verified presets --------------------------------------------------- #
    @classmethod
    def joint(cls) -> "TrainableSpec":
        """VERIFIED foundational recipe (openvla Phase-2 / starVLA baseline)."""
        return cls(
            lrs={"adapter": 1e-3, "action_head": 1e-3, "proprio": 1e-3, "kept_layers": 1e-5},
            note="boundary adapter + action_head + proprio + gently-unfrozen kept layers",
        )

    @classmethod
    def onpolicy(cls) -> "TrainableSpec":
        """VERIFIED ★ on-policy continuation (gentle LRs everywhere)."""
        return cls(
            lrs={"adapter": 1e-4, "action_head": 1e-5, "kept_layers": 1e-5},
            note="gentle co-adaptation while distilling on student's own states",
        )

    @classmethod
    def adapter_only(cls) -> "TrainableSpec":
        """VERIFIED only at shallow contiguous k (frozen sidecars). Fails at deep k."""
        return cls(lrs={"adapter": 1e-4}, note="adapter only; frozen head/proprio -- shallow k only")

    @classmethod
    def per_task_specialist(cls) -> "TrainableSpec":
        """VERIFIED-conditional specialist: full unfreeze, per-task filtered data."""
        return cls(
            lrs={"adapter": 2e-5, "action_head": 2e-5, "proprio": 2e-5},
            note="full sidecar unfreeze; adapter delta is high-rank so LoRA is insufficient",
        )


# --------------------------------------------------------------------------- #
#  The recipe = trainable set + objective + data + hyperparameters             #
# --------------------------------------------------------------------------- #
@dataclass
class RecoveryRecipe:
    """A fully-specified recovery run."""

    name: str
    trainable: TrainableSpec
    objective: str                       # registry name of a RecoveryObjective
    data: str                            # registry name of a data source
    steps: int = 10_000
    adapter_bottleneck: int = 1024
    teacher: str = "dense"               # frozen dense teacher by default
    hyper: Dict[str, float] = field(default_factory=dict)
    notes: str = ""


# --------------------------------------------------------------------------- #
#  Forward outputs the objectives consume                                      #
# --------------------------------------------------------------------------- #
@dataclass
class StudentOutput:
    """One student (or anchor) forward at action-token positions."""

    action: np.ndarray                                  # (N, chunk, action_dim)
    hidden: Optional[Dict[int, np.ndarray]] = None      # layer -> (N, D), kept layers only
    logits: Optional[np.ndarray] = None                 # (N, action_vocab) at action range
    meta: dict = field(default_factory=dict)


@dataclass
class TeacherOutput:
    """The frozen dense teacher's targets (cached or live)."""

    action: np.ndarray
    hidden: Optional[Dict[int, np.ndarray]] = None
    logits: Optional[np.ndarray] = None


@dataclass
class Batch:
    """A supervision batch. For on-policy/DAgger, ``on_policy`` marks student-visited
    states (teacher relabels them); ``advantage`` carries returns for AWR-style terms."""

    gt_action: np.ndarray                               # (N, chunk, action_dim)
    obs: Optional[object] = None                        # backend-defined model input
    on_policy: bool = False
    advantage: Optional[np.ndarray] = None              # (N,) returns - baseline, in [-1, 1]
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  Result                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class RecoveryResult:
    """Output of a recovery strategy."""

    recipe: str
    mask: Tuple[int, ...]
    final_sr: Optional[float] = None
    sr_curve: List[Tuple[int, float]] = field(default_factory=list)   # (step, sr)
    loss_curve: List[Tuple[int, float]] = field(default_factory=list)
    trainable: Optional[TrainableSpec] = None
    converged: Optional[bool] = None
    notes: str = ""

    @property
    def slope(self) -> float:
        """Mean SR slope over the curve -- the verified recovery decision signal
        (the F-series rule: judge recovery by the SR slope, NOT offline loss)."""
        if len(self.sr_curve) < 2:
            return 0.0
        steps = np.array([s for s, _ in self.sr_curve], dtype=float)
        srs = np.array([v for _, v in self.sr_curve], dtype=float)
        steps = (steps - steps.mean())
        denom = (steps ** 2).sum()
        return float((steps * (srs - srs.mean())).sum() / denom) if denom > 0 else 0.0
