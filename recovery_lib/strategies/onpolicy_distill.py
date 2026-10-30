"""
OnPolicyDistillTrustRegion -- the verified ★ recovery win (breaks the 0.80 ceiling).
====================================================================================

Source
------
* starVLA : ``finetune_onpolicy_distill.py`` + ``qwenoft_libero10_onpolicy_distill_k10anchor.yaml``.

Recipe
------
Continue-train a Phase-2 student with the **matched triad** that is the only thing
verified to beat the frozen-backbone 0.80 ceiling:

1. **on-policy states** -- distil on the student's OWN rollout states (``OnPolicyRolloutSource``),
   teacher relabelled live (covariate-shift fix);
2. **co-adapting weights** -- gentle unfreeze of adapter + head + kept layers (``TrainableSpec.onpolicy``);
3. **anti-collapse anchor** -- ``TrustRegionDistill`` adds a relative multi-hidden anchor +
   a trust-region smooth-L1 to a FROZEN snapshot of the pre-continuation student.

Verified evidence
-----------------
k10anchor 0.788 → **0.828** at 3000 steps (+4.0 pp, significant; first non-oracle
method past the 0.80 ceiling; fixed the hard transition task 28 %→50 %).

Ablations that prove each leg is load-bearing (all in negative_baselines):
offline multi-anchor 0.708 (no on-policy), frozen-downstream 0.005 (no co-adapt),
SWA/SCP/GRACE/best-of-N all ≤ baseline (wrong lever).
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..core.backend import RecoveryBackend
from ..core.registry import register
from ..core.types import RecoveryResult, TrainableSpec
from ..data.onpolicy_rollouts import OnPolicyRolloutSource
from ..objectives.trust_region import TrustRegionDistill
from .base import RecoveryStrategy
from .phase2_joint import PhaseTwoJointRecovery


@register("recovery_strategy", "onpolicy_distill")
class OnPolicyDistillTrustRegion(RecoveryStrategy):
    """On-policy DAgger distillation + trust-region anchor, continued from a Phase-2 student."""

    name = "onpolicy_distill"

    def __init__(self, steps: int = 3000, eval_every: int = 1000, eval_trials: int = 50,
                 trust_region_weight: float = 0.5, multi_hidden_weight: float = 10.0,
                 n_rollouts: int = 500, warm_start: Optional[object] = None,
                 base_strategy: Optional[PhaseTwoJointRecovery] = None):
        self.steps = steps
        self.eval_every = eval_every
        self.eval_trials = eval_trials
        self.trust_region_weight = trust_region_weight
        self.multi_hidden_weight = multi_hidden_weight
        self.n_rollouts = n_rollouts
        self.warm_start = warm_start
        self.base_strategy = base_strategy        # if set, run Phase-2 first to get the 0.788 student

    def recover(self, backend: RecoveryBackend, mask: Sequence[int]) -> RecoveryResult:
        mask = tuple(sorted(mask))

        # 0) obtain the Phase-2 student to continue from (run it, or assume built)
        base_result = None
        if self.base_strategy is not None:
            base_result = self.base_strategy.recover(backend, mask)
        # rebuild with the gentle on-policy trainable set, warm-started from current
        backend.build_student(mask, TrainableSpec.onpolicy(),
                              init_from=self.warm_start or "current")

        # 1) freeze a snapshot of the pre-continuation student for the trust region
        anchor = backend.snapshot_anchor()

        # 2) objective with the verified anchor terms; removed_after = boundary layers
        removed_after = self._boundaries(mask)
        objective = TrustRegionDistill(
            multi_hidden_weight=self.multi_hidden_weight,
            trust_region_weight=self.trust_region_weight,
            removed_after=removed_after,
        )

        # 3) on-policy data source (the covariate-shift fix)
        data = OnPolicyRolloutSource(n_rollouts=self.n_rollouts, success_only=False)

        start_sr = backend.evaluate_sr(self.eval_trials)
        sr_curve, loss_curve = self._train_loop(
            backend, data, objective, self.steps, self.eval_every,
            anchor_snapshot=anchor, eval_trials=self.eval_trials,
        )
        result = RecoveryResult(
            recipe=self.name, mask=mask, final_sr=sr_curve[-1][1],
            sr_curve=sr_curve, loss_curve=loss_curve, trainable=TrainableSpec.onpolicy(),
            notes=(f"on-policy distill + trust-region anchor (w={self.trust_region_weight}); "
                   f"continued from Phase-2 start SR={start_sr:.3f}"),
        )
        result.converged = result.final_sr is not None and result.final_sr >= start_sr
        return result

    @staticmethod
    def _boundaries(mask: Sequence[int]) -> list:
        """Kept-layer indices that immediately follow a removed segment (post-boundary)."""
        m = sorted(mask)
        if not m:
            return []
        bset = set(m)
        return [l + 1 for l in m if (l + 1) not in bset]
