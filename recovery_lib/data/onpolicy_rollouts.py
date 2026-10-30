"""
OnPolicyRolloutSource -- DAgger states (the covariate-shift fix).
=================================================================

Roll out the CURRENT recovered student, collect the states it actually visited
(not the demo distribution), and let the teacher relabel them on the fly. This is
the verified data lever behind the 0.828 win: distilling on the student's own
reachable states fixes the covariate shift that offline demos miss.

``success_only=False`` (keep ALL visited states, including from failed episodes) is
the verified setting -- the failures are exactly where the teacher's correction
matters.
"""

from __future__ import annotations

from typing import Iterable

from ..core.backend import RecoveryBackend
from ..core.registry import register
from ..core.types import Batch


@register("recovery_data", "onpolicy_rollouts")
class OnPolicyRolloutSource:
    """Wrap ``backend.collect_onpolicy_rollouts``; marks batches ``on_policy=True``."""

    name = "onpolicy_rollouts"

    def __init__(self, n_rollouts: int = 500, success_only: bool = False):
        self.n_rollouts = n_rollouts
        self.success_only = success_only

    def batches(self, backend: RecoveryBackend, steps: int) -> Iterable[Batch]:
        rollouts = list(backend.collect_onpolicy_rollouts(self.n_rollouts, self.success_only))
        if not rollouts:
            return
        for i in range(steps):
            b = rollouts[i % len(rollouts)]
            b.on_policy = True
            yield b
