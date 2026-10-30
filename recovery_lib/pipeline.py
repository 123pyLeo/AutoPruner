"""
recovery_lib.pipeline
=====================

End-to-end recovery orchestration. Picks up where ``mask_probe_lib`` leaves off:
given a chosen ``mask``, run a verified recovery strategy and gate the result by
its closed-loop SR slope.

    from recovery_lib import RecoveryPipeline
    from recovery_lib.backends.mock import MockRecoveryBackend

    pipe = RecoveryPipeline(MockRecoveryBackend(n_layers=32))
    out = pipe.run(mask=[3, 5, 8, 10, 13, 17, 21, 25, 27, 30])   # from mask_probe_lib
    print(out.result.final_sr, out.gate.passed)

Flow (each step is one recovery_lib layer):

    [3] strategy.recover(backend, mask)        # build student → train → SR curve
        (uses [1] adapter, [2] objective, [4] data internally)
        |
    validation.RecoverySlopeGate               # judge by SR slope, not offline loss
        |
    (optional) OnPolicyDistillTrustRegion       # continue to break the 0.80 ceiling
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from mask_probe_lib.core.types import ValidationResult

from .core.backend import RecoveryBackend
from .core.types import RecoveryResult
from .strategies.base import RecoveryStrategy
from .strategies.phase2_joint import PhaseTwoJointRecovery
from .strategies.onpolicy_distill import OnPolicyDistillTrustRegion
from .validation.recovery_funnel import AntiCollapseCheck, RecoverySlopeGate


@dataclass
class RecoveryPipelineResult:
    result: RecoveryResult
    gate: ValidationResult
    onpolicy_result: Optional[RecoveryResult] = None
    anti_collapse: Optional[ValidationResult] = None


class RecoveryPipeline:
    """Run a base recovery (+ optional on-policy continuation) with slope gating."""

    def __init__(self, backend: RecoveryBackend, strategy: Optional[RecoveryStrategy] = None,
                 gate: Optional[RecoverySlopeGate] = None):
        self.backend = backend
        self.strategy = strategy or PhaseTwoJointRecovery()
        self.gate = gate or RecoverySlopeGate(min_slope=0.0, min_final_lift=0.0)

    def run(self, mask: Sequence[int], continue_onpolicy: bool = False,
            onpolicy_steps: int = 3000) -> RecoveryPipelineResult:
        # [3] base recovery
        result = self.strategy.recover(self.backend, mask)
        gate = self.gate.evaluate(result)

        out = RecoveryPipelineResult(result=result, gate=gate)

        # optional: continue with the on-policy trust-region recipe to break the ceiling
        if continue_onpolicy and gate.passed:
            op = OnPolicyDistillTrustRegion(steps=onpolicy_steps)
            op_result = op.recover(self.backend, mask)
            out.onpolicy_result = op_result
            out.anti_collapse = AntiCollapseCheck().evaluate(op_result)
        return out
