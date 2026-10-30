"""
Recovery validation -- the discipline that decides if a recovery is real.
=========================================================================

Three verified guards, distilled from the long list of recovery failures where
offline loss dropped but closed-loop SR did not move (anchor-KD, E34, Theseus):

* **slope gate** (F-series) -- judge a recovery by its closed-loop SR *slope*, not
  offline loss. A flat/negative slope kills the recipe even if the loss fell.
* **frozen-sidecar ablation** -- re-run with head/proprio frozen; if SR barely
  changes, the gain came from the adapter alone; if SR collapses, recovery is the
  verified JOINT co-adaptation (openvla k16 frozen 85.5 % vs joint 98.0 %;
  starVLA frozen-downstream 0.005).
* **anti-collapse check** -- for on-policy continuation, confirm SR never drops far
  below the start (the trust-region anchor is doing its job); a deep dip means the
  basin collapsed (the failure mode SWA/GRACE/naive-RL hit).
"""

from __future__ import annotations

from typing import Optional, Sequence

from mask_probe_lib.core.types import ValidationResult

from ..core.backend import RecoveryBackend
from ..core.registry import register
from ..core.types import RecoveryResult, TrainableSpec


@register("validation", "recovery_slope_gate")
class RecoverySlopeGate:
    """Pass a recovery only if its SR slope (and final lift) clear a bar."""

    name = "recovery_slope_gate"

    def __init__(self, min_slope: float = 0.0, min_final_lift: float = 0.0):
        self.min_slope = min_slope
        self.min_final_lift = min_final_lift

    def evaluate(self, result: RecoveryResult) -> ValidationResult:
        start = result.sr_curve[0][1] if result.sr_curve else 0.0
        final = result.final_sr if result.final_sr is not None else start
        lift = final - start
        slope = result.slope
        passed = slope >= self.min_slope and lift >= self.min_final_lift
        return ValidationResult(
            name=self.name,
            passed=bool(passed),
            statistic={"slope": float(slope), "start_sr": float(start),
                       "final_sr": float(final), "lift": float(lift)},
            detail=(f"SR {start:.3f}→{final:.3f} (lift {lift:+.3f}, slope {slope:+.2e}); "
                    f"{'KEEP' if passed else 'KILL -- flat/negative slope (offline loss is not SR)'}"),
        )


@register("validation", "frozen_sidecar_ablation")
class FrozenSidecarAblation:
    """Attribute the recovery: adapter-only vs joint co-adaptation."""

    name = "frozen_sidecar_ablation"

    def __init__(self, strategy, collapse_margin: float = 0.10):
        self.strategy = strategy
        self.collapse_margin = collapse_margin

    def run(self, backend: RecoveryBackend, mask: Sequence[int], joint_sr: float) -> ValidationResult:
        # re-run the same strategy but with only the adapter unfrozen
        self.strategy.trainable = TrainableSpec.adapter_only()
        adapter_only = self.strategy.recover(backend, mask)
        gap = joint_sr - (adapter_only.final_sr or 0.0)
        joint_needed = gap >= self.collapse_margin
        return ValidationResult(
            name=self.name,
            passed=True,
            statistic={"joint_sr": float(joint_sr),
                       "adapter_only_sr": float(adapter_only.final_sr or 0.0), "gap": float(gap)},
            detail=(f"joint {joint_sr:.3f} vs adapter-only {adapter_only.final_sr:.3f} (gap {gap:+.3f}); "
                    f"recovery is {'JOINT co-adaptation (head/proprio required)' if joint_needed else 'adapter-dominated (shallow k)'}"),
        )


@register("validation", "anti_collapse_check")
class AntiCollapseCheck:
    """For on-policy continuation: confirm the basin never collapsed."""

    name = "anti_collapse_check"

    def __init__(self, max_dip: float = 0.10):
        self.max_dip = max_dip

    def evaluate(self, result: RecoveryResult) -> ValidationResult:
        if not result.sr_curve:
            return ValidationResult(self.name, False, {}, "no SR curve")
        start = result.sr_curve[0][1]
        lowest = min(v for _, v in result.sr_curve)
        dip = start - lowest
        passed = dip <= self.max_dip
        return ValidationResult(
            name=self.name,
            passed=bool(passed),
            statistic={"start_sr": float(start), "lowest_sr": float(lowest), "dip": float(dip)},
            detail=(f"start {start:.3f}, lowest {lowest:.3f} (dip {dip:.3f}); "
                    f"{'trust-region held' if passed else 'BASIN COLLAPSE -- anchor too weak / wrong lever'}"),
        )
