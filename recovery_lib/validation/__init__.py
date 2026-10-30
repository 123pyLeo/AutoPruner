"""
Recovery validation discipline.

    recovery_slope_gate     -- RecoverySlopeGate      (judge by SR slope, not offline loss)
    frozen_sidecar_ablation -- FrozenSidecarAblation  (adapter-only vs joint co-adaptation)
    anti_collapse_check     -- AntiCollapseCheck       (trust-region held the basin?)
"""

from .recovery_funnel import AntiCollapseCheck, FrozenSidecarAblation, RecoverySlopeGate

__all__ = ["RecoverySlopeGate", "FrozenSidecarAblation", "AntiCollapseCheck"]
