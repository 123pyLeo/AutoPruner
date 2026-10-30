"""
Layer 5 -- Validation discipline.

The meta-methods that decide whether a layer-selection signal can be trusted, and
that screen candidate masks cheaply before expensive training.

Verified components (see ``METHODS.md``):
    probe_vs_sr           -- ProbeVsSRValidation   (correlate metric vs SR; the arbiter)
    anti_anchor           -- AntiAnchorControl     (causal negative control)
    recoverability_funnel -- RecoverabilityFunnel  (F-series staged screening)
"""

from .correlation import ProbeVsSRValidation
from .anti_anchor import AntiAnchorControl
from .recoverability_funnel import RecoverabilityFunnel

__all__ = ["ProbeVsSRValidation", "AntiAnchorControl", "RecoverabilityFunnel"]
