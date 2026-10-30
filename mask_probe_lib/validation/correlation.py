"""
ProbeVsSRValidation -- the discipline that decides if a metric is trustworthy.
==============================================================================

Source
------
* openvla : ``scripts/validate_attn_rank_vs_sr.py``, ``scripts/perlayer_residual_vs_sr.py``,
            the correlation step inside ``scripts/probe_boundary_alignment.py``.

This is the *meta-method* behind the whole library: a probe metric is "verified"
iff it correlates with ground-truth closed-loop SR across a set of known-SR masks
(Spearman / Pearson).  This is exactly how the source repo separated the verified
methods from the misleading ones:

    VERIFIED        recoverability adapter cos, drift predictor, knockout dAll, planning R²
    MISLEADING      FIM (anti-correlated), CKA (hides L30), BI (rho=0.38), act3k (0.29)

Use it to (a) re-confirm a verified probe on your own model, and (b) screen any
*new* probe idea before trusting it -- never adopt a layer-selection metric that
has not been correlated against real SR.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import numpy as np

from ..core.kernels import pearson, spearman
from ..core.registry import register
from ..core.types import ValidationResult


@register("validation", "probe_vs_sr")
class ProbeVsSRValidation:
    """Correlate a per-mask probe metric against ground-truth SR."""

    name = "probe_vs_sr"

    def __init__(self, min_spearman: float = 0.6):
        #: a metric is accepted only if |Spearman| with SR clears this bar
        self.min_spearman = min_spearman

    def evaluate(
        self,
        metric_values: Sequence[float],
        sr_values: Sequence[float],
        higher_metric_means_higher_sr: bool = True,
    ) -> ValidationResult:
        """Given paired (metric, SR) over known masks, report correlation + verdict."""
        m = np.asarray(metric_values, dtype=float)
        s = np.asarray(sr_values, dtype=float)
        rho = spearman(m, s)
        r = pearson(m, s)
        signed = rho if higher_metric_means_higher_sr else -rho
        passed = signed >= self.min_spearman
        return ValidationResult(
            name=self.name,
            passed=bool(passed),
            statistic={"spearman": float(rho), "pearson": float(r),
                       "signed_spearman": float(signed), "n": int(m.size)},
            detail=(
                f"Spearman(metric, SR)={rho:+.3f} (signed {signed:+.3f}); "
                f"{'TRUSTWORTHY' if passed else 'REJECT -- below bar / wrong sign'} "
                f"@ threshold {self.min_spearman}."
            ),
        )

    def screen_probe(
        self,
        backend,
        masks: Sequence[Sequence[int]],
        metric_fn: Callable[[object, Sequence[int]], float],
        recovery: str = None,
        higher_metric_means_higher_sr: bool = True,
    ) -> ValidationResult:
        """End-to-end: compute ``metric_fn(backend, mask)`` and SR per mask, correlate.

        Requires ``backend.closed_loop_sr`` (the oracle).  Masks with unknown SR
        (oracle returns None) are skipped.
        """
        mv, sv = [], []
        for mask in masks:
            sr = backend.closed_loop_sr(list(mask), recovery=recovery)
            if sr is None:
                continue
            mv.append(metric_fn(backend, mask))
            sv.append(sr)
        return self.evaluate(mv, sv, higher_metric_means_higher_sr)
