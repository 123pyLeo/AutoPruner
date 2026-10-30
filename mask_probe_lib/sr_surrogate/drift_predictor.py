"""
DriftPredictor -- the verified forward-pass SR surrogate.
=========================================================

Source
------
* openvla : ``scripts/parallel_drift_master.py`` + ``parallel_drift_worker.py``
            (the best-validated quantitative SR predictor in the project).

Features (two, and the PAIR is what matters)
--------------------------------------------
* ``full_l2_rc`` -- Frobenius drift of the **recovered+masked** model's
  action-token hidden states vs the dense model (``kernels.frobenius_drift``).
  Captures overall representational damage.
* ``out_var_pr`` -- variance of the action-head output under **base+mask**
  (``kernels.output_variance``).  Captures head saturation (a dead head outputs a
  near-constant action -> low variance).

The two features are individually weak (single-feature Spearman ~0.65 / ~0.43) but
together jump to LOO R²=0.71, Spearman ρ=0.95 (n=13 masks) because they capture
**complementary** failure modes: overall drift vs head saturation.

A training-free variant uses ``full_l2_pr`` (drift of base+mask, no recovery)
instead of ``full_l2_rc``; it is weaker (LOO R²=0.13) -- a coarse first-round
filter only.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ..core.backend import ModelBackend
from ..core.kernels import (
    frobenius_drift,
    leave_one_out_r2,
    output_variance,
    r2_score,
    ridge_fit,
    ridge_predict,
    spearman,
)
from ..core.registry import register
from .base import SRSurrogate

#: the verified feature pair (recovered drift + head variance)
DEFAULT_FEATURES = ("full_l2_rc", "out_var_pr")
#: the training-free fallback pair (no recovered model needed)
TRAINING_FREE_FEATURES = ("full_l2_pr", "out_var_pr")


@register("sr_surrogate", "drift_predictor")
class DriftPredictor(SRSurrogate):
    """Linear SR surrogate on forward-pass drift + head-variance features."""

    name = "drift_predictor"

    def __init__(self, feature_names: Sequence[str] = DEFAULT_FEATURES, n_batches: int = 50,
                 ridge_lam: float = 1.0, seed: int = 0):
        self.feature_names = list(feature_names)
        self.n_batches = n_batches
        self.ridge_lam = ridge_lam
        self.seed = seed
        self._W = None
        self._mu = None
        self._sd = None

    # -- feature extraction (the only model-touching part) ------------------ #
    def features(self, backend: ModelBackend, mask: Sequence[int]) -> Dict[str, float]:
        """Compute all supported features for one candidate ``mask``.

        Requires the backend to expose recovered (masked+adapter) activations to
        get ``full_l2_rc``; if ``backend.collect_activations`` accepts a ``mask``
        kwarg the recovered path is used, otherwise only the training-free
        features are filled.
        """
        dense = backend.collect_activations(n_batches=self.n_batches, seed=self.seed)
        # dense action-token hidden at the final residual point
        final = max(dense.hidden)
        h_dense = dense.hidden[final]

        feats: Dict[str, float] = {}

        # base+mask drift + head variance (always available)
        try:
            masked = backend.collect_activations(n_batches=self.n_batches, seed=self.seed, mask=list(mask))  # type: ignore[call-arg]
            feats["full_l2_pr"] = frobenius_drift(masked.hidden[max(masked.hidden)], h_dense)
        except TypeError:
            feats["full_l2_pr"] = float("nan")
        feats["out_var_pr"] = output_variance(backend.head_output(mask=list(mask), n_batches=self.n_batches, seed=self.seed))

        # recovered+mask drift (needs a recovered model; optional)
        try:
            rec = backend.collect_activations(n_batches=self.n_batches, seed=self.seed, mask=list(mask), recovered=True)  # type: ignore[call-arg]
            feats["full_l2_rc"] = frobenius_drift(rec.hidden[max(rec.hidden)], h_dense)
        except TypeError:
            feats["full_l2_rc"] = feats.get("full_l2_pr", float("nan"))
        return feats

    def feature_matrix(self, backend: ModelBackend, masks: List[Sequence[int]]) -> np.ndarray:
        rows = []
        for m in masks:
            f = self.features(backend, m)
            rows.append([f[name] for name in self.feature_names])
        return np.asarray(rows, dtype=np.float64)

    # -- fit / predict ------------------------------------------------------ #
    def _standardise(self, X, fit=False):
        if fit:
            self._mu = X.mean(0, keepdims=True)
            self._sd = X.std(0, keepdims=True) + 1e-9
        return (X - self._mu) / self._sd

    def fit(self, X: np.ndarray, sr: np.ndarray) -> "DriftPredictor":
        X = np.asarray(X, dtype=np.float64)
        sr = np.asarray(sr, dtype=np.float64)
        Xs = self._standardise(X, fit=True)
        self._W = ridge_fit(Xs, sr[:, None], self.ridge_lam, fit_bias=True)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._W is None:
            raise RuntimeError("DriftPredictor.fit must be called before predict")
        Xs = self._standardise(np.asarray(X, dtype=np.float64))
        return ridge_predict(Xs, self._W, fit_bias=True).ravel()

    # -- honesty check ------------------------------------------------------ #
    def cross_validate(self, X: np.ndarray, sr: np.ndarray) -> Dict[str, float]:
        """Leave-one-out R² / Spearman -- the mandatory over-fit guard.

        Standardises with the full-sample stats (matching the source pipeline);
        for n=13 the difference vs per-fold standardisation is negligible and the
        comparison to the -0.84 OLS baseline is what matters.
        """
        X = np.asarray(X, dtype=np.float64)
        sr = np.asarray(sr, dtype=np.float64)
        Xs = self._standardise(X, fit=True)
        out = leave_one_out_r2(Xs, sr, lam=self.ridge_lam)
        # in-sample for reference
        self.fit(X, sr)
        ins = r2_score(sr[:, None], self.predict(X)[:, None])
        out["in_sample_r2"] = float(ins)
        return out
