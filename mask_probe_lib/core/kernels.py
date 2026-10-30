"""
mask_probe_lib.core.kernels
===========================

Pure, model-agnostic numerical kernels.  **Every formula here is a verified
metric distilled from openvla-oft / starVLA** (see ``METHODS.md`` for the source
script and the evidence that it predicts closed-loop SR).

Design rules
------------
* numpy-only (torch is optional and only used by the trainable MLP adapter).
* no model, no I/O -- these operate on already-extracted activation matrices,
  so they are unit-testable with synthetic arrays (see ``tests/``).
* one canonical implementation per metric.  The source repos had 2-3 slightly
  different copies of ``effective_rank`` / ``linear_cka``; we standardise here.

The kernels are grouped by the pipeline layer that consumes them.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_EPS = 1e-12


def _to_numpy(x) -> np.ndarray:
    """Accept torch tensors or numpy arrays; always return float64 ndarray."""
    if hasattr(x, "detach"):                 # torch.Tensor
        x = x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float64)


# ===========================================================================
#  Spectral / manifold-geometry kernels   (ManifoldGeometryProbe, boundary)  #
# ===========================================================================
def effective_rank(matrix) -> float:
    """Roy-Vetterli effective rank = ``exp(spectral entropy)``.

    A low effective rank means the matrix's energy lives in few directions, i.e.
    the transform is low-rank and **recoverable by a small adapter**.  This is
    the canonical form used across ``per_layer_manifold.py``, ``probe_mask_sweep.py``,
    ``compute_attention_rank.py``.
    """
    M = _to_numpy(matrix)
    s = np.linalg.svd(M, compute_uv=False)
    s = s[s > 1e-10]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    H = -(p * np.log(p + _EPS)).sum()
    return float(np.exp(H))


def cumulative_energy_rank(matrix, frac: float = 0.9) -> int:
    """Number of singular values needed to capture ``frac`` of the squared energy.

    ``probe_mask_sweep.py`` reports rank@50/90/99 % of the cumulative residual Δ;
    a Δ whose 50 % energy sits in ~2 singular values is trivially recoverable.
    """
    M = _to_numpy(matrix)
    s = np.linalg.svd(M, compute_uv=False)
    e = (s ** 2).cumsum() / max((s ** 2).sum(), _EPS)
    return int(np.searchsorted(e, frac) + 1)


def anisotropy(activations, n_pairs: int = 2000, seed: int = 0) -> float:
    """Mean cosine of random sample pairs (``per_layer_manifold.py``).

    High anisotropy (cosine -> 1) means the representation has collapsed toward a
    common direction = uninformative; low anisotropy = informative spread.
    """
    X = _to_numpy(activations)
    n = X.shape[0]
    if n < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    a, b = X[i], X[j]
    num = (a * b).sum(1)
    den = (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)) + _EPS
    return float((num / den).mean())


def isometry(h_in, h_out, n_pairs: int = 2000, seed: int = 0) -> float:
    """Spearman correlation of pairwise distances entering vs leaving a layer.

    ``per_layer_manifold.py``: ``Spearman(||d_in||, ||d_out||)`` over random pairs.
    Near 1.0 == the layer preserves metric structure.  NOTE: high isometry alone
    does **not** mean a layer is removable (L30 had isometry 0.985 yet was
    essential) -- use only inside :class:`ManifoldGeometryProbe`, never alone.
    """
    A, B = _to_numpy(h_in), _to_numpy(h_out)
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    d_in = np.linalg.norm(A[i] - A[j], axis=1)
    d_out = np.linalg.norm(B[i] - B[j], axis=1)
    return spearman(d_in, d_out)


# ===========================================================================
#  Linear-decodability kernels   (LinearActionProbe, PlanningProbe, boundary) #
# ===========================================================================
def ridge_fit(X_tr, Y_tr, lam: float, fit_bias: bool = True):
    """Closed-form ridge weights.  Bias column (if any) is left unregularised."""
    X = _to_numpy(X_tr)
    Y = _to_numpy(Y_tr)
    # Equivalent dual solve when samples are fewer than features. Centering
    # preserves the original unregularized intercept exactly. This avoids a
    # 4096-by-4096 solve for every lambda on a small calibration set.
    if lam > 0 and X.shape[0] < X.shape[1]:
        scalar_target = Y.ndim == 1
        if scalar_target:
            Y = Y[:,None]
        x_mean = X.mean(0, keepdims=True) if fit_bias else np.zeros((1,X.shape[1]))
        y_mean = Y.mean(0, keepdims=True) if fit_bias else np.zeros((1,Y.shape[1]))
        centered_x, centered_y = X-x_mean, Y-y_mean
        dual = np.linalg.solve(centered_x @ centered_x.T + lam*np.eye(len(X)), centered_y)
        weights = centered_x.T @ dual
        if fit_bias:
            weights = np.concatenate([weights,y_mean-x_mean@weights],axis=0)
        return weights[:,0] if scalar_target else weights
    if fit_bias:
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    D = X.shape[1]
    R = lam * np.eye(D)
    if fit_bias:
        R[-1, -1] = 0.0                       # do not regularise the bias
    W = np.linalg.solve(X.T @ X + R, X.T @ Y)
    return W


def ridge_predict(X, W, fit_bias: bool = True) -> np.ndarray:
    X = _to_numpy(X)
    if fit_bias:
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    return X @ W


def ridge_probe(
    X,
    Y,
    lambdas: Sequence[float] = (0.1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5),
    val_frac: float = 0.2,
    fit_bias: bool = True,
    seed: int = 0,
) -> Dict[str, float]:
    """Held-out linear decodability of ``Y`` from ``X``.

    Sweeps ``lambdas``, selects the best by validation L1, and returns both the
    L1 (``linear_action.py`` metric: lower == more action-ready) and R²
    (``run_subgoal_probe.py`` metric: planning info content).  Also returns a
    ``baseline_l1`` = predict-train-mean, so callers can normalise.
    """
    X = _to_numpy(X)
    Y = _to_numpy(Y)
    if Y.ndim == 1:
        Y = Y[:, None]
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_va = max(1, int(n * val_frac))
    va, tr = perm[:n_va], perm[n_va:]
    Xt, Yt, Xv, Yv = X[tr], Y[tr], X[va], Y[va]

    best = None
    for lam in lambdas:
        W = ridge_fit(Xt, Yt, lam, fit_bias)
        pred = ridge_predict(Xv, W, fit_bias)
        l1 = float(np.abs(pred - Yv).mean())
        if best is None or l1 < best["val_l1"]:
            best = {"val_l1": l1, "lam": float(lam), "r2": r2_score(Yv, pred)}
    best["baseline_l1"] = float(np.abs(Yv - Yt.mean(0, keepdims=True)).mean())
    return best


def r2_score(Y, Yhat) -> float:
    Y = _to_numpy(Y)
    Yhat = _to_numpy(Yhat)
    ss_res = ((Y - Yhat) ** 2).sum()
    ss_tot = ((Y - Y.mean(0, keepdims=True)) ** 2).sum()
    return float(1.0 - ss_res / max(ss_tot, _EPS))


# ===========================================================================
#  Similarity / redundancy kernels   (validation, negative baselines)        #
# ===========================================================================
def linear_cka(X, Y) -> float:
    """Centred linear CKA = ``||X^T Y||_F^2 / (||X^T X||_F ||Y^T Y||_F)``.

    Standardised form (``cka_between_layers.py``).  High adjacent CKA == redundant
    pair.  **Verified misleading as a standalone pruning signal** (L29-31 all CKA
    > 0.97 yet L30 essential); kept for redundancy diagnostics & validation only.
    """
    X = _to_numpy(X) - _to_numpy(X).mean(0, keepdims=True)
    Y = _to_numpy(Y) - _to_numpy(Y).mean(0, keepdims=True)
    num = (X.T @ Y) ** 2
    aa = (X.T @ X) ** 2
    bb = (Y.T @ Y) ** 2
    return float(num.sum() / (np.sqrt(aa.sum()) * np.sqrt(bb.sum()) + _EPS))


def block_influence(h_in, h_out) -> float:
    """ShortGPT Block-Influence = ``1 - mean cos(h_in, h_out)`` (``compute_layer_metrics.py``).

    **Verified weak** as a per-layer SR predictor (Spearman(cos, SR)=+0.38, wrong
    sign for some layers).  Lives here so the negative baseline can compute it.
    """
    A, B = _to_numpy(h_in), _to_numpy(h_out)
    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + _EPS
    return float(1.0 - (num / den).mean())


# ===========================================================================
#  Correlation kernels   (validation layer -- the discipline that decides    #
#                          whether a metric is trustworthy)                   #
# ===========================================================================
def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties shared), dependency-free Spearman support."""
    a = np.asarray(a, dtype=np.float64)
    order = a.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1)
    # average tied ranks
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def pearson(x, y) -> float:
    x = _to_numpy(x).ravel()
    y = _to_numpy(y).ravel()
    if x.size < 2 or np.std(x) < _EPS or np.std(y) < _EPS:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y) -> float:
    x = _to_numpy(x).ravel()
    y = _to_numpy(y).ravel()
    if x.size < 2:
        return 0.0
    return pearson(_rankdata(x), _rankdata(y))


def leave_one_out_r2(X, y, lam: float = 1.0) -> Dict[str, float]:
    """Leave-one-out R² for a small linear SR-surrogate fit (``DriftPredictor``).

    This is the metric that exposed over-fitting in the source repo: the
    structural-OLS baseline scored LOO R²=-0.84 (worse than predicting the mean)
    while the verified two-feature drift predictor scored LOO R²=0.71.
    """
    X = _to_numpy(X)
    y = _to_numpy(y).ravel()
    n = X.shape[0]
    preds = np.empty(n)
    for i in range(n):
        tr = np.arange(n) != i
        W = ridge_fit(X[tr], y[tr][:, None], lam, fit_bias=True)
        preds[i] = ridge_predict(X[i:i + 1], W, fit_bias=True).ravel()[0]
    return {
        "loo_r2": r2_score(y[:, None], preds[:, None]),
        "loo_spearman": spearman(y, preds),
        "preds": preds.tolist(),
    }


# ===========================================================================
#  Knockout / drift kernels   (AblationKnockoutProbe, DriftPredictor)         #
# ===========================================================================
def action_perturbation(action_ref, action_knockout, mask: Optional[np.ndarray] = None,
                        metric: str = "paper_l2_v1") -> float:
    """Versioned action deviation. Paper Eq. 3 uses mean flattened-chunk L2.

    The core of the verified **anchor-knockout** probe: identity-skip a layer,
    re-predict the action chunk, and measure how far it moved.  ``mask`` selects a
    subset of frames (e.g. transition frames for ``d_trans``).
    """
    a = _to_numpy(action_ref)
    b = _to_numpy(action_knockout)
    if a.shape != b.shape or a.ndim < 2 or a.shape[0] == 0:
        raise ValueError(f"Action batches must have identical nonempty shapes: {a.shape}, {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Non-finite action prediction")
    d = a - b
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.any():
            d = d[m]
    if metric == "paper_l2_v1":
        return float(np.linalg.norm(d.reshape(len(d), -1), axis=1).mean())
    if metric == "legacy_l1_v1":
        return float(np.abs(d).mean())
    raise ValueError(f"Unknown action deviation metric: {metric}")


def frobenius_drift(hidden_a, hidden_b) -> float:
    """Per-sample-mean Frobenius drift ``||A - B||_F`` at action tokens.

    The ``full_l2_rc`` feature of the verified Drift Predictor: drift of the
    recovered+masked model's action-token hidden states vs the dense model.
    """
    A, B = _to_numpy(hidden_a), _to_numpy(hidden_b)
    return float(np.linalg.norm(A - B, axis=1).mean())


def output_variance(head_output) -> float:
    """Cross-sample variance of the action-head output (``out_var`` feature).

    Distinguishes a *saturated* head (outputs near-constant -> low variance ->
    dead) from a responsive one.  Second half of the verified drift-predictor pair.
    """
    H = _to_numpy(head_output)
    return float(H.var(axis=0).mean())
