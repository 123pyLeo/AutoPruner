"""
BoundaryLowRankProbe -- can a small adapter recover the deleted block?
======================================================================

Source
------
* openvla : ``scripts/probe_mask_sweep.py``, ``scripts/train_boundary_adapter_msdp.py``,
            ``logs/bottleneck_ablation/`` (the "ultimate offline judge")
* starVLA : ``scripts/probe/probe_layer_redundancy.py`` (svd_report on the Δ)

Idea
----
For a contiguous ``mask``, take the boundary pair ``(h_pre, h_post)`` at action
tokens (``backend.boundary_pair``) and measure how recoverable the skip
``h_pre -> h_post`` is, via three cheap signals:

1. **Δ effective rank**  -- ``Δ = h_post - h_pre``; low effective rank (energy in a
   few singular values) => a tiny MLP can reproduce it.  (k=8 Δ had 50 % of its
   energy in 2 singular values.)
2. **ridge hold-out cos** -- closed-form linear ``h_pre -> h_post``; cosine on a
   held-out split.  Fast ms-level ranking pre-filter.
3. **MLP-adapter hold-out cos** (optional, needs torch) -- the deploy-time signal:
   a bottleneck MLP recovers the action-token reps far better than ridge
   (ridge SR 26 % vs MLP SR 93.5 % on the same boundary), so this is the most
   faithful offline judge.

Verified evidence
-----------------
k=8 ``[22-29]`` adapter hold-out cos 0.85 -> closed-loop SR 94.5 % (vs 12.5 %
naked); recoverable across the scaling curve to k=24.  Caveat: cos->SR is a
cliff (use to rank / gate, not to predict SR).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..core.backend import ModelBackend
from ..core.kernels import (
    cumulative_energy_rank,
    effective_rank,
    ridge_fit,
    ridge_predict,
)
from ..core.registry import register
from ..core.types import MaskScore
from .base import MaskRecoverabilityProbe


def _cosine_rows(A: np.ndarray, B: np.ndarray) -> float:
    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12
    return float((num / den).mean())


def _train_mlp_adapter(X, Y, bottleneck=1024, steps=2000, seed=0):
    """Optional torch MLP adapter; returns held-out cosine, or None if torch absent."""
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return None
    g = torch.Generator().manual_seed(seed)
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    Y = torch.as_tensor(np.asarray(Y), dtype=torch.float32)
    n = X.shape[0]
    perm = torch.randperm(n, generator=g)
    n_va = max(1, n // 5)
    va, tr = perm[:n_va], perm[n_va:]
    D = X.shape[1]
    net = nn.Sequential(nn.Linear(D, bottleneck), nn.GELU(), nn.Linear(bottleneck, Y.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    bs = min(256, len(tr))
    for _ in range(steps):
        idx = tr[torch.randint(0, len(tr), (bs,), generator=g)]
        opt.zero_grad()
        loss = lossf(net(X[idx]), Y[idx])
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = net(X[va]).numpy()
    return _cosine_rows(pred, Y[va].numpy())


@register("mask_recoverability", "boundary_lowrank")
class BoundaryLowRankProbe(MaskRecoverabilityProbe):
    """Δ effective-rank + ridge/MLP boundary-recovery cosine."""

    name = "boundary_lowrank"

    def __init__(
        self,
        n_batches: int = 50,
        use_mlp: bool = False,
        ridge_lam: float = 1.0,
        recoverable_cos: float = 0.80,
        seed: int = 0,
    ):
        # ridge_lam=1e3 massively over-regularized at the default sample size (≈40
        # train rows): the held-out boundary cosine collapsed to ~0.09, so EVERY mask
        # (even the tolerant k8 boundary the docstring says recovers at cos 0.85) was
        # flagged unrecoverable → the recoverability funnel returned an empty shortlist.
        # λ≈1 restores a meaningful cosine on the mock's unit-scale features.
        self.n_batches = n_batches
        self.use_mlp = use_mlp
        self.ridge_lam = ridge_lam
        self.recoverable_cos = recoverable_cos
        self.seed = seed

    def score_mask(self, backend: ModelBackend, mask: Sequence[int]) -> MaskScore:
        h_pre, h_post = backend.boundary_pair(mask, n_batches=self.n_batches, seed=self.seed)
        h_pre, h_post = np.asarray(h_pre), np.asarray(h_post)

        delta = h_post - h_pre
        er = effective_rank(delta)
        r50 = cumulative_energy_rank(delta, 0.5)
        r90 = cumulative_energy_rank(delta, 0.9)

        # ridge hold-out cosine
        n = h_pre.shape[0]
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)
        n_va = max(1, n // 5)
        va, tr = perm[:n_va], perm[n_va:]
        W = ridge_fit(h_pre[tr], h_post[tr], self.ridge_lam)
        ridge_cos = _cosine_rows(ridge_predict(h_pre[va], W), h_post[va])

        mlp_cos = None
        if self.use_mlp:
            mlp_cos = _train_mlp_adapter(h_pre, h_post, seed=self.seed)

        decisive_cos = mlp_cos if mlp_cos is not None else ridge_cos
        return MaskScore(
            mask=tuple(sorted(mask)),
            method=self.name,
            value=float(decisive_cos),
            recoverable=bool(decisive_cos >= self.recoverable_cos),
            raw={
                "delta_effective_rank": float(er),
                "delta_rank50": float(r50),
                "delta_rank90": float(r90),
                "ridge_cos": float(ridge_cos),
                "mlp_cos": float(mlp_cos) if mlp_cos is not None else float("nan"),
            },
        )
