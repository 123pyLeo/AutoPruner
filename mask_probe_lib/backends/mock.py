"""
MockBackend -- a fully-runnable synthetic VLA for testing the whole pipeline.
=============================================================================

No real model required.  It implements a tiny generative "model" whose internals
have a *planted* anchor structure, so every verified probe produces sensible,
checkable output and the examples / tests run in milliseconds.

Planted ground truth
--------------------
* an action latent ``g`` (== the action target) is embedded into hidden space;
* each layer ``l`` reveals a fixed increment ``imp[l]`` of that information into the
  residual stream (so linear decodability rises with depth);
* **anchors** (structural {0, 1, n-2, n-1} + one mid layer) get a large ``imp`` --
  identity-skipping them perturbs the predicted action a lot (high knockout score)
  and removing them tanks the (synthetic) closed-loop SR;
* tolerant layers get a tiny ``imp`` -- safe to prune.

This makes the knockout ranking, the anchor policy, the recoverability gate, the
anti-anchor control and the probe-vs-SR correlation all line up the way they do on
the real stacks -- a self-consistent sandbox, not real numbers.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ..core.backend import ModelBackend
from ..core.types import ActivationBundle


class MockBackend(ModelBackend):
    def __init__(self, n_layers: int = 32, chunk: int = 8, action_dim: int = 7,
                 hidden_dim: int = 80, sigma: float = 0.05, dense_sr: float = 0.95,
                 mid_anchor: Optional[int] = None, seed: int = 0):
        self._n_layers = n_layers
        self.chunk = chunk
        self.action_dim = action_dim
        self.A = chunk * action_dim
        self.D = max(hidden_dim, self.A + 8)
        self.sigma = sigma
        self._dense_sr = dense_sr

        p = np.random.default_rng(seed)
        # action -> hidden embedding with a right inverse (D >= A)
        self.E = p.standard_normal((self.A, self.D)) / np.sqrt(self.A)
        self.W = np.linalg.pinv(self.E)                 # (D, A) readout
        # planted per-layer importance (the reveal increment). Tolerant layers get
        # a small *varied* importance (real layers are never exactly tied), so the
        # quantile tagging behaves; anchors get a clearly larger increment.
        imp = 0.02 + 0.015 * p.random(n_layers)         # tolerant in [0.020, 0.035]
        for a in (0, 1, n_layers - 2, n_layers - 1):    # structural anchors
            imp[a] = 0.5
        self.mid_anchor = mid_anchor if mid_anchor is not None else n_layers // 2
        imp[self.mid_anchor] = 0.2
        self.imp = imp
        self.total = imp.sum()

    # -- identity ----------------------------------------------------------- #
    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def dense_sr(self) -> float:
        return self._dense_sr

    @property
    def suite(self) -> str:
        return "mock"

    # -- the synthetic forward simulation ----------------------------------- #
    def _data(self, N: int, seed: int):
        rng = np.random.default_rng(1000 + seed)
        g = rng.standard_normal((N, self.A))            # action latent == target
        noise = rng.standard_normal((self._n_layers + 1, N, self.D))
        trans = rng.random(N) < 0.3
        return g, noise, trans

    def _cumulative_reveal(self, mask=None, knockout_layer=None, recovered=False):
        inc = self.imp.copy()
        if not recovered:
            if mask is not None:
                inc[list(mask)] = 0.0
            if knockout_layer is not None:
                inc[knockout_layer] = 0.0
        F = np.concatenate([[0.0], np.cumsum(inc)]) / self.total   # length n+1
        return F

    def _hidden(self, N, seed, mask=None, knockout_layer=None, recovered=False):
        g, noise, trans = self._data(N, seed)
        gE = g @ self.E                                  # (N, D) signal direction
        F = self._cumulative_reveal(mask, knockout_layer, recovered)
        hidden = {p: (F[p] * gE + self.sigma * noise[p]) for p in range(self._n_layers + 1)}
        actions = g.reshape(N, self.chunk, self.action_dim)
        return hidden, actions, trans

    def _resolve_N(self, n_batches: int) -> int:
        return max(32, int(n_batches))

    # -- ModelBackend interface --------------------------------------------- #
    def collect_activations(self, n_batches: int = 200, seed: int = 0,
                            mask=None, recovered: bool = False) -> ActivationBundle:
        N = self._resolve_N(n_batches)
        hidden, actions, trans = self._hidden(N, seed, mask=mask, recovered=recovered)
        return ActivationBundle(hidden=hidden, actions=actions, transition_mask=trans,
                                meta={"suite": "mock", "n": N, "mask": list(mask) if mask else []})

    def predict_action(self, *, mask=None, knockout_layer=None, n_batches: int = 50,
                       seed: int = 0) -> np.ndarray:
        N = self._resolve_N(n_batches)
        hidden, _, _ = self._hidden(N, seed, mask=mask, knockout_layer=knockout_layer)
        final = hidden[self._n_layers]
        a = final @ self.W                               # (N, A)
        return a.reshape(N, self.chunk, self.action_dim)

    def boundary_pair(self, mask: Sequence[int], n_batches: int = 50, seed: int = 0
                      ) -> Tuple[np.ndarray, np.ndarray]:
        m = sorted(int(x) for x in mask)
        if not m:
            h, _, _ = self._hidden(self._resolve_N(n_batches), seed)
            return h[0], h[0]
        N = self._resolve_N(n_batches)
        hidden, _, _ = self._hidden(N, seed)             # captured on the DENSE teacher
        pre = hidden[m[0]]
        post = hidden[min(m[-1] + 1, self._n_layers)]
        return pre, post

    def closed_loop_sr(self, mask: Sequence[int], recovery: Optional[str] = None,
                       n_trials: int = 200) -> float:
        """Synthetic SR. Faithful to the verified finding that recovery restores
        *tolerant* removals well but *anchor* removals poorly: each removed layer's
        recoverability shrinks as its importance grows, so anti-anchor masks stay
        low even WITH an adapter (mirrors starVLA 0.484 vs anchor 0.788)."""
        m = list(mask)
        if not m:
            return float(self._dense_sr)
        imp = self.imp[m]
        # per-layer recoverability: ~0.95 for tolerant (low imp), ~0 for anchors
        rec_on = 1.0 if recovery else 0.0
        recoverability = np.clip(1.0 - imp / 0.5, 0.0, 1.0) * 0.95 * rec_on
        damage = float((imp * (1.0 - recoverability)).sum() / self.total)
        return float(np.clip(self._dense_sr * (1.0 - damage), 0.0, self._dense_sr))
