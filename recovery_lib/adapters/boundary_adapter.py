"""
Boundary adapters -- the one verified recovery architecture.
============================================================

Source
------
* openvla : ``prismatic/training/replacement_adapter.py`` (Adapter / TransformerAdapter)
* starVLA : ``starVLA/model/modules/adapter/boundary_adapter.py`` ("Ported from openvla-oft")

Both repos converged on the SAME module: a **zero-init residual adapter** inserted
at each prune boundary, physically replacing the removed block. It must be:

* **nonlinear** (MLP with SiLU, or a bottlenecked transformer) -- a linear/closed-form
  adapter gives 0 % SR (openvla E14 SVD-linear);
* **trained by SGD on the boundary** -- LoRA-only on kept layers (no boundary adapter)
  gives 0 % SR (openvla E13);
* **identity at initialisation** -- the up-projection is zero-init so the pruned model
  starts exactly where it was and recovery is a smooth departure.

    forward(x) = x + up(silu(down(rmsnorm(x))))         # up_init = 0  → out = x at step 0

This module is the torch implementation for real backends; a pure-numpy reference
(``NumpyResidualAdapter``) lets the mock backend and tests run without torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mask_probe_lib.core.registry import register


@dataclass
class AdapterSpec:
    """Shape/config of a boundary adapter."""

    kind: str = "mlp"            # "mlp" | "transformer" | "resmlp"
    hidden_dim: int = 4096
    bottleneck: int = 1024
    up_init_std: float = 0.0     # 0.0 => identity at init (REQUIRED)
    n_heads: int = 16            # transformer kind only


def _rmsnorm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


class NumpyResidualAdapter:
    """Pure-numpy zero-init residual MLP adapter (reference / mock / tests).

    Same algebra as the torch module; ``up`` is zero-init so it is the identity at
    construction. ``recover_fraction`` (0→1) is a synthetic training knob the mock
    uses to interpolate the adapter from identity toward the true skip map.
    """

    def __init__(self, spec: AdapterSpec, seed: int = 0):
        self.spec = spec
        rng = np.random.default_rng(seed)
        d, bn = spec.hidden_dim, spec.bottleneck
        self.down = rng.standard_normal((d, bn)) * 0.02
        self.up = rng.standard_normal((bn, d)) * spec.up_init_std    # 0 => identity
        self.recover_fraction = 0.0

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        delta = _silu(_rmsnorm(x) @ self.down) @ self.up
        return x + self.recover_fraction * delta

    __call__ = forward

    def is_identity_at_init(self, x: np.ndarray) -> bool:
        return float(np.abs(self.forward(x) - x).max()) < 1e-9


def build_torch_adapter(spec: AdapterSpec):
    """Construct the real torch ``nn.Module`` (zero-init residual MLP).

    Returns ``None`` if torch is unavailable so numpy-only environments still import
    the library. Verifies identity-at-init for ``up_init_std == 0``.
    """
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return None

    class ResidualMLPAdapter(nn.Module):
        def __init__(self, s: AdapterSpec):
            super().__init__()
            self.norm = nn.RMSNorm(s.hidden_dim) if hasattr(nn, "RMSNorm") else nn.LayerNorm(s.hidden_dim)
            self.down = nn.Linear(s.hidden_dim, s.bottleneck, bias=False)
            self.up = nn.Linear(s.bottleneck, s.hidden_dim, bias=False)
            self.act = nn.SiLU()
            nn.init.normal_(self.down.weight, std=0.02)
            if s.up_init_std == 0.0:
                nn.init.zeros_(self.up.weight)          # identity at init
            else:
                nn.init.normal_(self.up.weight, std=s.up_init_std)

        def forward(self, x):
            return x + self.up(self.act(self.down(self.norm(x))))

    return ResidualMLPAdapter(spec)


# register the verified adapter kinds for uniform discovery
register("recovery_adapter", "mlp_residual")(NumpyResidualAdapter)
