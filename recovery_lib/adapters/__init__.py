"""
[1] Recovery adapters -- the verified recovery module architecture.

    AdapterSpec            -- shape/config (kind, hidden_dim, bottleneck, up_init_std)
    NumpyResidualAdapter   -- pure-numpy zero-init residual MLP (mock / tests)
    build_torch_adapter    -- the real torch nn.Module for live backends

Verified constraint: nonlinear + SGD-trained boundary adapter, identity at init.
Linear / closed-form / LoRA-only alternatives gave 0 % SR (see negative_baselines).
"""

from .boundary_adapter import (
    AdapterSpec,
    NumpyResidualAdapter,
    build_torch_adapter,
)

__all__ = ["AdapterSpec", "NumpyResidualAdapter", "build_torch_adapter"]
