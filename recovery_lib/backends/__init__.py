"""
Recovery backends -- the training/rollout seam.

    MockRecoveryBackend        -- synthetic, fully runnable (simulates training→SR)
    OpenVLAOFTRecoveryBackend  -- skeleton citing openvla-oft finetune_phase2_adapter.py etc.
    QwenOFTRecoveryBackend     -- skeleton citing starVLA finetune_onpolicy_distill.py etc.
"""

from .mock import MockRecoveryBackend

__all__ = ["MockRecoveryBackend"]


def __getattr__(name):
    raise AttributeError(name)
