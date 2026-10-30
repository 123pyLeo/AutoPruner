"""
Model backends -- the seam between the model-agnostic probes and a concrete VLA.

    MockBackend        -- synthetic, fully runnable (tests / demos / CI)
    OpenVLAOFTBackend  -- reference skeleton for OpenVLA-OFT (LLaMA-2-7B, 32 layers)
    QwenOFTBackend     -- reference skeleton for QwenOFT (Qwen3-VL-4B, 36 layers)

Implement one ``ModelBackend`` subclass for your model and the whole library runs.
"""

from .mock import MockBackend

# the real-model adapters import heavy deps lazily; expose names without importing
__all__ = ["MockBackend"]


def __getattr__(name):
    raise AttributeError(name)
