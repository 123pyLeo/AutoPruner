"""Core infrastructure: shared types, pure kernels, the backend seam, registry."""

from .types import (
    ActivationBundle,
    LayerScore,
    LayerImportanceReport,
    MaskCandidate,
    MaskScore,
    ValidationResult,
)
from .backend import ModelBackend
from . import kernels
from .registry import register, build, available

__all__ = [
    "ActivationBundle",
    "LayerScore",
    "LayerImportanceReport",
    "MaskCandidate",
    "MaskScore",
    "ValidationResult",
    "ModelBackend",
    "kernels",
    "register",
    "build",
    "available",
]
