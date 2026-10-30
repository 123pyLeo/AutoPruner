"""Core: recovery types, verified loss kernels, the RecoveryBackend seam.

Reuses ``mask_probe_lib.core`` (ModelBackend, registry, kernels) -- recovery_lib is
the downstream half of the same prune→recover→eval loop.
"""

from mask_probe_lib.core.registry import register, build, available

from . import losses
from .backend import RecoveryBackend
from .types import (
    Batch,
    RecoveryRecipe,
    RecoveryResult,
    StudentOutput,
    TeacherOutput,
    TrainableSpec,
)

__all__ = [
    "RecoveryBackend",
    "TrainableSpec",
    "RecoveryRecipe",
    "RecoveryResult",
    "StudentOutput",
    "TeacherOutput",
    "Batch",
    "losses",
    "register",
    "build",
    "available",
]
