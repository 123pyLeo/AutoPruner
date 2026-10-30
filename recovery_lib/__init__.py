"""
recovery_lib -- verified post-pruning RECOVERY strategies for VLA models.
========================================================================

The downstream half of the prune→recover→eval loop. Sibling to ``mask_probe_lib``
(which selects the mask); this library restores the success rate the pruning cost,
distilling ONLY the VERIFIED-SUCCESSFUL recovery methods from openvla-oft and
starVLA-Robotwin2.0. Failed methods are documented in
``recovery_lib.negative_baselines``.

Reuses ``mask_probe_lib.core`` (ModelBackend, registry). Run from a directory where
``mask_probe_lib`` is importable (e.g. <research-workspace>

Logical hierarchy (call order)::

    core/                 recovery types, verified loss kernels, RecoveryBackend seam
    [1] adapters          the verified module       (zero-init residual boundary adapter)
    [2] objectives        the verified loss recipes (distill_bc, trust_region_distill)
    [3] strategies        the end-to-end recipes     (phase2_joint, onpolicy_distill, per_task_specialist)
    [4] data              supervision source         (offline_demos, onpolicy_rollouts)
    [5] deploy            composition                (task_router)
    validation            the discipline             (slope gate, frozen-sidecar ablation, anti-collapse)
    pipeline.py           RecoveryPipeline ties it together
    backends/             MockRecoveryBackend (runnable) + OpenVLAOFT/QwenOFT skeletons

Quick start::

    from recovery_lib import RecoveryPipeline
    from recovery_lib.backends import MockRecoveryBackend

    backend = MockRecoveryBackend(n_layers=32)
    pipe = RecoveryPipeline(backend)
    out = pipe.run(mask=[3, 5, 8, 10, 13, 17, 21, 25, 27, 30], continue_onpolicy=True)
    print(out.result.final_sr, "->", out.onpolicy_result.final_sr)

See ``METHODS.md`` for per-method evidence + source pointers.
"""

from .core import (
    RecoveryBackend,
    TrainableSpec,
    RecoveryRecipe,
    RecoveryResult,
    StudentOutput,
    TeacherOutput,
    Batch,
    losses,
)
from .pipeline import RecoveryPipeline, RecoveryPipelineResult

# import the layer packages for their @register side-effects
from . import adapters, objectives, strategies, data, deploy, validation  # noqa: E402,F401

__version__ = "0.1.0"

__all__ = [
    "RecoveryPipeline",
    "RecoveryPipelineResult",
    "RecoveryBackend",
    "TrainableSpec",
    "RecoveryRecipe",
    "RecoveryResult",
    "StudentOutput",
    "TeacherOutput",
    "Batch",
    "losses",
]
