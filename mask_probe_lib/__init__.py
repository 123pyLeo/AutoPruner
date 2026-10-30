"""
mask_probe_lib -- verified probe -> pruning-layer (mask) selection methods.
==========================================================================

A standardized, layered library distilling the **verified-successful** layer/mask
selection methodology from two VLA-pruning projects (openvla-oft, starVLA-Robotwin2.0)
into clean, callable interfaces.  Failed methods are kept separately as documented
negative baselines (``mask_probe_lib.negative_baselines``).

Logical hierarchy (call order)::

    core/                 shared types, pure kernels, ModelBackend seam, registry
    [1] layer_importance  rank individual layers   (knockout, linear_action, manifold, planning)
    [2] mask_recoverability  score a candidate mask (boundary_lowrank, planning_gap, multi_anchor)
    [3] sr_surrogate      predict SR cheaply        (drift_predictor)
    [4] mask_selection    build masks               (anchor_policy, ratio, qd_archive)
    [5] validation        decide what to trust      (probe_vs_sr, anti_anchor, recoverability_funnel)
    pipeline.py           MaskProbePipeline ties [1]->[5] together
    backends/             MockBackend (runnable) + OpenVLAOFT/QwenOFT skeletons

Quick start::

    from mask_probe_lib import MaskProbePipeline
    from mask_probe_lib.backends import MockBackend

    pipe = MaskProbePipeline(MockBackend(n_layers=32))
    out = pipe.run(prune_ratio=0.28)
    print("anchors:", out.report.anchors(), "-> mask:", out.chosen_mask)

See ``METHODS.md`` for the per-method evidence and source pointers, and
``core.registry.available()`` to list every registered method.
"""

from .core import (
    ActivationBundle,
    LayerScore,
    LayerImportanceReport,
    MaskCandidate,
    MaskScore,
    ValidationResult,
    ModelBackend,
    kernels,
    register,
    build,
    available,
)
from .pipeline import MaskProbePipeline, PipelineResult

# import the five pipeline-layer packages for their @register side-effects, so
# core.registry.available() is fully populated after a bare `import mask_probe_lib`.
from . import layer_importance, mask_recoverability, sr_surrogate, mask_selection, validation  # noqa: E402,F401

__version__ = "0.1.0"

__all__ = [
    "MaskProbePipeline",
    "PipelineResult",
    "ModelBackend",
    "ActivationBundle",
    "LayerScore",
    "LayerImportanceReport",
    "MaskCandidate",
    "MaskScore",
    "ValidationResult",
    "kernels",
    "register",
    "build",
    "available",
]
