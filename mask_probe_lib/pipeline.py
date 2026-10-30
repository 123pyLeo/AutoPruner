"""
mask_probe_lib.pipeline
=======================

End-to-end orchestration that wires the five layers together in their natural
hierarchy.  This is the single entry point most callers want::

    from mask_probe_lib import MaskProbePipeline
    from mask_probe_lib.backends.mock import MockBackend

    pipe = MaskProbePipeline(MockBackend(n_layers=32))
    result = pipe.run(prune_ratio=0.28)
    print(result.chosen_mask, result.report.anchors())

The flow (each step is one library layer):

    backend.collect_activations()              # one dense sweep, shared
        |
    [1] LayerImportanceProbe.score_layers      -> per-layer anchors / tolerant
        |
    [4] PruneRatioSweep + AnchorProtectionPolicy -> candidate mask(s)
        |
    [2] MaskRecoverabilityProbe.score_mask      -> recoverability gate
        |
    [3] DriftPredictor.predict (if calibrated)  -> predicted SR ranking
        |
    [5] RecoverabilityFunnel / validation       -> shortlist (+ optional closed-loop)

Every stage is optional and swappable; pass your own probe instances to override
the verified defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .core.backend import ModelBackend
from .core.types import LayerImportanceReport, MaskCandidate
from .layer_importance.knockout import AblationKnockoutProbe
from .layer_importance.manifold import ManifoldGeometryProbe
from .layer_importance.base import LayerImportanceProbe
from .mask_recoverability.base import MaskRecoverabilityProbe
from .mask_recoverability.boundary_lowrank import BoundaryLowRankProbe
from .mask_selection.anchor_policy import AnchorProtectionPolicy
from .mask_selection.ratio import PruneRatioSweep
from .mask_selection.qd_archive import QualityDiversityArchive
from .validation.recoverability_funnel import RecoverabilityFunnel


@dataclass
class PipelineResult:
    report: LayerImportanceReport
    candidates: List[MaskCandidate]
    chosen_mask: Optional[tuple] = None
    shortlist: List[MaskCandidate] = field(default_factory=list)
    archive: Optional[QualityDiversityArchive] = None


class MaskProbePipeline:
    """The verified probe -> mask-selection pipeline, with swappable components."""

    def __init__(
        self,
        backend: ModelBackend,
        importance_probe: Optional[LayerImportanceProbe] = None,
        recoverability_probe: Optional[MaskRecoverabilityProbe] = None,
        policy: Optional[AnchorProtectionPolicy] = None,
        surrogate=None,
        n_early: int = 2,
        n_late: int = 2,
        manifold_probe: Optional[LayerImportanceProbe] = None,
    ):
        self.backend = backend
        # verified default: the decisive knockout probe
        self.importance_probe = importance_probe or AblationKnockoutProbe()
        # geometry fallback: when knockout is FLAT (can't rank layers -- the CogACT
        # case, d_all max/median 1.17), the policy ranks the interior by this instead.
        self.manifold_probe = manifold_probe or ManifoldGeometryProbe()
        self.recoverability_probe = recoverability_probe or BoundaryLowRankProbe()
        self.policy = policy or AnchorProtectionPolicy(
            n_layers=backend.n_layers, n_early=n_early, n_late=n_late, distributed=True
        )
        self.surrogate = surrogate
        self.ratio = PruneRatioSweep(backend.n_layers)

    def run(
        self,
        prune_ratio: Optional[float] = None,
        k: Optional[int] = None,
        confirm: bool = False,
        recovery: str = None,
    ) -> PipelineResult:
        """Score layers -> build mask(s) at the budget -> screen recoverability."""
        # [1] per-layer importance (one shared dense sweep for both probes)
        bundle = self.backend.collect_activations()
        report = self.importance_probe.score_layers(self.backend, bundle=bundle)
        # geometry fallback from the SAME bundle: peak-relative eff_rank (protect the
        # ascending build-up + peak, prune the descending refinement tail) -- the correct
        # signal when knockout is flat. Falls back to the combined manifold score if the
        # probe has no peak-relative method (custom probes).
        if hasattr(self.manifold_probe, "effrank_protect_report"):
            manifold_report = self.manifold_probe.effrank_protect_report(self.backend, bundle=bundle)
        else:
            manifold_report = self.manifold_probe.score_layers(self.backend, bundle=bundle)

        # [4] budget: ratio -> k (default to the verified 28% sweet spot)
        if k is None:
            ratio = prune_ratio if prune_ratio is not None else self.ratio.SWEET_SPOT
            k = self.ratio.k_for_ratio(ratio)

        # [4] selection policy -> a candidate mask (manifold fallback if knockout is flat)
        candidate = self.policy.select(report, k, manifold_report=manifold_report)
        candidates = [candidate]

        # [2]+[3]+[5] funnel-screen the candidate(s)
        funnel = RecoverabilityFunnel(
            self.backend, recoverability_probe=self.recoverability_probe, surrogate=self.surrogate
        )
        staged = funnel.run(candidates, top_n=1, confirm=confirm, recovery=recovery)
        shortlist = staged.get("confirmed", staged["surrogate_ranked"])

        # record into the QD archive (handy when run() is called in a search loop)
        archive = QualityDiversityArchive(self.backend.n_layers)
        for c in shortlist:
            archive.add(c.mask, c.sr if c.sr is not None else c.predicted_sr)

        chosen = shortlist[0].mask if shortlist else candidate.mask
        return PipelineResult(
            report=report, candidates=candidates, chosen_mask=chosen,
            shortlist=shortlist, archive=archive,
        )
