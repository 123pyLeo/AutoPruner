"""
AnchorProtectionPolicy -- turn per-layer importance into an actual mask.
========================================================================

Source
------
* openvla : the headline k28 rule -- keep {L0,L1,L30,L31}, prune the contiguous
            middle -> 96.5 % == dense (``agentic_pruning/agent/object_mask_policy.py``).
* starVLA : ``k10anchor = {3,5,8,10,13,17,21,25,27,30}`` -- data-driven tolerant +
            distributed, anchors & transition band protected -> SR 0.788.

Rule
----
Given a :class:`LayerImportanceReport` (e.g. from the knockout probe) and a target
prune budget, build a mask that:

1. **always keeps structural anchors** -- early input layers (L0,L1) and the late
   action stack (the last ``n_late`` layers).  These are irreplaceable / the
   action-head interface (verified: dropping L30 cost 11.5 pts).
2. **keeps the highest-importance layers** from the probe (the data-driven anchors
   and transition band).
3. **removes the lowest-importance ("tolerant") layers** up to the budget, either
   as a contiguous block or a distributed set.

Distributed-vs-contiguous: at matched ratio + capacity the verified ordering was
contiguous 0.680 < distributed 0.758 < anchor-distributed 0.788, so
``distributed=True`` (spread the cuts, keep them non-adjacent) is the default.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from ..core.registry import register
from ..core.types import LayerImportanceReport, MaskCandidate


@register("mask_selection", "anchor_policy")
class AnchorProtectionPolicy:
    """Build a prune mask that protects anchors and removes tolerant layers."""

    name = "anchor_policy"

    def __init__(
        self,
        n_layers: int,
        n_early: int = 2,
        n_late: int = 2,
        protect_extra: Optional[Sequence[int]] = None,
        distributed: bool = True,
        min_gap: int = 1,
    ):
        """
        Parameters
        ----------
        n_early / n_late:
            number of structural input / action-stack anchors to always keep.
        protect_extra:
            explicit layers to always keep (e.g. a known transition band).
        distributed:
            if True, choose removed layers spread out (non-adjacent where
            possible); if False, choose a single contiguous block.
        min_gap:
            minimum spacing between removed layers in distributed mode.
        """
        self.n_layers = n_layers
        self.n_early = n_early
        self.n_late = n_late
        self.protect_extra = set(protect_extra or [])
        self.distributed = distributed
        self.min_gap = min_gap

    # -- the always-keep structural anchor set ------------------------------ #
    def structural_anchors(self) -> set:
        early = set(range(self.n_early))
        late = set(range(self.n_layers - self.n_late, self.n_layers))
        return early | late | self.protect_extra

    @staticmethod
    def _spread(values: Sequence[float]) -> float:
        """max/median of |importance| -- a flatness gauge for a per-layer signal.

        A spiky, discriminative signal (e.g. OpenVLA knockout: clean end anchors)
        has a large ratio (~32); a flat, non-discriminative one (e.g. CogACT
        knockout: every layer perturbs the action chunk by ~the same amount) sits
        near 1.0 (~1.17 measured).  Below ``flat_threshold`` the ranking is noise.
        """
        vals = sorted(abs(float(v)) for v in values)
        if not vals:
            return 0.0
        med = vals[len(vals) // 2]
        return (max(vals) / med) if med > 1e-9 else float("inf")

    def select(
        self,
        report: LayerImportanceReport,
        k: int,
        manifold_report: Optional[LayerImportanceReport] = None,
        flat_threshold: float = 2.0,
    ) -> MaskCandidate:
        """Return a mask removing ``k`` of the most-tolerant unprotected layers.

        ``report`` is the primary per-layer importance (usually the knockout probe).
        When that signal is *flat* -- ``_spread(report) < flat_threshold``, i.e. it
        cannot rank layers (the CogACT case: knockout d_all max/median 1.17) -- and a
        ``manifold_report`` (effective-rank geometry) is supplied, layer ranking falls
        back to the manifold signal.  Manifold ``score`` is already normalised so
        higher == more important (the informative high-rank core to protect), so the
        same "remove lowest-importance" logic applies unchanged.  Structural anchors
        are always protected regardless of which signal ranks the interior.
        """
        protect = self.structural_anchors()
        importance = report.as_dict()
        src = report.method
        spread = self._spread(importance.values())
        if manifold_report is not None and spread < flat_threshold:
            importance = manifold_report.as_dict()
            src = f"{report.method} flat (max/median={spread:.2f}<{flat_threshold}) -> {manifold_report.method}"

        # candidate removable layers, lowest importance first
        removable = [l for l in range(self.n_layers) if l not in protect]
        removable.sort(key=lambda l: importance.get(l, 0.0))   # tolerant (low) first

        if not self.distributed:
            chosen = self._contiguous_block(removable, importance, k, protect)
        else:
            chosen = self._distributed(removable, k)

        rationale = (
            f"keep structural anchors {sorted(protect)} + high-importance layers; "
            f"remove {len(chosen)} most-tolerant "
            f"({'distributed' if self.distributed else 'contiguous'}); "
            f"ranking signal = {src}."
        )
        return MaskCandidate(mask=tuple(sorted(chosen)), rationale=rationale)

    def _distributed(self, removable: List[int], k: int) -> List[int]:
        """Pick the k most-tolerant layers, enforcing a minimum spacing."""
        chosen: List[int] = []
        for l in removable:
            if len(chosen) >= k:
                break
            if all(abs(l - c) >= self.min_gap for c in chosen) or self.min_gap <= 1:
                chosen.append(l)
        # if spacing was too strict to reach k, top up ignoring the gap
        if len(chosen) < k:
            for l in removable:
                if l not in chosen:
                    chosen.append(l)
                if len(chosen) >= k:
                    break
        return chosen[:k]

    def _contiguous_block(self, removable, importance, k, protect) -> List[int]:
        """Slide a contiguous window over the unprotected span; pick the window of
        size k with the lowest total importance that avoids protected layers."""
        best, best_cost = None, float("inf")
        for start in range(self.n_layers - k + 1):
            block = list(range(start, start + k))
            if any(b in protect for b in block):
                continue
            cost = sum(importance.get(b, 0.0) for b in block)
            if cost < best_cost:
                best, best_cost = block, cost
        return best if best is not None else removable[:k]
