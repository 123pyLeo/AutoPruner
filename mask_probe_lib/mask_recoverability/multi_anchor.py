"""
MultiAnchorProbe -- non-contiguous anchor masks raise the recovery ceiling.
===========================================================================

Source
------
* openvla : Probe C in ``scripts/probe_deeper_designs.py`` (and the CONVERSATION_LOG
            Probe A/B/C/D suite).

Idea
----
Instead of recovering from a single contiguous-boundary layer, feed a *concatenation
of non-adjacent kept anchors* (e.g. L13 + L17 + L21 + L25) into the adapter and
measure the action-chunk R².  Scattered mid anchors carry complementary
information, so keeping them raises the ceiling above any single boundary.

Verified evidence
-----------------
``h_L13`` baseline chunk R²=0.883  ->  adding L17+L21+L25  ->  0.943
rpy R² 0.803 -> 0.914  (+0.05..0.10 R²).

This result explicitly **overturned the earlier effective-rank veto** of
non-contiguous masks (effective rank has no mapping to SR; the multi-anchor
ceiling does).  Use it to decide *which kept anchors are worth protecting* when
designing a sparse (non-contiguous) mask.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..core.backend import ModelBackend
from ..core.kernels import ridge_probe
from ..core.registry import register
from ..core.types import ActivationBundle, MaskScore
from .base import MaskRecoverabilityProbe


@register("mask_recoverability", "multi_anchor")
class MultiAnchorProbe(MaskRecoverabilityProbe):
    """Ridge action-R² from a concatenation of kept anchor layers.

    Unlike the other recoverability probes this scores a *set of kept anchors*
    rather than a remove-mask: pass the anchors via :meth:`score_anchors`, or pass
    a remove-mask to :meth:`score_mask` and it scores the kept layers' best
    incremental anchor set.
    """

    name = "multi_anchor"

    def __init__(self, target: str = "chunk", n_batches: int = 200, seed: int = 0,
                 bundle: Optional[ActivationBundle] = None):
        self.target = target
        self.n_batches = n_batches
        self.seed = seed
        self._bundle = bundle

    def _get_bundle(self, backend: ModelBackend) -> ActivationBundle:
        if self._bundle is None:
            self._bundle = backend.collect_activations(n_batches=self.n_batches, seed=self.seed)
        return self._bundle

    def score_anchors(self, backend: ModelBackend, anchors: Sequence[int]) -> MaskScore:
        """R² when concatenating the given kept anchor layers."""
        bundle = self._get_bundle(backend)
        Y = bundle.action_target(self.target)
        feats = [bundle.hidden[a] for a in anchors if a in bundle.hidden]
        X = np.concatenate(feats, axis=1)
        r2 = ridge_probe(X, Y, seed=self.seed)["r2"]
        return MaskScore(mask=tuple(sorted(anchors)), method=self.name, value=float(r2),
                         raw={"anchors": list(anchors), "chunk_r2": float(r2)})

    def incremental_gain(self, backend: ModelBackend, anchors: Sequence[int]):
        """R² after greedily adding each anchor -- shows which anchors add info."""
        bundle = self._get_bundle(backend)
        Y = bundle.action_target(self.target)
        order, gains, cur = list(anchors), [], None
        feats = []
        for a in order:
            feats.append(bundle.hidden[a])
            r2 = ridge_probe(np.concatenate(feats, axis=1), Y, seed=self.seed)["r2"]
            gains.append({"added": a, "r2": float(r2), "gain": float(r2 - cur) if cur is not None else float(r2)})
            cur = r2
        return gains

    def score_mask(self, backend: ModelBackend, mask: Sequence[int]) -> MaskScore:
        kept = backend.keep_layers(mask)
        return self.score_anchors(backend, kept)
