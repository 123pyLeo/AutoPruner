"""
mask_probe_lib.core.backend
===========================

The single seam between the model-agnostic probes and a concrete VLA stack.

Everything above this file (layer_importance / mask_recoverability / sr_surrogate
/ validation) is pure and reusable.  Everything model-specific (loading weights,
running forward passes, capturing residual-stream activations at action-token
positions, physically deleting layers, the closed-loop eval) lives **behind this
interface**.  To support a new model you implement one ``ModelBackend`` subclass;
the whole pipeline then runs unchanged.

Reference implementations / skeletons:
    * ``mask_probe_lib.backends.openvla_oft.OpenVLAOFTBackend``  (LLaMA-2-7B, 32 layers)
    * ``mask_probe_lib.backends.qwenoft.QwenOFTBackend``          (Qwen3-VL-4B, 36 layers)
    * ``mask_probe_lib.backends.mock.MockBackend``                (synthetic, fully runnable)

A subclass need only implement the abstract methods.  The non-abstract methods
have sensible defaults built on top of them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from .types import ActivationBundle


class ModelBackend(ABC):
    """Adapter a concrete VLA must provide for the probes to run.

    Coordinate conventions
    -----------------------
    * Layers are indexed ``0 .. n_layers-1``.
    * A ``mask`` is the list of layer indices to **physically remove**.
    * All activations are returned **at action-token positions only**.
    * Hidden-state key convention matches :class:`ActivationBundle`: key ``l`` is
      the input (residual stream) of layer ``l``; key ``n_layers`` is the output
      of the last layer.
    """

    # --- identity ---------------------------------------------------------- #
    @property
    @abstractmethod
    def n_layers(self) -> int:
        """Number of decoder layers in the dense backbone."""

    @property
    def dense_sr(self) -> Optional[float]:
        """Closed-loop SR of the unpruned model, if known (else ``None``)."""
        return None

    @property
    def suite(self) -> str:
        return "unknown"

    # --- the one dense sweep every per-layer probe consumes ---------------- #
    @abstractmethod
    def collect_activations(self, n_batches: int = 200, seed: int = 0) -> ActivationBundle:
        """Run the dense model once and capture per-layer action-token hidden
        states + the ground-truth action chunk (+ proprio / transition mask).

        Returns
        -------
        ActivationBundle with ``hidden`` keyed ``0 .. n_layers`` and ``actions``
        of shape ``(N, chunk, action_dim)``.
        """

    # --- action prediction under structural edits -------------------------- #
    @abstractmethod
    def predict_action(
        self,
        *,
        mask: Optional[Sequence[int]] = None,
        knockout_layer: Optional[int] = None,
        n_batches: int = 50,
        seed: int = 0,
    ) -> np.ndarray:
        """Predicted action chunk ``(N, chunk, action_dim)`` over a fixed eval set.

        * ``mask``           -> physically prune those layers first.
        * ``knockout_layer`` -> identity-skip exactly one layer (output := input)
                                without changing layer count.  This is the cheap
                                open-loop signal behind the anchor-knockout probe.
        The eval set must be **the same samples** for every call (fixed ``seed``)
        so that two predictions are directly comparable element-wise.
        """

    # --- boundary residual pair for a candidate mask ----------------------- #
    @abstractmethod
    def boundary_pair(
        self, mask: Sequence[int], n_batches: int = 50, seed: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(h_pre, h_post)`` at action tokens for a contiguous ``mask``.

        ``h_pre``  = residual stream entering the first removed layer ``min(mask)``.
        ``h_post`` = residual stream leaving the last removed layer ``max(mask)``
                     (== input of the first kept layer after the gap).
        These define the skip an adapter must learn: ``h_pre -> h_post``.
        """

    # --- optional: raw action-head output (for the out_var drift feature) --- #
    def head_output(
        self, *, mask: Optional[Sequence[int]] = None, n_batches: int = 50, seed: int = 0
    ) -> np.ndarray:
        """Raw action-head output ``(N, chunk*action_dim)``.

        Default = flattened :meth:`predict_action`.  Override if the head exposes
        pre-quantisation logits (more faithful to the source ``out_var`` feature).
        """
        a = self.predict_action(mask=mask, n_batches=n_batches, seed=seed)
        return a.reshape(a.shape[0], -1)

    # --- the ground-truth oracle (expensive / cached) ---------------------- #
    def closed_loop_sr(
        self, mask: Sequence[int], recovery: Optional[str] = None, n_trials: int = 200
    ) -> float:
        """Closed-loop success rate after pruning ``mask`` (+ optional recovery).

        This is the only ground-truth metric and the final arbiter (warning W5).
        It is intentionally **not abstract**: many workflows only need the cheap
        probes and validate against a *cached* SR table.  Implement it (real GPU
        eval or a cached lookup) when you want the validation layer to run.
        """
        raise NotImplementedError(
            "closed_loop_sr is the ground-truth oracle; wire it to a cached SR "
            "table or a real eval to use the validation layer."
        )

    # --- convenience ------------------------------------------------------- #
    def keep_layers(self, mask: Sequence[int]) -> list:
        m = set(int(x) for x in mask)
        return [l for l in range(self.n_layers) if l not in m]
