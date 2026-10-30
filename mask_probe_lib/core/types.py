"""
mask_probe_lib.core.types
=========================

Shared data structures passed between the five layers of the library.

Naming conventions (kept identical to the source repos so results stay comparable):

* ``mask``      : the set of layer indices to **REMOVE** (prune), NOT to keep.
                  This mirrors ``openvla-oft`` and ``starVLA`` exactly.
* ``keep``      : ``set(range(n_layers)) - set(mask)``.
* ``sr``        : closed-loop success rate -- the only ground-truth metric.
* action token  : the hidden-state positions that decode into the action chunk.
                  Every probe reads activations **at action-token positions only**
                  (full-sequence averaging was a verified failure mode, W3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
#  Activations captured from one dense forward sweep                           #
# --------------------------------------------------------------------------- #
@dataclass
class ActivationBundle:
    """Everything the per-layer probes need from a single dense forward sweep.

    A :class:`~mask_probe_lib.core.backend.ModelBackend` produces one of these so
    that the layer-importance / recoverability probes never touch the model
    directly -- they consume this bundle and call the pure kernels.

    Attributes
    ----------
    hidden:
        ``{layer_index: ndarray (N, D)}`` residual-stream hidden states at the
        action-token positions.  ``hidden[l]`` is the **input** of decoder layer
        ``l`` (i.e. the residual stream entering layer ``l``); the key
        ``n_layers`` holds the **output** of the final layer (post-L31 / post-L35).
        So a model with 32 layers exposes keys ``0..32`` (33 residual points).
    actions:
        ``ndarray (N, chunk, action_dim)`` ground-truth action chunk for each
        captured sample (the probe target).
    proprio:
        optional ``ndarray (N, P)`` proprioceptive embedding.
    transition_mask:
        optional ``ndarray (N,) bool`` -- True on sub-goal / gripper-flip
        transition frames.  Used by the knockout probe's ``d_trans`` and by the
        transition-signal router probe.
    meta:
        free-form provenance (suite name, n_batches, model id, ...).
    """

    hidden: Dict[int, np.ndarray]
    actions: np.ndarray
    proprio: Optional[np.ndarray] = None
    transition_mask: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    @property
    def layer_indices(self) -> List[int]:
        return sorted(self.hidden.keys())

    @property
    def n_samples(self) -> int:
        return int(self.actions.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(next(iter(self.hidden.values())).shape[1])

    def action_target(self, kind: str = "chunk") -> np.ndarray:
        """Flatten the action target the way each probe expects.

        ``"chunk"``  -> (N, chunk*action_dim)   full chunk (linear_action default)
        ``"step0"``  -> (N, action_dim)          immediate action  (P1)
        ``"future"`` -> (N, 3)                    last-step xyz      (P2, planning)
        ``"delta"``  -> (N, 3)                    last-step xyz minus step0 xyz (P5)
        ``"gripper"``-> (N, chunk)               gripper channel sequence (P3)
        """
        a = self.actions
        if kind == "chunk":
            return a.reshape(a.shape[0], -1)
        if kind == "step0":
            return a[:, 0, :]
        if kind == "future":
            return a[:, -1, :3]
        if kind == "delta":
            return a[:, -1, :3] - a[:, 0, :3]
        if kind == "gripper":
            return a[:, :, 6]
        raise ValueError(f"unknown action-target kind: {kind!r}")


# --------------------------------------------------------------------------- #
#  Results emitted by each layer of the pipeline                              #
# --------------------------------------------------------------------------- #
@dataclass
class LayerScore:
    """Per-layer output of a :class:`LayerImportanceProbe`.

    ``score`` is normalised so that **higher == more important / protect**,
    regardless of the underlying metric's natural direction (the probe is
    responsible for flipping signs).  ``tag`` is a coarse label in
    ``{"anchor", "soft-anchor", "tolerant"}`` decided by the probe's thresholds.
    """

    layer: int
    score: float
    tag: str = "tolerant"
    raw: Dict[str, float] = field(default_factory=dict)


@dataclass
class LayerImportanceReport:
    """Full table returned by a layer-importance probe."""

    method: str
    scores: List[LayerScore]
    higher_is_more_important: bool = True
    notes: str = ""

    def ranked(self, most_important_first: bool = True) -> List[LayerScore]:
        return sorted(self.scores, key=lambda s: s.score, reverse=most_important_first)

    def anchors(self) -> List[int]:
        return [s.layer for s in self.scores if s.tag in ("anchor", "soft-anchor")]

    def tolerant(self) -> List[int]:
        return [s.layer for s in self.scores if s.tag == "tolerant"]

    def as_dict(self) -> Dict[int, float]:
        return {s.layer: s.score for s in self.scores}


@dataclass
class MaskCandidate:
    """A candidate pruning mask plus whatever scores have been attached to it."""

    mask: Tuple[int, ...]
    recovery: str = "adapter+head+proprio"
    sr: Optional[float] = None                       # ground-truth, if known
    predicted_sr: Optional[float] = None             # from sr_surrogate
    recoverability: Optional[float] = None           # from mask_recoverability
    scores: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""

    def __post_init__(self):
        self.mask = tuple(sorted(int(m) for m in self.mask))

    @property
    def k(self) -> int:
        return len(self.mask)

    def keep(self, n_layers: int) -> List[int]:
        return [l for l in range(n_layers) if l not in set(self.mask)]

    def is_contiguous(self) -> bool:
        m = self.mask
        return len(m) == 0 or list(m) == list(range(m[0], m[0] + len(m)))


@dataclass
class MaskScore:
    """Output of a :class:`MaskRecoverabilityProbe`."""

    mask: Tuple[int, ...]
    method: str
    value: float                       # higher == more recoverable / safer
    recoverable: Optional[bool] = None  # decision under the method's threshold
    raw: Dict[str, float] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Output of the validation layer (correlation / control / funnel)."""

    name: str
    passed: bool
    statistic: Dict[str, float] = field(default_factory=dict)
    detail: str = ""
