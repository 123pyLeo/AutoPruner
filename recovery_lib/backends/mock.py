"""
MockRecoveryBackend -- fully-runnable synthetic recovery backend.
=================================================================

Extends ``mask_probe_lib.backends.MockBackend`` (reusing its planted per-layer
importance) and simulates training→SR so the verified recovery strategies actually
run with no real model/GPU. The simulation is faithful to the verified findings:

* naked pruning is catastrophic (SR ~0.12) regardless of which layers;
* recovery climbs toward a ceiling set by (a) whether the removed set respects
  anchors -- removing high-importance anchors caps recovery low; (b) the trainable
  set -- joint (adapter+head+kept) recovers far better than adapter-only; (c)
  whether training is on-policy -- offline joint saturates at a frozen-backbone
  ceiling that only on-policy + trust-region can exceed.

So on a tolerant mask PhaseTwoJointRecovery lands ~0.75 here, OnPolicyDistillTrustRegion
lifts it above that offline ceiling (~0.82), adapter-only stalls (~0.50), and an
anti-anchor mask stays stuck at the naked floor (~0.12, never below) -- mirroring
openvla/starVLA at the structural level.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from mask_probe_lib.backends.mock import MockBackend

from ..core.backend import RecoveryBackend
from ..core.types import Batch, StudentOutput, TeacherOutput, TrainableSpec

_NAKED_SR = 0.12
_OFFLINE_CEILING_FRAC = 0.90      # frozen-backbone oracle ceiling (offline saturates here)
_ONPOLICY_CEILING_FRAC = 1.00     # on-policy + trust-region can reach the full ceiling
_STEP_RATE = 0.02                 # synthetic SGD convergence rate per step


class MockRecoveryBackend(MockBackend, RecoveryBackend):
    def __init__(self, n_layers: int = 32, hidden_probe_dim: int = 16, **kw):
        super().__init__(n_layers=n_layers, **kw)
        self._hidden_probe_dim = hidden_probe_dim
        self._rng = np.random.default_rng(0)
        self._batch_counter = 0
        # recovery state (set by build_student)
        self._current_mask: Optional[tuple] = None
        self._kept: list = []
        self.recovered_frac = 0.0
        self._quality = 0.0
        self._on_policy = False
        self._current_recovery = "joint"

    def model_config(self) -> dict:
        """Nominal OpenVLA-OFT (LLaMA-2-7B) hyperparams so the param/compute report
        prints realistic numbers in the no-GPU demo. Flagged `nominal` so callers can
        label it an estimate rather than a measured count."""
        return {"hidden_size": 4096, "intermediate_size": 11008,
                "num_hidden_layers": self.n_layers, "vocab_size": 32064, "nominal": True}

    # --- build / reset the student ---------------------------------------- #
    def _quality_of(self, trainable: TrainableSpec) -> float:
        g = set(trainable.groups)
        if "action_head" not in g:
            return 0.62                       # adapter-only (frozen sidecars)
        if "kept_layers" in g:
            return 0.93                       # joint / on-policy co-adaptation
        return 0.95                           # full sidecar unfreeze (per-task)

    def build_student(self, mask: Sequence[int], trainable: TrainableSpec,
                      adapter_bottleneck: int = 1024, init_from: Optional[object] = None) -> None:
        keep_frac = (init_from == "current") and (self._current_mask == tuple(sorted(mask)))
        self._current_mask = tuple(sorted(int(m) for m in mask))
        self._kept = self.keep_layers(self._current_mask)
        self._quality = self._quality_of(trainable)
        self._current_recovery = trainable.note
        if not keep_frac:
            self.recovered_frac = 0.0         # fresh prune (else warm-start preserves progress)
        # on-policy flag is (re)enabled only when rollouts are collected
        self._on_policy = False

    # --- ceiling / SR model ------------------------------------------------ #
    def _anchor_mass_of(self, mask) -> float:
        """Fraction of total importance sitting on the REMOVED layers (caps recovery)."""
        m = list(mask) if mask else []
        if not m:
            return 0.0
        return float(self.imp[m].sum() / self.total)

    def _anchor_mass(self) -> float:
        return self._anchor_mass_of(self._current_mask)

    def _ceiling_for(self, mask, quality: float, on_policy: bool) -> float:
        """Converged-recovery SR ceiling for (mask, quality, on_policy), floored at the
        naked-prune SR. The anchor penalty is 1x anchor_mass: the old 2x drove the
        ceiling BELOW _NAKED_SR for anchor-heavy masks (so 'training' made SR go *down*
        past do-nothing) and held even tolerant masks ~0.1 under the documented anchors."""
        base = self.dense_sr * (1.0 - min(1.0, self._anchor_mass_of(mask)))
        cap = _ONPOLICY_CEILING_FRAC if on_policy else _OFFLINE_CEILING_FRAC
        full = base * quality
        ceiling = _NAKED_SR + (full - _NAKED_SR) * cap
        return float(np.clip(ceiling, _NAKED_SR, self.dense_sr))

    def _target_ceiling(self) -> float:
        return self._ceiling_for(self._current_mask, self._quality, self._on_policy)

    def evaluate_sr(self, n_trials: int = 50) -> float:
        if self._current_mask is None:
            raise RuntimeError("build_student must be called before evaluate_sr")
        ceiling = self._target_ceiling()
        sr = _NAKED_SR + (ceiling - _NAKED_SR) * self.recovered_frac
        # never below the naked-prune floor, never above dense (recovery can't beat dense)
        return float(np.clip(sr, _NAKED_SR, self.dense_sr))

    def closed_loop_sr(self, mask: Sequence[int], recovery: Optional[str] = None,
                       n_trials: int = 200) -> float:
        """Oracle SR for (mask, recovery) = the CONVERGED ceiling, so the probe-side
        validation (RecoverabilityFunnel / correlation / anti-anchor, which call this)
        agrees with the recovery loop's final SR. Overrides MockBackend.closed_loop_sr
        (a separate, contradictory SR model); the base RecoveryBackend.evaluate_sr is
        designed to route through closed_loop_sr, so both sides now share one model."""
        m = list(mask)
        if not m:
            return float(self.dense_sr)
        r = str(recovery or "").lower()
        if r in ("", "direct", "none"):
            return float(_NAKED_SR)                     # naked prune floor
        # adapter-only stalls low; joint / on-policy co-adaptation recovers far better
        adapter_only = ("adapter" in r) and not any(k in r for k in ("head", "proprio", "joint", "onpolicy"))
        quality = 0.62 if adapter_only else 0.93
        return self._ceiling_for(m, quality, on_policy="onpolicy" in r)

    # --- synthetic teacher hidden (deterministic per batch) ---------------- #
    def _teacher_hidden(self, batch: Batch, layer: int) -> np.ndarray:
        bid = batch.meta.get("id", 0)
        rng = np.random.default_rng(10_000 + bid * 131 + layer)
        n = batch.gt_action.shape[0]
        return rng.standard_normal((n, self._hidden_probe_dim))

    # --- forwards ---------------------------------------------------------- #
    def student_forward(self, batch: Batch) -> StudentOutput:
        f = self.recovered_frac
        n = batch.gt_action.shape[0]
        noise = self._rng.standard_normal(batch.gt_action.shape) * (1.0 - f)
        action = batch.gt_action + 0.5 * noise
        hidden = {l: f * self._teacher_hidden(batch, l)
                  + (1.0 - f) * self._rng.standard_normal((n, self._hidden_probe_dim))
                  for l in self._kept}
        return StudentOutput(action=action, hidden=hidden)

    def teacher_forward(self, batch: Batch) -> TeacherOutput:
        hidden = {l: self._teacher_hidden(batch, l) for l in self._kept}
        return TeacherOutput(action=batch.gt_action.copy(), hidden=hidden)

    def apply_gradients(self, loss) -> float:
        self.recovered_frac = float(min(1.0, self.recovered_frac + _STEP_RATE * (1.0 - self.recovered_frac)))
        return float(loss)

    # --- trust-region anchor ----------------------------------------------- #
    def snapshot_anchor(self) -> object:
        return {"frac": self.recovered_frac}

    def anchor_forward(self, snapshot: object, batch: Batch) -> StudentOutput:
        f = snapshot["frac"]
        noise = self._rng.standard_normal(batch.gt_action.shape) * (1.0 - f)
        return StudentOutput(action=batch.gt_action + 0.5 * noise)

    # --- data sources ------------------------------------------------------ #
    def _make_batch(self, on_policy: bool = False) -> Batch:
        self._batch_counter += 1
        gt = self._rng.standard_normal((8, self.chunk, self.action_dim))
        return Batch(gt_action=gt, on_policy=on_policy, meta={"id": self._batch_counter})

    def iter_offline_batches(self, steps: int, batch_size: int = 4,
                             task_filter: Optional[str] = None) -> Iterable[Batch]:
        for _ in range(steps):
            yield self._make_batch(on_policy=False)

    def collect_onpolicy_rollouts(self, n_rollouts: int = 500,
                                  success_only: bool = False) -> Iterable[Batch]:
        self._on_policy = True                # unlock the higher (ceiling-breaking) target
        return [self._make_batch(on_policy=True) for _ in range(min(64, n_rollouts))]
