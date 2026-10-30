"""
LibEngine -- the bridge between the agentic loop and the two libraries.
======================================================================

Holds ONE ``recovery_lib.RecoveryBackend`` (also a ``mask_probe_lib.ModelBackend``)
and exposes exactly what the three agentic seams need:

* ``query(probe, layer_ids)``     -> per-layer probe values (mask_probe_lib)
* ``recover_and_eval(mask, rec)`` -> recovered closed-loop SR (recovery_lib)
* ``recoverability(mask)``        -> mask-level recoverability meta (mask_probe_lib)

Per-layer probe values are computed once from a single shared activation sweep and
cached; recovery results are cached per (mask, recovery). Everything degrades
gracefully (returns None) if a probe/strategy is unavailable on a given backend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# mask_probe_lib (probes / kernels)
from mask_probe_lib.core import kernels
from mask_probe_lib.layer_importance import (
    AblationKnockoutProbe,
    LinearActionProbe,
    ManifoldGeometryProbe,
    PlanningInfoProbe,
)
from mask_probe_lib.mask_recoverability import BoundaryLowRankProbe, PlanningGapRule

# recovery_lib (strategies)
from recovery_lib.core.types import TrainableSpec
from recovery_lib.strategies import (
    OnPolicyDistillTrustRegion,
    PhaseTwoJointRecovery,
)

#: recovery-recipe id  ->  how to run it with recovery_lib
#: (the agent proposes one of these ids; see LibRecoveryModule.list_recipes)
RECIPE_IDS = ["direct", "adapter_only", "adapter_mlp_0.5M", "adapter+head+proprio", "onpolicy_distill"]


class LibEngine:
    def __init__(self, backend, *, recovery_steps: int = 150, eval_every: int = 50,
                 knockout_batches: int = 8, bundle_batches: int = 32, seed: int = 0,
                 cache_dir=None):
        # NOTE: the probe table is computed ONCE then cached to disk (see _build_layer_table),
        # so these sample counts only set the one-time cost. Smaller = faster first build;
        # the probes are for RANKING layers (anchors stand out hugely), not absolute values.
        self.backend = backend
        self.recovery_steps = recovery_steps
        self.eval_every = eval_every
        self.knockout_batches = knockout_batches
        self.bundle_batches = bundle_batches
        self.seed = seed
        self._bundle = None
        self._layer_table: Dict[str, Dict[int, float]] = {}
        self._recover_cache: Dict[tuple, dict] = {}
        # On-disk cache for the per-layer probe table: compute ONCE per (model, suite,
        # ckpt), then every later run loads it instantly instead of re-running the
        # 32-layer knockout. Set AUTOPRUNE_NO_PROBE_CACHE=1 to force recompute.
        self.cache_dir = Path(cache_dir or (Path(getattr(backend, "work_dir", ".")) / "probe_evidence"))

    # ----------------------------------------------------------------- probes #
    def _ensure_bundle(self):
        if self._bundle is None:
            self._bundle = self.backend.collect_activations(n_batches=self.bundle_batches, seed=self.seed)
        return self._bundle

    def _cache_path(self) -> Path:
        model = getattr(getattr(self.backend, "plugin", None), "name", type(self.backend).__name__)
        suite = getattr(self.backend, "suite", "unknown")
        return self.cache_dir / f"probe_table_{model}_{suite}.json"

    def _build_layer_table(self) -> None:
        """Per-layer probe table: load from disk if cached for this (model, suite, ckpt),
        else compute once (one activation sweep + the knockout) and SAVE for reuse."""
        if self._layer_table:
            return
        import os
        ckpt = str(getattr(self.backend, "full_ckpt", ""))
        cp = self._cache_path()
        # Persist evidence for inspection, but never reuse a previous run's table.
        # The legacy on-disk key did not identify weights, data, preprocessing or metric.

        import time
        print(f"[lib_engine] computing probe table ONCE (then cached -> {cp.name}); "
              f"bundle={self.bundle_batches}, knockout={self.knockout_batches}/layer ...", flush=True)
        _t = time.time()
        b = self._ensure_bundle()
        print(f"[lib_engine]   activation bundle done in {time.time() - _t:.0f}s", flush=True)
        t: Dict[str, Dict[int, float]] = {}

        def safe(fn, name):
            try:
                t[name] = fn()
            except Exception as e:  # noqa: BLE001 - a probe may be unsupported on a backend
                t[name] = {}
                t.setdefault("_errors", {})[name] = repr(e)

        # the decisive verified probe: identity-skip perturbation per layer
        _t = time.time()
        safe(lambda: {s.layer: s.raw["d_all"]
                      for s in AblationKnockoutProbe(n_batches=self.knockout_batches, seed=self.seed)
                      .score_layers(self.backend, bundle=b).scores}, "knockout_dall")
        if not t.get("knockout_dall"):
            raise RuntimeError(f"Required functional probe failed: {t.get('_errors', {})}")
        print(f"[lib_engine]   knockout done in {time.time() - _t:.0f}s", flush=True)
        # planning information (future-xyz R²) per layer
        safe(lambda: {s.layer: s.raw["p2_r2"]
                      for s in PlanningInfoProbe(seed=self.seed).score_layers(self.backend, bundle=b).scores},
             "planning_p2_r2")
        # manifold geometry: effective_rank + isometry (+ anisotropy)
        man = ManifoldGeometryProbe(seed=self.seed).score_layers(self.backend, bundle=b).scores
        t["effective_rank"] = {s.layer: s.raw["effective_rank"] for s in man}
        t["isometry"] = {s.layer: s.raw["isometry"] for s in man}
        # linear action decodability (relative L1; LOW = action-ready)
        safe(lambda: {s.layer: s.raw["l1_rel"]
                      for s in LinearActionProbe(seed=self.seed).score_layers(self.backend, bundle=b).scores},
             "linear_probe_l1")
        # kernel-derived per-layer signals from the residual stream
        cka, l2, cosm = {}, {}, {}
        idx = b.layer_indices
        for L in idx:
            nxt = L + 1
            if nxt in b.hidden:
                cka[L] = kernels.linear_cka(b.hidden[L], b.hidden[nxt])
                delta = b.hidden[nxt] - b.hidden[L]
                l2[L] = float(((delta ** 2).sum(1) ** 0.5).mean())   # mean per-sample L2 of the residual write
                cosm[L] = float(kernels.block_influence(b.hidden[L], b.hidden[nxt]))  # 1 - cos
        t["adjacent_cka_to_next"] = cka
        t["residual_l2_mean"] = l2
        t["residual_cos_mean"] = {L: 1.0 - v for L, v in cosm.items()}  # back to a cosine
        # fim_avg: a gradient metric (and a verified-misleading one) -> not computed here
        t["fim_avg"] = {}
        self._layer_table = t

        # persist for instant reuse next run (skip the _errors meta; floats only)
        try:
            cp.parent.mkdir(parents=True, exist_ok=True)
            out = {p: {str(k): float(v) for k, v in d.items()}
                   for p, d in t.items() if not p.startswith("_")}
            out["_ckpt"] = ckpt
            out["_metric"] = "paper_l2_v1"
            out["_seed"] = self.seed
            out["_requested_knockout_samples"] = self.knockout_batches
            out["_actual_bundle_samples"] = int(b.actions.shape[0])
            cp.write_text(json.dumps(out, indent=2))
            print(f"[lib_engine] saved probe table -> {cp}")
        except Exception:  # noqa: BLE001 - caching is best-effort
            pass

    def query(self, probe: str, layer_ids: Sequence[int]) -> Dict[int, Optional[float]]:
        self._build_layer_table()
        table = self._layer_table.get(probe, {})
        return {int(L): (float(table[L]) if L in table else None) for L in layer_ids}

    #: probes this engine can compute (for the cheap availability check — answering
    #: "does this probe exist" must NOT trigger the expensive table build; that only
    #: happens when an agent actually query()s a probe).
    COMPUTABLE_PROBES = frozenset({
        "knockout_dall", "planning_p2_r2", "effective_rank", "isometry",
        "linear_probe_l1", "adjacent_cka_to_next", "residual_l2_mean", "residual_cos_mean",
    })

    def probe_is_available(self, probe: str) -> bool:
        # already computed/loaded -> answer from the table; otherwise just check the
        # catalog (cheap) so build_context's list_available() doesn't pay the table cost.
        if self._layer_table:
            return bool(self._layer_table.get(probe))
        return probe in self.COMPUTABLE_PROBES

    # --------------------------------------------------------------- recovery #
    def _strategy_for(self, recovery: str):
        r = (recovery or "direct").lower()
        steps, ev = self.recovery_steps, self.eval_every
        if r in ("direct", "none", ""):
            return None                                            # naked prune, no training
        if r == "adapter_only":
            return PhaseTwoJointRecovery(steps=steps, eval_every=ev, trainable=TrainableSpec.adapter_only())
        if r in ("adapter_mlp_0.5m",):
            return PhaseTwoJointRecovery(steps=steps, eval_every=ev,
                                         trainable=TrainableSpec.adapter_only(), bottleneck=256)
        if r in ("adapter+head+proprio", "phase2_joint", "joint"):
            return PhaseTwoJointRecovery(steps=steps, eval_every=ev)
        if r in ("onpolicy_distill", "onpolicy"):
            return OnPolicyDistillTrustRegion(
                steps=steps, eval_every=ev,
                base_strategy=PhaseTwoJointRecovery(steps=steps, eval_every=ev))
        # unknown -> fall back to the verified foundational recipe
        return PhaseTwoJointRecovery(steps=steps, eval_every=ev)

    def recover_and_eval(self, mask: Sequence[int], recovery: str) -> dict:
        """Run the recovery strategy for (mask, recovery) and return its closed-loop SR.

        If the backend exposes ``train_recovery(mask, recovery)`` (real backends like
        GenericVLABackend delegate recovery to the model plugin's real recovery+eval,
        e.g. OftPlugin -> finetune_phase2_adapter + run_libero_eval), use it. Otherwise
        run the recovery_lib per-step strategy (the synthetic mock path).
        """
        key = (tuple(sorted(int(m) for m in mask)), recovery)
        if key in self._recover_cache:
            return self._recover_cache[key]

        if hasattr(self.backend, "train_recovery"):
            out = dict(self.backend.train_recovery(list(mask), recovery))
            out.setdefault("source", f"backend:{recovery}")
            out.setdefault("recipe", recovery)
            out.setdefault("recoverability", self.recoverability(mask))
            self._recover_cache[key] = out
            return out

        strategy = self._strategy_for(recovery)
        meta = {"recoverability": self.recoverability(mask)}
        if strategy is None:
            # naked prune: build the student, do not train -> the catastrophic floor
            self.backend.build_student(list(mask), TrainableSpec.joint())
            sr = float(self.backend.evaluate_sr())
            out = {"sr": sr, "source": "lib:direct", "recipe": recovery,
                   "sr_curve": [(0, sr)], "slope": 0.0, **meta}
        else:
            result = strategy.recover(self.backend, list(mask))
            out = {"sr": float(result.final_sr), "source": f"lib:{result.recipe}",
                   "recipe": recovery, "sr_curve": result.sr_curve, "slope": result.slope, **meta}
        self._recover_cache[key] = out
        return out

    def known_recoveries_for_mask(self, mask: Sequence[int]) -> List[str]:
        m = tuple(sorted(int(x) for x in mask))
        return [rec for (mm, rec) in self._recover_cache if mm == m]

    def recovery_ladder(self, mask: Sequence[int]) -> Dict[str, float]:
        """Run every recipe on one mask -> {recipe: SR} (the recovery_lib ladder)."""
        return {r: self.recover_and_eval(mask, r)["sr"] for r in RECIPE_IDS}

    # --------------------------------------------------------- recoverability #
    def recoverability(self, mask: Sequence[int]) -> Optional[float]:
        """Mask-level boundary recoverability cos (mask_probe_lib), or None."""
        try:
            return float(BoundaryLowRankProbe().score_mask(self.backend, list(mask)).value)
        except Exception:  # noqa: BLE001
            return None
