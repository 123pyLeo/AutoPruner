"""
Lib-backed drop-in replacements for the three live agentic seams.
=================================================================

Each class mirrors the EXACT interface the driver calls, but delegates to a shared
:class:`LibEngine` (mask_probe_lib + recovery_lib) instead of cached JSON:

* ``LibProbeQueryService``  <- ProbeQueryService  (list_available / query)
* ``LibRecoveryModule``     <- RecoveryModule      (list_recipes / is_valid / get_summary)
* ``LibEvalModule``         <- EvalModule          (lookup_cached / list_known_recoveries_for_mask)

``build_lib_module_set(data_dir, backend)`` assembles a ``ModuleSet`` that reuses the
JSON-backed ProbeModule / MaskSearchModule / DiagnosisModule (static priors,
search-space, diagnosis taxonomy) and swaps in these three. The driver / run_loop /
rule_store / diagnosis are used UNCHANGED.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from agentic_pruning.modules import ModuleSet
from agentic_pruning.modules.probe import ProbeModule
from agentic_pruning.modules.mask_search import MaskSearchModule
from agentic_pruning.modules.diagnosis import DiagnosisModule
from agentic_pruning.modules.probe_query import PROBE_CATALOG

from .lib_engine import RECIPE_IDS, LibEngine

# verified probes the libraries add on top of the original catalog
EXTRA_PROBE_CATALOG = {
    "knockout_dall": {
        "description": "Identity-skip a layer and measure mean flattened-action-chunk L2 change. "
                       "Higher means a larger action change on these calibration inputs. "
                       "Use it to rank proposals; closed-loop success must still be measured.",
        "value_range": "[0, ~]",
        "real_cost_estimate": "~1 forward pass/layer (cheap, open-loop)",
    },
    "planning_p2_r2": {
        "description": "Held-out ridge R² decoding the final action's first three components. "
                       "Interpret according to the model's action units (for LIBERO these are action deltas). "
                       "This diagnostic is not a closed-loop success gate.",
        "value_range": "[0, 1]",
        "real_cost_estimate": "~1 forward sweep + ridge",
    },
}

# recovery recipes the agent may choose (enriched by recovery_lib)
RECIPES = [
    {"id": "direct", "trainable": [],
     "description": "naked prune, no recovery -> catastrophic floor (verifies pruning is invalid alone)"},
    {"id": "adapter_only", "trainable": ["adapter"],
     "description": "boundary adapter only, frozen sidecars. VERIFIED at shallow k; fails deep."},
    {"id": "adapter_mlp_0.5M", "trainable": ["adapter"],
     "description": "tiny bn=256 adapter only. k8 [22-29] -> 94.5%."},
    {"id": "adapter+head+proprio", "trainable": ["adapter", "action_head", "proprio"],
     "description": "VERIFIED foundational joint recovery (Phase-2). k16->98%, k28 keep-anchors->96.5%."},
    {"id": "adapter_multi", "trainable": ["adapter"],
     "description": "one boundary adapter PER removed run (skip-pruning models, e.g. StarVLA k10anchor -> 10 adapters)."},
    {"id": "onpolicy_distill", "trainable": ["adapter", "action_head", "kept_layers"],
     "description": "VERIFIED on-policy DAgger distill + trust-region anchor; breaks the frozen-backbone ceiling (0.788->0.828)."},
]


class LibProbeQueryService:
    """probe_query seam -> mask_probe_lib per-layer probes (via LibEngine)."""

    def __init__(self, engine: LibEngine):
        self.engine = engine
        self.catalog = {**PROBE_CATALOG, **EXTRA_PROBE_CATALOG}
        self._calls: list[dict] = []

    def list_available(self) -> List[dict]:
        out = []
        for name, meta in self.catalog.items():
            out.append({"name": name, "available": self.engine.probe_is_available(name), **meta})
        return out

    def query(self, probe: str, layer_ids: list[int]) -> dict:
        if probe not in self.catalog:
            raise KeyError(f"unknown probe '{probe}'; available: {list(self.catalog)}")
        res = self.engine.query(probe, [int(L) for L in layer_ids])
        self._calls.append({"probe": probe, "layer_ids": list(layer_ids), "result": res})
        return res

    def query_log(self) -> list[dict]:
        return list(self._calls)

    def reset_log(self) -> None:
        self._calls = []


class LibRecoveryModule:
    """recovery seam -> recovery_lib strategies (recipe metadata only; SR via LibEvalModule)."""

    def __init__(self, engine: LibEngine):
        self.engine = engine
        # The standard menu is the shared VOCABULARY; each model implements a subset,
        # declared in its plugin.recovery_strategies. Restrict the agent's options to
        # that subset (fall back to the full menu for mock / plugin-less backends).
        allowed = getattr(getattr(engine, "backend", None), "plugin", None)
        allowed = getattr(allowed, "recovery_strategies", None)
        if allowed:
            by_id = {r["id"]: r for r in RECIPES}
            self._recipes = [dict(by_id.get(i, {"id": i, "trainable": [],
                                           "description": f"(model recovery recipe '{i}')"})
                             )
                             for i in allowed]   # preserve the plugin's declared order
            descriptions = {
                "direct": "Evaluate the shortened policy without recovery training.",
                "adapter_only": "Train the boundary adapter; freeze the backbone, action head and proprio projector.",
                "adapter+head+proprio": "Train boundary adapter, action head and proprio projector with demonstrations and teacher supervision.",
            }
            for recipe in self._recipes:
                recipe["description"] = descriptions.get(recipe["id"], "Model-declared recovery recipe")
        else:
            self._recipes = list(RECIPES)

    def restrict(self, recipe_ids):
        """Restrict the search to an explicit subset of model-supported recipes."""
        ids = list(recipe_ids)
        available = {r["id"]: r for r in self._recipes}
        if not ids or len(set(ids)) != len(ids) or any(i not in available for i in ids):
            raise ValueError(f"Recoveries must be a nonempty unique subset of {list(available)}")
        self._recipes = [available[i] for i in ids]

    def list_recipes(self) -> List[dict]:
        return [dict(r) for r in self._recipes]

    def get_recipe(self, recipe_id: str) -> Optional[dict]:
        return next((dict(r) for r in self._recipes if r["id"] == recipe_id), None)

    def is_valid(self, recipe_id: str) -> bool:
        return any(r["id"] == recipe_id for r in self._recipes)

    def get_summary(self) -> dict:
        return {"recipes": [{"id": r["id"], "trainable": r["trainable"],
                             "description": r["description"]} for r in self._recipes]}

    def train(self, mask: list[int], recipe_id: str, **kwargs):
        """Actually runs the recovery_lib strategy (the no-op stub is now live)."""
        return self.engine.recover_and_eval(mask, recipe_id)


class LibEvalModule:
    """eval seam -> recovery_lib recovers the student, backend reports SR."""

    def __init__(self, engine: LibEngine, suite: str = "libero_object"):
        self.engine = engine
        self._suite = suite

    @property
    def suite(self) -> str:
        return self._suite

    def lookup_cached(self, mask: list[int], recovery: str) -> Optional[dict]:
        """Runs (memoised) recovery + eval and returns the SR -- same shape as EvalModule."""
        out = self.engine.recover_and_eval(mask, recovery)
        return {**out, "sr": out.get("sr"), "n_trials": out.get("episodes", out.get("n_trials")), "raw": None,
                "name": out.get("source", "lib"), "source": out.get("source", "lib"),
                "slope": out.get("slope"), "recoverability": out.get("recoverability"),
                # carried through so the report can point at the recovered weights:
                "run_dir": out.get("run_dir"), "adapter_step": out.get("adapter_step")}

    def list_known_recoveries_for_mask(self, mask: list[int]) -> List[str]:
        return self.engine.known_recoveries_for_mask(mask)

    def run_closed_loop(self, ckpt_path=None, n_trials: int = 50) -> dict:
        raise NotImplementedError("lib mode evaluates via recover_and_eval(mask, recovery).")


def _resolve_prior_path(d: Path, model: Optional[str], suite: Optional[str]) -> Optional[Path]:
    """Find a static-prior file that genuinely belongs to this (model, suite), else None.

    Tries ``probe_<model>_<suite>.json`` then ``probe_<suite>.json``; the legacy
    ``probe_object.json`` counts ONLY for OFT's libero_object. Returns None for an unseen
    model/suite -> the caller seeds an EMPTY prior and the agent relies on the live,
    model-specific probe table (LibEngine), per project policy ("for new models this must
    be computed").
    """
    names = []
    if model and suite:
        names.append(f"probe_{model}_{suite}.json")
    if suite:
        names.append(f"probe_{suite}.json")
    for name in names:
        p = d / name
        if p.exists():
            return p
    legacy = d / "probe_object.json"                       # OFT libero_object only
    if legacy.exists() and model in (None, "oft") and suite in (None, "libero_object"):
        return legacy
    return None


def build_lib_module_set(data_dir, backend, *, recovery_steps: int = 150,
                         eval_every: int = 50, align_dense_sr: bool = True) -> ModuleSet:
    """Assemble a ModuleSet whose probe_query / recovery / eval are lib-backed.

    Model-agnostic: the BACKEND is the ground truth for shape (n_layers), suite, and
    dense SR. The static-prior JSON only SEEDS per-layer single-delete facts, and only
    when it genuinely belongs to this (model, suite); for an unseen model the seed is
    empty and the agent derives everything from the live probe table. MaskSearchModule
    (search space + QD archive) and DiagnosisModule are reused unchanged.
    """
    d = Path(data_dir)
    model = getattr(getattr(backend, "plugin", None), "name", None)
    suite = getattr(backend, "suite", None)

    prior = _resolve_prior_path(d, model, suite)
    probe = ProbeModule(prior or (d / "probe_object.json"))   # legacy file = structural template

    # --- backend is ground truth: override the seed file's shape/baseline so a NEW model
    #     is never stuck with OFT's 32 layers / libero_object facts. ---
    n = getattr(backend, "n_layers", None)
    if n:
        probe._data["n_layers_total"] = int(n)
    if suite:
        probe._data["suite"] = suite
    if align_dense_sr and getattr(backend, "dense_sr", None) is not None:
        probe._data["dense_sr"] = float(backend.dense_sr)
    # per-layer single-delete facts transfer ONLY if the prior is really this model's
    # (matching layer count); otherwise drop them — they don't carry across models.
    facts = probe._data.get("layers", [])
    if prior is None or len(facts) != int(n or 0):
        probe._data["layers"] = []

    engine = LibEngine(backend, recovery_steps=recovery_steps, eval_every=eval_every)
    search = MaskSearchModule(d / "mask_search_config.json")
    search._data["total_layers"] = int(n or probe.n_layers)
    search._data["suite"] = suite or probe.suite
    search._data["k_options"] = [k for k in search._data.get("k_options", []) if 0 < k < search.total_layers]
    if not search._data["k_options"]:
        search._data["k_options"] = list(range(1, search.total_layers))
    mode = getattr(getattr(backend, "plugin", None), "pruning_mode", "contiguous")
    search._data["mask_form"] = "skip" if mode == "skip" else "single_contiguous_block"
    search._data["mask_form_doc"] = (
        "A nonempty subset of original layer indices; leave at least one layer."
        if mode == "skip" else "One consecutive run of original layer indices; leave at least one layer.")
    search._data["constraint"] = search._data["mask_form_doc"]
    search._data["start_constraint_doc"] = f"Valid original indices: 0..{search.total_layers-1}."
    return ModuleSet(
        probe=probe,
        mask_search=search,
        recovery=LibRecoveryModule(engine),
        eval_=LibEvalModule(engine, suite=probe.suite),
        diagnosis=DiagnosisModule(
            d / "diagnosis_taxonomy.json",
            dense_sr=probe.dense_sr,
            layer_static_facts=probe.get_layer_static_facts(),
            n_layers=probe.n_layers,
        ),
        probe_query=LibProbeQueryService(engine),
    )
