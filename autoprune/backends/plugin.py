"""
ModelPlugin -- the small, model-specific surface a new VLA must provide.
=========================================================================

The lightweight extension point. To support a new model family (OpenVLA-OFT,
StarVLA, a World Model, ...) you do NOT write a whole RecoveryBackend. You write
ONE ``ModelPlugin`` (these 4 methods + 3 identity fields) and register it; the
shared :class:`GenericVLABackend` turns it into a full backend that the agentic
search, recovery, and eval all run on unchanged. Native export uses the optional
``export_checkpoint`` method so the framework never guesses a model format.

What is SHARED (written once, in GenericVLABackend -- you never touch it):
    * contiguous-block mask -> (start, end) handling
    * boundary_pair(mask)        -- sliced from the dense activations you return
    * the 5 library method-classes' wiring + the RecoveryBackend ABC
    * param/compute reporting from the checkpoint config
    * policy caching, work dirs, eval-scope plumbing

What is PER-MODEL (this file -- the irreducible ~20%):
    identity     : n_layers, default_dense_sr, suites
    load_policy  : how to load the dense model
    collect_activations : one dense sweep -> per-layer action-token hidden + actions
    predict_action      : action chunk under a mask / single-layer knockout
    recover_and_eval    : (mask, recovery, suite) -> closed-loop SR  (usually a
                          subprocess to the model's real recovery + eval scripts)

``boundary_pair`` is intentionally NOT here: GenericVLABackend derives it from
``collect_activations`` (h_pre = hidden[min(mask)], h_post = hidden[max(mask)+1]),
so every plugin gets it for free.

Suites (e.g. LIBERO object/goal/spatial/long) are just a ``suite`` argument to
collect_activations / predict_action / recover_and_eval -- a plugin that threads
it through supports all of a model's dimensions with no extra code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class EvalScope:
    """How much closed-loop eval to run. ``None`` caps == the model's full suite.

    Set the caps for a fast smoke (1 task / few steps); leave them None for the
    real number that goes in a result.
    """
    num_trials: int = 10          # rollouts per task
    max_eval_tasks: Optional[int] = None
    task_max_steps: Optional[int] = None


class ModelPlugin(ABC):
    """The model-specific surface. Implement these for a new VLA family."""

    # --- identity ---------------------------------------------------------- #
    #: registry key + ``--backend`` value (e.g. "oft", "starvla").
    name: str = "unnamed"
    #: True only for explicit software demonstrations with synthetic outcomes.
    synthetic: bool = False
    #: number of decoder layers in the dense backbone.
    n_layers: int = 0
    #: closed-loop SR of the unpruned model (reported alongside results).
    default_dense_sr: Optional[float] = None
    #: task suites this model supports; suites[0] is the default.
    suites: Tuple[str, ...] = ()

    #: natural-language names that resolve to this plugin (so the PROMPT can name
    #: the model, e.g. "剪 OpenVLA-OFT"). Matched case-insensitively as substrings.
    aliases: Tuple[str, ...] = ()

    #: default pruning mode: "contiguous" | "skip" | "auto".
    #: The Agent confirms/overrides this from the knockout anchor-geometry rule
    #: (see decide_pruning_mode); "auto" = no prior, decide purely from the probe.
    pruning_mode: str = "auto"

    #: recovery recipe ids this model supports (the standard-menu vocabulary it
    #: implements). The Agent may only pick a recovery from this set.
    recovery_strategies: Tuple[str, ...] = ("direct",)

    #: unified recovery-ladder declaration: canonical tier id -> this model's native
    #: recipe id (None = the architecture does not support that rung). Tiers and the
    #: escalation order are defined once in :mod:`autoprune.backends.recovery_ladder`.
    #: Subclasses REPLACE this with their own map; ``validate_ladder`` keeps it in
    #: sync with ``recovery_strategies``.
    recovery_ladder: Dict[str, Optional[str]] = {"direct": "direct"}

    #: weight registry: ``{suite -> checkpoint path}``. Lets the Agent locate the
    #: ckpt from (model name + suite) instead of being handed a path. Subclasses
    #: REPLACE this with their own map (do not mutate the shared default).
    checkpoints: Dict[str, str] = {}

    #: per-suite dense (un-pruned) SR for the report baseline. Subclasses populate the
    #: values they have VERIFIED; unlisted suites fall back to ``default_dense_sr``.
    dense_sr_by_suite: Dict[str, float] = {}

    @property
    def default_suite(self) -> str:
        return self.suites[0] if self.suites else "default"

    def resolve_checkpoint(self, suite: str) -> Optional[str]:
        """Look up this model's checkpoint for ``suite`` from the registry (or None)."""
        return self.checkpoints.get(suite)

    def resolve_dense_sr(self, suite: str) -> Optional[float]:
        """Per-suite dense SR if verified, else the model-wide ``default_dense_sr``."""
        return self.dense_sr_by_suite.get(suite, self.default_dense_sr)

    def supports_recovery(self, recipe: str) -> bool:
        """True if ``recipe`` is a native recipe id OR a supported ladder tier id."""
        return recipe in self.recovery_strategies or self.resolve_recovery(recipe) is not None

    def resolve_recovery(self, name: str) -> Optional[str]:
        """Resolve ``name`` (a ladder tier id or a native recipe id) to this model's
        native recipe id, or None if the model does not support it. This lets the
        agent speak the unified tier vocabulary while ``recover_and_eval`` keeps
        receiving the native ids it already dispatches on."""
        if name in self.recovery_strategies:
            return name
        return self.recovery_ladder.get(name)

    def supported_tiers(self) -> Tuple[str, ...]:
        """Supported ladder rungs, cheapest first (the agent's escalation order)."""
        from autoprune.backends.recovery_ladder import TIER_ORDER
        return tuple(t for t in TIER_ORDER if self.recovery_ladder.get(t) is not None)

    @staticmethod
    def mask_tag(mask: Sequence[int]) -> str:
        """Compact, filename-safe id for a mask (a full 28-layer list overflows the
        255-char filename limit used in run dirs). Contiguous -> 'k28_2-29';
        scattered/skip -> 'k10_<hash>'. Shared by every plugin's run/dir naming."""
        import hashlib
        m = sorted(int(x) for x in mask)
        if m and m == list(range(m[0], m[-1] + 1)):
            return f"k{len(m)}_{m[0]}-{m[-1]}"
        h = hashlib.md5(",".join(map(str, m)).encode()).hexdigest()[:8]
        return f"k{len(m)}_{h}"

    # --- model load (cached by the backend) -------------------------------- #
    @abstractmethod
    def load_policy(self, full_ckpt: str):
        """Load and return the dense policy/model handle (opaque to the backend).

        Called at most once per backend; the result is cached and passed back to
        ``collect_activations`` / ``predict_action``. Recovery/eval re-load inside
        their own subprocess, so this handle is only for the cheap probes.
        """

    # --- probe captures (model-specific forward) --------------------------- #
    @abstractmethod
    def collect_activations(
        self, policy, *, suite: str, n_samples: int, seed: int
    ) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        """One dense forward sweep over a fixed calibration set.

        Returns ``(hidden, actions)`` where
            hidden[l]  -> (N, hidden_dim) residual stream ENTERING layer ``l``
                          at action-token positions, for every ``l`` in
                          ``0 .. n_layers`` (key ``n_layers`` = output of last layer).
            actions    -> (N, chunk, action_dim) actions aligned to those inputs.
                          Document whether these are predictions or demonstrations.

        Same ``seed`` must yield the same N samples (so predictions are comparable).
        """

    @abstractmethod
    def predict_action(
        self, policy, *, mask: Optional[Sequence[int]] = None,
        knockout_layer: Optional[int] = None, n_samples: int = 50, seed: int = 0,
    ) -> np.ndarray:
        """Predicted action chunk ``(N, chunk, action_dim)`` over the fixed eval set.

        * ``mask``           -> physically skip those layers first.
        * ``knockout_layer`` -> identity-skip exactly ONE layer (output := input)
                                without changing layer count (the cheap anchor probe).
        Fixed ``seed`` -> same samples every call (element-wise comparable).
        """

    # --- recovery + eval (coarse; usually a subprocess to the real scripts) -- #
    @abstractmethod
    def recover_and_eval(
        self, *, full_ckpt: str, mask: Sequence[int], recovery: str,
        suite: str, scope: EvalScope, work_dir, seed: int,
    ) -> dict:
        """Recover the pruned model and return its closed-loop SR.

        This is the ground-truth path the agentic loop routes through. Typically
        shells out to the model's REAL recovery (e.g. finetune_phase2_adapter) then
        its REAL eval (e.g. run_libero_eval), both parameterised by ``suite``.

        ``recovery == "direct"`` means *no* training: prune + eval only (the SR
        floor). Other ``recovery`` ids select the model's recovery recipe.

        A measured result requires completed status, SR, successes and episodes;
        the framework validates their consistency. Return the executed native
        recipe and a unique run_dir containing that candidate's saved artifacts.
        Failed candidates must not reuse a previous candidate's weights or score.
        """

    def export_checkpoint(self, *, evaluation: dict, mask: Sequence[int], recovery: str,
                          destination: Path) -> str:
        """Optional native export of the selected, evaluated candidate.

        Write reloadable model files into the new empty destination and return
        a nonempty format identifier. Use evaluation's exact saved artifacts;
        do not retrain, choose a newer step, or copy the dense model unchanged.
        The framework adds checksums and a selection receipt after validation.
        Implement with this architecture's native save/export routine. Synthetic
        plugins must keep their artifact format explicitly synthetic.
        """
        raise NotImplementedError(f"{self.name} must implement export_checkpoint for native model export")


# --------------------------------------------------------------------------- #
# registry -- a plugin registers itself on import; make_backend looks it up.
# --------------------------------------------------------------------------- #
PLUGIN_REGISTRY: Dict[str, ModelPlugin] = {}


def register_plugin(plugin: ModelPlugin) -> ModelPlugin:
    """Register a plugin instance under ``plugin.name`` (idempotent)."""
    PLUGIN_REGISTRY[plugin.name] = plugin
    return plugin


def get_plugin(name: str) -> Optional[ModelPlugin]:
    return PLUGIN_REGISTRY.get(name)


def resolve_plugin_by_name(text: str) -> Optional[ModelPlugin]:
    """Resolve a plugin from free text (e.g. a prompt naming the model).

    Matches ``text`` against each plugin's registry name and ``aliases``
    (case-insensitive substring). Returns the first match, or None. Longer alias
    matches win so "openvla-oft" beats a bare "openvla".
    """
    t = (text or "").lower()
    best, best_len = None, 0
    for plugin in PLUGIN_REGISTRY.values():
        for key in (plugin.name, *plugin.aliases):
            k = key.lower()
            if k and k in t and len(k) > best_len:
                best, best_len = plugin, len(k)
    return best
