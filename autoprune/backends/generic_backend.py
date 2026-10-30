"""
GenericVLABackend -- one RecoveryBackend for every model, driven by a ModelPlugin.
==================================================================================

Written ONCE. It implements the full library interface
(``mask_probe_lib.ModelBackend`` + ``recovery_lib.RecoveryBackend``) by delegating
the few model-specific bits to a :class:`~autoprune.backends.plugin.ModelPlugin`.

    GenericVLABackend(plugin=OftPlugin(), full_ckpt=..., suite="libero_object")

So a new model family = write a plugin (4 methods), not a backend. The agentic
search, recovery and eval then run on it with zero changes.

Recovery path: like the validated AutoPrune flow, recovery+eval is COARSE -- the
backend exposes ``train_recovery(mask, recovery)`` and the lib engine routes
through it (one subprocess to the model's real recovery + eval), instead of the
per-step in-process loop. The fine-grained RecoveryBackend training primitives
(build_student / *_forward / apply_gradients / iter_offline_batches) are therefore
not used here and raise a clear message if called.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from recovery_lib.core.backend import RecoveryBackend
from recovery_lib.core.types import Batch, StudentOutput, TeacherOutput, TrainableSpec
from mask_probe_lib.core.types import ActivationBundle

from autoprune.backends.plugin import EvalScope, ModelPlugin
from autoprune.backends.pruning_mode import split_into_runs

REPO_ROOT = Path(__file__).resolve().parents[2]


def mask_boundaries(mask: Sequence[int]) -> list[tuple[int, int]]:
    """Map a mask to its 断口 boundaries — ONE pair per removed run.

    Each removed contiguous run ``[a..b]`` leaves a boundary the adapter must bridge:
        (h_pre = residual entering layer a,  h_post = residual leaving layer b == input of b+1)
    Contiguous mask -> 1 boundary. Skip mask -> several (StarVLA k10anchor -> 10).
    Returns ``[(a, b+1), ...]`` (residual-stream keys).
    """
    m = [int(x) for x in mask]
    if not m:
        raise ValueError("empty mask")
    return [(r[0], r[-1] + 1) for r in split_into_runs(m)]


class GenericVLABackend(RecoveryBackend):
    """RecoveryBackend over any VLA, parameterised by a ModelPlugin + checkpoint."""

    def __init__(self, plugin: ModelPlugin, full_ckpt: str, suite: Optional[str] = None, *,
                 dense_sr: Optional[float] = None, num_trials: int = 10, seed: int = 7,
                 work_dir: Optional[str] = None,
                 max_eval_tasks: Optional[int] = None, task_max_steps: Optional[int] = None):
        self.plugin = plugin
        self.full_ckpt = str(Path(full_ckpt).expanduser().resolve())
        self.task_suite = suite or plugin.default_suite
        if plugin.suites and self.task_suite not in plugin.suites:
            raise ValueError(f"{plugin.name}: suite {self.task_suite!r} not in {plugin.suites}")
        plugin.active_suite = self.task_suite
        self._dense_sr = dense_sr if dense_sr is not None else plugin.resolve_dense_sr(self.task_suite)
        self.num_trials = num_trials
        self.seed = seed
        # Env knobs let a fast smoke cap eval scope (e.g. AUTOPRUNE_MAX_EVAL_TASKS=1).
        if max_eval_tasks is None and os.environ.get("AUTOPRUNE_MAX_EVAL_TASKS"):
            max_eval_tasks = int(os.environ["AUTOPRUNE_MAX_EVAL_TASKS"])
        if task_max_steps is None and os.environ.get("AUTOPRUNE_TASK_MAX_STEPS"):
            task_max_steps = int(os.environ["AUTOPRUNE_TASK_MAX_STEPS"])
        self.scope = EvalScope(num_trials=num_trials, max_eval_tasks=max_eval_tasks,
                               task_max_steps=task_max_steps)
        self.work_dir = Path(work_dir or (Path.cwd() / "runs" / f"{plugin.name}_{uuid.uuid4().hex}")).expanduser().resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._policy = None
        self._current_mask = None
        self._current_recovery = None

    # ------------------------------------------------------------- identity -- #
    @property
    def n_layers(self) -> int:
        return self.plugin.n_layers

    @property
    def dense_sr(self) -> Optional[float]:
        return self._dense_sr

    @property
    def suite(self) -> str:
        return self.task_suite

    # ------------------------------------------------------------- probes ---- #
    def _load_policy(self):
        if self._policy is None:
            self._policy = self.plugin.load_policy(self.full_ckpt)
        return self._policy

    def collect_activations(self, n_batches: int = 200, seed: int = 0, **kw) -> ActivationBundle:
        hidden, actions = self.plugin.collect_activations(
            self._load_policy(), suite=self.task_suite, n_samples=n_batches, seed=seed)
        actions = self._actions(actions, n_batches)
        if (not isinstance(hidden, dict) or
                any(not isinstance(k, (int, np.integer)) or isinstance(k, bool) for k in hidden) or
                set(hidden) != set(range(self.n_layers + 1))):
            raise ValueError("Plugin activations must provide every layer boundary 0..n_layers")
        hidden = {int(k): np.asarray(v) for k, v in hidden.items()}
        shape = hidden[0].shape
        if (len(shape) != 2 or shape[0] != len(actions) or shape[1] < 1 or
                any(v.shape != shape or not np.isfinite(v).all() for v in hidden.values())):
            raise ValueError("Plugin hidden arrays must be finite, aligned (N, hidden_dim) residuals")
        return ActivationBundle(hidden=hidden, actions=actions, meta={"suite": self.task_suite})

    @staticmethod
    def _actions(values, requested):
        actions = np.asarray(values)
        if (actions.ndim != 3 or not 1 <= actions.shape[0] <= requested or
                min(actions.shape[1:]) < 1 or not np.isfinite(actions).all()):
            raise ValueError("Plugin actions must be finite (N, chunk, action_dim), with 1 <= N <= requested")
        return actions

    def predict_action(self, *, mask=None, knockout_layer=None, n_batches: int = 50, seed: int = 0):
        return self._actions(self.plugin.predict_action(self._load_policy(), mask=mask,
                             knockout_layer=knockout_layer, n_samples=n_batches, seed=seed), n_batches)

    def boundary_pairs(self, mask: Sequence[int], n_batches: int = 50, seed: int = 0):
        """Shared: slice the dense activation bundle at EVERY 断口 of the mask.

        Returns ``[(h_pre, h_post), ...]`` — one pair per removed run, so it works
        for both contiguous (1 pair) and skip (several) masks. Each pair:
            h_pre  = residual entering the run's first removed layer  -> hidden[a]
            h_post = residual leaving the run's last removed layer     -> hidden[b+1]
        """
        b = self.collect_activations(n_batches=n_batches, seed=seed)
        pairs = []
        for pre, post in mask_boundaries(mask):
            if pre not in b.hidden or post not in b.hidden:
                raise KeyError(f"activation bundle lacks boundary keys {pre} / {post}")
            pairs.append((np.asarray(b.hidden[pre]), np.asarray(b.hidden[post])))
        return pairs

    def boundary_pair(self, mask: Sequence[int], n_batches: int = 50, seed: int = 0):
        """Single-断口 convenience (contiguous masks). Raises for skip masks —
        use :meth:`boundary_pairs` there."""
        pairs = self.boundary_pairs(mask, n_batches=n_batches, seed=seed)
        if len(pairs) != 1:
            raise ValueError(f"boundary_pair() is single-断口 (contiguous) only; mask has "
                             f"{len(pairs)} boundaries — use boundary_pairs() for skip masks.")
        return pairs[0]

    # ------------------------------------------- recovery + eval (the real path) #
    def _free_policy(self) -> None:
        """Release the in-process PROBE model from the GPU before a recovery subprocess.

        The probe loads the dense policy on one GPU and the per-layer table is then cached
        (in-memory + on disk), so the policy is no longer needed. Recovery training is a
        SUBPROCESS that wants the same GPUs — if the ~16GB probe model stays resident, the
        first candidate's training OOMs. Freeing it here is safe: later probe queries read
        the cached table; if one ever needs the policy again, _load_policy reloads it."""
        if self._policy is None:
            return
        try:
            import gc
            import torch
            self._policy = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - best-effort GPU release
            self._policy = None

    def train_recovery(self, mask: Sequence[int], recovery: str) -> dict:
        """Recover (mask, recovery) on this model's real pipeline; return SR.

        Mode-agnostic: accepts contiguous OR skip masks (the plugin's recover_and_eval
        handles multi-断口). The lib engine routes recover_and_eval() here.
        """
        if not list(mask):
            raise ValueError("empty mask")
        self._current_mask = tuple(sorted(int(x) for x in mask))
        self._current_recovery = recovery
        # Dry-run: exercise the FULL agentic loop (probe -> gpt mask selection -> search/report)
        # WITHOUT the expensive recovery+eval subprocess. Returns SR=None so the loop still
        # records the agent's probe-informed mask proposals (the search trace) — a feasibility
        # check that the GPT-led pruning step works on this model.
        dry = os.environ.get("AUTOPRUNE_DRY_RECOVERY", "0")
        if dry not in {"0", "1"}:
            raise ValueError("AUTOPRUNE_DRY_RECOVERY must be 0 or 1")
        if dry == "1":
            m = sorted(int(x) for x in mask)
            print(f"[generic_backend] DRY recovery (no training): mask={m} recovery={recovery}", flush=True)
            return {"sr": None, "status": "dry_run", "source": f"{getattr(self.plugin, 'name', 'model')}:{recovery}:dry",
                    "recipe": recovery, "mask": m, "dry": True}
        self._free_policy()                  # release the probe GPU model before the subprocess
        try:
            result = self.plugin.recover_and_eval(
                full_ckpt=self.full_ckpt, mask=mask, recovery=recovery, suite=self.task_suite,
                scope=self.scope, work_dir=self.work_dir, seed=self.seed)
            from autoprune.validation import validate_evaluation
            result = validate_evaluation(result, require_counts=not self.plugin.synthetic)
            if result.get('sr') is not None and (
                    result.get('recipe', recovery) != recovery or
                    result.get('executed_recovery', recovery) != recovery):
                raise ValueError('Plugin executed a different recovery recipe from the requested one')
            return result
        except Exception as e:  # noqa: BLE001
            # One candidate's recovery/eval failing (e.g. a transient subprocess crash) must
            # NOT kill the whole search — record it as SR=None and let the loop continue.
            print(f"[generic_backend] recover_and_eval failed for mask={sorted(int(x) for x in mask)} "
                  f"recovery={recovery}: {type(e).__name__}: {str(e)[:200]} -> SR=None (skipped)", flush=True)
            return {"sr": None, "status": "failed", "source": f"{getattr(self.plugin, 'name', 'model')}:{recovery}:failed",
                    "recipe": recovery, "error": str(e)[:300]}

    def closed_loop_sr(self, mask: Sequence[int], recovery: Optional[str] = None,
                       n_trials: int = 10) -> Optional[float]:
        return self.train_recovery(mask, recovery or "adapter+head+proprio").get("sr")

    # ------------------------------------------------------- param/compute --- #
    def model_config(self) -> Optional[dict]:
        """Transformer hyperparams from the ckpt config (no weights loaded), for the
        param/compute reduction report. Best-effort across OFT/LLaMA config shapes."""
        cfg_path = Path(self.full_ckpt) / "config.json"
        if not cfg_path.exists():
            return None
        try:
            raw = json.loads(cfg_path.read_text())
        except Exception:  # noqa: BLE001
            return None
        candidates = [raw]
        for key in ("text_config", "llm_config", "language_model", "llm"):
            v = raw.get(key)
            if isinstance(v, dict):
                candidates.append(v)
        for c in candidates:
            h, i, n = c.get("hidden_size"), c.get("intermediate_size"), c.get("num_hidden_layers")
            if h and i and n:
                return {"hidden_size": int(h), "intermediate_size": int(i),
                        "num_hidden_layers": int(n), "vocab_size": int(c.get("vocab_size", 0) or 0)}
        # ckpt config carries no LLM hyperparams (e.g. OpenVLA/Prismatic configs) ->
        # fall back to the plugin's DECLARED decoder shape (marked nominal in the report).
        nom = getattr(self.plugin, "llm_config", None)
        if nom and nom.get("hidden_size") and nom.get("intermediate_size"):
            return {**{k: int(v) for k, v in nom.items() if isinstance(v, (int, float))},
                    "nominal": True}
        return None

    # ---------------- fine-grained training primitives: not used (coarse path) -- #
    _COARSE = ("GenericVLABackend recovers via train_recovery() -> the plugin's "
               "real recovery+eval subprocess; per-step training primitives are not used.")

    def build_student(self, *a, **k):
        raise NotImplementedError(self._COARSE)

    def student_forward(self, batch: Batch) -> StudentOutput:
        raise NotImplementedError(self._COARSE)

    def teacher_forward(self, batch: Batch) -> TeacherOutput:
        raise NotImplementedError(self._COARSE)

    def apply_gradients(self, loss) -> float:
        raise NotImplementedError(self._COARSE)

    def iter_offline_batches(self, steps, batch_size=4, task_filter=None) -> Iterable[Batch]:
        raise NotImplementedError(self._COARSE)
