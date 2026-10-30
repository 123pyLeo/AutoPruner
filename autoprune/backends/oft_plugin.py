"""OFT integration using extracted probe helpers and the pinned custom training overlay."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import shutil
import json
import uuid
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from autoprune.backends.plugin import EvalScope, ModelPlugin, register_plugin

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Paths are supplied by the user; no research-machine defaults are active.
OFT_REPO = Path(os.environ.get("OFT_REPO", ".")).expanduser().resolve()
DATA_ROOT = str(Path(os.environ["OFT_DATA_ROOT"]).expanduser().resolve()) if os.environ.get("OFT_DATA_ROOT") else ""
HDF5_ROOT = str(Path(os.environ["OFT_HDF5_ROOT"]).expanduser().resolve()) if os.environ.get("OFT_HDF5_ROOT") else ""
LIBERO_ROOT = str(Path(os.environ["OFT_LIBERO_ROOT"]).expanduser().resolve()) if os.environ.get("OFT_LIBERO_ROOT") else ""
PYTHON_BIN = os.environ.get("AUTOPRUNE_PY", sys.executable)
TORCHRUN = os.environ.get("AUTOPRUNE_TORCHRUN", str(Path(PYTHON_BIN).with_name("torchrun")))
TRAIN_GPUS = os.environ.get("AUTOPRUNE_TRAIN_GPUS", "0")
EVAL_GPU = os.environ.get("AUTOPRUNE_EVAL_GPU", "0")
NUM_IMAGES_IN_INPUT = 2
CACHE_GEN_SCRIPT = OFT_REPO / "vla-scripts/cache_teacher_outputs.py"
NCCL_PRELOAD = os.environ.get("AUTOPRUNE_NCCL_PRELOAD", "")


def _cuda_visibility(selection):
    """Map logical selections through the caller's existing GPU visibility."""
    requested = [x.strip() for x in selection.split(",") if x.strip()]
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited is None:
        return ",".join(requested)
    visible = [x.strip() for x in inherited.split(",") if x.strip() and x.strip() != "-1"]
    try:
        indices = [int(x) for x in requested]
        if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= len(visible) for i in indices):
            raise ValueError
    except ValueError:
        raise ValueError("GPU selections must be valid logical indices within CUDA_VISIBLE_DEVICES") from None
    return ",".join(visible[i] for i in indices)


def _with_nccl_preload(env: dict) -> dict:
    """Prepend the known-good libnccl.so.2 to LD_PRELOAD (no-op if disabled/missing)."""
    if NCCL_PRELOAD and Path(NCCL_PRELOAD).exists():
        prev = env.get("LD_PRELOAD", "")
        env["LD_PRELOAD"] = NCCL_PRELOAD + (os.pathsep + prev if prev else "")
    return env

#: suite -> RLDS dataset name / unnorm key (also the teacher-cache subdir name).
SUITE_DATASET = {
    "libero_object":  "libero_object_no_noops",
    "libero_goal":    "libero_goal_no_noops",
    "libero_spatial": "libero_spatial_no_noops",
    "libero_10":      "libero_10_no_noops",
}

#: number of tasks per suite (to split the closed-loop eval across GPUs).
SUITE_NTASKS = {"libero_object": 10, "libero_goal": 10, "libero_spatial": 10, "libero_10": 10}

#: per-suite dense (un-pruned) SR for the report baseline. Only verified values go here;
#: unmeasured suites fall back to default_dense_sr (so the baseline is honest, not assumed).
DENSE_SR_BY_SUITE = {}  # Baseline is supplied explicitly for the actual checkpoint.

#: recovery-recipe id -> phase2 adapter knobs. The verified default is the JOINT
#: recipe (adapter + action_head + proprio); "adapter_only" freezes the sidecars.
RECIPE_TO_ARGS = {
    "direct":               None,                                          # no training
    "adapter+head+proprio": dict(freeze_sidecars=False, adapter_type="mlp"),
    "adapter_only":         dict(freeze_sidecars=True,  adapter_type="mlp"),
}
DEFAULT_RECIPE = "adapter+head+proprio"


class OftPlugin(ModelPlugin):
    name = "oft"
    n_layers = 32
    default_dense_sr = None
    dense_sr_by_suite = DENSE_SR_BY_SUITE       # per-suite baseline (resolve_dense_sr)
    suites = ("libero_object", "libero_goal", "libero_spatial", "libero_10")
    #: prompt can name the model
    aliases = ("openvla-oft", "openvla-ft", "openvla", "oft")
    #: OFT's anchors cluster at the ends -> contiguous (Agent confirms via knockout)
    pruning_mode = "contiguous"
    #: recovery recipes OFT implements (keys of RECIPE_TO_ARGS)
    recovery_strategies = ("direct", "adapter_only", "adapter+head+proprio")
    #: declared decoder shape (LLaMA-2-7B backbone) — the OpenVLA ckpt config.json does
    #: not carry LLM hyperparams, so the param/compute report falls back to this
    #: (flagged "nominal" in the report).
    llm_config = {"hidden_size": 4096, "intermediate_size": 11008, "num_hidden_layers": 32}
    #: unified ladder: OFT's action head reads the LAST LLM layer, so a cut needs a
    #: boundary adapter -- tier 1 = adapter alone, tier 2 = adapter + head + proprio
    #: joint distill (the verified deep-prune recipe). No on-policy stage implemented.
    recovery_ladder = {
        "direct":           "direct",
        "module_finetune":  "adapter_only",
        "joint_distill":    "adapter+head+proprio",
        "onpolicy_distill": None,
    }
    #: weight registry: suite -> ckpt path
    checkpoints = {}  # --full-ckpt is required

    def __init__(self, *, train_max_steps: Optional[int] = None, adapter_bottleneck: int = 1024,
                 kl_weight: float = 1.0, action_weight: float = 1.0,
                 num_steps_before_decay: int = 8_000, calib_samples: int = 256,
                 global_batch: Optional[int] = None, shuffle_buffer_size: int = 100_000):
        # AUTOPRUNE_TRAIN_MAX_STEPS lets a fast smoke shrink recovery training (e.g. 50).
        train_max_steps = int(train_max_steps if train_max_steps is not None else os.environ.get("AUTOPRUNE_TRAIN_MAX_STEPS", 10_000))
        self.train_max_steps = train_max_steps
        self.adapter_bottleneck = adapter_bottleneck
        self.kl_weight = kl_weight
        self.action_weight = action_weight
        self.num_steps_before_decay = min(num_steps_before_decay, max(1, train_max_steps - 10))
        self.calib_samples = calib_samples
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        # DDP global (effective) batch is FIXED to the verified value (32); the per-GPU
        # batch is DERIVED from the GPU count in _train (8 GPUs -> 4, 4 GPUs -> 8, ...),
        # so the data seen per step matches the verified anchor runs regardless of #GPUs.
        self.global_batch = int(global_batch if global_batch is not None else os.environ.get("AUTOPRUNE_GLOBAL_BATCH", 32))
        self._batch_override = os.environ.get("AUTOPRUNE_TRAIN_BATCH")  # optional per-GPU override
        #: (suite, n) -> list of (inputs, proprio) preprocessed ONCE and reused across
        #: every knockout layer (the 32-layer knockout must not re-read HDF5 + re-preprocess).
        self._prepared_cache: dict = {}
        if min(self.train_max_steps, self.global_batch, self.adapter_bottleneck, self.shuffle_buffer_size) < 1 or self.calib_samples < 2:
            raise ValueError("Positive recovery steps/batch/bottleneck and at least two calibration samples are required")

    def runtime_config(self):
        return {"train_max_steps":self.train_max_steps,"global_batch":self.global_batch,
                "per_gpu_batch_override":self._batch_override,"adapter_bottleneck":self.adapter_bottleneck,
                "kl_weight":self.kl_weight,"action_weight":self.action_weight,
                "num_steps_before_decay":self.num_steps_before_decay,"calib_samples":self.calib_samples,
                "shuffle_buffer_size":self.shuffle_buffer_size,
                "repo":str(OFT_REPO),"data_root":DATA_ROOT,"hdf5_root":HDF5_ROOT,"libero_root":LIBERO_ROOT,
                "python":PYTHON_BIN,"torchrun":TORCHRUN,"train_gpus":TRAIN_GPUS,"eval_gpu":EVAL_GPU,
                "nccl_preload":NCCL_PRELOAD,"teacher_cache_format":"autoprune-teacher-v2"}


    def preflight(self, full_ckpt, *, training=True, calibration=True):
        if not os.environ.get("OFT_REPO"):
            raise ValueError("Set OFT_REPO to the checkout produced by autoprune.prepare_oft")
        required = [("OFT_LIBERO_ROOT", LIBERO_ROOT)]
        if calibration:
            required.append(("OFT_HDF5_ROOT", HDF5_ROOT))
        if training:
            required.append(("OFT_DATA_ROOT", DATA_ROOT))
        for name, value in required:
            if not value or not Path(value).is_dir():
                raise FileNotFoundError(f"Set {name} to an existing directory")
        from autoprune.libero_resources import resource_paths
        resource_paths(LIBERO_ROOT,dataset_root=HDF5_ROOT,suite=getattr(self,"active_suite",None))
        if not (Path(full_ckpt)/"config.json").is_file():
            raise FileNotFoundError(f"Missing checkpoint config: {full_ckpt}")
        from autoprune.prepare_oft import verify_checkout
        verify_checkout(OFT_REPO)
        for program in [PYTHON_BIN, TORCHRUN]:
            if not Path(program).is_file() and not shutil.which(program):
                raise FileNotFoundError(f"Missing executable: {program}")
        if NCCL_PRELOAD and not Path(NCCL_PRELOAD).is_file():
            raise FileNotFoundError("Explicit AUTOPRUNE_NCCL_PRELOAD does not exist")

    # ===================================================================== #
    # probe side (in-process, single GPU, cheap)
    # ===================================================================== #
    def _hdf5_dir(self, suite: str) -> str:
        """Calibration HDF5 demos for a suite (OFT_HDF5_ROOT/<suite>)."""
        return str(Path(HDF5_ROOT) / suite)

    def load_policy(self, full_ckpt: str):
        from autoprune.oft_observations import _load_oft_policy
        suite = getattr(self, "active_suite", None)
        if suite not in self.suites:
            raise ValueError("Set the OFT task suite before loading the policy")
        return _load_oft_policy(full_ckpt, device=f"cuda:{EVAL_GPU}", benchmark_name=suite)

    def collect_activations(self, policy, *, suite: str, n_samples: int, seed: int
                            ) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        n = min(n_samples, self.calib_samples)
        if self._has_hdf5(suite):
            import autoprune.oft_observations as tc
            hidden, actions = tc.collect_all_layer_activations(policy, n_samples=n, dataset_dir=self._hdf5_dir(suite))
        else:                                          # no HDF5 demos -> calibrate from RLDS
            hidden, actions = self._collect_activations_rlds(policy, SUITE_DATASET[suite], n)
        return {int(k): np.asarray(v) for k, v in hidden.items()}, np.asarray(actions)

    def _prepared_inputs(self, policy, suite: str, n_samples: int):
        """Preprocess the calibration examples ONCE (HDF5 read + image/tokenize) and cache
        them. Knockout calls predict_action 32× (one per layer) on the SAME inputs, so this
        avoids re-reading HDF5 + re-preprocessing 32 times — only the forwards repeat."""
        key = (suite, n_samples)
        cached = self._prepared_cache.get(key)
        if cached is None:
            cached = []
            for ex in self._calib_examples(suite, n_samples):
                inputs, proprio, _ = policy._prepare_batch_inputs([ex])
                cached.append((inputs, proprio))
            self._prepared_cache[key] = cached
        return cached

    def predict_action(self, policy, *, mask: Optional[Sequence[int]] = None,
                       knockout_layer: Optional[int] = None, n_samples: int = 50, seed: int = 0
                       ) -> np.ndarray:
        """Action chunk over a fixed eval set, optionally with layers structurally
        skipped (mask) or a single layer identity-knocked-out. Uses forward hooks
        (output := input) so no checkpoint re-load is needed for the open-loop signal.
        Reuses preprocessed inputs across calls (see _prepared_inputs)."""
        import torch

        layers = policy.openvla_model.language_model.model.layers
        skip = set(int(x) for x in (mask or []))
        if knockout_layer is not None:
            skip.add(int(knockout_layer))

        def identity_skip(module, inp, out):
            hs_in = inp[0]
            return (hs_in,) + tuple(out[1:]) if isinstance(out, tuple) else hs_in

        if any(l < 0 or l >= len(layers) for l in skip):
            raise ValueError("OFT deletion/knockout index out of range")
        handles = [layers[l].register_forward_hook(identity_skip) for l in skip]
        preds = []
        try:
            with torch.no_grad():
                for inputs, proprio in self._prepared_inputs(policy, self._suite_of(policy), n_samples):
                    result = policy.openvla_model.predict_action(
                        **inputs, unnorm_key=policy.unnorm_key, do_sample=False,
                        proprio=None if proprio is None else proprio[0].detach().float().cpu().numpy(),
                        proprio_projector=policy.proprio_projector if proprio is not None else None,
                        action_head=policy.action_model)
                    preds.append(np.asarray(result[0] if isinstance(result, tuple) else result))
        finally:
            for h in handles:
                h.remove()
        return np.stack(preds, axis=0)

    @staticmethod
    def _suite_of(policy) -> str:
        # unnorm_key encodes the suite (e.g. "libero_object_no_noops"); fall back to object.
        key = getattr(policy, "unnorm_key", "") or ""
        return key.replace("_no_noops", "") if key else "libero_object"

    @staticmethod
    def _iter_examples(hdf5_dir: str, n_samples: int):
        """Yield up to n_samples OFT calibration examples from a suite's HDF5 demos
        (mirrors collect_all_layer_activations' sampling; same fixed traversal order)."""
        import json
        import h5py
        from autoprune.oft_observations import _example_from_hdf5
        files = sorted(Path(hdf5_dir).glob("*.hdf5"))
        if not files:
            raise RuntimeError(f"No calibration HDF5 in {hdf5_dir}")
        per_file = max(1, n_samples // len(files))
        total = 0
        for hf_path in files:
            if total >= n_samples:
                break
            with h5py.File(hf_path, "r") as f:
                demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
                from autoprune.calibration import read_language
                lang = read_language(f["data"].attrs, hf_path.name)
                n = 0
                for demo in demos:
                    if n >= per_file or total >= n_samples:
                        break
                    imgs = f[f"data/{demo}/obs/agentview_rgb"]
                    for idx in np.linspace(0, imgs.shape[0] - 1, min(5, imgs.shape[0]), dtype=int):
                        if n >= per_file or total >= n_samples:
                            break
                        yield _example_from_hdf5(f, demo, int(idx), lang)
                        n += 1
                        total += 1

    # --- RLDS calibration fallback (when a suite has no probe HDF5 demos) ---- #
    def _has_hdf5(self, suite: str) -> bool:
        d = Path(self._hdf5_dir(suite))
        return d.is_dir() and bool(list(d.glob("*.hdf5")))

    def _calib_examples(self, suite: str, n_samples: int):
        """Calibration examples for the probe: HDF5 demos if present, else RLDS (the same
        dataset recovery trains on — so the probe works wherever recovery does, no separate
        HDF5 download). Both yield the SAME ex dict {image:[a,wrist], lang, state(8D)}."""
        if self._has_hdf5(suite):
            yield from self._iter_examples(self._hdf5_dir(suite), n_samples)
        else:
            yield from self._iter_examples_rlds(SUITE_DATASET[suite], n_samples)

    @staticmethod
    def _iter_examples_rlds(dataset_name: str, n_samples: int):
        """Yield up to n_samples probe examples from the OFT RLDS (TFDS) build — same
        {image:[agentview, wrist], lang, state(8D)} schema as _example_from_hdf5. Samples a
        few evenly-spaced steps per episode for diversity (mirrors the HDF5 5-per-demo)."""
        import tensorflow_datasets as tfds
        from autoprune.oft_observations import _prepare_libero_image
        base = Path(DATA_ROOT) / dataset_name
        builder_dir = base                                  # TFDS stores under a version subdir
        if not (base / "dataset_info.json").exists():
            versions = sorted(d for d in base.iterdir() if (d / "dataset_info.json").exists())
            if versions:
                builder_dir = versions[-1]
        ds = tfds.builder_from_directory(str(builder_dir)).as_dataset(split="train")
        total = 0
        for ep in ds:
            if total >= n_samples:
                break
            steps = list(ep["steps"])
            if not steps:
                continue
            pick = np.linspace(0, len(steps) - 1, min(5, len(steps)), dtype=int)
            for i in pick:
                if total >= n_samples:
                    break
                step = steps[int(i)]
                obs = step["observation"]
                state = np.asarray(obs["state"].numpy(), dtype=np.float32).reshape(-1)
                lang_t = step.get("language_instruction")
                lang = lang_t.numpy().decode("utf-8") if lang_t is not None else ""
                if not lang.strip():
                    raise ValueError("RLDS calibration sample has no language instruction")
                yield {"image": [_prepare_libero_image(obs["image"].numpy()),
                                 _prepare_libero_image(obs["wrist_image"].numpy())],
                       "lang": str(lang), "state": state}
                total += 1

    def _collect_activations_rlds(self, policy, dataset_name: str, n_samples: int):
        """Per-layer action-token hidden + actions from RLDS examples — the same capture as
        train_cutoff_object_kl_act.collect_all_layer_activations, but sourced from RLDS (it
        only re-creates the layer hooks; per-example span/forward/pool reuse the verified
        module helpers)."""
        import torch
        import autoprune.oft_observations as tc
        layers = policy.openvla_model.language_model.model.layers
        n_layers = len(layers)
        span = {"a": 0, "b": 0}
        latest: dict = {}

        def _slice(hs):
            return hs[0, span["a"]:span["b"], :].detach().float().cpu()

        def out_hook(k):
            def h(m, i, o):
                latest[k] = _slice(o[0] if isinstance(o, tuple) else o)
            return h

        def in_hook(k):
            def h(m, i):
                latest[k] = _slice(i[0] if isinstance(i, tuple) else i)
            return h

        handles = [layers[0].register_forward_pre_hook(in_hook(0))]
        for l in range(n_layers):
            handles.append(layers[l].register_forward_hook(out_hook(l + 1)))
        per_layer = {l: [] for l in range(n_layers + 1)}
        actions = []
        try:
            with torch.no_grad():
                got = 0
                for ex in self._iter_examples_rlds(dataset_name, n_samples):
                    try:
                        a, b, seq = tc._compute_action_span_from_inputs(policy, ex)
                        if b > seq:
                            raise ValueError(f"span [{a},{b}) exceeds seq {seq}")
                        span["a"], span["b"] = a, b
                        pred = tc._predict_oft_raw_actions(policy, ex)   # fires hooks
                        if len(latest) != n_layers + 1:
                            raise ValueError(f"got {len(latest)} captures != {n_layers + 1}")
                        width = b - a
                        for l in range(n_layers + 1):
                            hh = latest[l]
                            if hh.shape[0] != width:
                                raise ValueError(f"point {l}: {hh.shape[0]} rows != {width}")
                            per_layer[l].append(hh.mean(0).numpy())
                        chunk = np.asarray(pred, dtype=np.float32).reshape(-1)
                        if chunk.size != tc.ACTION_TOKEN_COUNT:
                            raise ValueError("Unexpected action chunk size")
                        actions.append(chunk.reshape(tc.ACTION_CHUNK_SIZE, tc.ACTION_DIM))
                        got += 1
                    except Exception as exc:
                        raise RuntimeError("Invalid RLDS calibration sample") from exc
                    finally:
                        latest.clear()
        finally:
            for h in handles:
                h.remove()
        if not actions:
            raise RuntimeError(f"RLDS calibration ({dataset_name}) collected no samples")
        hidden = {l: np.stack(per_layer[l], axis=0) for l in range(n_layers + 1)}
        return hidden, np.stack(actions, axis=0)

    # ===================================================================== #
    # recovery + eval (coarse: subprocess to the real production scripts)
    # ===================================================================== #
    def recover_and_eval(self, *, full_ckpt: str, mask: Sequence[int], recovery: str,
                         suite: str, scope: EvalScope, work_dir, seed: int) -> dict:
        if suite not in SUITE_DATASET:
            raise ValueError(f"unknown suite {suite!r}; one of {tuple(SUITE_DATASET)}")
        dataset_name = SUITE_DATASET[suite]
        mask_str = ",".join(str(x) for x in sorted(int(x) for x in mask))
        tag = self.mask_tag(mask)    # compact id for dir/run names (full 28-layer mask
                                     # blows the 255-char filename limit; --remove_layers stays full)
        work_dir = Path(work_dir).expanduser().resolve()
        recipe = recovery
        if recipe not in RECIPE_TO_ARGS:
            raise ValueError(f"Unsupported OFT recovery: {recipe}")

        # --- direct: prune + eval only (the SR floor, no training) ---------- #
        if RECIPE_TO_ARGS[recipe] is None:
            eval_dir = work_dir / f"direct_{tag}_{suite}"
            evaluated = self._eval(pretrained=full_ckpt, mask_str=mask_str, tag=tag, suite=suite,
                            dataset_name=dataset_name, adapter_path=None, adapter_type="mlp",
                            out_dir=eval_dir, scope=scope, seed=seed)
            return {**evaluated, "source": "oft:direct", "recipe": "direct", "run_dir": str(Path(full_ckpt).resolve()), "checkpoint": str(Path(full_ckpt).resolve())}

        # --- recover: ensure the teacher cache exists (auto-build if missing) - #
        cache_dir = work_dir / "teacher_cache" / dataset_name
        self._ensure_teacher_cache(full_ckpt, dataset_name, cache_dir)
        args = RECIPE_TO_ARGS[recipe]
        run_root = work_dir / f"recover_{tag}_{suite}_{recipe.replace('+', '_')}_{uuid.uuid4().hex}"
        run_root.mkdir(parents=True, exist_ok=True)

        # 1) train the boundary adapter (+head+proprio unless frozen) ---------- #
        self._train(full_ckpt, mask_str, tag, dataset_name, str(cache_dir), run_root, args, seed=seed)
        run_dir, adapter_path, step = self._latest_adapter(run_root)
        if step != self.train_max_steps:
            raise RuntimeError(f"Recovery checkpoint step {step} does not match requested budget {self.train_max_steps}")

        # 2) make the trained run_dir a loadable checkpoint (base weights symlinked in,
        #    so eval picks up the TRAINED action_head/proprio that live there) -- #
        self._symlink_base(full_ckpt, run_dir)

        # 3) closed-loop eval of the recovered student ----------------------- #
        evaluated = self._eval(pretrained=str(run_dir), mask_str=mask_str, tag=tag, suite=suite,
                        dataset_name=dataset_name, adapter_path=str(adapter_path),
                        adapter_type=args["adapter_type"], out_dir=run_dir / "eval", scope=scope, seed=seed)
        return {**evaluated, "source": f"oft:{recipe}", "recipe": recipe,
                "run_dir": str(run_dir), "adapter_step": step}

    # --- teacher cache (managed prerequisite, not a manual step) ------------ #
    def _ensure_teacher_cache(self, full_ckpt, dataset_name, cache_dir):
        """Create fresh teacher supervision once within this run, from its exact teacher."""
        import torch
        cache_dir = Path(cache_dir)
        identity = (str(Path(full_ckpt).resolve()), dataset_name, str(cache_dir.resolve()))
        ready = getattr(self, "_ready_teacher_caches", set())
        if identity in ready:
            return
        if cache_dir.exists():
            raise FileExistsError(f"Refusing unverified existing teacher cache: {cache_dir}")
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        cmd = [TORCHRUN, "--standalone", "--nnodes", "1", "--nproc_per_node", "1",
               str(CACHE_GEN_SCRIPT), "--vla_path", str(full_ckpt),
               "--dataset_name", dataset_name, "--data_root_dir", DATA_ROOT,
               "--shuffle_buffer_size", str(self.shuffle_buffer_size),
               "--cache_output_dir", str(cache_dir), "--image_aug", "False",
               "--use_proprio", "True", "--num_images_in_input", str(NUM_IMAGES_IN_INPUT)]
        self._run(cmd, "cache", cuda=EVAL_GPU)
        metadata = cache_dir / "metadata.pt"
        if not metadata.is_file() or not list(cache_dir.glob("cache_shard_*.pt")):
            raise RuntimeError("Teacher generation finished without metadata and data shards")
        info = torch.load(metadata, map_location="cpu", weights_only=False)
        if int(info.get("num_patches", -1)) != 513:
            raise RuntimeError("Teacher cache does not describe the two-image/proprio input")
        if (info.get("format") != "autoprune-teacher-v2" or info.get("complete") is not True
                or not isinstance(info.get("expected_samples"), int)
                or info["expected_samples"] < 1 or info.get("num_samples") != info["expected_samples"]):
            raise RuntimeError("Teacher cache is legacy or incomplete; generate a fresh v2 cache")
        (cache_dir / "autoprune_source.json").write_text(json.dumps({
            "teacher_checkpoint": identity[0], "dataset": dataset_name,
            "num_images": 2, "image_aug": False, "use_proprio": True,
            "cache_format": "autoprune-teacher-v2", "identity": "TFDS record and original frame plus tokens",
            "reuse_scope": "this search run only"}, indent=2) + "\n")
        self._ready_teacher_caches = ready | {identity}


    # --- subprocess steps --------------------------------------------------- #
    def _train(self, full_ckpt, mask_str, tag, dataset_name, cache_dir, run_root, args, seed=7) -> None:
        nproc = max(1, len([g for g in TRAIN_GPUS.split(",") if g.strip()]))
        # per-GPU batch derived so global batch == self.global_batch (e.g. 4 GPUs -> 8)
        per_gpu = int(self._batch_override) if self._batch_override else max(1, self.global_batch // nproc)
        if nproc * per_gpu != self.global_batch:
            raise ValueError("GPU count and per-GPU batch must exactly match configured global batch")
        print(f"[oft:train] {nproc} GPUs x batch {per_gpu} = global {nproc * per_gpu} "
              f"(target {self.global_batch})", flush=True)
        cmd = [
            TORCHRUN, "--standalone", "--nnodes", "1",
            "--nproc_per_node", str(nproc),
            str(OFT_REPO / "vla-scripts" / "finetune_phase2_adapter.py"),
            "--vla_path", full_ckpt,
            "--seed", str(seed),
            "--num_images_in_input", str(NUM_IMAGES_IN_INPUT),
            "--data_root_dir", DATA_ROOT,
            "--dataset_name", dataset_name,
            "--shuffle_buffer_size", str(self.shuffle_buffer_size),
            "--teacher_cache_dir", cache_dir,
            "--run_root_dir", str(run_root),
            "--remove_layers", mask_str,
            "--max_steps", str(self.train_max_steps),
            "--num_steps_before_decay", str(self.num_steps_before_decay),
            "--save_freq", str(max(1, self.train_max_steps // 2)),
            "--batch_size", str(per_gpu), "--learning_rate", "1e-3",
            "--adapter_bottleneck", str(self.adapter_bottleneck),
            "--adapter_type", args["adapter_type"], "--image_aug", "True",
            "--kl_loss_weight", str(self.kl_weight),
            "--action_loss_weight", str(self.action_weight),
            "--freeze_sidecars", str(bool(args["freeze_sidecars"])),
            "--run_id_note", f"autoprune_{tag}",
        ]
        self._run(cmd, "train", cuda=TRAIN_GPUS, retries=0)

    def _eval(self, *, pretrained, mask_str, tag, suite, dataset_name, adapter_path,
              adapter_type, out_dir, scope: EvalScope, seed=7, physically_pruned=False):
        """Every requested shard must finish and report its full episode budget."""
        out_dir = Path(out_dir).expanduser().resolve() / f"evaluation_{uuid.uuid4().hex}"
        out_dir.mkdir(parents=True)
        n_tasks = SUITE_NTASKS[suite]
        if scope.max_eval_tasks is not None:
            n_tasks = min(n_tasks, scope.max_eval_tasks)
        if n_tasks < 1 or scope.num_trials < 1:
            raise ValueError("Evaluation task and trial counts must be positive")
        if scope.task_max_steps is not None:
            raise ValueError("Step-capped evaluation is not implemented by this integration")
        gpus = [g.strip() for g in TRAIN_GPUS.split(",") if g.strip()]
        if not gpus or len(set(gpus)) != len(gpus):
            raise ValueError("GPU list must be nonempty and have no duplicate IDs")
        groups = {}
        for task in range(n_tasks):
            groups.setdefault(gpus[task % len(gpus)], []).append(task)
        base = [PYTHON_BIN, str(OFT_REPO / "experiments/robot/libero/run_libero_eval.py"),
                "--pretrained_checkpoint", str(pretrained), "--task_suite_name", suite,
                "--unnorm_key", dataset_name, "--use_l1_regression", "True",
                "--num_images_in_input", "2", "--use_proprio", "True", "--center_crop", "True",
                "--remove_layers", mask_str, "--num_trials_per_task", str(scope.num_trials),
                "--seed", str(seed), "--save_videos", "False"]
        if physically_pruned:
            base += ["--physically_pruned", "True"]
        state_path = os.environ.get("OFT_INITIAL_STATES_PATH", "DEFAULT")
        if state_path != "DEFAULT":
            state_path = str(Path(state_path).expanduser().resolve())
            if not Path(state_path).is_file():
                raise FileNotFoundError(f"Initial state file does not exist: {state_path}")
        base += ["--initial_states_path", state_path]
        if adapter_path:
            base += ["--adapter_path", str(adapter_path), "--adapter_bottleneck", str(self.adapter_bottleneck),
                     "--adapter_type", adapter_type]
        procs = []
        try:
            for gpu, tasks in groups.items():
                sub = out_dir / f"gpu{gpu}"
                sub.mkdir()
                cmd = base + ["--task_ids", ",".join(map(str,tasks)), "--local_log_dir", str(sub),
                              "--run_id_note", f"autoprune_{tag}_g{gpu}"]
                procs.append((self._popen_eval(cmd, cuda=gpu, log=sub/"eval.log"), sub, tasks))
            records, errors = [], []
            for proc, sub, tasks in procs:
                rc = proc.wait()
                if rc != 0:
                    errors.append(f"{sub.name}: exit {rc}")
                    continue
                try:
                    successes, episodes = self._parse_counts(sub/"eval.log", sub)
                    expected = len(tasks) * scope.num_trials
                    if episodes != expected or not 0 <= successes <= episodes:
                        raise RuntimeError(f"observed {successes}/{episodes}, expected {expected} episodes")
                    records.append({"tasks":tasks, "successes":successes, "episodes":episodes,
                                    "log":str(sub/"eval.log")})
                except RuntimeError as exc:
                    errors.append(f"{sub.name}: {exc}")
            if errors:
                raise RuntimeError("Incomplete evaluation: " + "; ".join(errors))
            successes=sum(r["successes"] for r in records)
            episodes=sum(r["episodes"] for r in records)
            result={"status":"completed", "sr":successes/episodes, "successes":successes,
                    "episodes":episodes, "shards":records, "initial_states_path":state_path,
                    "initial_state_indices":list(range(scope.num_trials)), "seed":seed,
                    "held_out_from_search":False, "evaluation_dir":str(out_dir)}
            (out_dir/"result.json").write_text(json.dumps(result,indent=2)+"\n")
            return result
        finally:
            for proc, _, _ in procs:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=30)


    @staticmethod
    def _eval_env(cuda: str) -> dict:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(OFT_REPO), LIBERO_ROOT, env.get("PYTHONPATH", "")])
        env["WANDB_MODE"] = "disabled"
        gl = os.environ.get("AUTOPRUNE_MUJOCO_GL", "osmesa")
        env["MUJOCO_GL"], env["PYOPENGL_PLATFORM"] = gl, gl
        env["CUDA_VISIBLE_DEVICES"] = _cuda_visibility(cuda)
        return _with_nccl_preload(env)

    @staticmethod
    def _libero_config(env, directory):
        """Bind all LIBERO resources to the selected checkout for this invocation."""
        from autoprune.libero_resources import resource_paths
        paths = resource_paths(LIBERO_ROOT,dataset_root=HDF5_ROOT)
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        # JSON is valid YAML and avoids importing LIBERO or changing ~/.libero.
        (directory/"config.yaml").write_text(json.dumps(paths,indent=2)+"\n")
        env["LIBERO_CONFIG_PATH"] = str(directory)
        return env

    def _popen_eval(self, cmd, *, cuda: str, log) -> "subprocess.Popen":
        """Launch one eval subprocess NON-blocking on a given card (output -> log)."""
        print(f"[oft:eval] CUDA={cuda} -> {Path(log).parent.name}", flush=True)
        fh = open(log, "w")
        try:
            p = subprocess.Popen(cmd, env=self._libero_config(self._eval_env(cuda), Path(log).parent/"libero_config"), cwd=str(OFT_REPO),
                                 stdin=subprocess.PIPE, stdout=fh, stderr=subprocess.STDOUT, text=True)
        finally:
            fh.close()
        try:
            p.stdin.write("N\n"); p.stdin.flush(); p.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        return p

    @staticmethod
    def _parse_counts(log, out_dir):
        """Only this invocation's stdout is accepted, never historical sibling text files."""
        content = Path(log).read_text(errors="replace")
        successes = re.findall(r"Total successes:\s*(\d+)", content)
        episodes = re.findall(r"Total episodes:\s*(\d+)", content)
        if not successes or not episodes:
            raise RuntimeError("Final evaluation counters are absent")
        return int(successes[-1]), int(episodes[-1])


    # --- helpers ------------------------------------------------------------ #
    def _latest_adapter(self, run_root: Path) -> Tuple[Path, Path, int]:
        """Find the highest-step adapter checkpoint under run_root/<run_id>/."""
        cands = sorted(run_root.glob("*/adapter--*_checkpoint.pt"),
                       key=lambda p: int(re.search(r"adapter--(\d+)_", p.name).group(1)))
        if not cands:
            raise FileNotFoundError(f"no adapter--*_checkpoint.pt produced under {run_root}")
        adapter = cands[-1]
        step = int(re.search(r"adapter--(\d+)_", adapter.name).group(1))
        return adapter.parent, adapter, step

    @staticmethod
    def _symlink_base(full_ckpt: str, run_dir: Path) -> None:
        """Symlink the base model files (weights, config, tokenizer, modeling code) from
        the full ckpt into the adapter run_dir, so it loads as a complete checkpoint that
        also carries the TRAINED action_head/proprio saved there during recovery."""
        base = Path(full_ckpt).expanduser().resolve()
        names = [p.name for p in base.iterdir()
                 if not p.name.startswith(("adapter--", "action_head--", "proprio_projector--",
                                           "dataset_statistics"))]
        for name in names:
            dst = run_dir / name
            if not dst.exists():
                dst.symlink_to(base / name)

    @staticmethod
    def _parse_sr(log: Path, out_dir: Path) -> Optional[float]:
        """Pull 'Overall success rate: <float>' from the eval output (stdout tee or the
        run_libero_eval .txt log)."""
        texts = []
        if log.exists():
            texts.append(log.read_text(errors="ignore"))
        for txt in out_dir.glob("*.txt"):
            texts.append(txt.read_text(errors="ignore"))
        rate = None
        for t in texts:
            for m in re.finditer(r"Overall success rate:\s*([0-9.]+)", t):
                rate = float(m.group(1))   # take the LAST (final) occurrence
        return rate

    @staticmethod
    def _run(cmd, label, *, cuda: str, stdin: Optional[str] = None, tee: Optional[Path] = None,
             retries: int = 0) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(OFT_REPO), LIBERO_ROOT, env.get("PYTHONPATH", "")])
        # The real recovery/eval scripts call wandb.init; without a key, rank 0 dies and
        # takes down the whole DDP group (NCCL connection reset). Disable it (override-able).
        env["WANDB_MODE"] = "disabled"
        gl = os.environ.get("AUTOPRUNE_MUJOCO_GL", "osmesa")
        env["MUJOCO_GL"], env["PYOPENGL_PLATFORM"] = gl, gl
        env["CUDA_VISIBLE_DEVICES"] = _cuda_visibility(cuda)
        _with_nccl_preload(env)
        output = None
        for flag in ("--run_root_dir", "--cache_output_dir"):
            if flag in cmd:
                output = Path(cmd[cmd.index(flag)+1]).resolve()
                break
        if output is None:
            raise ValueError("OFT subprocess requires an explicit artifact directory")
        OftPlugin._libero_config(env, output.parent / (output.name + "_libero_config"))
        rc = 1
        for attempt in range(retries + 1):
            note = f"  (attempt {attempt + 1}/{retries + 1})" if retries else ""
            print(f"[oft:{label}] CUDA={cuda}{note} {' '.join(str(c) for c in cmd)}")
            if tee:
                with open(tee, "w") as fh:
                    p = subprocess.Popen(cmd, env=env, cwd=str(OFT_REPO), stdin=subprocess.PIPE,
                                         stdout=fh, stderr=subprocess.STDOUT, text=True)
                    p.communicate(input=stdin)
                    rc = p.returncode
            else:
                p = subprocess.Popen(cmd, env=env, cwd=str(OFT_REPO), stdin=subprocess.PIPE, text=True)
                p.communicate(input=stdin)
                rc = p.returncode
            if rc == 0:
                return
            if attempt < retries:
                # transient DDP-startup flakiness (e.g. "torchvision::nms does not exist",
                # NCCL races) — a fresh subprocess usually succeeds.
                print(f"[oft:{label}] exit {rc} -> retrying (transient subprocess failure)", flush=True)
        raise RuntimeError(f"oft {label} subprocess failed (exit {rc}) after {retries + 1} attempt(s)")


# register on import
register_plugin(OftPlugin())
