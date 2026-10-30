"""OFT probe helpers extracted from the validated experimental implementation.

Private wrapper imports are namespaced; release changes enforce explicit suite,
calibration directory and valid task language. See provenance/SELF_REVIEW.md.
"""
from __future__ import annotations
import json, logging, math, os, time
from pathlib import Path
import h5py
import numpy as np
import torch
import cv2 as cv
from omegaconf import OmegaConf
from autoprune._vendor.oft_probe.starVLA.model.framework.VLM4A.OpenVLAOFT import OpenVLAOFT
logger = logging.getLogger(__name__)
POLICY_IMAGE_SIZE = 224
PRE_RESIZE_IMAGES = True
ACTION_DIM = 7
ACTION_CHUNK_SIZE = 8
ACTION_TOKEN_COUNT = 56

def _read_norm_stats(ckpt_path: str) -> dict:
    stats_path = os.path.join(ckpt_path, "dataset_statistics.json")
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            return json.load(f)
    return {}

def _resolve_unnorm_key(norm_stats: dict, benchmark_name: str) -> str:
    for key in (f"{benchmark_name}_no_noops", benchmark_name):
        if key in norm_stats:
            return key
    raise KeyError(f"No normalization statistics for requested suite {benchmark_name!r}; available: {sorted(norm_stats)}")


def _make_oft_config(ckpt_path: str, benchmark_name: str) -> OmegaConf:
    """Minimal config matching the native StarVLA OpenVLA-OFT wrapper."""
    unnorm_key = _resolve_unnorm_key(_read_norm_stats(ckpt_path), benchmark_name)
    return OmegaConf.create(
        {
            "framework": {
                "name": "OpenVLAOFT",
                "openvla": {
                    "pretrained_checkpoint": ckpt_path,
                    "trust_remote_code": True,
                },
                "action_model": {
                    "action_model_type": "OpenVLAL1Regression",
                    "action_dim": 7,
                    "future_action_window_size": 7,
                    "past_action_window_size": 0,
                },
                "inference": {
                    "num_images_in_input": 2,
                    "use_proprio": True,
                    "center_crop": True,
                    "image_size": 224,
                    "unnorm_key": unnorm_key,
                },
            },
            "trainer": {"pretrained_checkpoint": None},
        }
    )

def _load_oft_policy(ckpt_path: str, device: str | torch.device, *, benchmark_name: str) -> OpenVLAOFT:
    """Load the native OpenVLA-OFT stack: VLM + proprio projector + action head."""
    policy = OpenVLAOFT(config=_make_oft_config(ckpt_path, benchmark_name))
    policy.openvla_model = policy.openvla_model.to(device)
    policy.action_model = policy.action_model.to(device=device, dtype=torch.bfloat16)
    policy.action_head = policy.action_model
    if policy.proprio_projector is not None:
        policy.proprio_projector = policy.proprio_projector.to(device=device, dtype=torch.bfloat16)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad = False
    return policy

def _libero_rot180(img_array):
    img = np.asarray(img_array)
    return np.ascontiguousarray(img[::-1, ::-1])

def _prepare_libero_image(img_array: np.ndarray) -> np.ndarray:
    """Match OpenVLA-OFT LIBERO eval: rotate env image, then resize to policy input size."""
    img = _libero_rot180(img_array)
    if PRE_RESIZE_IMAGES and img.shape[:2] != (POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE):
        img = cv.resize(img, (POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE), interpolation=cv.INTER_AREA)
    return np.ascontiguousarray(img)

def _example_from_hdf5(f: h5py.File, demo_name: str, idx: int, lang: str) -> dict:
    base = f[f"data/{demo_name}"]
    img = _prepare_libero_image(base["obs/agentview_rgb"][idx])
    wrist_img = _prepare_libero_image(base["obs/eye_in_hand_rgb"][idx])
    state = np.concatenate(
        (
            np.asarray(base["obs/ee_pos"][idx], dtype=np.float32).reshape(-1),
            np.asarray(base["obs/ee_ori"][idx], dtype=np.float32).reshape(-1),
            np.asarray(base["obs/gripper_states"][idx], dtype=np.float32).reshape(-1),
        ),
        axis=0,
    )
    if state.shape != (8,):
        raise ValueError(f"Expected 8D HDF5 proprio state, got {state.shape}")
    return {"image": [img, wrist_img], "lang": str(lang), "state": state}

def _predict_oft_raw_actions(policy: OpenVLAOFT, example: dict) -> np.ndarray:
    """Run native OpenVLA-OFT preprocessing and return unnormalized action chunk."""
    inputs, proprio, _ = policy._prepare_batch_inputs([example])
    result = policy.openvla_model.predict_action(
        **inputs,
        unnorm_key=policy.unnorm_key,
        do_sample=False,
        proprio=None if proprio is None else proprio[0].detach().float().cpu().numpy(),
        proprio_projector=policy.proprio_projector if proprio is not None else None,
        action_head=policy.action_model,
    )
    return result[0] if isinstance(result, tuple) else result

def _compute_action_span_from_inputs(policy: OpenVLAOFT, example: dict) -> tuple[int, int, int]:
    """Return action-token span in the multimodal LM sequence for one OFT example."""
    inputs, proprio, _ = policy._prepare_batch_inputs([example])
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
        )
    num_prompt_tokens = input_ids.shape[-1] - 1
    input_ids, attention_mask = policy.openvla_model._prepare_input_for_action_prediction(input_ids, attention_mask)
    if attention_mask.shape[0] != 1:
        raise ValueError("Calibration currently expects batch size 1")

    num_patches = (
        policy.openvla_model.vision_backbone.get_num_patches()
        * policy.openvla_model.vision_backbone.get_num_images_in_input()
    )
    if proprio is not None:
        num_patches += 1
    action_start = int(num_patches + num_prompt_tokens)
    action_end = int(action_start + ACTION_TOKEN_COUNT)
    seq_len = int(num_patches + input_ids.shape[-1])
    if seq_len != int(attention_mask.shape[-1] + num_patches):
        raise ValueError(f"Unexpected multimodal sequence length: seq_len={seq_len}, mask={attention_mask.shape}")
    return action_start, action_end, seq_len

def collect_all_layer_activations(policy, n_samples: int = 256, *, dataset_dir):
    """Per-layer action-token hidden states from the DENSE teacher, for the
    mask_probe_lib probes (knockout/manifold/planning/linear). Generalises
    collect_calibration_from_hdf5 to hook EVERY decoder layer instead of just
    src/tgt/norm.

    Returns ``(hidden, actions)`` matching mask_probe_lib's ActivationBundle contract:

      hidden : ``{l: ndarray (N, D)}`` for ``l`` in ``0..n_layers``. ``hidden[l]`` is the
               residual stream ENTERING decoder layer ``l`` (so ``hidden[0]`` = embedding
               output, ``hidden[n_layers]`` = final layer output). Each row is the MEAN
               over that sample's action-token positions ONLY (per W3, never the full
               sequence) — one (D,) vector per sample per layer.
      actions: ``ndarray (N, chunk, action_dim)`` — the policy's predicted action chunk
               per sample (the probe regression target; using the model's own output
               avoids new HDF5 action-key assumptions and keeps hidden→action aligned).

    Memory note: hooking all 33 residual points with full-sequence buffers would be tens
    of GB, so the hooks slice to the action-token span ON-GPU and keep only (N, 33, D).
    Run on the openvla-oft GPU env (validated there: 33 residual points, finite, probe
    values rise with depth). The capture path mirrors the verified
    collect_calibration_from_hdf5, which uses the same single-forward _predict helper.
    """
    if n_samples < 1:
        raise ValueError("Calibration sample count must be positive")
    hdf5_files = sorted(Path(dataset_dir).glob("*.hdf5"))
    if not hdf5_files:
        raise RuntimeError(f"No HDF5 files in {dataset_dir}")
    policy.eval()
    lm = policy.openvla_model.language_model
    layers = lm.model.layers
    n_layers = len(layers)
    # action-token span for the CURRENT sample (set before each forward); the hooks
    # slice to it ON-GPU so only the ~56 action-token rows per layer reach CPU, not the
    # full [1, S, 4096] sequence (S can be hundreds-1000+ → that copy is ~90% waste).
    span = {"a": 0, "b": 0}
    latest: dict[int, torch.Tensor] = {}     # residual point -> (n_action_tokens, D) on CPU

    def _slice(hs) -> torch.Tensor:
        return hs[0, span["a"]:span["b"], :].detach().float().cpu()

    def out_hook(key):
        def hook(module, inp, out):
            latest[key] = _slice(out[0] if isinstance(out, tuple) else out)
        return hook

    def in_hook(key):
        def hook(module, inp):
            latest[key] = _slice(inp[0] if isinstance(inp, tuple) else inp)
        return hook

    # hidden[0] = input to layer 0 (embedding output); hidden[l+1] = output of layer l
    handles = [layers[0].register_forward_pre_hook(in_hook(0))]
    for l in range(n_layers):
        handles.append(layers[l].register_forward_hook(out_hook(l + 1)))

    per_layer: dict[int, list] = {l: [] for l in range(n_layers + 1)}
    actions: list = []

    hdf5_files = sorted(Path(dataset_dir).glob("*.hdf5"))
    if not hdf5_files:
        raise RuntimeError(f"No HDF5 files in {dataset_dir}")
    samples_per_file = max(1, n_samples // len(hdf5_files))
    logger.info(f"collect_all_layer_activations: {n_layers + 1} residual points, "
                f"{len(hdf5_files)} HDF5 files, ~{samples_per_file} calls/file, target N={n_samples}")

    try:
        with torch.no_grad():
            total = 0
            t0 = time.time()
            for hf_path in hdf5_files:
                if total >= n_samples:
                    break
                with h5py.File(hf_path, "r") as f:
                    demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
                    from autoprune.calibration import read_language
                    lang = read_language(f["data"].attrs, hf_path.name)
                    n = 0
                    for demo_name in demos:
                        if n >= samples_per_file or total >= n_samples:
                            break
                        images = f[f"data/{demo_name}/obs/agentview_rgb"]
                        n_img = images.shape[0]
                        idxs = np.linspace(0, n_img - 1, min(5, n_img), dtype=int)
                        for idx in idxs:
                            if n >= samples_per_file or total >= n_samples:
                                break
                            try:
                                example = _example_from_hdf5(f, demo_name, int(idx), lang)
                                a_start, a_end, seq_len = _compute_action_span_from_inputs(policy, example)
                                if a_end > seq_len:
                                    raise ValueError(f"action span [{a_start},{a_end}) exceeds seq_len {seq_len}")
                                span["a"], span["b"] = a_start, a_end
                                pred = _predict_oft_raw_actions(policy, example)  # fires hooks; they slice to span
                                if len(latest) != n_layers + 1:
                                    raise ValueError(f"expected {n_layers + 1} layer captures, got {len(latest)}")
                                width = a_end - a_start
                                for l in range(n_layers + 1):      # validate + pool in one pass
                                    h = latest[l]
                                    if h.shape[0] != width:
                                        raise ValueError(f"residual point {l}: {h.shape[0]} action-token rows != {width}")
                                    per_layer[l].append(h.mean(0).numpy())            # (D,) action-token mean
                                chunk = np.asarray(pred, dtype=np.float32).reshape(-1)
                                if chunk.size != ACTION_TOKEN_COUNT:
                                    raise ValueError("Unexpected action chunk size")
                                actions.append(chunk.reshape(ACTION_CHUNK_SIZE, ACTION_DIM))
                                n += 1; total += 1
                            except Exception as e:
                                raise RuntimeError(f"Invalid HDF5 calibration sample in {hf_path.name}") from e
                            finally:
                                latest.clear()
                logger.info(f"  {hf_path.name}: {n} calls | total={total} | {time.time() - t0:.0f}s")
    finally:
        for h in handles:
            h.remove()

    if not actions:
        raise RuntimeError("collect_all_layer_activations collected no samples")
    hidden = {l: np.stack(per_layer[l], axis=0) for l in range(n_layers + 1)}   # each (N, D)
    actions_arr = np.stack(actions, axis=0)                                     # (N, chunk, action_dim)
    logger.info(f"collect_all_layer_activations: N={actions_arr.shape[0]}, "
                f"residual points={n_layers + 1}, D={hidden[0].shape[1]}")
    return hidden, actions_arr
