"""Load OFT, predict on a few calibration inputs, optionally run bounded rollouts.

This command never trains and never calls an LLM. It is an environment smoke check,
not a paper success-rate evaluation.
"""
from __future__ import annotations

import argparse
import gc
import inspect
import json
from pathlib import Path
import uuid

import numpy as np

from autoprune.run_e2e import stage_checkpoint


def run(checkpoint, *, suite="libero_object", samples=2, knockout_layer=None,
        evaluate=False, trials=1, run_root="runs"):
    from autoprune.backends.oft_plugin import OftPlugin, SUITE_DATASET
    from autoprune.backends.plugin import EvalScope
    from mask_probe_lib.core.kernels import action_perturbation
    if suite not in SUITE_DATASET or not 1 <= samples <= 24 or not 1 <= trials <= 50:
        raise ValueError("Use a LIBERO suite, 1–24 calibration samples and 1–50 trials")
    if knockout_layer is not None and not 0 <= knockout_layer < 32:
        raise ValueError("Knockout layer must be in [0,32)")
    plugin = OftPlugin()
    plugin.active_suite = suite
    plugin.preflight(checkpoint, training=False)
    output = Path(run_root).expanduser().resolve() / f"oft_smoke_{uuid.uuid4().hex}"
    output.mkdir(parents=True)
    staged = stage_checkpoint(checkpoint, output)
    result = {"kind":"environment_smoke", "training":False, "llm_calls":0,
              "suite":suite, "requested_calibration_samples":samples,
              "source_checkpoint":str(Path(checkpoint).resolve())}
    (output/"request.json").write_text(json.dumps(result,indent=2)+"\n")
    policy = plugin.load_policy(str(staged))
    dense = plugin.predict_action(policy, n_samples=samples, seed=0)
    if dense.ndim != 3 or dense.shape[0] < 1 or dense.shape[-2:] != (8,7) or not np.isfinite(dense).all():
        raise RuntimeError(f"Invalid OFT action output shape or values: {dense.shape}")
    result["action_shape"] = list(dense.shape)
    result["model_class"] = type(policy.openvla_model).__qualname__
    model_source = inspect.getfile(type(policy.openvla_model))
    import hashlib
    result["model_code_file"] = model_source
    result["model_code_sha256"] = hashlib.sha256(Path(model_source).read_bytes()).hexdigest()
    np.savez_compressed(output/"actions.npz",dense=dense)
    if knockout_layer is not None:
        intervened = plugin.predict_action(policy, knockout_layer=knockout_layer, n_samples=samples, seed=0)
        result["knockout_layer"] = knockout_layer
        result["mean_action_chunk_l2"] = action_perturbation(dense,intervened)
        np.savez_compressed(output/"actions.npz",dense=dense,knockout=intervened)
    del policy
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    if evaluate:
        result["evaluation"] = plugin._eval(pretrained=str(staged),mask_str="",tag="dense_smoke",
            suite=suite,dataset_name=SUITE_DATASET[suite],adapter_path=None,adapter_type="mlp",
            out_dir=output/"rollout",scope=EvalScope(num_trials=trials,max_eval_tasks=1),seed=7)
    result["status"] = "completed"
    (output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    return output, result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--suite",default="libero_object")
    p.add_argument("--samples",type=int,default=2)
    p.add_argument("--knockout-layer",type=int)
    p.add_argument("--evaluate",action="store_true")
    p.add_argument("--trials",type=int,default=1)
    p.add_argument("--run-root",default="runs")
    a=p.parse_args()
    try:
        output,result=run(a.checkpoint,suite=a.suite,samples=a.samples,knockout_layer=a.knockout_layer,
                          evaluate=a.evaluate,trials=a.trials,run_root=a.run_root)
    except (ValueError,RuntimeError,FileNotFoundError) as exc:
        p.exit(2,f"OFT smoke: {exc}\n")
    print(json.dumps({"run_dir":str(output),**result},indent=2))


if __name__=="__main__":main()
