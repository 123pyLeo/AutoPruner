"""Report OFT environment readiness without loading policy weights or calling an LLM."""
from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys

EXPECTED = {"torch":"2.2.0", "torchvision":"0.17.0", "transformers":"4.40.1",
            "timm":"0.9.10", "numpy":"1.26.4", "omegaconf":"2.3.0"}
FORK_SHA = "bc339d9ad707454c0c115970db43c260067c61ab"


def inspect_environment(checkpoint=None, suite="libero_object", require_training_data=False, skip_gpu=False):
    issues=[]
    report={"python":sys.version.split()[0],"packages":{},"issues":issues,"weights_loaded":False,"llm_calls":0}
    if sys.version_info[:2] != (3,10):issues.append("The OFT reference environment requires Python 3.10")
    for name,expected in EXPECTED.items():
        try:
            actual=metadata.version(name)
            report["packages"][name]=actual
            if actual.split('+')[0]!=expected:issues.append(f"{name}: expected {expected}, observed {actual}")
        except metadata.PackageNotFoundError:issues.append(f"Missing package: {name}")
    try:
        source=json.loads(metadata.distribution("transformers").read_text("direct_url.json") or "{}")
        actual=source.get("vcs_info",{}).get("commit_id")
        report["transformers_git_commit"]=actual
        if actual!=FORK_SHA:issues.append("Transformers must be the pinned OFT fork, not just a matching version number")
    except (metadata.PackageNotFoundError,json.JSONDecodeError):
        issues.append("Cannot identify the installed Transformers fork")
    repo=os.environ.get("OFT_REPO","")
    try:
        if not repo:raise ValueError("Set OFT_REPO")
        from autoprune.prepare_oft import verify_checkout
        report["oft_source"]=verify_checkout(repo)
    except (ValueError,RuntimeError) as exc:issues.append(str(exc))
    hdf5=Path(os.environ.get("OFT_HDF5_ROOT","."))/suite
    report["calibration_hdf5_files"]=len(list(hdf5.glob("*.hdf5")))
    if not report["calibration_hdf5_files"]:issues.append(f"No calibration HDF5 files for {suite} under OFT_HDF5_ROOT")
    libero=os.environ.get("OFT_LIBERO_ROOT","")
    try:
        if not libero:raise FileNotFoundError("Set OFT_LIBERO_ROOT to a complete LIBERO checkout")
        from autoprune.libero_resources import resource_paths
        report["libero_resources"]=resource_paths(libero,suite=suite)
    except FileNotFoundError as exc:issues.append(str(exc))
    if require_training_data:
        data=os.environ.get("OFT_DATA_ROOT","")
        if not data or not (Path(data)/(suite+"_no_noops")).is_dir():
            issues.append(f"Missing RLDS training dataset for {suite}")
    if checkpoint:
        p=Path(checkpoint)
        required=["config.json","dataset_statistics.json","tokenizer_config.json","modeling_prismatic.py"]
        for name in required:
            if not (p/name).is_file():issues.append(f"Checkpoint resource missing: {name}")
        for name in ["action_head","proprio_projector"]:
            if len(list(p.glob(f"{name}--*_checkpoint.pt"))) != 1:issues.append(f"Checkpoint requires exactly one unambiguous component: {name}")
        try:
            stats = json.loads((p/"dataset_statistics.json").read_text())
            if not any(key in stats for key in (suite, suite+"_no_noops")):
                issues.append(f"Checkpoint has no normalization statistics for requested suite {suite}")
        except (OSError,ValueError,TypeError):
            issues.append("Checkpoint normalization statistics are missing or malformed")
        index=p/"model.safetensors.index.json"
        if index.is_file():
            try:
                shards=set(json.loads(index.read_text())["weight_map"].values())
                if any(not (p/name).is_file() for name in shards):issues.append("Checkpoint weight shards are incomplete")
                report["weight_shards"]=len(shards)
            except (ValueError,KeyError):issues.append("Invalid checkpoint weight index")
        elif not (p/"model.safetensors").is_file():issues.append("Checkpoint safetensors weights are missing")
    if not skip_gpu:
        try:
            result=subprocess.run(["nvidia-smi","--query-gpu=index,name,memory.total,memory.free","--format=csv,noheader"],
                                   capture_output=True,text=True,timeout=10)
            report["gpus"]=result.stdout.strip().splitlines()
            if result.returncode or not result.stdout.strip():issues.append("nvidia-smi cannot see a working GPU")
        except (OSError,subprocess.TimeoutExpired):issues.append("nvidia-smi is unavailable")
    report["status"]="ready_for_smoke" if not issues else "incomplete"
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint")
    p.add_argument("--suite",default="libero_object")
    p.add_argument("--require-training-data",action="store_true")
    p.add_argument("--skip-gpu",action="store_true")
    p.add_argument("--output",default="runs/environment.json")
    a=p.parse_args()
    report=inspect_environment(a.checkpoint,a.suite,a.require_training_data,a.skip_gpu)
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return 0 if report["status"]=="ready_for_smoke" else 1


if __name__=="__main__":raise SystemExit(main())
