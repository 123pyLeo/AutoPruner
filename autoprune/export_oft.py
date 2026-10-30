"""Physically shorten an OFT checkpoint, retaining the selected recovery sidecars.

Adapts scripts/prune_layers_8_23.py's tensor filtering and index rebuilding.
The destination is new, self-contained, and contains no symlinks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

LAYER = re.compile(r"(^|\.)language_model\.model\.layers\.(\d+)\.")
METADATA = ("configuration_prismatic.py", "modeling_prismatic.py", "processing_prismatic.py",
            "preprocessor_config.json", "processor_config.json", "tokenizer.json", "tokenizer.model",
            "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "generation_config.json",
            "dataset_statistics.json")


def export_checkpoint(source, destination, removed, step, *, recovery="adapter+head+proprio"):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Export destination must be new: {destination}")
    if recovery not in {"direct", "adapter_only", "adapter+head+proprio"}:
        raise ValueError("Unsupported OFT recovery")
    config = json.loads((source/"config.json").read_text())
    index = json.loads((source/"model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    actual = {int(m.group(2)) for key in weight_map if (m := LAYER.search(key))}
    if actual != set(range(32)):
        raise ValueError("Expected a dense 32-layer OFT base; refusing to double-prune or guess its depth")
    removed = list(removed)
    if (not removed or any(not isinstance(i,int) or isinstance(i,bool) for i in removed)
            or len(set(removed)) != len(removed) or not set(removed) < actual):
        raise ValueError("Deletion indices must be unique, in range, nonempty, and leave a backbone")
    kept = sorted(actual-set(removed))
    remap = {old:new for new,old in enumerate(kept)}
    components = {}
    if recovery != "direct":
        for name in ["adapter", "action_head", "proprio_projector"]:
            p = source/f"{name}--{step}_checkpoint.pt"
            if not p.is_file():
                raise FileNotFoundError(f"Explicit recovery component is missing: {p.name}")
            components[name] = p
    else:
        for name in ["action_head", "proprio_projector"]:
            matches=list(source.glob(f"{name}--*_checkpoint.pt"))
            if len(matches)!=1:
                raise ValueError(f"Direct export requires exactly one unambiguous {name} checkpoint")
            components[name]=matches[0]
    adapter_config = None
    if recovery != "direct":
        saved = torch.load(components["adapter"],map_location="cpu",weights_only=True)
        meta = saved.get("meta",{})
        if sorted(meta.get("remove_layers",[])) != sorted(removed) or meta.get("step") != step:
            raise ValueError("Adapter metadata does not match the selected mask and checkpoint step")
        if not isinstance(meta.get("bottleneck"),int) or meta["bottleneck"]<1:
            raise ValueError("Adapter metadata lacks its bottleneck size")
        adapter_config = {"bottleneck":meta["bottleneck"],"type":meta.get("adapter_type","mlp")}
    for name in ["modeling_prismatic.py", "configuration_prismatic.py", "tokenizer_config.json", "dataset_statistics.json"]:
        if not (source/name).is_file():
            raise FileNotFoundError(f"Missing model resource: {name}")
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".autoprune-export-",dir=destination.parent) as td:
        output=Path(td)/"model";output.mkdir()
        files={}
        for key,name in weight_map.items():
            if Path(name).name!=name:
                raise ValueError("Checkpoint shard filenames must be local basenames")
            files.setdefault(name,[]).append(key)
        new_map={};size=0
        for name,keys in files.items():
            tensors={}
            with safe_open(source/name,framework="pt",device="cpu") as shard:
                for key in keys:
                    match=LAYER.search(key)
                    if match:
                        old=int(match.group(2))
                        if old not in remap:
                            continue
                        begin,end=match.span(2)
                        new_key=key[:begin]+str(remap[old])+key[end:]
                    else:
                        new_key=key
                    tensor=shard.get_tensor(key)
                    tensors[new_key]=tensor
                    size += tensor.numel()*tensor.element_size()
                    new_map[new_key]=name
            if tensors:
                save_file(tensors,str(output/name),metadata={"format":"pt"})
        config.setdefault("text_config",{})["num_hidden_layers"]=len(kept)
        if "num_hidden_layers" in config:
            config["num_hidden_layers"]=len(kept)
        (output/"config.json").write_text(json.dumps(config,indent=2)+"\n")
        (output/"model.safetensors.index.json").write_text(json.dumps(
            {"metadata":{"total_size":size},"weight_map":new_map},indent=2)+"\n")
        for name in METADATA:
            if (source/name).is_file():
                shutil.copyfile(source/name,output/name)
        for p in components.values():
            shutil.copyfile(p,output/p.name)
        manifest={"format":"autoprune-oft-physical-v1","original_layers":32,"removed_layers":sorted(removed),
                  "kept_layers":kept,"recovery":recovery,"adapter_step":step if recovery!="direct" else None,
                  "tensor_bytes":size,"adapter_config":adapter_config,"files":{}}
        for p in output.iterdir():
            digest=hashlib.sha256()
            with p.open('rb') as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''):
                    digest.update(chunk)
            manifest["files"][p.name]=digest.hexdigest()
        (output/"autoprune_export.json").write_text(json.dumps(manifest,indent=2)+"\n")
        shutil.move(str(output),str(destination))
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",required=True,help="Explicit dense-base/recovered run directory")
    p.add_argument("--destination",required=True)
    p.add_argument("--remove-layers",required=True)
    p.add_argument("--step",type=int,default=10000)
    p.add_argument("--recovery",choices=["direct","adapter_only","adapter+head+proprio"],default="adapter+head+proprio")
    a=p.parse_args()
    result=export_checkpoint(a.source,a.destination,[int(v) for v in a.remove_layers.split(',')],a.step,recovery=a.recovery)
    print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))


if __name__=="__main__":main()
