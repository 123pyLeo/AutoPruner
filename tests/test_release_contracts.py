import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from autoprune.run_e2e import DATA_DIR, run_prompt
from autoprune.backends.plugin import EvalScope
from autoprune.backends import oft_plugin as oft
from agentic_pruning.integration.lib_modules import build_lib_module_set, LibEvalModule
from mask_probe_lib.core.kernels import action_perturbation


def test_paper_metric_and_legacy_are_explicit():
    zero=np.zeros((1,1,2))
    a=np.array([[[3.,0.]]]);b=np.array([[[2.,2.]]])
    assert action_perturbation(zero,a)>action_perturbation(zero,b)
    assert action_perturbation(zero,a,metric="legacy_l1_v1")<action_perturbation(zero,b,metric="legacy_l1_v1")


@pytest.mark.parametrize("other", [np.ones((2,1,2)),np.array([[[np.nan,0.]]])])
def test_metric_refuses_invalid_actions(other):
    with pytest.raises(ValueError):action_perturbation(np.zeros((1,1,2)),other)


def test_run_isolation_and_recovery_escalation(tmp_path):
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA_DIR.glob("*.json")}
    runs=[run_prompt("prune libero_object, SR>=40%",run_root=tmp_path) for _ in range(2)]
    assert runs[0]["run_dir"]!=runs[1]["run_dir"]
    for out in runs:
        rows=out["summary"]["trace"]
        assert len(rows)==6 and all(r["sr"] is not None for r in rows)
        assert rows[0]["mask"]==rows[1]["mask"] and rows[0]["recovery"]!=rows[1]["recovery"]
        assert (Path(out["run_dir"])/"experience").is_dir()
        assert json.loads((Path(out["run_dir"])/"report.json").read_text())["synthetic"]
    assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA_DIR.glob("*.json")}


def test_model_shape_and_prior_do_not_transfer_by_layer_count(tmp_path):
    shutil.copytree(DATA_DIR,tmp_path/"data")
    for depth in [30,32]:
        model=SimpleNamespace(n_layers=depth,suite="libero_object",dense_sr=.9,
                              plugin=SimpleNamespace(name="unseen",recovery_strategies=("direct",)))
        modules=build_lib_module_set(tmp_path/"data",model)
        assert modules.mask_search.total_layers==depth
        assert modules.probe.get_per_layer_table()==[]


def test_execution_metadata_is_retained():
    record={"sr":.9,"episodes":500,"successes":450,"ckpt":"selected.pt","status":"completed"}
    module=LibEvalModule(SimpleNamespace(recover_and_eval=lambda *args:record))
    result=module.lookup_cached([2],"direct")
    for key in record:assert result[key]==record[key]
    assert result["n_trials"]==500


class Process:
    def __init__(self,rc):self.rc=rc
    def wait(self,**kw):return self.rc
    def poll(self):return self.rc


@pytest.mark.parametrize("mode",["crash","partial","missing","invalid_count"])
def test_oft_rejects_incomplete_evaluation(tmp_path,monkeypatch,mode):
    monkeypatch.setattr(oft,"TRAIN_GPUS","0,1")
    def launch(cmd,*,cuda,log):
        s,n=(45,50) if cuda=="0" else (9,10)
        if cuda=="1" and mode=="invalid_count":s,n=51,50
        Path(log).write_text("crashed" if cuda=="1" and mode=="missing" else f"Total successes: {s}\nTotal episodes: {n}\n")
        return Process(1 if cuda=="1" and mode=="crash" else 0)
    monkeypatch.setattr(oft.OftPlugin,"_popen_eval",staticmethod(launch))
    with pytest.raises(RuntimeError,match="Incomplete evaluation"):
        oft.OftPlugin()._eval(pretrained="fake",mask_str="2",tag="one",suite="libero_object",
            dataset_name="libero_object_no_noops",adapter_path=None,adapter_type="mlp",
            out_dir=tmp_path,scope=EvalScope(num_trials=50,max_eval_tasks=2))


def test_oft_full_evaluation_keeps_counts_and_run_identity(tmp_path,monkeypatch):
    monkeypatch.setattr(oft,"TRAIN_GPUS","0,1")
    def launch(cmd,*,cuda,log):
        Path(log).write_text("Total successes: 45\nTotal episodes: 50\n")
        return Process(0)
    monkeypatch.setattr(oft.OftPlugin,"_popen_eval",staticmethod(launch))
    kwargs=dict(pretrained="fake",mask_str="2",tag="one",suite="libero_object",
            dataset_name="libero_object_no_noops",adapter_path=None,adapter_type="mlp",
            out_dir=tmp_path,scope=EvalScope(num_trials=50,max_eval_tasks=2))
    first=oft.OftPlugin()._eval(**kwargs)
    second=oft.OftPlugin()._eval(**kwargs)
    assert (first["successes"],first["episodes"],first["sr"])==(90,100,.9)
    assert first["evaluation_dir"]!=second["evaluation_dir"]


def test_teacher_generation_uses_requested_checkpoint(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(load=lambda *a,**kw:{"num_patches":513,"format":"autoprune-teacher-v2","complete":True,"expected_samples":2,"num_samples":2}))
    calls=[];cache=tmp_path/"cache"
    def execute(cmd,*args,**kw):
        calls.append(cmd);cache.mkdir();(cache/"metadata.pt").touch();(cache/"cache_shard_000.pt").touch()
    monkeypatch.setattr(oft.OftPlugin,"_run",staticmethod(execute))
    plugin=oft.OftPlugin()
    plugin._ensure_teacher_cache("custom-teacher","libero_object_no_noops",cache)
    plugin._ensure_teacher_cache("custom-teacher","libero_object_no_noops",cache)
    assert len(calls)==1
    assert calls[0][calls[0].index("--vla_path")+1]=="custom-teacher"
    with pytest.raises(FileExistsError):
        oft.OftPlugin()._ensure_teacher_cache("other-teacher","libero_object_no_noops",cache)


def test_scripted_rule_agent_cannot_claim_a_real_search(tmp_path):
    with pytest.raises(ValueError,match="synthetic demo"):
        run_prompt("prune",backend_name="oft",agent_name="rule",run_root=tmp_path)


def test_upstream_metadata_updates_cannot_modify_input_checkpoint(tmp_path,monkeypatch):
    from autoprune.run_e2e import make_backend
    source=tmp_path/"original";source.mkdir()
    (source/"config.json").write_text('{"original":true}')
    (source/"model.safetensors").write_bytes(b"read-only weights")
    monkeypatch.setattr(oft.OftPlugin,"preflight",lambda *args:None)
    backend=make_backend("oft",full_ckpt=str(source),suite="libero_object",num_trials=10,
                         work_dir=tmp_path/"run",dense_sr=.982)
    staged=Path(backend.full_ckpt)
    (staged/"config.json").write_text('{"updated":true}')
    assert (source/"config.json").read_text()=='{"original":true}'
    assert (staged/"model.safetensors").is_symlink()


def test_old_probe_table_is_not_loaded(tmp_path):
    from recovery_lib.backends import MockRecoveryBackend
    from agentic_pruning.integration.lib_engine import LibEngine
    backend=MockRecoveryBackend(n_layers=4)
    engine=LibEngine(backend,knockout_batches=4,bundle_batches=8,cache_dir=tmp_path)
    path=engine._cache_path()
    path.write_text(json.dumps({"_ckpt":"","knockout_dall":{"0":123.456}}))
    engine._build_layer_table()
    assert engine._layer_table["knockout_dall"][0]!=123.456
    assert json.loads(path.read_text())["_metric"]=="paper_l2_v1"


def test_physical_export_reindexes_language_layers_and_is_portable(tmp_path):
    torch=pytest.importorskip("torch")
    safetensors=pytest.importorskip("safetensors.torch")
    from autoprune.export_oft import export_checkpoint
    source=tmp_path/"source";source.mkdir()
    tensors={f"language_model.model.layers.{i}.weight":torch.tensor([float(i)]) for i in range(32)}
    tensors["vision.layers.2.weight"]=torch.tensor([123.])
    safetensors.save_file(tensors,str(source/"model-00001-of-00001.safetensors"))
    (source/"model.safetensors.index.json").write_text(json.dumps({"weight_map":{k:"model-00001-of-00001.safetensors" for k in tensors}}))
    (source/"config.json").write_text(json.dumps({"text_config":{"num_hidden_layers":32}}))
    for name in ["configuration_prismatic.py","modeling_prismatic.py","tokenizer_config.json","dataset_statistics.json"]:
        (source/name).write_text("{}")
    torch.save({"boundaries":{},"meta":{"remove_layers":list(range(2,30)),"step":10000,"bottleneck":1024,"adapter_type":"mlp"}},source/"adapter--10000_checkpoint.pt")
    for name in ["action_head","proprio_projector"]:
        torch.save({"weight":torch.ones(1)},source/f"{name}--10000_checkpoint.pt")
    adapter_bytes=(source/"adapter--10000_checkpoint.pt").read_bytes()
    output=tmp_path/"export"
    export_checkpoint(source,output,list(range(2,30)),10000)
    shutil.rmtree(source)
    result=safetensors.load_file(str(output/"model-00001-of-00001.safetensors"))
    assert {k:result[f"language_model.model.layers.{k}.weight"].item() for k in range(4)}=={0:0.,1:1.,2:30.,3:31.}
    assert result["vision.layers.2.weight"].item()==123.
    assert json.loads((output/"config.json").read_text())["text_config"]["num_hidden_layers"]==4
    assert all(not p.is_symlink() for p in output.iterdir())
    assert (output/"adapter--10000_checkpoint.pt").read_bytes()==adapter_bytes
    with pytest.raises(FileExistsError):export_checkpoint(output,output,[2],10000)
