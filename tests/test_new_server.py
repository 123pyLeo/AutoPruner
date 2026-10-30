import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autoprune.backends import oft_plugin as oft


def test_gpu_selection_preserves_inherited_visibility(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","2,3")
    assert oft._cuda_visibility("0")=="2"
    assert oft._cuda_visibility("1")=="3"
    assert oft._cuda_visibility("0,1")=="2,3"
    with pytest.raises(ValueError):oft._cuda_visibility("2")


def test_doctor_rejects_plain_transformers_even_if_version_matches(tmp_path,monkeypatch):
    from autoprune import doctor
    monkeypatch.setattr(doctor.metadata,"version",lambda name:doctor.EXPECTED[name])
    monkeypatch.setattr(doctor.metadata,"distribution",lambda name:SimpleNamespace(read_text=lambda _:None))
    report=doctor.inspect_environment(skip_gpu=True)
    assert report["status"]=="incomplete"
    assert any("pinned OFT fork" in issue for issue in report["issues"])
    assert report["weights_loaded"] is False and report["llm_calls"]==0


class FakeModel:
    pass


def test_real_smoke_entrypoint_uses_no_training_or_llm(tmp_path,monkeypatch):
    from autoprune import smoke_oft
    import sys
    checkpoint=tmp_path/"checkpoint";checkpoint.mkdir()
    (checkpoint/"config.json").write_text("{}")
    class Plugin:
        def preflight(self,path,*,training):assert training is False
        def load_policy(self,path):return SimpleNamespace(openvla_model=FakeModel())
        def predict_action(self,policy,**kw):return np.zeros((2,8,7))
        def _eval(self,**kw):
            assert kw["scope"].max_eval_tasks==1
            return {"status":"completed","successes":1,"episodes":1,"sr":1.0}
    monkeypatch.setattr(oft,"OftPlugin",Plugin)
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda:None)))
    output,result=smoke_oft.run(checkpoint,knockout_layer=22,evaluate=True,run_root=tmp_path/"runs")
    assert result["training"] is False and result["llm_calls"]==0
    assert result["evaluation"]["episodes"]==1
    assert result["action_shape"]==[2,8,7]
    assert json.loads((output/"result.json").read_text())["status"]=="completed"
