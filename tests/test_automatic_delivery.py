import json
from pathlib import Path

import pytest

from autoprune.run_e2e import run_prompt
from examples.custom_model import ExamplePlugin

OPTIONS={'n_layers':6,'suite':'custom_bench','chunk':3,'action_dim':4,'recovery_steps':2}


def search(tmp_path,**kwargs):
    return run_prompt('prune custom_bench, SR>=0%',backend_name='example',
        plugin_spec='examples.custom_model:ExamplePlugin',backend_options=OPTIONS,
        n_iters=2,recoveries=['direct','adapter_only'],run_root=tmp_path/'runs',
        export_to=tmp_path/'exported',**kwargs)


def test_search_can_deliver_its_exact_qualifying_winner(tmp_path):
    out=search(tmp_path)
    delivery=json.loads((Path(out['run_dir'])/'delivery.json').read_text())
    assert delivery['status']=='exported' and delivery['synthetic']
    assert not delivery['action_equivalence_verified']
    assert not delivery['full_robot_performance_verified']
    receipt=json.loads((Path(delivery['checkpoint'])/'autoprune_selection.json').read_text())
    assert receipt['removed_layers']==out['selection']['winner']['mask']
    assert receipt['recovery']==out['selection']['winner']['recovery']


def test_search_does_not_export_a_nonqualifying_frontier(tmp_path):
    out=search(tmp_path,sr_threshold=1.)
    assert not out['selection']['winner_meets_spec']
    assert out['delivery']['status']=='no_qualifying_checkpoint'
    assert not (tmp_path/'exported').exists()
    assert (Path(out['run_dir'])/'selection.json').is_file()


def test_automatic_delivery_refuses_existing_output_before_search(tmp_path):
    (tmp_path/'exported').mkdir()
    with pytest.raises(FileExistsError):search(tmp_path)
    assert not (tmp_path/'runs').exists()


def test_automatic_delivery_preserves_search_when_export_fails(tmp_path,monkeypatch):
    def fail(*args,**kwargs):raise RuntimeError('native saver failed')
    monkeypatch.setattr(ExamplePlugin,'export_checkpoint',fail)
    with pytest.raises(RuntimeError,match='Search records were saved'):search(tmp_path)
    run=next((tmp_path/'runs').glob('search_*'))
    assert (run/'selection.json').is_file()
    assert json.loads((run/'delivery.json').read_text())['status']=='failed'
    assert not (tmp_path/'exported').exists()


def test_native_oft_verification_is_not_claimed_for_an_unrelated_model(tmp_path):
    with pytest.raises(ValueError,match='built-in OFT'):search(tmp_path,verify_export=True)
    assert not (tmp_path/'runs').exists()


def test_downloaded_oft_export_uses_manifest_without_search_or_dense_model(tmp_path,monkeypatch):
    from autoprune import eval_oft_export as entry
    from autoprune.backends import oft_plugin as oft
    model=tmp_path/'download';model.mkdir();(model/'config.json').write_text('{}')
    manifest={'recovery':'adapter+head+proprio','adapter_config':{'bottleneck':128,'type':'mlp'},
              'adapter_step':2,'removed_layers':[2,3,4]}
    monkeypatch.setattr(entry,'verify_files',lambda path:manifest)
    calls=[]
    monkeypatch.setattr(oft.OftPlugin,'preflight',lambda self,path,training,calibration:calls.append((str(path),training,calibration)))
    def evaluate(self,**kwargs):
        assert kwargs['physically_pruned'] is True and kwargs['mask_str']=='2,3,4'
        assert str(kwargs['adapter_path']).endswith('adapter--2_checkpoint.pt')
        assert self.adapter_bottleneck==128
        assert kwargs['scope'].num_trials==1 and kwargs['scope'].max_eval_tasks==1
        return {'status':'completed','sr':0.,'successes':0,'episodes':1,'held_out_from_search':False}
    monkeypatch.setattr(oft.OftPlugin,'_eval',evaluate)
    output,result=entry.evaluate(model,run_root=tmp_path/'runs',max_eval_tasks=1)
    assert calls==[(str(model),False,False)] and result['episodes']==1
    assert json.loads((output/'request.json').read_text())['dense_reference_required'] is False
    assert not (model/'autoprune_selection.json').exists()
