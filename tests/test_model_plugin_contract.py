"""Integration contracts across non-OFT geometry, configuration and native export."""
import json
from pathlib import Path

import numpy as np
import pytest

from autoprune.backends.loader import load_plugin
from autoprune.backends.plugin import ModelPlugin
from autoprune.check_export import verify_files
from autoprune.finalize import finalize
from autoprune.run_e2e import make_backend,run_prompt
from examples.custom_model import ExamplePlugin

SPEC='examples.custom_model:ExamplePlugin'
OPTIONS={'n_layers':6,'suite':'custom_bench','chunk':3,'action_dim':4,'recovery_steps':2}


def run_example(root):
    return run_prompt('prune custom_bench, SR>=0%',backend_name='example',plugin_spec=SPEC,
        backend_options=OPTIONS,task_suite='custom_bench',n_iters=2,run_root=root,
        recoveries=['direct','adapter_only'])


def test_non_oft_configuration_search_feedback_and_export(tmp_path):
    out=run_example(tmp_path/'runs')
    run=Path(out['run_dir']);request=json.loads((run/'request.json').read_text())
    assert request['plugin_options']==OPTIONS
    assert request['backend_config']==OPTIONS
    assert request['spec']['suite']=='custom_bench'
    assert out['summary']['total_layers']==6
    rows=out['summary']['trace']
    assert [r['recovery'] for r in rows]==['direct','adapter_only']
    assert rows[0]['mask']==rows[1]['mask']
    assert all(r['sr'] is not None and 0<min(r['mask'])+1<=max(r['mask'])+1<=6 for r in rows)
    # The second candidate's context contains the first candidate's real simulator feedback.
    context=json.loads((run/'iter_001/context.json').read_text())
    assert context['last_result']['sr']==rows[0]['sr']
    assert out['selection']['winner_meets_spec']
    manifest,receipt=finalize(run,tmp_path/'exported')
    assert manifest['format']=='autoprune-plugin-v1' and manifest['synthetic']
    assert manifest['native_format']=='autoprune-synthetic-plan-v1'
    assert receipt['synthetic'] and not receipt['action_equivalence_verified']
    saved=json.loads((tmp_path/'exported/simulation.json').read_text())
    assert saved['configuration']==OPTIONS and saved['mask']==out['selection']['winner']['mask']
    assert saved['recovery']==out['selection']['winner']['recovery']
    assert verify_files(tmp_path/'exported')==manifest
    (tmp_path/'exported/simulation.json').write_text('tampered')
    with pytest.raises(ValueError,match='checksum'):verify_files(tmp_path/'exported')


def test_non_oft_live_probe_shapes_and_layer_indices(tmp_path):
    from agentic_pruning.integration.lib_engine import LibEngine
    backend=make_backend('example',plugin_spec=SPEC,full_ckpt=None,suite='custom_bench',
        backend_options=OPTIONS,num_trials=1,work_dir=tmp_path)
    bundle=backend.collect_activations(n_batches=8,seed=0)
    assert bundle.actions.shape==(8,3,4) and set(bundle.hidden)==set(range(7))
    table=LibEngine(backend).query('knockout_dall',range(6))
    assert set(table)==set(range(6)) and all(np.isfinite(v) for v in table.values())


def test_preflight_receives_the_selected_suite(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(ExamplePlugin,'preflight',lambda self,ckpt:calls.append(self.active_suite),raising=False)
    make_backend('example',plugin_spec=SPEC,full_ckpt=None,suite='custom_bench',
        backend_options=OPTIONS,num_trials=1,work_dir=tmp_path)
    assert calls==['custom_bench']


@pytest.mark.parametrize('failure',['missing_boundary','nonfinite','misaligned'])
def test_plugin_capture_contract_refuses_invalid_arrays(tmp_path,monkeypatch,failure):
    backend=make_backend('example',plugin_spec=SPEC,full_ckpt=None,suite='custom_bench',
        backend_options=OPTIONS,num_trials=1,work_dir=tmp_path)
    capture=backend.plugin.collect_activations
    def broken(*args,**kwargs):
        hidden,actions=capture(*args,**kwargs)
        if failure=='missing_boundary':hidden.pop(6)
        elif failure=='nonfinite':actions[0,0,0]=np.nan
        else:hidden[2]=hidden[2][:-1]
        return hidden,actions
    monkeypatch.setattr(backend.plugin,'collect_activations',broken)
    with pytest.raises(ValueError,match='Plugin'):backend.collect_activations(n_batches=8)


def test_native_export_is_required_and_not_guessed(tmp_path,monkeypatch):
    out=run_example(tmp_path/'runs')
    monkeypatch.setattr(ExamplePlugin,'export_checkpoint',ModelPlugin.export_checkpoint)
    with pytest.raises(ValueError,match='no native'):finalize(out['run_dir'],tmp_path/'exported')
    assert not (tmp_path/'exported').exists()


@pytest.mark.parametrize('failure',['empty','symlink'])
def test_incomplete_native_export_does_not_publish_destination(tmp_path,monkeypatch,failure):
    out=run_example(tmp_path/'runs')
    def broken(self,*,destination,**kwargs):
        if failure=='symlink':(destination/'weights').symlink_to(Path(out['run_dir'])/'request.json')
        return 'test-format'
    monkeypatch.setattr(ExamplePlugin,'export_checkpoint',broken)
    with pytest.raises(ValueError):finalize(out['run_dir'],tmp_path/'exported')
    assert not (tmp_path/'exported').exists()


def test_plugin_different_executed_recipe_is_a_failure(tmp_path,monkeypatch):
    backend=make_backend('example',plugin_spec=SPEC,full_ckpt=None,suite='custom_bench',
        backend_options=OPTIONS,num_trials=1,work_dir=tmp_path)
    monkeypatch.setattr(backend.plugin,'recover_and_eval',lambda **kw:{'sr':.9,'recipe':'different'})
    result=backend.train_recovery([2],'direct')
    assert result['status']=='failed' and result['sr'] is None


def test_plugin_configuration_errors_are_explicit():
    with pytest.raises(ValueError,match='constructor configuration'):load_plugin(SPEC,{'unknown':3})


def test_custom_plugin_default_suite_does_not_become_libero_object(tmp_path):
    out=run_prompt('prune, SR>=0%',backend_name='example',plugin_spec=SPEC,
        backend_options=OPTIONS,n_iters=1,run_root=tmp_path)
    assert out['spec'].suite=='custom_bench'
    assert out['summary']['trace'][0]['sr'] is not None


def test_prompt_resolves_a_plugin_declared_suite(tmp_path,monkeypatch):
    original=ExamplePlugin.__init__
    def constructor(self,**kwargs):
        original(self,**kwargs);self.suites=('bench_alpha','bench_beta')
    monkeypatch.setattr(ExamplePlugin,'__init__',constructor)
    backend=make_backend('example',plugin_spec=SPEC,full_ckpt=None,suite=None,
        prompt='Prune BENCH_BETA, SR>=0%',backend_options=OPTIONS,num_trials=1,work_dir=tmp_path)
    assert backend.suite==backend.plugin.active_suite=='bench_beta'
    with pytest.raises(ValueError,match='one suite'):
        make_backend('example',plugin_spec=SPEC,full_ckpt=None,suite=None,
            prompt='Prune bench_alpha and bench_beta',backend_options=OPTIONS,num_trials=1,work_dir=tmp_path)


def test_plugin_suite_name_takes_priority_over_libero_keyword_heuristic(tmp_path):
    options={**OPTIONS,'suite':'robot_object'}
    out=run_prompt('prune robot_object, SR>=0%',backend_name='example',plugin_spec=SPEC,
        backend_options=options,n_iters=1,run_root=tmp_path)
    assert out['spec'].suite=='robot_object' and out['summary']['trace'][0]['sr'] is not None
