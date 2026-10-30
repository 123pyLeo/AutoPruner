import ast,json
from pathlib import Path
from types import SimpleNamespace
import pytest
from autoprune.finalize import resolve_winner
from autoprune.backends import oft_plugin as oft


def test_real_policy_loader_does_not_depend_on_calibration_variables():
    source=Path(__file__).parents[1]/'autoprune/oft_observations.py'
    fn=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_load_oft_policy')
    class Module:
        def to(self,*a,**kw):return self
    policy=SimpleNamespace(openvla_model=Module(),action_model=Module(),proprio_projector=Module(),eval=lambda:None,parameters=lambda:[])
    ns={'OpenVLAOFT':lambda **kw:policy,'_make_oft_config':lambda *a:{},'torch':SimpleNamespace(bfloat16='bf16')}
    tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),ns)
    assert ns['_load_oft_policy']('checkpoint','cpu',benchmark_name='libero_object') is policy


def test_explicit_recovery_configuration_wins_and_is_recorded(monkeypatch):
    monkeypatch.setenv('AUTOPRUNE_TRAIN_MAX_STEPS','99');monkeypatch.setenv('AUTOPRUNE_GLOBAL_BATCH','32')
    plugin=oft.OftPlugin(train_max_steps=2,global_batch=1,shuffle_buffer_size=4)
    assert plugin.runtime_config()['train_max_steps']==2
    assert plugin.runtime_config()['global_batch']==1
    assert plugin.runtime_config()['shuffle_buffer_size']==4


def test_libero_configuration_is_invocation_local(tmp_path,monkeypatch):
    repo=tmp_path/'LIBERO';root=repo/'libero/libero';(root/'benchmark').mkdir(parents=True);(root/'benchmark/__init__.py').touch()
    for name in ['bddl_files','init_files','assets']:
        (root/name).mkdir();(root/name/'resource').touch()
    monkeypatch.setattr(oft,'LIBERO_ROOT',str(repo));monkeypatch.setattr(oft,'HDF5_ROOT',str(tmp_path/'demos'))
    env={'LIBERO_CONFIG_PATH':'stale-config'}
    oft.OftPlugin._libero_config(env,tmp_path/'run/libero_config')
    cfg=json.loads((Path(env['LIBERO_CONFIG_PATH'])/'config.yaml').read_text())
    assert cfg['benchmark_root']==str(root) and cfg['datasets']==str(tmp_path/'demos')
    assert env['LIBERO_CONFIG_PATH']!= 'stale-config'


def make_run(tmp_path):
    (tmp_path/'checkpoint').mkdir();(tmp_path/'iter_000').mkdir()
    (tmp_path/'request.json').write_text(json.dumps({'backend':'oft','synthetic':False}))
    winner={'iter':0,'mask':[8,9],'recovery':'adapter_only','sr':.9}
    (tmp_path/'selection.json').write_text(json.dumps({'winner_meets_spec':True,'winner':winner}))
    (tmp_path/'iter_000/proposal.json').write_text(json.dumps({'mask':[8,9]}))
    result={'status':'completed','successes':9,'episodes':10,'sr':.9,'recipe':'adapter_only','executed_recovery':'adapter_only','run_dir':str(tmp_path/'checkpoint'),'adapter_step':2}
    (tmp_path/'iter_000/result.json').write_text(json.dumps(result))
    return result


def test_export_handoff_uses_executed_recipe_and_checkpoint_step(tmp_path):
    make_run(tmp_path);winner=resolve_winner(tmp_path)
    assert winner['mask']==[8,9] and winner['step']==2 and winner['recovery']=='adapter_only'
    selection=json.loads((tmp_path/'selection.json').read_text());selection['winner_meets_spec']=False
    (tmp_path/'selection.json').write_text(json.dumps(selection))
    with pytest.raises(ValueError,match='qualifying'):resolve_winner(tmp_path)


def test_export_handoff_rejects_mixed_iteration_artifacts(tmp_path):
    result=make_run(tmp_path);result['adapter_step']=None
    (tmp_path/'iter_000/result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError,match='step'):resolve_winner(tmp_path)


@pytest.mark.parametrize('bias',[True,False])
@pytest.mark.parametrize('regularization',[.1,1.,1000.])
def test_dual_probe_solver_matches_original_unregularized_intercept(bias,regularization):
    import numpy as np
    from mask_probe_lib.core.kernels import ridge_fit,ridge_predict
    rng=np.random.default_rng(8);x=rng.normal(size=(8,25));y=rng.normal(size=(8,7))+4
    design=np.column_stack([x,np.ones(len(x))]) if bias else x
    penalty=regularization*np.eye(design.shape[1])
    if bias:penalty[-1,-1]=0
    original=np.linalg.solve(design.T@design+penalty,design.T@y)
    actual=ridge_fit(x,y,regularization,fit_bias=bias)
    np.testing.assert_allclose(actual,original,rtol=1e-8,atol=1e-8)
    heldout=rng.normal(size=(4,25))
    np.testing.assert_allclose(ridge_predict(heldout,actual,bias),ridge_predict(heldout,original,bias),rtol=1e-8,atol=1e-8)


def test_agent_protocol_rejects_prose_in_numeric_gates():
    from agentic_pruning.agents.base import AgentContext,decision_from_json
    ctx=AgentContext(0,[],n_layers=32,recovery_options=[{'id':'direct'}])
    obj={'mask':[8,9],'recovery':'direct','success_gate':.9,'stop_gate':.1}
    assert decision_from_json(obj,ctx).success_gate==.9
    obj['success_gate']='SR must exceed 90 percent'
    with pytest.raises(ValueError,match='success_gate must be a JSON number'):
        decision_from_json(obj,ctx)


def test_dual_probe_solver_preserves_scalar_targets():
    import numpy as np
    from mask_probe_lib.core.kernels import ridge_fit,ridge_predict
    rng=np.random.default_rng(5);x=rng.normal(size=(4,10));y=rng.normal(size=4)
    w=ridge_fit(x,y,1.)
    assert w.shape==(11,)
    assert ridge_predict(x,w).shape==(4,)


def test_search_recovery_menu_is_an_explicit_subset():
    from agentic_pruning.integration.lib_modules import LibRecoveryModule
    engine=SimpleNamespace(backend=SimpleNamespace(plugin=SimpleNamespace(recovery_strategies=['direct','adapter+head+proprio'])))
    menu=LibRecoveryModule(engine)
    with pytest.raises(ValueError,match='subset'):menu.restrict(['imaginary'])
    menu.restrict(['adapter+head+proprio'])
    assert not menu.is_valid('direct')
    assert menu.list_recipes()[0]['id']=='adapter+head+proprio'


@pytest.mark.parametrize('prompt',['prune all suites','prune object within 2 hours','prune object remove 120%'])
def test_unimplemented_or_invalid_search_constraints_fail_before_execution(prompt,tmp_path):
    from autoprune.run_e2e import run_prompt
    with pytest.raises(ValueError):run_prompt(prompt,run_root=tmp_path)
    assert list(tmp_path.iterdir())==[]
