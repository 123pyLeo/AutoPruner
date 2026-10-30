import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autoprune.calibration import read_language
from autoprune.run_e2e import stage_checkpoint
from autoprune.selection import select_winner
from autoprune.validation import validate_evaluation
from agentic_pruning.e2e.prompt_parser import ProblemSpec
from autoprune.backends.generic_backend import GenericVLABackend
from examples.custom_model import ExamplePlugin


@pytest.mark.parametrize('score', [float('nan'), float('inf'), -0.1, 1.1, True])
def test_invalid_success_rates_cannot_win(score):
    row = {'iter':0,'mask':[2],'k':1,'sr':score,'recovery':'direct'}
    result = select_winner({'dense_sr':.95,'total_layers':32,'trace':[row]}, ProblemSpec())
    assert result['winner'] is None and result['winner_meets_spec'] is False


def test_relative_paths_survive_a_subprocess_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = Path('weights'); source.mkdir()
    (source/'model.safetensors').write_bytes(b'weight')
    staged = stage_checkpoint(source, 'runs/search')
    assert staged.is_absolute()
    monkeypatch.chdir(tmp_path.parent)
    assert (staged/'model.safetensors').read_bytes() == b'weight'
    backend = GenericVLABackend(ExamplePlugin(), str(source), work_dir=tmp_path/'work')
    assert backend.work_dir.is_absolute() and Path(backend.full_ckpt).is_absolute()


@pytest.mark.parametrize('attrs', [{}, {'problem_info':'not-json'}, {'problem_info':'{}'},
                                    {'problem_info':'{"language_instruction":""}'}])
def test_missing_language_cannot_silently_change_task(attrs):
    with pytest.raises(ValueError, match='language'):
        read_language(attrs, 'calibration.hdf5')


def test_explicit_normalization_suite_never_falls_back_to_object():
    # Compile the small actual resolver without importing GPU/framework dependencies.
    source = Path(__file__).parents[1]/'autoprune/oft_observations.py'
    module = ast.parse(source.read_text())
    fn = next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='_resolve_unnorm_key')
    namespace = {}
    exec(compile(ast.Module(body=[fn],type_ignores=[]), str(source), 'exec'), namespace)
    resolve = namespace['_resolve_unnorm_key']
    stats = {'libero_object_no_noops':{},'libero_goal_no_noops':{}}
    assert resolve(stats,'libero_goal') == 'libero_goal_no_noops'
    with pytest.raises(KeyError):
        resolve({'libero_object_no_noops':{}},'libero_goal')


@pytest.mark.parametrize('record', [
    {'sr':.9},
    {'sr':float('nan'),'status':'completed','episodes':10,'successes':9},
    {'sr':.9,'status':'failed','episodes':10,'successes':9},
    {'sr':.9,'status':'completed','episodes':10,'successes':8},
    {'sr':.9,'status':'completed','episodes':0,'successes':0},
])
def test_real_plugin_results_need_consistent_measured_counts(record):
    with pytest.raises(ValueError):
        validate_evaluation(record,require_counts=True)


def test_real_plugin_validation_failure_cannot_enter_search(tmp_path):
    plugin = ExamplePlugin()
    plugin.synthetic = False
    plugin.recover_and_eval = lambda **kw: {'sr':.99,'status':'completed'}
    backend = GenericVLABackend(plugin, 'checkpoint', work_dir=tmp_path)
    result = backend.train_recovery([2], 'direct')
    assert result['sr'] is None and result['status'] == 'failed'
    assert 'counts' in result['error']


def test_zero_dry_flag_still_executes_and_suite_is_explicit(tmp_path,monkeypatch):
    plugin = ExamplePlugin()
    calls = []
    def evaluate(**kwargs):
        calls.append(kwargs)
        return {'sr':.9,'status':'completed'}
    plugin.recover_and_eval = evaluate
    monkeypatch.setenv('AUTOPRUNE_DRY_RECOVERY','0')
    backend = GenericVLABackend(plugin, 'checkpoint', suite='libero_object', work_dir=tmp_path)
    assert plugin.active_suite == 'libero_object'
    assert backend.train_recovery([2], 'direct')['sr'] == .9
    assert len(calls) == 1
    monkeypatch.setenv('AUTOPRUNE_DRY_RECOVERY','1')
    assert backend.train_recovery([2], 'direct')['status'] == 'dry_run'
    assert len(calls) == 1


def test_small_skip_model_does_not_inherit_oft_search_instructions(tmp_path):
    import shutil
    from autoprune.run_e2e import DATA_DIR
    from agentic_pruning.integration.lib_modules import build_lib_module_set
    plugin = ExamplePlugin()
    plugin.n_layers = 3
    plugin.pruning_mode = 'skip'
    backend = GenericVLABackend(plugin, 'checkpoint', work_dir=tmp_path/'execution')
    shutil.copytree(DATA_DIR, tmp_path/'data')
    modules = build_lib_module_set(tmp_path/'data', backend)
    space = modules.mask_search.get_search_space()
    assert space['total_layers'] == 3
    assert space['mask_form'] == 'skip'
    assert space['k_options'] == [1, 2]
    assert 'consecutive' not in space['mask_form_doc']


def test_openai_agent_never_borrows_anthropic_credentials(monkeypatch):
    from agentic_pruning.agents.gpt_agent import GptAgent
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('OPENAI_BASE_URL', raising=False)
    monkeypatch.setenv('ANTHROPIC_AUTH_TOKEN', 'test-provider-token')
    monkeypatch.setenv('ANTHROPIC_BASE_URL', 'https://different-provider.invalid')
    monkeypatch.setenv('AGENT_MODEL', 'test-model')
    with pytest.raises(RuntimeError, match='OPENAI_API_KEY'):
        GptAgent()
    monkeypatch.setenv('OPENAI_API_KEY', 'test-openai-token')
    agent = GptAgent()
    assert agent.base_url == 'https://api.openai.com'
    assert agent.auth_token == 'test-openai-token'
