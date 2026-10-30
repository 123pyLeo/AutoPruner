"""Run the actual packaged overlay cache code, without a model or GPU."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


def cache_class(tmp_path):
    torch = pytest.importorskip('torch')
    from importlib.resources import files
    patch = files('autoprune.integrations.openvla_oft').joinpath('overlay.patch').read_bytes()
    # The cache is a release-added upstream file; git apply can extract that file
    # without installing the model training stack.
    subprocess.run(['git','apply','--include=prismatic/training/teacher_cache.py','-'],
                   cwd=tmp_path,input=patch,check=True)
    source=tmp_path/'prismatic/training/teacher_cache.py'
    spec=importlib.util.spec_from_file_location('cache_under_test',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return torch,module.TeacherOutputCache


def test_distinct_observations_with_identical_tokens_retain_distinct_targets(tmp_path):
    torch,Cache=cache_class(tmp_path)
    ids=torch.tensor([4,5,6]);cache=Cache()
    for frame in range(14):
        assert cache.save_sample(ids,torch.full((2,3),float(frame)),float(frame),sample_id=f'data/1.0|record|{frame}')
    cache.save_to_disk(tmp_path/'cache',{'expected_samples':14})
    restored=Cache(tmp_path/'cache')
    order=list(reversed(range(14)))
    padded=torch.stack([torch.cat([ids,torch.tensor([0,0])]) for _ in order])
    masks=torch.tensor([[1,1,1,0,0]]*len(order))
    result=restored.lookup_batch(padded,masks,sample_ids=[f'data/1.0|record|{i}' for i in order])
    assert result['found_mask'].all()
    assert result['teacher_logits_action_range'][:,0,0].float().tolist()==order
    assert len(restored)==14


def test_misses_and_missing_identity_are_errors_not_silent_kl_skips(tmp_path):
    torch,Cache=cache_class(tmp_path);ids=torch.tensor([[1,2]])
    cache=Cache()
    with pytest.raises(ValueError,match='sample_id'):
        cache.lookup_batch(ids)
    with pytest.raises(KeyError,match='missing'):
        cache.lookup_batch(ids,sample_ids=['unseen'])


def test_legacy_and_partial_caches_cannot_be_loaded_as_complete(tmp_path):
    torch,Cache=cache_class(tmp_path)
    legacy=tmp_path/'legacy';legacy.mkdir();torch.save({'num_samples':1},legacy/'metadata.pt')
    with pytest.raises(ValueError,match='Legacy'):
        Cache(legacy)
    cache=Cache();cache.save_sample(torch.tensor([1]),torch.zeros(2,3),0,sample_id='sample')
    with pytest.raises(ValueError,match='incomplete'):
        cache.save_to_disk(tmp_path/'partial',{'expected_samples':2})
    assert not (tmp_path/'partial').exists()
    cache.save_to_disk(tmp_path/'valid',{'expected_samples':1})
    (tmp_path/'valid/cache_shard_0.pt').unlink()
    with pytest.raises(ValueError,match='incomplete'):
        Cache(tmp_path/'valid')


def test_key_ignores_right_padding_and_rejects_invalid_masks(tmp_path):
    torch,Cache=cache_class(tmp_path);ids=torch.tensor([1,2])
    key=Cache.compute_hash(ids,sample_id='record|0')
    assert key==Cache.compute_hash(torch.tensor([1,2,0]),torch.tensor([1,1,0]),sample_id='record|0')
    assert key!=Cache.compute_hash(ids,sample_id='record|1')
    with pytest.raises(ValueError,match='right-padded'):
        Cache.compute_hash(torch.tensor([0,1,2]),torch.tensor([0,1,1]),sample_id='record|0')
    with pytest.raises(ValueError,match='integer'):
        Cache.compute_hash(torch.tensor([1.5]),sample_id='record|0')


@pytest.mark.parametrize('metadata', [
    {'num_patches':513},
    {'num_patches':513,'format':'autoprune-teacher-v2','complete':False,'expected_samples':1,'num_samples':1},
    {'num_patches':513,'format':'autoprune-teacher-v2','complete':True,'expected_samples':2,'num_samples':1},
])
def test_oft_generation_cannot_accept_legacy_or_incomplete_output(tmp_path,monkeypatch,metadata):
    import sys
    from types import SimpleNamespace
    from autoprune.backends import oft_plugin as oft
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(load=lambda *a,**kw:metadata))
    cache=tmp_path/'generated'
    def execute(*args,**kw):
        cache.mkdir();(cache/'metadata.pt').touch();(cache/'cache_shard_0.pt').touch()
    monkeypatch.setattr(oft.OftPlugin,'_run',staticmethod(execute))
    with pytest.raises(RuntimeError,match='legacy or incomplete'):
        oft.OftPlugin()._ensure_teacher_cache('checkpoint','libero_object_no_noops',cache)
