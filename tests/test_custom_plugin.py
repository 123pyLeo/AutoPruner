import json
from pathlib import Path

import pytest

from autoprune.backends.loader import load_plugin
from autoprune.run_e2e import make_backend, run_prompt
from examples.custom_model import ExamplePlugin


def test_explicit_plugin_runs_through_core_and_reports_synthetic(tmp_path):
    out = run_prompt('prune libero_object, SR>=40%', backend_name='example',
                     plugin_spec='examples.custom_model:ExamplePlugin', n_iters=2, run_root=tmp_path)
    root = Path(out['run_dir'])
    assert out['summary']['synthetic'] is True
    assert len(out['summary']['trace']) == 2
    assert all(row['sr'] is not None for row in out['summary']['trace'])
    assert json.loads((root/'request.json').read_text())['plugin'] == 'examples.custom_model:ExamplePlugin'
    assert json.loads((root/'report.json').read_text())['synthetic'] is True


@pytest.mark.parametrize('spec', ['bad', 'examples.custom_model:missing', 'examples.custom_model:ModelPlugin'])
def test_plugin_rejects_invalid_declarations(spec):
    with pytest.raises((ValueError, TypeError)):
        load_plugin(spec)


def test_real_plugin_requires_explicit_inputs_and_forbids_rule_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(ExamplePlugin, 'synthetic', False)
    spec = 'examples.custom_model:ExamplePlugin'
    with pytest.raises(ValueError, match='full-ckpt'):
        make_backend('example', plugin_spec=spec, full_ckpt=None, suite='libero_object', num_trials=1)
    with pytest.raises(ValueError, match='dense-sr'):
        make_backend('example', plugin_spec=spec, full_ckpt='model', suite='libero_object', num_trials=1)
    with pytest.raises(ValueError, match='scripted rule agent'):
        run_prompt('prune libero_object', backend_name='example', plugin_spec=spec,
                   full_ckpt='model', dense_sr=.9, run_root=tmp_path)


def test_starvla_reference_preserves_kept_layers_and_adapter_identity():
    torch = pytest.importorskip('torch')
    from types import SimpleNamespace
    from examples.starvla import apply_layer_mask, install_adapters
    nn = torch.nn

    class Layer(nn.Module):
        def __init__(self, idx):
            super().__init__()
            self.self_attn = SimpleNamespace(layer_idx=idx)
        def forward(self, x):
            return x + 1

    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([Layer(i) for i in range(6)])
    model.config = SimpleNamespace(num_hidden_layers=6)
    originals = list(model.model.layers)
    info = apply_layer_mask(model, [1, 2, 4])
    assert info['kept_layers'] == [0, 3, 5]
    assert list(model.model.layers) == [originals[i] for i in [0, 3, 5]]
    assert [layer.self_attn.layer_idx for layer in model.model.layers] == [0, 1, 2]
    assert model.config.num_hidden_layers == 3
    adapters = install_adapters(model.model.layers, [1, 2, 4], total_layers=6,
                                hidden_size=4, bottleneck=2, dtype=torch.float32)
    assert set(adapters) == {1, 2}
    x = torch.randn(2, 4)
    for adapter in adapters.values():
        torch.testing.assert_close(adapter(x), x, rtol=0, atol=0)
    torch.testing.assert_close(model.model.layers[1](x), x+1)
