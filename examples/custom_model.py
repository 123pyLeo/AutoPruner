"""Runnable synthetic plugin showing configuration, search and native export.

The simulator is reused from recovery_lib. Its outcomes and exported plan are
not robot measurements or deployable weights. A real adapter replaces the four
model methods and uses its architecture's saver in export_checkpoint.
"""
import json
from pathlib import Path
import shutil
import uuid

from autoprune.backends.plugin import ModelPlugin
from agentic_pruning.integration.lib_engine import LibEngine
from recovery_lib.backends import MockRecoveryBackend


class ExamplePlugin(ModelPlugin):
    name = 'example'
    synthetic = True
    pruning_mode = 'contiguous'
    default_dense_sr = 0.95
    recovery_strategies = ('direct','adapter_only','adapter_mlp_0.5M','adapter+head+proprio')

    def __init__(self,n_layers=32,suite='libero_object',chunk=8,action_dim=7,recovery_steps=120):
        self.n_layers = n_layers
        self.suites = (suite,)
        self.chunk = chunk
        self.action_dim = action_dim
        self.recovery_steps = recovery_steps
        for value in (chunk,action_dim,recovery_steps):
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError('Example dimensions and steps must be positive integers')

    def runtime_config(self):
        return {'n_layers':self.n_layers,'suite':self.suites[0],'chunk':self.chunk,
                'action_dim':self.action_dim,'recovery_steps':self.recovery_steps}

    def _backend(self):
        return MockRecoveryBackend(n_layers=self.n_layers,chunk=self.chunk,
                                   action_dim=self.action_dim,seed=0)

    def load_policy(self,full_ckpt):
        return self._backend()

    def collect_activations(self,policy,*,suite,n_samples,seed):
        if suite != self.suites[0]:raise ValueError('Unexpected suite')
        bundle = policy.collect_activations(n_batches=n_samples,seed=seed)
        return {layer:values[:n_samples] for layer,values in bundle.hidden.items()},bundle.actions[:n_samples]

    def predict_action(self,policy,*,mask=None,knockout_layer=None,n_samples=50,seed=0):
        return policy.predict_action(mask=mask,knockout_layer=knockout_layer,n_batches=n_samples,seed=seed)[:n_samples]

    def recover_and_eval(self,*,full_ckpt,mask,recovery,suite,scope,work_dir,seed):
        result = LibEngine(self._backend(),recovery_steps=self.recovery_steps).recover_and_eval(mask,recovery)
        output = Path(work_dir)/f'example_{uuid.uuid4().hex}'
        output.mkdir()
        (output/'simulation.json').write_text(json.dumps({'synthetic':True,'configuration':self.runtime_config(),
            'suite':suite,'mask':list(mask),'recovery':recovery,'sr':result['sr']},indent=2)+'\n')
        return {**result,'synthetic':True,'status':'completed','source':'example:synthetic','run_dir':str(output)}

    def export_checkpoint(self,*,evaluation,mask,recovery,destination):
        source = Path(evaluation['run_dir'])/'simulation.json'
        saved = json.loads(source.read_text())
        if saved['mask']!=list(mask) or saved['recovery']!=recovery:
            raise ValueError('Synthetic artifacts do not match the evaluated selection')
        shutil.copyfile(source,Path(destination)/source.name)
        return 'autoprune-synthetic-plan-v1'
