"""Verify exported files, then compare native recovered/exported actions in separate processes."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import numpy as np


def verify_files(exported):
    exported=Path(exported).expanduser().resolve()
    manifest=json.loads((exported/'autoprune_export.json').read_text())
    if manifest.get('format')!='autoprune-oft-physical-v1':raise ValueError('Unsupported export format')
    for name,expected in manifest['files'].items():
        path=exported/name
        if Path(name).name!=name or not path.is_file() or path.is_symlink():
            raise ValueError(f'Export resource is missing or not self-contained: {name}')
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
        if digest.hexdigest()!=expected:raise ValueError(f'Export resource checksum differs: {name}')
    return manifest


def worker(args):
    from autoprune.backends import oft_plugin as oft
    from autoprune.run_e2e import stage_checkpoint
    from experiments.robot.libero.run_libero_eval import GenerateConfig,initialize_model
    from experiments.robot.openvla_utils import get_vla_action
    manifest=json.loads((Path(args.exported)/'autoprune_export.json').read_text())
    output=Path(args.worker_output);output.mkdir(parents=True)
    source=Path(args.source) if args.worker=='reference' else Path(args.exported)
    staged=stage_checkpoint(source,output)
    cfg=GenerateConfig(pretrained_checkpoint=str(staged),task_suite_name=args.suite,
        unnorm_key=oft.SUITE_DATASET[args.suite],use_l1_regression=True,use_proprio=True,
        num_images_in_input=2,center_crop=True,save_videos=False,
        physically_pruned=args.worker=='export',remove_layers=','.join(map(str,manifest['removed_layers'])))
    if manifest['recovery']!='direct':
        cfg.adapter_path=str(staged/f"adapter--{manifest['adapter_step']}_checkpoint.pt")
        cfg.adapter_bottleneck=manifest['adapter_config']['bottleneck']
        cfg.adapter_type=manifest['adapter_config']['type']
    model,head,proprio,noise,processor=initialize_model(cfg)
    predictions=[]
    for example in oft.OftPlugin._iter_examples(str(Path(oft.HDF5_ROOT)/args.suite),args.samples):
        observation={'full_image':example['image'][0],'wrist_image':example['image'][1],
                     'state':np.asarray(example['state']).copy()}
        predictions.append(np.asarray(get_vla_action(cfg,model,processor,observation,example['lang'],
            action_head=head,proprio_projector=proprio,noisy_action_projector=noise)))
    actions=np.stack(predictions)
    if actions.shape!=(args.samples,8,7) or not np.isfinite(actions).all():
        raise RuntimeError(f'Unexpected native prediction shape or values: {actions.shape}')
    np.save(output/'actions.npy',actions)
    result={'role':args.worker,'action_shape':list(actions.shape),
            'loaded_decoder_layers':len(model.language_model.model.layers)}
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')


def verify(exported,*,source=None,suite='libero_object',samples=2,run_root='runs',atol=1e-5,rtol=1e-5,evaluate=False,trials=1):
    from autoprune.backends import oft_plugin as oft
    if trials<1 or trials>50:raise ValueError('Use 1–50 bounded rollout trials')
    if samples<1 or samples>24:raise ValueError('Use 1–24 fixed calibration samples')
    exported=Path(exported).expanduser().resolve();manifest=verify_files(exported)
    if source is None:
        receipt=json.loads((exported/'autoprune_selection.json').read_text())
        result=json.loads((Path(receipt['search_run'])/f"iter_{receipt['iteration']:03d}"/'result.json').read_text())
        source=result['run_dir']
    source=Path(source).expanduser().resolve()
    plugin=oft.OftPlugin();plugin.active_suite=suite;plugin.preflight(source,training=False)
    output=Path(run_root).expanduser().resolve()/f'export_check_{uuid.uuid4().hex}';output.mkdir(parents=True)
    for role in ['reference','export']:
        work=output/role
        env=oft.OftPlugin._eval_env(oft.EVAL_GPU)
        env['PYTHONPATH']=os.pathsep.join([str(Path(__file__).resolve().parents[1]),env.get('PYTHONPATH','')])
        oft.OftPlugin._libero_config(env,output/f'{role}_libero_config')
        command=[sys.executable,'-m','autoprune.verify_oft_export','--worker',role,
                 '--exported',str(exported),'--source',str(source),'--suite',suite,
                 '--samples',str(samples),'--worker-output',str(work)]
        with (output/f'{role}.log').open('w') as log:
            completed=subprocess.run(command,env=env,cwd=oft.OFT_REPO,stdout=log,stderr=subprocess.STDOUT)
        if completed.returncode:raise RuntimeError(f'{role} reload failed; inspect {output/ (role+".log")}')
    reference=np.load(output/'reference/actions.npy');physical=np.load(output/'export/actions.npy')
    result={'status':'passed' if np.allclose(reference,physical,atol=atol,rtol=rtol) else 'failed',
            'action_shape':list(reference.shape),'max_abs_action_error':float(np.max(np.abs(reference-physical))),
            'atol':atol,'rtol':rtol,'files_verified':len(manifest['files']),
            'reference':json.loads((output/'reference/result.json').read_text()),
            'physical':json.loads((output/'export/result.json').read_text()),
            'full_robot_evaluation':False}
    if result['status']=='passed' and evaluate:
        from autoprune.backends.plugin import EvalScope
        from autoprune.run_e2e import stage_checkpoint
        staged=stage_checkpoint(exported,output)
        if manifest['recovery']!='direct':plugin.adapter_bottleneck=manifest['adapter_config']['bottleneck']
        adapter=staged/f"adapter--{manifest['adapter_step']}_checkpoint.pt" if manifest['recovery']!='direct' else None
        result['bounded_rollout']=plugin._eval(pretrained=staged,mask_str=','.join(map(str,manifest['removed_layers'])),
            tag='physical_export',suite=suite,dataset_name=oft.SUITE_DATASET[suite],adapter_path=adapter,
            adapter_type=(manifest.get('adapter_config') or {}).get('type','mlp'),out_dir=output/'rollout',
            scope=EvalScope(num_trials=trials,max_eval_tasks=1),seed=7,physically_pruned=True)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['status']!='passed':raise RuntimeError(f'Exported actions differ; inspect {output/"result.json"}')
    return output,result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exported',required=True);parser.add_argument('--source')
    parser.add_argument('--suite',default='libero_object');parser.add_argument('--samples',type=int,default=2)
    parser.add_argument('--run-root',default='runs');parser.add_argument('--files-only',action='store_true')
    parser.add_argument('--evaluate',action='store_true',help='Also run the exported policy on one task; bounded smoke only')
    parser.add_argument('--trials',type=int,default=1)
    parser.add_argument('--worker',choices=['reference','export'],help=argparse.SUPPRESS)
    parser.add_argument('--worker-output',help=argparse.SUPPRESS)
    args=parser.parse_args()
    try:
        if args.worker:worker(args);return
        if args.files_only:
            data=verify_files(args.exported);print(json.dumps({'status':'files_verified','files':len(data['files'])}));return
        output,result=verify(args.exported,source=args.source,suite=args.suite,samples=args.samples,run_root=args.run_root,evaluate=args.evaluate,trials=args.trials)
    except (OSError,KeyError,ValueError,RuntimeError) as exc:parser.exit(2,f'Export check: {exc}\n')
    print(json.dumps({'run_dir':str(output),**result},indent=2))


if __name__=='__main__':main()
