"""Evaluate a downloaded physical OFT export without its original search directory.

The export manifest supplies the original deletion mask and exact adapter step.
No API key, teacher cache, training dataset or dense reference model is needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

from autoprune.run_e2e import stage_checkpoint
from autoprune.verify_oft_export import verify_files


def evaluate(exported,*,suite='libero_object',trials=1,max_eval_tasks=None,run_root='runs',seed=7):
    from autoprune.backends.oft_plugin import OftPlugin,SUITE_DATASET
    from autoprune.backends.plugin import EvalScope
    if suite not in SUITE_DATASET or not 1<=trials<=50:
        raise ValueError('Use a supported LIBERO suite and 1–50 trials per task')
    if max_eval_tasks is not None and max_eval_tasks<1:
        raise ValueError('max_eval_tasks must be positive')
    exported=Path(exported).expanduser().resolve()
    manifest=verify_files(exported)
    plugin=OftPlugin();plugin.active_suite=suite
    plugin.preflight(exported,training=False,calibration=False)
    output=Path(run_root).expanduser().resolve()/f'export_eval_{uuid.uuid4().hex}'
    output.mkdir(parents=True)
    staged=stage_checkpoint(exported,output)
    recovered=manifest['recovery']!='direct'
    adapter_config=manifest.get('adapter_config') or {}
    if recovered:
        plugin.adapter_bottleneck=adapter_config['bottleneck']
        adapter=staged/f"adapter--{manifest['adapter_step']}_checkpoint.pt"
    else:adapter=None
    request={'kind':'downloaded_oft_export_evaluation','exported':str(exported),'suite':suite,
        'num_trials_per_task':trials,'max_eval_tasks':max_eval_tasks,'seed':seed,
        'removed_layers':manifest['removed_layers'],'recovery':manifest['recovery'],
        'adapter_step':manifest['adapter_step'],'training':False,'llm_calls':0,
        'dense_reference_required':False}
    (output/'request.json').write_text(json.dumps(request,indent=2)+'\n')
    result=plugin._eval(pretrained=str(staged),mask_str=','.join(map(str,manifest['removed_layers'])),
        tag='downloaded_export',suite=suite,dataset_name=SUITE_DATASET[suite],adapter_path=adapter,
        adapter_type=adapter_config.get('type','mlp'),out_dir=output/'rollout',
        scope=EvalScope(num_trials=trials,max_eval_tasks=max_eval_tasks),seed=seed,physically_pruned=True)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return output,result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exported',required=True)
    parser.add_argument('--suite',default='libero_object')
    parser.add_argument('--trials',type=int,default=1)
    parser.add_argument('--max-eval-tasks',type=int)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--run-root',default='runs')
    args=parser.parse_args()
    try:
        output,result=evaluate(args.exported,suite=args.suite,trials=args.trials,
            max_eval_tasks=args.max_eval_tasks,seed=args.seed,run_root=args.run_root)
    except (OSError,ValueError,KeyError,RuntimeError) as exc:parser.exit(2,f'OFT export evaluation: {exc}\n')
    print(json.dumps({'run_dir':str(output),'status':result['status'],'sr':result['sr'],
        'successes':result['successes'],'episodes':result['episodes'],'held_out_from_search':result['held_out_from_search']},indent=2))


if __name__=='__main__':main()
