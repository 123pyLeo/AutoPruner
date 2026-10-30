"""Export the exact qualifying winner through OFT or a model plugin's native exporter."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from autoprune.validation import validate_evaluation, valid_sr


def resolve_winner(run):
    run = Path(run).expanduser().resolve()
    request = json.loads((run/'request.json').read_text())
    selection = json.loads((run/'selection.json').read_text())
    if not request.get('plugin') and (request.get('synthetic') or request.get('backend') != 'oft'):
        raise ValueError('Export requires OFT or an explicit model plugin with a native exporter')
    if selection.get('winner_meets_spec') is not True or not selection.get('winner'):
        raise ValueError('No qualifying winner; review selection.json before deployment')
    winner = selection['winner']
    iteration = winner.get('iter')
    if not isinstance(iteration,int) or isinstance(iteration,bool) or iteration < 0:
        raise ValueError('Winner has no valid iteration')
    result = json.loads((run/f'iter_{iteration:03d}'/'result.json').read_text())
    proposal = json.loads((run/f'iter_{iteration:03d}'/'proposal.json').read_text())
    validate_evaluation(result,require_counts=not request.get('synthetic', False))
    if not valid_sr(result.get('sr')):
        raise ValueError('Winning evaluation has no valid SR')
    if sorted(proposal['mask']) != sorted(winner['mask']):
        raise ValueError('Winning mask does not match its recorded proposal')
    recovery = result.get('executed_recovery',result.get('recipe'))
    if (recovery != winner.get('recovery') or recovery != result.get('recipe') or
            proposal.get('recovery', recovery) != recovery):
        raise ValueError('Winning recovery recipe does not match execution')
    if result['sr'] != winner['sr']:
        raise ValueError('Winning SR does not match its evaluation')
    source = Path(result['run_dir']).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f'Winning checkpoint directory is absent: {source}')
    step = result.get('adapter_step')
    if request['backend'] == 'oft' and recovery != 'direct' and (not isinstance(step,int) or isinstance(step,bool) or step < 1):
        raise ValueError('Recovered winner has no explicit checkpoint step')
    return {'source':str(source),'mask':sorted(winner['mask']),'recovery':recovery,
            'step':step,'iteration':iteration,'search_run':str(run),'evaluation':result,'request':request}


def _plugin_export(winner, destination):
    from autoprune.backends.loader import load_plugin
    from autoprune.backends.plugin import ModelPlugin
    request = winner['request']
    plugin = load_plugin(request['plugin'], request.get('plugin_options', {}))
    if plugin.name != request['backend'] or plugin.synthetic != request.get('synthetic', False):
        raise ValueError('Export plugin identity does not match the evaluated search')
    if type(plugin).export_checkpoint is ModelPlugin.export_checkpoint:
        raise ValueError(f'{plugin.name} has no native export_checkpoint implementation')
    plugin.active_suite = request['spec']['suite']
    output = Path(destination).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Export destination must be new: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.autoprune-plugin-export-',dir=output.parent) as temporary:
        staged = Path(temporary)/'checkpoint'
        staged.mkdir()
        native_format = plugin.export_checkpoint(evaluation=dict(winner['evaluation']),
            mask=list(winner['mask']), recovery=winner['recovery'], destination=staged)
        if staged.is_symlink() or not staged.is_dir():
            raise ValueError('Native exporter must preserve the provided output directory')
        if not isinstance(native_format, str) or not native_format.strip():
            raise ValueError('Native exporter must return its nonempty format identifier')
        if (staged/'autoprune_export.json').exists() or (staged/'autoprune_selection.json').exists():
            raise ValueError('Export manifest and selection receipt names are reserved by the framework')
        resources = list(staged.rglob('*'))
        if any(p.is_symlink() for p in resources):
            raise ValueError('Native export must be self-contained; symlinks are not allowed')
        files = {str(p.relative_to(staged)): _sha256(p) for p in resources if p.is_file()}
        if not files:
            raise ValueError('Native exporter produced no model files')
        manifest = {'format':'autoprune-plugin-v1','native_format':native_format,
                    'backend':plugin.name,'synthetic':plugin.synthetic,'removed_layers':winner['mask'],
                    'recovery':winner['recovery'],'files':files}
        (staged/'autoprune_export.json').write_text(json.dumps(manifest,indent=2)+'\n')
        shutil.move(str(staged),str(output))
    return manifest


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def finalize(run,destination):
    winner = resolve_winner(run)
    if winner['request'].get('plugin'):
        manifest = _plugin_export(winner,destination)
    else:
        from autoprune.export_oft import export_checkpoint
        manifest = export_checkpoint(winner['source'],destination,winner['mask'],winner['step'],recovery=winner['recovery'])
    receipt = {'search_run':winner['search_run'],'iteration':winner['iteration'],
               'backend':winner['request']['backend'],'synthetic':winner['request'].get('synthetic',False),
               'recovery':winner['recovery'],'removed_layers':winner['mask'],
               'source_step':winner['step'],'evaluation':winner['evaluation'],
               'export_manifest':'autoprune_export.json','action_equivalence_verified':False}
    output=Path(destination).expanduser().resolve()
    path=output/'autoprune_selection.json'
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    manifest['files'][path.name]=_sha256(path)
    (output/'autoprune_export.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest,receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--destination',required=True)
    args=parser.parse_args()
    try:
        _,receipt=finalize(args.run,args.destination)
    except (ValueError,RuntimeError,KeyError,OSError) as exc:
        parser.exit(2,f'Finalize: {exc}\n')
    print(json.dumps({'destination':str(Path(args.destination).expanduser().resolve()),'iteration':receipt['iteration'],
                      'recovery':receipt['recovery'],'removed_layers':receipt['removed_layers'],
                      'synthetic':receipt['synthetic']},indent=2))


if __name__=='__main__':main()
