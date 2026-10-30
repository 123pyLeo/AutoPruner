"""Check exported resources and checksums for OFT or a native model plugin.

This checks file integrity only. Use the model's native loader and fixed-input
predictions separately to establish numerical equivalence.
"""
import argparse
import json
from pathlib import Path

from autoprune.finalize import _sha256


def verify_files(exported):
    root = Path(exported).expanduser().resolve()
    manifest = json.loads((root/'autoprune_export.json').read_text())
    if manifest.get('format') not in {'autoprune-oft-physical-v1','autoprune-plugin-v1'}:
        raise ValueError('Unsupported framework export manifest')
    files = manifest.get('files')
    if not isinstance(files,dict) or not files:
        raise ValueError('Export manifest has no resource checksums')
    for name,digest in files.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or name in {'','.','autoprune_export.json'}:
            raise ValueError(f'Invalid export resource name: {name}')
        path = root/relative
        current = path
        linked = False
        while current != root:
            linked = linked or current.is_symlink()
            current = current.parent
        if linked:
            raise ValueError(f'Export resource must not use symlinks: {name}')
        if not path.is_file() or _sha256(path)!=digest:
            raise ValueError(f'Export resource missing or checksum differs: {name}')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exported',required=True)
    args=parser.parse_args()
    try:manifest=verify_files(args.exported)
    except (OSError,ValueError,KeyError) as exc:parser.exit(2,f'Export files: {exc}\n')
    print(json.dumps({'status':'files_verified','format':manifest['format'],
        'synthetic':manifest.get('synthetic',False),'files':len(manifest['files']),
        'action_equivalence_verified':False},indent=2))


if __name__=='__main__':main()
