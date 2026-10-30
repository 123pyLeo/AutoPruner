"""Resolve selected LIBERO resources without consulting the user's home config."""
from pathlib import Path


def resource_paths(checkout, *, dataset_root=None, suite=None):
    root=Path(checkout).expanduser().resolve()/'libero/libero'
    if not (root/'benchmark/__init__.py').is_file():
        raise FileNotFoundError('OFT_LIBERO_ROOT must be a complete LIBERO checkout')
    paths={'benchmark_root':root,'bddl_files':root/'bddl_files','init_states':root/'init_files',
           'assets':root/'assets','datasets':Path(dataset_root).expanduser().resolve() if dataset_root else root.parent/'datasets'}
    for key in ['bddl_files','init_states','assets']:
        folder=paths[key]
        if not folder.is_dir() or not any(folder.iterdir()):
            raise FileNotFoundError(f'LIBERO resource directory is missing or empty: {folder}')
    if suite:
        if not any((paths['init_states']/suite).glob('*.pruned_init')):
            raise FileNotFoundError(f'LIBERO initial states are absent for {suite}')
        if not any((paths['bddl_files']/suite).glob('*.bddl')):
            raise FileNotFoundError(f'LIBERO task definitions are absent for {suite}')
    return {key:str(value) for key,value in paths.items()}
