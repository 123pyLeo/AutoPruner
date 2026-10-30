#!/usr/bin/env bash
# Run only after activating a fresh Python 3.10 environment.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python -c 'import sys; assert sys.version_info[:2] == (3, 10), "Activate a fresh Python 3.10 environment first"'
# Older pip releases reject normalized dependency names in current wheel metadata
# and may fall back to source builds against the CUDA-only package index.
python -m pip install --upgrade 'pip>=24.2,<27'
python -m pip install 'setuptools>=68,<81' wheel packaging ninja
python -m pip install 'numpy==1.26.4'
python -m pip install 'torch==2.2.0' 'torchvision==0.17.0' 'torchaudio==2.2.0' --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e .

oft_checkout="$project_root/third_party/openvla-oft"
if [[ -d "$oft_checkout" ]]; then
    python -m autoprune.prepare_oft --destination "$oft_checkout" --verify-only
else
    python -m autoprune.prepare_oft --destination "$oft_checkout"
fi
python -m pip install -r requirements/oft-reference.txt
python -m pip install -e "$oft_checkout" --no-deps

libero_checkout="$project_root/third_party/LIBERO"
libero_commit=8f1084e3132a39270c3a13ebe37270a43ece2a01
if [[ ! -e "$libero_checkout" ]]; then
    git init -q "$libero_checkout"
    git -C "$libero_checkout" fetch --depth=1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "$libero_commit"
    git -C "$libero_checkout" checkout --detach FETCH_HEAD
fi
actual_commit="$(git -C "$libero_checkout" rev-parse HEAD)"
if [[ "$actual_commit" != "$libero_commit" ]]; then
    echo 'Existing LIBERO checkout differs from the pinned version; choose a fresh directory.' >&2
    exit 2
fi
python -m pip install -e "$libero_checkout" --no-deps

if [[ "${AUTOPRUNE_SKIP_FLASH_ATTN:-0}" != 1 ]]; then
    if ! command -v nvcc >/dev/null 2>&1; then
        echo 'Flash Attention needs nvcc. Install a matching CUDA toolkit or set AUTOPRUNE_SKIP_FLASH_ATTN=1 for import diagnostics only.' >&2
        exit 2
    fi
    # flash-attn's prebuilt-wheel installer uses rename(); a persistent cache on
    # another filesystem can cause EXDEV. Keep its temporary build/cache local.
    python -m pip install --no-cache-dir 'flash-attn==2.5.5' --no-build-isolation
fi
python -m pip check
# Dependency metadata can be consistent while generated TFDS protos cannot
# import. Exercise the actual OFT import closure before declaring installation.
python -c 'import tensorflow_datasets; import dlimp; from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig'
python -m pytest -q
echo 'Environment installed. Follow docs/NEW_SERVER.md to download data and run a real model smoke check.'
