# AutoPruner

**Agentic Depth Pruning for Vision-Language-Action Models**

[Paper](https://arxiv.org/abs/2610.23880) · [OFT checkpoint](https://huggingface.co/123pyLeo/AutoPruner-OFT-Object) · [中文安装指南](docs/NEW_SERVER.md)

AutoPruner uses an LLM agent to search for layer deletions and recovery strategies in VLA models. It measures layer redundancy, tries pruning candidates, runs recovery training, and uses robot task evaluations to guide the next iteration. The selected model is exported as a checkpoint.

This release includes the pruning framework, a complete **OpenVLA-OFT** integration, and an OFT Object checkpoint with **4 of 32 decoder layers retained**. StarVLA and FastWAM pruning components are included as references; their standalone model packages are still being prepared.

## Installation

For the framework and CPU demo, use Python 3.10 or newer:

```bash
git clone https://github.com/123pyLeo/AutoPruner.git
cd AutoPruner
python -m pip install -e .
```

Run the CPU demo:

```bash
python -m autoprune.run_e2e --backend mock --agent rule \
  --prompt "prune libero_object, SR>=40%" --run-root runs
```

The demo uses a simulated model and a scripted agent. It checks the software flow and does not measure robot performance.

Real OFT runs require Linux, an NVIDIA GPU, Python 3.10, a CUDA toolkit, and LIBERO simulation assets. Create a fresh environment, then install the pinned dependencies and model code:

```bash
conda create -n autopruner-oft python=3.10 -y
conda activate autopruner-oft
bash scripts/setup_oft_env.sh
```

The script prepares OFT and LIBERO in `third_party/`. It does not download weights or datasets. Headless servers also need MuJoCo rendering libraries. Follow the [installation guide](docs/NEW_SERVER.md) for system dependencies, downloads, and checks. Keep the pinned Transformers fork; a stock package with the same version number is not interchangeable.

## Use the released OFT checkpoint

The [OFT Object checkpoint](https://huggingface.co/123pyLeo/AutoPruner-OFT-Object) includes the physically pruned model, boundary adapter, action head, proprio projector, tokenizer, and normalization statistics. It retains original decoder layers 0, 1, 30, and 31.

After installing the OFT environment:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "123pyLeo/AutoPruner-OFT-Object",
    local_dir="checkpoints/oft-object-k28",
)
PY

export OFT_REPO="$PWD/third_party/openvla-oft"
export OFT_LIBERO_ROOT="$PWD/third_party/LIBERO"
export CUDA_VISIBLE_DEVICES=0
export AUTOPRUNE_TRAIN_GPUS=0
export AUTOPRUNE_EVAL_GPU=0
export AUTOPRUNE_MUJOCO_GL=osmesa

python -m autoprune.check_export --exported checkpoints/oft-object-k28
python -m autoprune.eval_oft_export --exported checkpoints/oft-object-k28 \
  --suite libero_object --trials 1 --max-eval-tasks 1 --run-root runs
```

This runs one task episode. It needs the complete LIBERO simulation assets, but no LLM API, training dataset, calibration demos, original search directory, or dense model. Remove `--max-eval-tasks 1` and use `--trials 50` to request the full 500-episode Object suite. See [checkpoint details](models/README.md).

## Prune an OFT model

Prepare a compatible dense OFT checkpoint, LIBERO HDF5 calibration demos, and the RLDS recovery-training dataset. [The setup guide](docs/NEW_SERVER.md) provides downloads and a small model check before starting a long run.

Configure local paths and the agent's model service:

```bash
export OFT_REPO="$PWD/third_party/openvla-oft"
export OFT_LIBERO_ROOT="$PWD/third_party/LIBERO"
export OFT_HDF5_ROOT=/absolute/path/to/libero_hdf5
export OFT_DATA_ROOT=/absolute/path/to/modified_libero_rlds
export OFT_CHECKPOINT=/absolute/path/to/dense-oft-checkpoint
export CUDA_VISIBLE_DEVICES=0
export AUTOPRUNE_TRAIN_GPUS=0
export AUTOPRUNE_EVAL_GPU=0
export AUTOPRUNE_MUJOCO_GL=osmesa

read -rsp "API key: " OPENAI_API_KEY; echo
export OPENAI_API_KEY
export AGENT_MODEL=your-api-model-name
# For an OpenAI-compatible service, set its base URL, ending in /v1:
# export OPENAI_BASE_URL=https://your-provider.example/v1
```

Credentials belong in your environment. `.env.example` documents the settings; the package does not automatically load `.env`.

Check the environment, then start search:

```bash
python -m autoprune.doctor --checkpoint "$OFT_CHECKPOINT" \
  --require-training-data --output runs/environment.json

python -m autoprune.run_e2e --backend oft --agent openai \
  --task-suite libero_object --full-ckpt "$OFT_CHECKPOINT" \
  --dense-sr 0.982 --num-trials 20 --n-iters 4 \
  --prompt "Prune libero_object while keeping success rate at least 90%" \
  --run-root runs --export-to exports/oft-object --verify-export
```

Replace `0.982` with the measured dense success rate for your checkpoint and evaluation protocol. The example value is the paper's Object baseline. A small installation check can use `--max-eval-tasks 1 --num-trials 1`; such a run is not a benchmark.

The agent probes the model, proposes deleted layers and a recovery strategy, and receives evaluation feedback. If a candidate meets the requested constraints, the command exports the selected checkpoint and checks its actions after an independent reload. The destination must be a new directory.

Each run writes a separate `runs/search_*` directory. Start with `request.json` for effective settings and parsed constraints, and `delivery.json` for the export result. Search results and candidate evaluations remain in that run directory.

`delivery.json` reports `exported_and_action_verified` when the automatic OFT export and action comparison pass. If no candidate qualifies, it reports `no_qualifying_checkpoint`, exports no model, and exits with code 3. Completed search records are retained if export fails.

### Search settings

Pass model settings with `--backend-config oft.json`, for example:

```json
{"train_max_steps": 10000, "global_batch": 64, "adapter_bottleneck": 1024, "calib_samples": 64, "shuffle_buffer_size": 100000}
```

Pass API options such as `{"max_tokens": 4096, "timeout": 180}` with `--agent-config agent.json`. Use `--recoveries direct` to disable recovery training, or `--recoveries adapter+head+proprio` to restrict the search to joint recovery. Omitting this option exposes OFT's supported recovery menu.

Use explicit acceptance constraints, such as `SR>=90%`. The command prints the parsed constraints before search and records them in `request.json`. Search does not guarantee a target score or a fixed completion time. Long searches currently have no automatic interruption/resume support; use your server's job manager to keep them running.

### Export an existing search result

If search was run without `--export-to`, export its qualifying winner later:

```bash
python -m autoprune.finalize --run runs/search_RUN_ID --destination exports/oft-object
python -m autoprune.verify_oft_export --exported exports/oft-object \
  --suite libero_object --samples 2 --evaluate --run-root runs
```

The exporter uses the recorded deletion mask, recovery recipe, and exact saved training step. The verifier compares actions before and after export in separate processes. That comparison needs the reference recovered model and calibration inputs. Normal use of a downloaded checkpoint uses `eval_oft_export` instead.

## Other VLA models

The framework separates the search method from model operations. A new architecture needs a `ModelPlugin` for loading/probing, pruning, recovery, and evaluation. Add a native exporter to use the same automatic checkpoint delivery. An arbitrary VLA weight file cannot run without this integration.

See [Adding a model](docs/ADDING_A_MODEL.md) and the executable synthetic example in [examples/custom_model.py](examples/custom_model.py). Custom plugins receive constructor settings through `--backend-config`.

| Integration | Included here |
|---|---|
| OpenVLA-OFT | Complete search, recovery, LIBERO evaluation, physical export, and released Object weights |
| [StarVLA](examples/starvla/README.md) | Pruning and boundary-adapter reference components; standalone package in progress |
| [FastWAM](examples/fastwam/README.md) | MoT pruning patch and recorded settings; standalone package in progress |

## Evaluation and tests

The paper reports **463/500** successes for the released OFT Object pruning configuration. Release checks verified identical actions before and after physical export on two inputs, 9/10 task episodes in a bounded check, and a separate 1/1 check in a fresh GPU environment. These checks do not replace the paper's full evaluation. The fresh-environment API-to-checkpoint run used two recovery steps and tested the execution flow, not recovered performance.

The default LIBERO initial-state pool is not claimed to be held out from search. A separately validated pool can be supplied through `OFT_INITIAL_STATES_PATH`. See [validation status](STATUS.md) for evidence and remaining limitations.

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

There are 106 passing tests in the validated GPU environment. CPU-only installs skip tests that require model dependencies. Logs, caches, and checkpoints are generated locally and are excluded from the source release. Teacher caches use stable sample identities; legacy caches are rejected rather than silently reused.

## Citation

```bibtex
@article{lin2026autopruner,
  title={AutoPruner: Agentic Depth Pruning for Vision-Language-Action Models},
  author={Lin, Tao and Chen, Yikeng and Liang, Qiwei and Xie, Zinan and Liang, Zhengtu and Xu, Renjing},
  journal={arXiv preprint arXiv:2610.23880},
  year={2026}
}
```

## License and acknowledgments

The framework is released under the [MIT License](LICENSE). We build on [OpenVLA-OFT](https://github.com/moojink/openvla-oft), [StarVLA](https://github.com/starVLA/starVLA), [FastWAM](https://github.com/ZhaoRunyi/fastwam), and [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO). Model weights, datasets, and external assets retain their applicable terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for included code and attribution.
