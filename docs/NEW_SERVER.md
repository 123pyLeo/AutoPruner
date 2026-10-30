# 新服务器验证步骤

先确认安装和真实模型的小流程能运行，再启动长时间搜索。OFT 提供完整流程；StarVLA、FastWAM 暂时提供参考组件。

## 1 拉取源码并跑 CPU 演示

```bash
git clone https://github.com/123pyLeo/AutoPruner.git
cd AutoPruner
conda create -n autopruner-oft python=3.10 -y
conda activate autopruner-oft
python -m pip install -e '.[test]'
python -m pytest -q
python -m autoprune.run_e2e --backend mock --agent rule \
  --prompt 'prune libero_object, SR>=40%' --run-root runs
```

mock 是软件演示，其 SR 是模拟值。纯 CPU 环境会跳过需要 torch 的物理导出张量测试。

## 2 安装 OFT 环境

以下脚本安装依赖、生成带自定义补丁的 OFT checkout，并安装固定版本的 LIBERO。不会下载模型、数据，不会启动训练或调用 LLM。请在刚创建的环境中执行：

```bash
bash scripts/setup_oft_env.sh
```

参考环境为 Linux、Python 3.10、PyTorch 2.2.0 CUDA 12.1，以及 OFT 的 Transformers fork。**Transformers 版本号同为 4.40.1 也可能不是同一份代码**，脚本固定了 fork 的 commit，不要再用普通 `pip install transformers` 覆盖。

Flash Attention 需要可用的 `nvcc` 和兼容的 CUDA 编译环境。仅检查 Python 安装时可以设置 `AUTOPRUNE_SKIP_FLASH_ATTN=1` 后执行安装脚本；实际模型运行前仍需确认其 attention 实现所需扩展。参考安装依据见 [OFT 官方安装说明](https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/SETUP.md)和 [PyTorch 历史版本安装说明](https://pytorch.org/get-started/previous-versions/)。

无显示器的 Linux 服务器需要 MuJoCo 的渲染依赖。Ubuntu/Debian 上可由管理员安装 `libosmesa6`、`libgl1`、`libegl1`，然后使用 `AUTOPRUNE_MUJOCO_GL=osmesa`。如果服务器使用 EGL，改为 `egl` 并确认 NVIDIA EGL 驱动可用；不需要复制旧服务器的 userspace 库。

## 3 准备一个模型和 Object 校准数据

如果已有模型与数据，直接填写实际路径即可。没有模型时，以下示例下载上游公开的 dense OFT Object 权重；它不是本论文剪枝后的权重：

```bash
mkdir -p models data/libero_hdf5
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('moojink/openvla-7b-oft-finetuned-libero-object', local_dir='models/oft-object')
PY

python third_party/LIBERO/benchmark_scripts/download_libero_datasets.py \
  --datasets libero_object --download-dir "$PWD/data/libero_hdf5" --use-huggingface
```

确认 `data/libero_hdf5/libero_object/` 中有 `.hdf5` 文件。如果下载器多套了一层目录，按实际解压位置设置 `OFT_HDF5_ROOT`。模型中需有权重、tokenizer、modeling code、`dataset_statistics.json`、action head 和 proprio projector。

```bash
export OFT_REPO="$PWD/third_party/openvla-oft"
export OFT_LIBERO_ROOT="$PWD/third_party/LIBERO"
export OFT_HDF5_ROOT="$PWD/data/libero_hdf5"
export OFT_CHECKPOINT="$PWD/models/oft-object"
export CUDA_VISIBLE_DEVICES=0
export AUTOPRUNE_TRAIN_GPUS=0
export AUTOPRUNE_EVAL_GPU=0
export AUTOPRUNE_MUJOCO_GL=osmesa
```

例如想用物理 2 号卡，只改 `CUDA_VISIBLE_DEVICES=2`，后面两个逻辑编号继续设为 `0`。子进程会保留这个映射。先选择空闲且显存足够的卡。

## 4 检查环境和路径

```bash
python -m autoprune.doctor --checkpoint "$OFT_CHECKPOINT" --output runs/environment.json
```

`status=ready_for_smoke` 表示前置检查通过，不代表模型已经成功执行。该命令不加载模型权重，不调用 LLM。它检查版本、fork、OFT 补丁、模型文件、校准数据和 GPU 可见性。

## 5 跑真实模型的小验证

以下命令加载模型，用两个校准输入生成动作，再 identity-skip 一层计算动作变化。不训练、不调用 LLM：

```bash
python -m autoprune.smoke_oft --checkpoint "$OFT_CHECKPOINT" \
  --samples 2 --knockout-layer 22 --run-root runs
```

通过后再加一个任务、一次 rollout，检查渲染与闭环评测：

```bash
python -m autoprune.smoke_oft --checkpoint "$OFT_CHECKPOINT" \
  --samples 2 --evaluate --trials 1 --run-root runs
```

输入两张相机图像和 proprio 后，动作应为 `(N,8,7)` 且数值有限。执行的模型代码路径和 SHA256 也会保存，用于核对之前发现的缓存代码差异。单次 rollout 成败都不是论文指标；这里先检查环境和链路。

## 6 再准备训练和搜索

前五步不需要 RLDS 训练集。实际恢复还需要 [OFT 使用的 RLDS 数据](https://huggingface.co/datasets/openvla/modified_libero_rlds)，按 [上游 LIBERO 说明](https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/LIBERO.md)下载，然后设置：

```bash
export OFT_DATA_ROOT=/absolute/path/to/modified_libero_rlds
python -m autoprune.doctor --checkpoint "$OFT_CHECKPOINT" \
  --require-training-data --output runs/environment-training.json
```

实际搜索再设置 `OPENAI_API_KEY`、`AGENT_MODEL`（可选 `OPENAI_BASE_URL`），按 README 的真实搜索命令运行。dense SR 必须与选定 checkpoint 和评测协议对应。默认恢复为 10000 步，可能很耗时；先完成小验证再启动。

## 返回哪些记录

如果失败，保留失败命令和错误堆栈，以及 `runs/environment.json`。如果成功，保留 `runs/oft_smoke_*/result.json`；若有 rollout，再保留其 `evaluation` 指向的结果和日志。不要发送 API key 或 `.env` 内容。

## 缓存 v2 更新

旧版本 OFT checkout 和 teacher cache 不能直接用于新格式。请按 [缓存修复说明](../provenance/CACHE_V2_FIX.md) 准备新的 OFT checkout，重新设置 OFT_REPO；框架会在新运行中生成完整 v2 缓存。原数据无需重新下载。

## 7 搜索后导出与数值核对

搜索命令打印本次 `run_dir`。将它原样传给下面的命令，无需手工抄删层编号、恢复目录和 step：

```bash
python -m autoprune.finalize --run runs/search_RUN_ID --destination exports/oft-object
python -m autoprune.verify_oft_export --exported exports/oft-object \
  --suite libero_object --samples 2 --evaluate --run-root runs
```

检查结果要求 `status=passed`，并保存最大动作误差、实际层数和独立重载日志。`--files-only` 只查文件完整性，不能代替动作核对。完整链路的本机验收范围见 [FRAMEWORK_ACCEPTANCE.md](FRAMEWORK_ACCEPTANCE.md)。

如需调整训练步数、batch 或校准数量，使用 README 中的 `--backend-config`，有效值保存在 `request.json`。先用小预算确认流程，再启动完整训练。小预算不能用于判断剪层方法的性能。

LIBERO checkout 必须包含真实的 `assets`、`bddl_files` 和 `init_files`。框架为每次调用写独立配置，不依赖用户主目录里旧的 LIBERO 配置。渲染方式通过 `AUTOPRUNE_MUJOCO_GL` 选择；本次服务器使用 `osmesa`，其他服务器按驱动环境配置。

## 8 自动交付 checkpoint

在真实搜索命令后增加：

```bash
--export-to exports/oft-object --verify-export
```

它会在搜索结束后自动导出达标 winner，并独立重载比较两个固定输入上的动作。结果保存在本次 `delivery.json`：`exported_and_action_verified` 表示导出及动作对照完成；`no_qualifying_checkpoint` 表示没有达标方案，不生成交付模型，命令退出码为 3。失败时保留已完成的搜索和训练记录供排查。

动作对照不等于独立机器人性能验收。正式训练需要完整数据、合适的训练预算和完整闭环评估，不能照搬两步 smoke 的成功门槛。当前没有搜索中断后的自动恢复功能；长任务应使用服务器的可靠作业管理方式，不能承诺“任何模型几天后一定达到目标”。

只使用作者已导出的 OFT 权重时，不需要重新训练、调用 API 或提供原搜索记录，见 [权重使用说明](../models/README.md)。
