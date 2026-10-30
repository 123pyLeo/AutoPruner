# 剪枝权重

> OFT checkpoint upload is in progress. The code is available; the download link will be enabled after file verification.

本次发布 [OFT Object 剪枝权重](https://huggingface.co/123pyLeo/AutoPruner-OFT-Object)。StarVLA 和 FastWAM 的独立使用包仍在准备，暂不提供下载。

| 权重 | 状态 |
|---|---|
| OFT Object，保留 4/32 层 | 完整物理 checkpoint，包含 adapter、action head、proprio projector、tokenizer 和归一化统计 |
| StarVLA Object，保留 26/36 层 | 参考代码已公开，独立推理包仍在整理 |
| FastWAM Object/Spatial/Goal | 参考补丁已公开，独立推理包仍在整理 |

## 下载 OFT 权重

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('123pyLeo/AutoPruner-OFT-Object', local_dir='checkpoints/oft-object-k28')
PY
```

原始层号按 0 开始计数：删除 2–29，保留 0、1、30、31。恢复组件来自第 10000 步，adapter bottleneck 为 1024。请保留完整文件，不要再次对该 checkpoint 执行相同删层操作。

FastWAM 当前文件仍保存 30 个 video blocks 和 30 个 action blocks。L2–L7 是推理时两类专家共同跳过的层，不是已经从文件中物理删除的层。不能只发一个 `.pt` 文件，省略 mask 后还称它为相同的剪枝模型。Long 使用另一个五层方案，该历史路径下的权重仍未找到。

本轮新做的 9/10 是 OSMesa 下的小规模检查，不替代论文的 OFT Object 463/500，也不等同于独立最终性能验收。其他两类的论文结果仍是历史记录，并未在本轮重跑。

## 运行 OFT 权重

先按 [安装指南](../docs/NEW_SERVER.md) 准备 OFT 和完整 LIBERO 资源。仅使用该权重时，完成安装指南的环境安装步骤即可，跳过 dense 模型、HDF5 和 RLDS 下载步骤。设置：

```bash
export OFT_REPO="$PWD/third_party/openvla-oft"
export OFT_LIBERO_ROOT="$PWD/third_party/LIBERO"
export CUDA_VISIBLE_DEVICES=0
export AUTOPRUNE_TRAIN_GPUS=0
export AUTOPRUNE_EVAL_GPU=0
export AUTOPRUNE_MUJOCO_GL=osmesa
```

然后直接评估物理导出的模型：

```bash
python -m autoprune.eval_oft_export --exported checkpoints/oft-object-k28 \
  --suite libero_object --trials 1 --max-eval-tasks 1 --run-root runs
```

该命令从导出清单读取 mask、adapter 尺寸和 step，不需要原搜索目录、dense 参考模型、LLM API、HDF5 校准演示、RLDS 训练数据或 teacher cache。它仍需要 GPU、匹配的 OFT 环境及 LIBERO 的任务、初始状态和仿真资产。

`--trials 50` 且不设置任务上限，会请求整个 suite 的 500 episodes；必须完整执行才能统计。标准 LIBERO 状态不会因此自动变成与搜索独立的状态。

文件完整性可另查：

```bash
python -m autoprune.check_export --exported checkpoints/oft-object-k28
```

要做导出前后的动作等价性比较，还需要明确的参考恢复模型，使用 `autoprune.verify_oft_export --source ...`。下载后的正常使用和这种作者侧对照检查是不同步骤。

## 三类权重的来源

- OFT 基础权重：[moojink/OpenVLA-OFT Object](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-object)。
- StarVLA 基础模型及动作约定：[Qwen3-VL-OFT-LIBERO-4in1](https://huggingface.co/StarVLA/Qwen3-VL-OFT-LIBERO-4in1)。
- FastWAM 基础模型、所需资产及数据统计：[官方模型卡](https://huggingface.co/yuanty/fastwam)、[固定源码](https://github.com/ZhaoRunyi/fastwam/tree/7ddd3669d8d1d8d914844c153585faf5e2021196)。

源码的 MIT 许可不能替代各基础模型及传递资产的适用条款；分发时保留对应来源和声明。原始训练配置中的本机目录不作为公开使用路径。具体文件大小、当前哈希和未完成项见 [manifest.json](manifest.json)。
