# Teacher cache 真实数据检查

2026-10-24 更新：下文记录的是修复前的问题。缓存 v2 已修复并验证，当前状态及使用方式见 [CACHE_V2_FIX.md](CACHE_V2_FIX.md)。

这次查了现有四套 RLDS 训练数据的全部记录，不再只凭代码推测。结论是：**确有少量不同观测共用缓存输出，但没有发现大面积错配，也没有证据据此推翻论文成功率。**

## 全量结果

| 任务 | 演示条数 | 训练动作块数 | 涉及共用输出的样本数 | 没有旧缓存的样本数 |
|---|---:|---:|---:|---:|
| Object | 454 | 63,806 | 9，集中在 1 组 | 0 |
| Spatial | 432 | 49,946 | 0 | 10 |
| Goal | 428 | 49,046 | 0 | 4 |
| Long | 379 | 98,816 | 5，集中在 2 组 | 16 |
| 合计 | 1,693 | 261,614 | 14，集中在 3 组 | 30 |

14 个样本占约 **0.0054%**。这里统计的是“处于共用输出组中的样本”，不等于已经证明 14 个 teacher 答案都不正确，也不等于机器人任务失败率。

## 如何确认不是检查脚本算错了

1. 读取原始 TFRecord 全部训练演示，使用候选固定实现的真实 `normalize_action_and_proprio`、`ActionTokenizer`、`PurePromptBuilder` 和本机原 tokenizer，还原完整 8 步动作块及未 padding 的 int64 token 字节。
2. 保留原实现中 NumPy 统计数组和 TensorFlow 运算的 dtype 行为。最初尝试用纯 NumPy float32 手工还原归一化会制造假的 cache miss，已排除该近似，正式结果不使用它。
3. 与四套旧 cache 的全部键核对：261,584 个样本命中，30 个未命中。Object 的 63,798 个唯一键与旧 cache 全部对应。
4. 对发现的 3 组共 14 个样本，再调用实际 `RLDSBatchTransform` 核对 token；经真实图像 resize、image processor 和 BF16 转换后，图像和 proprio 仍分别不同。调用实际 `lookup_batch`，确认各组确实得到同一份保存的 teacher logits。

机器可读结果在 [cache_data_audit.json](cache_data_audit.json)。其中保留 TFRecord 文件、记录和时间点位置，以及输入哈希；没有公开原始图像、模型或 teacher logits。

## 问题意味着什么

缓存键包含语言和离散动作块，却没有独立的观测身份。真实数据中，Object 有一段连续 9 个时间点的动作块 token 相同，画面和状态却在变化；Long 也出现两组类似情况。因此缓存无法分辨这些输入。这是实际存在的实现缺陷，后续版本应使用稳定的原始样本身份，并独立记录 tokenizer、数据及预处理版本。

30 个没有旧缓存的样本也已确认。当前训练代码对未命中样本不计算 cached-teacher KL，演示动作损失仍然参与；不是整批训练被丢弃。未在本轮判定这些缺失的历史原因，不能将其自动归因于哪一次运行。

另一个需要说准确的地方是“teacher 不增强、student 增强”：这本身不等于错配。已有蒸馏方法会明确采用不同视图，例如 [DINO 官方实现](https://github.com/facebookresearch/dino/blob/main/main_dino.py) 中 teacher 使用 global views、student 使用多个 views。这个先例只说明不能把视图不同直接判成 bug；并不能证明本项目这套增强对机器人行为有效。这里需要明确训练约定和闭环实验，而不是擅自关掉增强来改变历史方法。

## 本轮没有证明什么

没有重新运行大模型 teacher，也没有测量旧缓存输出与在线 teacher 的误差，更没有重新训练或评测机器人成功率。检查时各 GPU 空闲显存约 1.2–13.9 GiB，并且有其他进程运行，因此没有启动大模型争抢显存。

这些结果针对现存数据和现存旧缓存。论文每个训练任务实际使用哪份缓存、对应什么 checkpoint，还需要和历史运行记录绑定，不能自动等同。

本轮只读源数据和缓存，未改动实验仓库、旧权重或训练方法。要修复缓存身份并重新生成新 cache，应保持独立版本，保留旧记录；之后在固定输入上比较 teacher 输出，再做恢复和闭环验证，才能回答性能影响。

## 复查命令

在 OFT 的 Python 环境、安装本框架后运行：

```bash
python scripts/audit_teacher_cache.py \
  --oft-repo /path/to/prepared/openvla-oft \
  --data-root /path/to/modified_libero_rlds \
  --checkpoint-root /path/to/oft/checkpoints \
  --cache-root /path/to/teacher_cache_without_image_aug \
  --output-dir runs/cache-audit
```

默认扫描所有四套数据。需要标准的 `libero_*_no_noops/1.0.0` 数据目录和 `openvla-7b-oft-finetuned-libero-*` checkpoint 名称。输出目录必须是新目录。`--per-shard 2` 只做抽样，不能替代上面的全量检查。此入口做键与原始观测检查；实际模型输入及 cache lookup 的 14 个样本对照结果另记录在 JSON 的 `actual_input_witnesses` 中。

源实现参考：[OFT 数据变换](https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/prismatic/vla/datasets/datasets.py)。本地自定义 teacher-cache 不属于该上游提供的正确性保证。
