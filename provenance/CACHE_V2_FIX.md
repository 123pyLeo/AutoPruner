# Teacher cache v2 修复

后续完整框架的小预算实跑已补充，见 [FRAMEWORK_ACCEPTANCE.md](../docs/FRAMEWORK_ACCEPTANCE.md)。下文保留该次检查的范围与历史记录。

缓存匹配代码已修复，并做了真实数据和小规模 GPU 验证。全量新缓存和修复后的完整恢复评测尚未执行；旧缓存及历史权重保持原样。

## 改了什么

1. 从 TFDS 的官方 `tfds_id` 取得原始记录身份，再结合数据集名称、版本和原始 frame index。该编号在打乱顺序、分批、chunk 和图像增强前确定，一路传到 PyTorch collator。
2. 新缓存键同时包含这个样本编号和去 padding 的动作 token。即使动作完全相同，两个时间点也不再共用答案。
3. 缓存生成改为单进程、有限遍历整套数据；完整样本数核对通过后才写入完成标记。删除“连续若干批没新键就提前结束”的旧逻辑。达到安全步数上限或数量不符会失败，不发布不完整缓存。
4. 训练遇到缺失的样本直接报错，不再静默省略其 teacher KL。旧 token-only 缓存明确拒绝加载，不能直接改名或转换成新格式。
5. 保留 teacher 原始视图、student 增强视图的既有训练设置。顺带修复 collator 在 batch size 为 1 时把 proprio 的 batch 维 squeeze 掉的问题。

实际修改在 OFT overlay 中：RLDS 身份适配、数据变换、collator、缓存生成器、缓存读写及两条训练查找路径。框架入口也检查 v2 格式和完整性。生成器不再允许随机初始化或含糊选择 teacher sidecar。

## 验证结果

| 检查 | 结果 |
|---|---|
| 四套真实数据的所有 261,614 个动作块 | 261,614 个不同样本编号；没有冲突 |
| 打乱记录顺序 | 抽查 5,074 个样本，编号仍与完整集合对应 |
| 实际数据管线的图像增强 | 图像变化，样本编号和动作 token 不变；collator 保留编号 |
| 原来共用答案的 14 个真实样本 | 形成 14 个 v2 键；保存、重载、反序和 padding 后匹配正确 |
| GPU 真模型重新生成这些样本 | Object 9 个、Long 5 个；每个原冲突组都得到不同的真实 teacher 输出，读取与各自保存值完全一致 |
| 原生 Phase-2 训练函数 | 两个真实样本完成前向及反向；缓存 2/2 命中，loss 与 action-head 梯度有限；未执行 optimizer step |
| 安装补丁 | 在新的固定上游 checkout 上应用、核对 94 个文件；CPU 回归和 wheel 检查另记录在 validation.json |

训练函数检查使用 dense student 对照，仅用于验证新的读取路径，不冒充剪层恢复训练。真模型生成调用与完整 CLI 相同的 `cache_batch` 函数。GPU 结果不等于论文成功率重测。

详细数值见 [cache_v2_validation.json](cache_v2_validation.json)。其中诊断 marker 验证、实际 teacher 输出验证和原生训练函数检查分别记录。

## 使用新版本

已有 OFT checkout 包含旧补丁时，准备一个新目录：

```bash
python -m autoprune.prepare_oft --destination third_party/openvla-oft-v2
python -m autoprune.prepare_oft --destination third_party/openvla-oft-v2 --verify-only
python -m pip install -e third_party/openvla-oft-v2 --no-deps
export OFT_REPO=/absolute/path/to/third_party/openvla-oft-v2
```

在模型环境中安装该 checkout，继续按 `docs/NEW_SERVER.md` 设置数据和 checkpoint 路径，使用原来的 `autoprune.run_e2e` 搜索入口。它在本次运行的独立目录生成 v2 缓存，然后再训练；生成全部 teacher 输出仍有计算成本。

不需要重下载数据，不会覆盖旧缓存。旧缓存没有保存完整样本身份，无法可靠补上那 14 个观测各自的真实输出，因此新格式需要重新运行 teacher。全量缓存生成和恢复训练会在正常新运行时执行，不能用本次 14 个样本的诊断缓存替代。

## 依据和适用条件

查阅了 [TFDS ReadConfig](https://www.tensorflow.org/datasets/api_docs/python/tfds/ReadConfig) 的 `add_tfds_id`，并检查所固定 dlimp 的 `from_rlds` 与 metadata 广播实现。dlimp 自带的 `_traj_index` 来自迭代顺序，不能在 shuffle 后作为稳定身份；采用官方记录 ID 加原始 `_frame_index`，不重写 dlimp。

`tfds_id` 的稳定性限定在同一数据快照和依赖版本内。TFDS 已固定版本；修改数据文件、tokenizer、归一化或预处理后必须重新生成缓存，不复用旧目录。完整数量公式针对当前四套 LIBERO 数据：原始 transition 数减去每条演示尾部不能组成 8 步 chunk 的 7 帧，已与全量实际读取核对。生成器明确限制为这四套数据，不向未知数据集套用该公式。

引入的最小假设是官方 ID 和原始 frame index 经现有 transforms 后保持对应关系；全量唯一性、shuffle 和增强检查验证了这个假设。匹配正确性已验证，修复对机器人表现的影响仍需完整恢复与闭环实验判断。
