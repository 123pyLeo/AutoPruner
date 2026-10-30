# FastWAM 参考补丁

`mot-pruning.patch` 提取本机实验实现中的 MoT 层跳过逻辑，基于 [FastWAM 固定版本](https://github.com/ZhaoRunyi/fastwam/tree/7ddd3669d8d1d8d914844c153585faf5e2021196)，保留上游 MIT 许可。来源、哈希与变更说明见 `../source_manifest.json`。

在该版本的独立 checkout 内先检查，再应用补丁：

```bash
git apply --check /path/to/AutoPruner/examples/fastwam/mot-pruning.patch
git apply /path/to/AutoPruner/examples/fastwam/mot-pruning.patch
```

在加载模型后显式指定要删除的层：

```python
policy.mot.set_layer_mask(drop_video=removed_layers, drop_action=removed_layers)
```

这里的 mask 是要删除的零起始层号。视频和动作专家必须一起跳过这些层，训练 joint forward 拒绝非对称 mask。推理路径保留了独立专家诊断功能。参考补丁已移除从环境变量自动加载 mask 的行为；仅设置旧的 `FASTWAM_PRUNE_LAYERS` 不会生效，训练和评测入口都必须显式调用该 API。

补丁覆盖 joint forward、视频 KV cache 和动作推理三条路径；只在某一条路径跳过层是不够的。它是计算跳过，不会自动缩小 checkpoint，也不包含完整训练入口或物理导出器。

`paper_settings.json` 是历史设置索引，不是可执行配置。已知论文恢复使用 full-MoT，不能替换成旧插件的 action-only 设置。原 trainer 还存在 DeepSpeed 初始化前加载权重的本地修复，不能忽略；但它与当前固定上游差异较大，尚未单独移植和验证，因此没有整份覆盖进本补丁。完整恢复训练需要继续核实这部分。Long 的历史恢复权重仍未在记录路径找到；Goal 的 93.7% 来自两次 500-episode 结果。

当前只完成补丁应用检查，未完成本候选的真实 FastWAM GPU 链路验证；不能使用 `--backend fastwam` 直接调用它。

2026-10-30 补充：已找到并校验共用恢复权重 step 150、配置和数据统计；文件仍保存两个专家各 30 个 blocks，运行时必须显式应用共同 mask。本轮未完成独立推理包验证。权重准备状态见 [模型索引](../../models/README.md)。
