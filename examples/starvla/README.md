# StarVLA 参考组件

本目录提取实验实现的 `layer_pruning.py`、`boundary_adapter.py` 和导出接口，保留 MIT 许可。来源及哈希见 `../source_manifest.json`。这些文件不自动注册为 `--backend starvla`，也不安装进核心包。

在匹配的 StarVLA/QwenOFT 环境中，可从这些组件开始接入：

```python
from examples.starvla import apply_layer_mask, install_adapters

backbone = framework.qwen_vl_interface.model
info = apply_layer_mask(backbone, remove_layers)
# install_adapters 的参数、原始层号与边界对应规则见 boundary_adapter.py。
```

这会修改模型结构；使用新加载的模型操作。必须在剪层前记录原始层数，adapter 的位置使用原始删除集合计算，不能拿重编号后的删除集合再次剪层。调用方负责恢复权重、动作头及归一化配置的匹配。

`paper_settings.json` 记录已核实的删除集合和部分训练设置，不能直接当作完整训练配置。Long 的示范恢复与其他任务的 teacher-guided 方案不同。此目录未包含完整历史 trainer、教师数据路径或评测服务，也未完成候选 GPU 验证；完整复现仍需补齐这些部分。所查上游为 [StarVLA 固定版本](https://github.com/starVLA/starVLA/tree/985c2238627481f25476fbee478083812dd82a21)。

2026-10-30 补充：原环境 checkpoint 完整重载和单输入预测已检查；固定公开上游加这些组件也能完整加载，但同输入动作与原实现有约 0.002518 的差异，尚未通过数值等价验收。不能仅凭严格参数匹配称公开推理包已完成。详见 [本轮验收](../../docs/OPEN_SOURCE_REVIEW.md)。
