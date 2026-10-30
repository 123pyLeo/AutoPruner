# 接入自己的模型

开发者只适配模型相关操作；提示词决策、探针计算、搜索记录、恢复调度、评估反馈、选择方案和导出调度由框架复用。完成适配后，使用者通过同一个 `autoprune.run_e2e` 入口启动搜索。

## 先运行完整的接入演示

在仓库根目录执行，无需 GPU 或 API key：

```bash
python -m autoprune.run_e2e --backend example \
  --plugin examples.custom_model:ExamplePlugin --backend-config examples/custom_model.json \
  --task-suite custom_bench --agent rule --recoveries direct adapter_only \
  --prompt "prune custom_bench, SR>=0%" --n-iters 2 --run-root runs
```

配置刻意使用 **6 层、自定义任务名、3 步动作块和 4 维动作**，检查通用流程没有要求 OFT 的 32 层、LIBERO 任务名或固定动作形状。这个示例调用现有 simulator，模型结果和训练都是合成的；`rule` 只枚举当前模型范围内未试过的方案，不使用写死的 OFT 层号或假称已有测量。

命令会打印本次 `run_dir`。然后运行：

```bash
python -m autoprune.finalize --run runs/search_RUN_ID --destination exports/example
python -m autoprune.check_export --exported exports/example
```

示例导出的是明确标为 synthetic 的模拟方案，不是可部署权重。它验证配置、搜索、评估反馈、胜出结果与插件导出之间的接口；不证明第二种真实机器人模型已经接通。`examples/` 是源码 checkout 的参考目录，不安装到核心 wheel。

## 模型相关的四个必需方法

继承 `autoprune.backends.plugin.ModelPlugin`。声明名称、真实层数、任务集、剪层模式和实际支持的恢复方法。构造函数接收 `--backend-config` JSON 对象的字段，因此模型路径、训练设置和数据配置不需要写进框架源码。凭据通过环境变量提供。

| 方法 | 开发者适配什么 | 框架复用什么 |
|---|---|---|
| `load_policy` | 加载指定的 dense checkpoint；保持源文件不变 | 模型句柄复用与探针后释放 |
| `collect_activations` | 在固定输入上采集各层 action-token residual 和对应动作 | 线性、几何、边界可恢复性等探针 |
| `predict_action` | 在相同输入上运行 dense、指定删除集合或单层 identity knockout | 逐层功能扰动计算及搜索工具查询 |
| `recover_and_eval` | 调用该模型的剪层、恢复和闭环评估实现，正确传递 mask、配方、scope 和 seed | 方案调度、错误记录、结果反馈和胜出选择 |

动作是 `(N, chunk, action_dim)`；chunk 和 action_dim 由模型决定。hidden 必须包含 `0..n_layers` 全部边界，每个数组是 `(N, hidden_dim)`，其中 `n_layers` 对应最后一层输出。框架检查键、形状、样本对齐及有限数值；token 选择、动作单位、normalization 和物理语义仍须由开发者核对，不能仅凭形状认定正确。

同一 seed、同一数量的输入必须在 dense、删层和 knockout 之间保持一致。`collect_activations` 接收 suite；加载前框架也设置 `plugin.active_suite`，供 `load_policy`、`predict_action` 和可选 `preflight` 使用。未显式传 `--task-suite` 时，框架匹配插件所声明的任务名；提示词未指明时使用插件默认任务。插件可实现 `runtime_config()` 返回有效配置，它会与原始构造参数一起保存到 `request.json`。

真实评估结果必须包含：

```python
{
    "status": "completed",
    "sr": successes / episodes,
    "successes": successes,
    "episodes": episodes,
    "recipe": recovery,
    "run_dir": str(exact_candidate_directory),
    # 本模型导出所需的精确 checkpoint step / 文件等信息也在这里返回
}
```

框架检查 SR 与计数、状态和实际配方是否一致。插件负责核对所有计划任务、trial 和分片是否完整；任何训练或评估失败应抛出异常。不得从旧目录取另一个模型或补一个旧分数。

## 使用同一个命令导出胜出模型

如果要通过 `autoprune.finalize` 导出，插件再实现一个模型相关方法：

```python
def export_checkpoint(self, *, evaluation, mask, recovery, destination):
    # 从 evaluation 的确切候选目录及保存步骤恢复模型。
    # 使用本架构的 native saver，将可独立加载的模型与所需资源写入 destination。
    # destination 已由框架创建，为空目录；不要写框架保留的 manifest/receipt 文件。
    return "my-model-checkpoint-v1"
```

导出不重新训练，不选择“最新”checkpoint，不把 dense 权重当成剪层结果。框架负责核对 winner、proposal、完成的 evaluation 和配方，重新构造同一个插件配置，并给导出文件生成校验清单及 selection receipt。缺少 native exporter、没有产物或含软链接时明确失败，不套用 OFT 的导出逻辑。

`autoprune.check_export` 对任意插件导出检查资源和校验值；它只验证文件完整性。**动作等价性必须通过该模型的 native loader，在固定相同输入上独立重载验证。** OFT 提供现成的 `autoprune.verify_oft_export`；其他架构不能直接使用 OFT 校验器。

## 在真实模型上启动

将插件安装到当前 Python 环境，或把它所在目录放到 `PYTHONPATH`：

```bash
python -m autoprune.run_e2e --backend my_model \
  --plugin my_package.plugin:MyPlugin --backend-config my_model.json \
  --agent openai --full-ckpt /path/to/checkpoint --task-suite my_suite --dense-sr 0.90 \
  --prompt "prune my_suite, SR>=85%" --run-root runs
python -m autoprune.finalize --run runs/search_RUN_ID --destination exports/my-model
python -m autoprune.check_export --exported exports/my-model
```

`my_model` 必须匹配插件的 `name`。真实模型需要 checkpoint、该评估协议下实测的 dense SR 和模型服务配置；`rule` 不可用于真实模型。指定的插件模块会执行其 Python 代码，和安装普通模型依赖一样，需要使用可信源码。

## 接入正确性的验收顺序

1. dense 动作与原生实现一致；确认 token 位置、预处理、动作单位和样本对应。
2. 检查真实层数、删除层号、单层 knockout 和插件声明的恢复方法。
3. 执行一个真实的小预算候选；核对训练产物、任务范围、状态、成功数和 episode 数。
4. 搜索后通过同一入口导出，用 native loader 独立重载，比较相同输入的动作。
5. 使用正常训练预算和完整评估检验任务效果。训练 loss 和 probe 只用于诊断，不能替代闭环任务结果。

现有证据及边界见 [模型接口复查](MODEL_PLUGIN_REVIEW.md)。模型实现参考：[OFT 完整流程](NEW_SERVER.md)、[StarVLA 组件](../examples/starvla/README.md)、[FastWAM 补丁](../examples/fastwam/README.md)。
