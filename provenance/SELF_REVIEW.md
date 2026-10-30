# 开源候选二次自查

后续完整框架的小预算实跑已补充，见 [FRAMEWORK_ACCEPTANCE.md](../docs/FRAMEWORK_ACCEPTANCE.md)。下文保留该次检查的范围与历史记录。

2026-10-24 更新：下文记录的是修复前的问题。缓存 v2 已修复并验证，当前状态及使用方式见 [CACHE_V2_FIX.md](CACHE_V2_FIX.md)。

结论：此前“整理完成”的说法过早。此前 31 项测试只覆盖部分软件行为，没有覆盖下面这些实际问题。目前应保持私有验证状态，不能据此宣布论文复现或正式发布验收完成。

## 已发现并修复

| 问题 | 实际影响 | 本轮处理与证据 |
|---|---|---|
| 相对路径跨进程失效 | 默认 `runs/` 和本地 checkpoint 路径进入以 OFT checkout 为工作目录的子进程后，可能指错目录；恢复软链接也受影响 | 运行目录、staging、模型、数据、初始状态路径转绝对路径；切换工作目录的回归测试 |
| probe 未绑定所选 suite | 多套统计共存时，原加载器默认 Object；只有一套但不属于请求 suite 时还会自动接受 | backend 将明确 suite 交给 OFT 加载器；找不到对应统计即失败；多套/错套统计的 resolver 回归测试。真实模型仍需对照 |
| 缺失任务语言被替换为固定指令 | HDF5/RLDS 数据损坏时可能用错误任务做 probe，流程仍能运行 | 拒绝缺失、空和无效语言；两个 HDF5 读取路径共用检查 |
| NaN 成功率能被判达标 | 已在 CPU 复现 `sr=NaN` 导致 `winner_meets_spec=True` | selector 和 frontier 拒绝非有限及越界 SR |
| 自定义真实插件结果不校验 | 插件可以只返回 SR，甚至失败状态携带 SR，进入搜索 | 真实结果必须完成、具有有效成功/episode 数，且 SR 与计数一致；无效结果记录为失败。完整评测预算仍由插件执行检查 |
| GPT agent 借用其他服务凭据 | 原历史网关兼容逻辑会自动读取 Anthropic 的地址和 token，还强制绕过系统代理 | 只读取显式 OpenAI 配置，恢复标准代理行为；无网络的凭据隔离测试 |
| `AUTOPRUNE_DRY_RECOVERY=0` 仍跳过恢复 | 使用字符串存在性判断，字符串 `0` 也为真 | 只接受 `0`/`1`；测试确认 `0` 执行、`1` 不执行 |
| 新模型继承 OFT 剪层说明 | skip 模型仍读到连续剪层约束；小层数模型没有适用 k 选项 | 根据插件深度和模式生成并公开搜索约束；三层 skip 模型测试 |
| 恢复目录和 direct 结果路径不够严格 | 直接使用 backend 时可能共享默认目录；direct 报告指向评测目录而非模型 | 默认目录及每次恢复使用独立标识；direct 记录实际 checkpoint；软链接失败不再吞掉 |
| 校准资源清理与无效索引 | 缺少 HDF5 时可能留下 hooks；错误发生时文件未按上下文关闭；越界 knockout 被静默忽略 | 安装 hooks 前检查输入；文件上下文关闭；越界显式报错 |
| 来源清单的 release 哈希已过期 | 后续修改后，`source_manifest.json` 中的 release hash 未同步，无法据此核对候选 | 更新当前 release 哈希，保留原始 source 哈希；与最终文件清单一起核验 |

复现前的结果和 teacher-cache 最小实验见 `review_evidence.json`。新测试位于 `tests/test_release_review.py`。这些测试验证具体工程故障，不能代替 GPU 动作和闭环验收。

## 仍然影响正式发布的缺口

1. **Teacher cache 样本身份不足。** 当前历史实现只对去 padding 的 `input_ids` 做哈希，不包含视觉或 proprio 状态。同一 token 序列对应不同观测时无法区分。最小实验确认第二份不同 teacher 输出不会写入；后续已全量检查 261,614 个训练样本：14 个样本集中在 3 个共用输出组，30 个未命中旧缓存，详见 [真实数据报告](CACHE_DATA_AUDIT.md)。这不是大面积错配，也不能据此推翻论文指标。生成 cache 不增强图像，训练开启增强，两者的监督对应关系也没有验收。保持历史行为便于追溯，不能因此声明其正确；本轮没有凭猜测更改蒸馏方法。
2. **真实数值对照没有归档。** probe 的 HDF5、RLDS 与 rollout 预处理链路并非同一实现，独立服务器“跑通”没有提供同输入动作误差、归一化和 action-span 的对照记录。suite 修复后也需要重新做多套统计检查。CPU stub 和 import 检查不能证明它们等价。
3. **论文协议证据尚未闭合。** 搜索与最终评测的初始状态独立性、部分历史训练参数、FastWAM Long 权重等问题仍在 STATUS 中；不能靠本轮代码修复补写历史事实。
4. **GitHub 整库公开范围没有完成审计。** 已核实该私有仓库还包含 `main`、`cyk`、`xie_zinan_cutoff` 等分支；干净验证分支并不意味着所有历史提交和分支都已清理。此轮没有修改可见性、删除分支或重写历史。旧明文凭据的更换仍须确认。
5. **远程 CI 未证明通过。** 之前 GitHub Actions 在创建 job 前启动失败；本机测试通过不等于远程 CI 成功。本轮也未重新进行 GPU 训练或 rollout。

## 下一步需要的证据

真实训练数据上的比例和实际输入对照已完成，见 [CACHE_DATA_AUDIT.md](CACHE_DATA_AUDIT.md)。下一步应比较缓存输出与同输入在线 teacher 输出。区分原始样本身份、增强策略和 padding 影响；确认后再选择最小修改，不能直接换一种 cache key 就宣告修好。验收必须包含监督匹配、cache hit/miss 记录以及恢复后的闭环行为。

随后归档 dense / probe / masked / physical-export 在固定输入上的动作对照和完整评测计数。正式公开应只包含经过审计的提交历史与分支；现有私有仓库暂不直接改为 public。

## 检查依据

检查了候选入口、backend、probe 采样、恢复调度、结果选择、导出、打包和来源清单，并对照固定上游及现有自定义 overlay。上游参考：[OFT 评测实现](https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/experiments/robot/libero/run_libero_eval.py)。现有公开实现没有提供本地自定义 teacher-cache 的正确性证明，因此这部分仍需用本地数据实测。
