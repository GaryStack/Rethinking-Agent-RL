<div align="center">

<h1>Rethinking Agentic RL</h1>

<p><strong>多轮强化学习究竟改善了 LLM 智能体的什么能力？</strong></p>
<p>在不同交互预算下评测，诊断决策与错误恢复，测试 BudgetMix RL。</p>

<p>
  <a href="#研究概览">研究概览</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="#诊断工具">诊断工具</a> ·
  <a href="paper/README.md">方法覆盖</a> ·
  <a href="README.md">English</a>
</p>

<p>Python 3.11+ · 核心仅依赖标准库 · MIT · 开发预览版</p>

</div>

![概念图：从采样次数 K 和交互轮数 T 出发，诊断同状态决策质量、错误恢复与训练动态，再测试 BudgetMix 和过程奖励。图中不包含实验数值。](docs/assets/research-overview.svg)

## 研究概览

最终成功率提高之后，还有一个问题：**智能体每一步做得更好了，还是更擅长在更多轮交互中纠错？**
我们从采样广度 **K** 和交互深度 **T** 两个维度研究多轮 agentic RL。
**Pass@(K,T)** 衡量：采样 K 条轨迹，其中至少一条在 T 轮交互内成功的概率。

配套研究关注三个发现：

- **收益依赖预算。** 交互预算收紧时，轨迹级优势可能缩小，甚至反转。
- **单步改善并不均匀。** RL 的优势更集中在错误恢复和长轨迹的后期阶段。
- **局部行为与整条轨迹可能不同步。** Likelihood 分析和 checkpoint 监测帮助解释：局部合理的动作如何组合成成功轨迹。

由此引出 **BudgetMix RL**：在训练任务组之间改变交互预算，并进一步探索过程监督信号。

本仓库提供 **Agent RL Diagnostics** 包（命令行入口 `ard`），包含方法实现、可复用诊断工具和小型可运行示例。
上述发现属于研究背景；仓库中的示例和工程验证没有重新复现论文 benchmark 数值。
详见[方法覆盖](paper/coverage.md)。

## 快速开始

在源码仓库根目录运行，要求 Python 3.11 或更新版本：

```bash
git clone https://github.com/GaryStack/Rethinking-Agent-RL.git
cd Rethinking-Agent-RL
python -m pip install -e .
ard doctor --json
ard toolkit-demo --output runs/toolkit-walkthrough
```

打开 `runs/toolkit-walkthrough/index.html`，查看五份对比报告：预算下的成功率、
同状态单步质量、错误状态单步质量、续跑恢复，以及固定 Base 池对比。
底层运行记录经过审计后才生成报告；示例还会生成一份 BudgetMix 采样计划。

示例使用**脚本策略**，Base/RL 标签不代表训练后的模型。安装完成后无需 GPU、模型权重或网络。
每次运行请使用新的输出目录。这个演示不验证训练 checkpoint，详见 [CPU 完整教程](docs/tutorials/offline-toolkit.md)。

### 选择你的使用入口

| 你的目标 | 从这里开始 |
|---|---|
| 评测自己的 Base/RL checkpoint | [模型 quickstart](docs/quickstart.md)：HTTP chat 服务，或可选的本地 Transformers backend |
| 比较已有 Agent 版本 | [JSONL 日志教程](docs/tutorials/external-traces.md)：同任务成功率、token/耗时/费用覆盖、原始记录和可选 CI 回归门槛 |
| 接入 Inspect 或工具智能体 | [Inspect 教程](docs/tutorials/inspect-agent.md)：原生 `.eval`/JSON 导入和可执行 SQLite 工具示例 |
| 研究或扩展方法 | [论文配套代码](paper/README.md)、[评测协议](docs/protocols.md)与[自定义 scorer](docs/extensions.md) |

核心评测和 JSONL 分析仅依赖 Python 标准库；本地 Transformers 推理、Inspect 和训练接入均为可选依赖。

## 诊断工具

| 工具 | 回答什么问题 | 入口与教程 |
|---|---|---|
| **Tight-budget evaluation** | 成功率如何随采样次数 K 和交互预算 T 变化？ | `ard eval budget`、`ard compare` · [教程](docs/tutorials/tight-budget.md) |
| **Error-prefix evaluation** | 在相同状态和剩余预算下，哪个策略决策更好、更容易恢复？ | `ard pool errors`、`ard eval step`、`ard eval recovery` · [教程](docs/tutorials/error-prefix.md) |
| **Training dynamics** | 不同 checkpoint 的轨迹成功率和固定状态决策质量是否同步改善？ | `ard eval checkpoint`、`ard dynamics` · [教程](docs/tutorials/training-dynamics.md) |
| **BudgetMix RL** | 训练任务组使用不同且对模型可见的交互预算，会发生什么？ | `ard budgetmix plan/audit`、可选的锁定 VeRL adapter · [教程](docs/tutorials/budgetmix.md) |

另外提供两条方法分析与测试路径：

- **Likelihood / NLL / LLR：** teacher-forced response likelihood、动作 token NLL、对齐后的 Base/RL 对数似然比和长度控制。[输入与命令](docs/likelihood.md)。
- **Process signals：** oracle/judge 评分、版本化缓存、显式奖励混合，以及可运行的奖励到参数更新示例。[方法与训练接入](docs/process-signals.md)。

对比复用冻结输入和已记录的协议；缺失样本、基础设施失败保持可见，不会被悄悄改成任务失败或成功。
历史日志导入保留调用方报告的结果，不代表受控错误恢复实验，也不独立认证任务成功。

## 支持范围与验证

当前预览版：**`0.1.0.dev11`**。

| 项目 | 当前范围 |
|---|---|
| 可执行环境 | 确定性 FrozenLake 和派生 stdin/stdout coding；coding 需要受限 Linux namespace executor |
| 外部数据 | 通用 JSONL 和可选 Inspect 日志；BFCL、SQL、Terminal 仅作外部分析输入，未打包完整 benchmark 环境 |
| 训练接入 | 锁定 VeRL 版本的有限单 rank 验收；完整 PPO/GiGPO 训练、多 rank 和 optimizer resume 尚未验收 |
| 论文复现 | 方法代码已提供，完整 benchmark 数值复现尚未验证 |

最近一轮 CPU 回归记录为 **333 通过、4 跳过**；独立的可选 Inspect 验收记录为 **27 通过**。
真实 **Qwen3-1.7B** 单 rank Ray/vLLM/FSDP 工程验证包含 BudgetMix 和两步 process-enabled REINFORCE++ 更新。
这些记录验证指定执行路径，不代表方法效果。已保留 seeded diagnostic off/on 对照中一条 rollout 和第二次更新状态的差异；
完整 trainer 无干扰性仍未验证。

详见[当前状态](STATUS.md)、[支持矩阵](docs/support-matrix.md)、
[预览版验证](docs/verification/dev11.md)与 [GPU process 验证](docs/verification/process-gpu.md)。

## 仓库导航

| 路径 | 内容 |
|---|---|
| [`src/agent_rl_diagnostics/`](src/agent_rl_diagnostics/) | 指标、评测器、冻结状态池、分析、报告与可选接入 |
| [`configs/examples/`](configs/examples/) | 评测与 backend 配置示例 |
| [`examples/`](examples/) | Likelihood 输入、过程奖励更新、外部日志和工具智能体示例 |
| [`docs/tutorials/`](docs/tutorials/) | 四个工具与外部数据的可运行教程 |
| [`paper/`](paper/) | 研究方法、代码和验证范围的对应关系 |

开发检查：

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
ruff check .
ruff format --check src tests scripts examples
```

欢迎贡献，见 [CONTRIBUTING.md](CONTRIBUTING.md)。版本变化与发布步骤见
[CHANGELOG](CHANGELOG.md) 和[发布指南](docs/release.md)。

## 许可证与引用

原创代码采用 [MIT 许可证](LICENSE)，第三方资产保留各自条款，见 [NOTICE](NOTICE)。
仓库不包含模型权重、私有 benchmark 测试、原始轨迹或完整实验结果。

软件引用元数据见 [CITATION.cff](CITATION.cff)，作者与论文链接待最终确认后补充。
