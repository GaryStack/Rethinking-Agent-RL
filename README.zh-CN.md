# Agent RL Diagnostics

用于多轮智能体的论文配套代码和诊断工具。Python 3.11+，核心仅依赖标准库。
[English](README.md) · [方法覆盖](paper/coverage.md) · [支持范围](docs/support-matrix.md)

## 安装与一键示例

从源码目录安装：

```bash
python -m pip install -e .
ard doctor --json
ard toolkit-demo --output runs/toolkit-walkthrough
```

打开 `runs/toolkit-walkthrough/index.html`。它用明确标注的脚本策略生成预算
评测、普通/错误状态单步评测、恢复评测、Base 固定池评测和 BudgetMix 计划，
审计原始记录后生成五份对比报告。无需网络、模型权重或 GPU；输出目录须是新目录。
脚本策略标签 Base/RL 不代表经过训练的模型。

## 评测自己的模型

按 [quickstart](docs/quickstart.md) 修改 HTTP 配置中的模型 ID、revision 和 endpoint，
保留任务、预算、采样参数一致。先采集 rollout，再构建冻结状态池，后续模型复用
同一份 pool。支持用户自备 HTTP chat 服务和本地 Transformers checkpoint；
后者需安装 `python -m pip install -e '.[hf]'`。工具不会自动下载模型或执行远程代码。

| 工具 | 用途 | 入口与教程 |
|---|---|---|
| Tight-budget Eval | Pass@(K,T)、确定性最短预算和 slack | `ard eval budget` / [教程](docs/tutorials/tight-budget.md) |
| Error-prefix Eval | 同预算错误干预、单步质量和续跑恢复 | `ard pool errors` / [教程](docs/tutorials/error-prefix.md) |
| Training Dynamics | checkpoint 多轮/单步双验证与曲线 | `ard eval checkpoint`、`ard dynamics` / [教程](docs/tutorials/training-dynamics.md) |
| BudgetMix | occurrence 级采样、组内共享预算 | `ard budgetmix plan/audit` / [教程](docs/tutorials/budgetmix.md) |

[Likelihood](docs/likelihood.md) 提供原始 teacher-forced NLL、action mask、Base/RL LLR
和长度控制；[Process signals](docs/process-signals.md) 提供 oracle/judge、缓存、显式
奖励混合及锁定 VeRL 接入；[外部 scorer](docs/extensions.md) 示例可在不修改核心的
情况下接入自己的单步评分器。

## 验证与限制

真实 Qwen3-1.7B/A100 的两步单 rank Ray/vLLM/FSDP process-enabled 工程测试已通过，
包括奖励导出核对、terminal-only 验证、梯度和 checkpoint 内容审计，详见
[验证记录](docs/verification/process-gpu.md)。这不构成论文数值复现或性能提升证据。
既有训练诊断 off/on 差异保留；PPO/GiGPO 完整训练、多 rank 和 resume 尚未验收。

可运行环境为确定性 FrozenLake 和派生 stdin/stdout coding；coding 程序只通过
受限 Linux namespace executor 执行。BFCL/SQL/Terminal 仅支持外部输入分析，
没有打包完整任务环境。缺失样本或基础设施失败不会被转换为零分或成功。

许可证为 MIT；不包含模型、私有任务/测试、原始训练日志或 W&B 二进制。
当前发布信息见 [release](docs/release.md)，引用信息见 `CITATION.cff`。
