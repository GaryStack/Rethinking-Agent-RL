# Agent RL Diagnostics

[中文说明](README.zh-CN.md) · [Changes](CHANGELOG.md)

A paper companion and diagnostic toolkit for multi-turn agents: budget-dependent
success, local decision quality, error-prefix recovery, training dynamics and
BudgetMix. Python 3.11+; the core uses only the standard library.

Development preview `0.1.0.dev10`. Start with the [quickstart](docs/quickstart.md)
to evaluate your own model, or [paper methods and coverage](paper/README.md) to
understand the accompanying methods. This release supplies method code and
small authored examples; formal paper numerical reproduction is unverified.

```bash
python -m pip install -e .
ard doctor --json
ard toolkit-demo --output runs/toolkit-walkthrough
```

Open `runs/toolkit-walkthrough/index.html`. The [CPU walkthrough](docs/tutorials/offline-toolkit.md)
collects scripted agent-v2 rollouts, freezes matched and Base-monitor pools,
evaluates error prefixes and recovery, and audits every run before drawing five
comparisons. No GPU, network or model weights are needed. The original
`ard demo --offline` remains available for the simpler legacy trajectory example.

| Tool | Tutorial | Main entry |
|---|---|---|
| Tight-budget evaluation | [Budget tutorial](docs/tutorials/tight-budget.md) | `ard eval budget`, `ard compare` |
| Error-prefix evaluation | [Recovery tutorial](docs/tutorials/error-prefix.md) | `ard pool errors`, `ard eval step/recovery` |
| Training dynamics monitor | [Dynamics tutorial](docs/tutorials/training-dynamics.md) | `ard eval checkpoint`, `ard dynamics` |
| BudgetMix plug-in | [BudgetMix tutorial](docs/tutorials/budgetmix.md) | `ard budgetmix plan/audit`, optional pinned trainer adapter |

Mechanism analysis: `ard likelihood score/analyze` reports raw teacher-forced
response likelihood, action NLL, aligned Base/RL LLR and explicit length controls.
See [likelihood inputs](docs/likelihood.md). Process rewards support oracle/judge
inputs, a versioned cache, explicit mixing and an executable training example;
see [process signals](docs/process-signals.md). The pinned reward-to-native-actor
path is additionally verified on CPU with scripted actions and a tiny random model.
The process-enabled path also passed two real Qwen3-1.7B one-rank Ray/vLLM/FSDP
REINFORCE++ updates with independently audited native rewards and validation.
See [GPU process verification](docs/verification/process-gpu.md); wider algorithm
and distributed coverage remain unverified.
[External scorers](docs/extensions.md) use a public Python adapter and reuse the
fixed-state metric rather than implementing another aggregation rule.

Deterministic FrozenLake and derived stdin/stdout coding are executable environments.
BFCL, SQL and Terminal are **external analysis inputs only**, not bundled runnable
benchmarks. Coding runs require the bounded Linux namespace executor.
See the [support matrix](docs/support-matrix.md), [input requirements](docs/inputs.md)
and [protocols](docs/protocols.md).

Real 1.7B one-rank Ray/vLLM/FSDP and BudgetMix engineering smoke evidence exists.
The seeded diagnostic off/on pair still differs in one rollout and second-update
state; full trainer noninterference is unverified. These tests are not paper
benchmark numbers. [Current status](STATUS.md) retains these limits, with
[historical verification](docs/verification.md) available for the evidence.

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
ruff check .
ruff format --check src tests scripts examples
```

Original code is MIT; third-party assets retain their own terms. No model weights,
private tests, original traces or full experiment results are bundled.
Citation authors and paper URL await confirmation; see `CITATION.cff` and
[release checklist](docs/release.md). No remote publishing is performed.
