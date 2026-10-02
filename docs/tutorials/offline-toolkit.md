# One-command CPU walkthrough

Install the core package, then run from any working directory:

```bash
ard toolkit-demo --output runs/toolkit-walkthrough
```

Open `runs/toolkit-walkthrough/index.html`. This needs no model weights, HTTP
server, GPU, Torch or network access. The command refuses to overwrite existing
output and commits its directory only after every evaluation and audit succeeds.

The two policies are authored scripts. “Base” wastes alternating actions by
choosing UP in a horizontal corridor; “RL” always chooses RIGHT. These labels
illustrate comparison roles and do not describe trained models. Repeated samples
are deterministic, so Pass@K does not demonstrate stochastic search gains.

The output contains complete agent-v2 budget rollouts at T=3,5,9; an equal-source
matched pool; clean/error intervention pairs; single-step and recovery samples;
a separately frozen Base monitor pool; five audited comparison reports; and an
audited BudgetMix assignment plan. Pools cover two tasks. Infeasible budgets and
unrecoverable error states are retained. Each raw evaluator directory contains
its config, hashed manifest, records and summary. `walkthrough.json` lists the
outputs and evidence scope. For example, replay an error continuation audit:

```bash
ard audit --input runs/toolkit-walkthrough/recovery-RL --pool runs/toolkit-walkthrough/errors
```

The monitor example evaluates both scripts on identical Base-collected states.
Actual checkpoint dual validation and training curves use the separate
[training-dynamics tutorial](training-dynamics.md). This command creates no
checkpoint, optimizer update or training result. BudgetMix output is a plan only.

To use your own model, follow [quickstart](../quickstart.md) with an HTTP or local
HF configuration. The walkthrough's policy adapter implements the same
`ChatModel.generate(messages, seed=...)` interface; its recorded HTTP placeholder
is never contacted because the scripted model is explicitly injected. Keep the
budget, prompt and sample configuration identical for Base/RL comparisons, and
reuse frozen pool content rather than recollecting states per evaluated model.
