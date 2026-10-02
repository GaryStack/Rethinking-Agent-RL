# BudgetMix without a GPU, then optional pinned training

The paper sampling rule is implemented independently of any framework:
`unique(ceil(k*Tmax/10), k=1..10)`, uniform over the unique values. Integer rounding
avoids floating-point boundary errors. Small Tmax does not duplicate probability
mass. Every pre-repeat row is a distinct task occurrence, even for duplicate task
IDs; all completions in that occurrence share T. No structural-minimum filtering
or feasibility rejection is performed.

## CPU planner and audit

```bash
ard budgetmix plan --task-ids configs/examples/budgetmix_task_ids.json \
  --tmax 24 --seed 5422 --update 1 --group-size 8 --output runs/budgetmix-plan
ard budgetmix audit --input runs/budgetmix-plan
```

Input is a JSON list of task IDs; order defines occurrence rows. Outputs contain
assignments, histogram/mean and a hashed manifest. Repeated task IDs are allowed,
not collapsed. The audit recomputes assignments and statistics even after artifact
hashes are rewritten. Plans are immutable, CPU-only and **not executed rollouts**.
Sampling uses a private occurrence-seeded PRNG and leaves global RNG untouched.
Identity includes sampler seed, logical update, row and task ID; determinism is
for identical ordering/settings, not an arbitrary resumed/reordered dataloader.

## Actual upstream CPU verification

Use the isolated pinned optional environment and clean source from `ray-smoke.md`:

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/verify_budgetmix_cpu.py --verl-source /absolute/clean/verl \
  --tokenizer-path /absolute/local/Qwen3-1.7B --output runs/budgetmix-native-cpu
```

This calls the real locked `RayPPOTrainer._get_gen_batch` without constructing a
trainer or cluster, native `DataProto.repeat`, and local tokenizer chat templating.
Two logical updates each include three occurrences and four repeated completions
per occurrence. Duplicate task IDs retain separate identities, native uid grouping
survives, prompts match sampled T, and all three infeasible groups are retained.
It loads no weights, initializes no CUDA and performs no optimizer updates.
Default stdlib tests use separately labeled trainer fixtures, not GPU evidence.

## Optional Ray grid adapter (one-rank fixture verified)

Dev7's actual A100/Qwen3-1.7B run completed two REINFORCE++ updates with Tmax=2.
Raw groups, pre-repeat assignments, masks/rewards, checkpoint content changes,
dual validation and actual offline W&B histograms passed independent auditing.
See [portable acceptance](remote-acceptance.md). This verifies this narrow training
integration, not learning improvement, PPO/GiGPO or larger-budget paper replication.

After a healthy bounded CUDA preflight, add `--budgetmix` to the Ray smoke recipe.
`--train-budget` becomes Tmax (the tiny recipe allows 1–8, not paper Tmax=70).
The adapter wraps the chosen native/checkpoint-diagnostic trainer before its
pre-repeat generation hook. It supports the locked raw text-grid dataset's dummy
tensor only; stale tokenized/multimodal inputs and resume are rejected. Validation
keeps its original fixed budgets. Existing default commands keep fixed training T.

Each update saves `budgetmix/update-XXXXXX/{assignments,metrics}.json` before native
generation. Every completed raw grid trace carries its actual assignment, and
per-action budget feedback/masks are replayed by the shared grid auditor. The
adapter records `train/budgetmix/budget_mean`, group count and per-budget counts
through native validation into the trainer's logger/offline W&B at its cadence.
These assignment metrics are not measured completions or learning performance.
Cadence is every update in the smoke recipe; custom infrequent validation delays
or omits pending histogram logging and is not an accepted configuration.

Replay completed groups separately (requires actual traces, not planner output):

```bash
python scripts/audit_budgetmix_traces.py --traces runs/ray-mix/rollouts/grid-traces \
  --update 1 --group-size 16 --output runs/ray-mix/budgetmix-update1-audit.json
```

This verifies budgets/snapshots/rewards/token masks, occurrence membership/counts
and uniform weight version. It reports actual actions, generated tokens, budget
termination and zero-reward group fractions. Missing groups or inconsistent
assignments fail; all-zero rewards remain visible. The existing worker deliberately
rejects all-zero training advantages rather than declaring an optimizer update.
Other updates/native validation traces are counted as out of scope, not hidden.

## PPO/GiGPO boundaries, not acceptance claims

The stdlib `terminal_bootstrap` contract returns zero on action-budget exhaustion
and natural terminals, preserves continuing values and rejects infrastructure
failures. `gigpo_anchor` includes occurrence ID, full restorable state and remaining
budget. Population-normalized step rewards are zero for singleton/constant groups;
episode advantages are not modified by this helper.

These functions have CPU tests; they are NOT wired into a real PPO critic or GiGPO
optimizer. The current Ray smoke still uses REINFORCE++, not paper PPO/GiGPO.
Critic conditioning/save/resume, exact upstream GAE verification, GiGPO anchor
grouping/training and process reward coefficients remain pending. No GPU BudgetMix
learning improvement or formal paper reproduction has been verified. Toolkit
construction can proceed while those explicit acceptance tasks wait.
