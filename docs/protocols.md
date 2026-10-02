# Diagnostic protocols

`ard-v1` is the trajectory schema, not the original research-workspace schema.
It remains supported for scripted demos. `ard-agent-v2` adds complete messages,
raw text, environment-neutral snapshots and typed action payloads. `ard-frozen-state-v1`
stores actual generation inputs. There is no implicit conversion of research
workspace records; legacy imports require an explicit migration.

## Pass@(K,T)

Within each task and T, `n` independent completions have `c` successes. Compute
`1 - C(n-c,K)/C(n,K)` for `n >= K`, then weight tasks equally. Duplicate sample
slots, mixed checkpoint versions and mixed protocols are errors.

The full estimate requires all expected tasks and sample slots to finish. Missing
records and infrastructure failures make the full estimate unavailable; the
separately labeled observed-task mean does not replace it. Task failures and
budget exhaustion are completed unsuccessful trials, so they enter the denominator.

The percentile CI resamples tasks. It captures between-task uncertainty, not all
within-task Monte Carlo uncertainty. Paired comparison resamples per-task
differences using the exact same task IDs.

## Budget and state

A grid move is one agent action. `remaining_budget` is measured before the action
and includes it; `turn_index` starts at zero. A boundary move still consumes an
action. A hole or goal terminates the rollout. Each T is rerun independently with
budget supplied on every action; the scripted policies may ignore that input.

Long-trajectory truncation belongs to `budget_blind_prefix`, which the current
runner rejects. In later imports, it requires an explicit online/replay success
verification convention; it will not be inferred from final success.

Deterministic BFS gives an `exact_minimum` or a proven unreachable state. Thus
`U(T) = mean(T >= d)` is a strict structural ceiling for the current maps, for
every K. Include unreachable tasks in this mean. Feasible-subset performance is
an additional conditional result.

Discounted Q* uses gamma=0.95 and counts the terminal goal reward once. Optimal
finite-horizon success uses gamma=1. The solver also supports analytical slippery
dynamics (three equiprobable directions), but the current executable runner uses
deterministic transitions only. Stochastic tasks cannot use a deterministic BFS
certificate as a probability-of-success bound.

## Paper presets reserved for later adapters

Scripted demos accept `toolkit_v1 / demo_synthetic`. Agent evaluators accept
`toolkit_v1` with `raw_recomputable` or explicitly synthetic test evidence.
`paper_v1` is reserved and rejected until original protocols are recovered.
The final paper requires temperature 0.6 and top_p 0.9 during evaluation; training
uses 1.0/1.0. Scripted policies have no such decoding parameters (manifest: null).

Final Tmax: FrozenLake 70, DeepCoder-MT 10, BFCL v3 24, InterCode-SQL 10,
Terminal-Bench 50. Section 3.3 uses Base/RL equal-contribution matched states,
16 sampled actions per policy/state. Training dynamics freezes 512 held-out Base
states and samples 8 actions per checkpoint/state.

Paper BudgetMix uniformly samples the unique set `ceil(k*Tmax/10), k=1..10`,
shares T within each task occurrence group, displays budgets every turn and retains
structurally infeasible budgets. PPO budget termination bootstraps zero; GiGPO
anchors include remaining budget. Dev6 implements the stdlib sampler/plan audit
and opt-in raw-grid pre-repeat trainer adapter. Native hook/repeat/tokenizer CPU
verification passes. PPO bootstrap and GiGPO anchors/degenerate step rewards have
pure CPU contracts only: critic training, resume and GiGPO integration remain open.

Final figure-specific settings, process-reward weights, the NLL sum/mean convention
and Table 1 rounding are tracked as unresolved in the asset ledger.

## Local quality and controlled recovery

The text-grid parser accepts exactly a direction, optionally with final punctuation,
after a closing reasoning tag. Invalid replies cost one action and score zero.
Its prompt is `ard-grid-text-v1`, a derived toolkit prompt, not a verified copy of
the paper's final prompt. Deterministic transitions and Q* use the foundation solver.

The coding environment is `deepcoder_stdin`, with `derived_stdin_test_fraction_v1`.
RUN and SUBMIT both consume one action. The score is the current program's fraction
of configured tests passed in fresh processes. RUN feedback is scratch execution
with empty stdin, not hidden-test outcomes. SUBMIT always terminates. Bare fenced
programs are implicit SUBMIT; surrounding prose is invalid. This is not the official
LiveCodeBench checker. No exact minimum coding-budget certificate is claimed.

Single-step evaluation restores every state for every sample, computes action means
within each state, then weights states equally. Frozen inputs are deduplicated by
complete messages/snapshot/task/budget identity, not by scalar reward. A Base-only
monitor is separate from an equal Base/RL source-contribution pool. Without explicit
training IDs no held-out proof is claimed; task-ID disjointness is not content-level
leakage detection.

Grid wrong-prefix interventions increase shortest distance, consume one action in
both clean/error conditions and keep nonterminal states. Remaining budgets match.
Coding interventions use explicitly identified, tested independent references and
a syntax mutation, not the paper's Qwen3-1.7B generated perturbations. By default
infeasible grid interventions are retained; `--require-recoverable` filters them
explicitly and changes the estimand. No evaluated model outcome selects pairs.

Recovery first estimates each pair's clean/error success probabilities, then averages
pairs equally. The difference is error minus clean. Conditional action lengths use
successful continuations only, not all trials. CIs for local quality and recovery
resample task clusters and retain within-cluster state/pair weighting; they capture
between-task variation, not all generation uncertainty. Early v1 summaries without
additive CI fields remain auditable but are not assigned invented intervals.
