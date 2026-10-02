# Next step: finish P3 distributed integration, then P4

Latest dev8: one-time pre-native-init actor seeding is implemented and exercised
on an A100; all actor RNG fingerprints match, but one second-update rollout and
subsequent input/gradient/optimizer/parameter state differ. Investigate async
rollout/server reproducibility separately; do not reset actor RNG at capture or
remove mismatching samples. Fixed/BudgetMix training and same-export standalone
raw comparison remain accepted in this narrow one-rank scope. The new
`ard dynamics` core command draws audited curves and descriptive warnings without
optional libraries. See `training-dynamics.md` and `integrations/actor-initialization.md`.
Next experiment: an independently labeled diagnostics-off/off repeatability
control, then fixed-rollout update comparison to separate backend rollout
variation from actor update effects. Do not relabel a control as off/on or turn
the training sampler greedy merely to make equality pass. Record both request
seeds and timing/worker versions while retaining every original differing sample.

Latest dev7 update: remote one-rank A100 acceptance resumed with explicit operator
authorization. Fixed-budget and BudgetMix training passed two real updates and
independent raw/tensor/W&B auditing. Off/on rollouts and non-RNG actor state match,
and standalone CLI raw budget/action records match the same Ray step-1 export;
but RNG fingerprints differ. Preserve this result and add an explicitly recorded,
identical one-time worker/model RNG initialization protocol for a subsequent
experiment, not a reset at capture boundaries. Complete controller/server RNG,
distributed and optimizer-resume acceptance remains open. See
`integrations/remote-acceptance.md`; older host-deferred instructions below are
historical and do not override the new remote authorization.

2026-10-01 operator update: defer GPU acceptance and progress independent CPU
implementation first. Dev6's BudgetMix sampler, planner, hook and raw-group audit
are available; see `integrations/budgetmix.md`. GPU work is now explicitly pending,
not a prerequisite for further stdlib tools, paper acquisition or release engineering.
When resumed, only physical 0/1 are authorized. The latest bounded checks both
returned cuInit=100; no model/training starts were made.

When GPU acceptance resumes, start with `scripts/check_runtime.py` on an idle
authorized physical 0/1. Once healthy, rerun the
public CLI on the Ray step-1 export, then use
`scripts/compare_checkpoint_validation.py` for replay/metric/hash comparison.
See `integrations/runtime-and-comparison.md`. Do not repeatedly start new CUDA
jobs when the driver probe times out and cannot be reaped.

Dev3 has two independently audited real Qwen3-1.7B updates through the locked
RayPPOTrainer, vLLM server and one-rank FSDP actor. The common checkpoint validators,
raw token/state replay and the actual offline W&B history work in that scope.
See `integrations/ray-smoke.md`, the compatibility matrix and dated verification.

Remaining P3 work: rerun the public checkpoint CLI on the same Ray export and
compare raw outputs; establish full-trainer diagnostics off/on optimizer/RNG
equivalence; extend the single-worker seed/version contracts to synchronized
multi-rank rollout and validation. Export/barrier costs are measured; scheduling
and asynchronous cost optimization remain follow-up work.

The 2026-09-30 host failed/timed out on CUDA initialization (including a minimal
driver call), despite idle cards reported by NVML. Separate real tensor checks
on all physical 0–4 also timed out. New GPU work may use 0–4 per the operator
and must first pass a bounded CUDA preflight. Never reset a GPU
or restart the driver on this shared machine. Three namespace integration checks
also timed out at launch on this date; preserve the fail-closed executor and
diagnose host availability before rerunning. The default CPU suite passes.
At 11:29 UTC the old residual process was absent, but NVIDIA device nodes were
also absent and `nvidia-smi` could not connect. Check driver/device availability
before starting CUDA probes; do not assume the process disappearance fixed CUDA.

Dev5 provides opt-in before/after actor-state capture, a diagnostics-off switch
and exact comparison. See `integrations/training-state-parity.md`. The actual
CPU AdamW check passes; the new Ray/GPU path remains unexecuted. Use identical
seeds/configuration and immutable separate output directories for the next GPU
pair. Captured actor state does not include controller RNG or rollout-server state,
so even matching proofs cannot complete the full-trainer acceptance by themselves.

P2 standalone toolkit diagnostics have genuine model smoke evidence. Keep formal
paper migration separate: final prompts, splits, checker and original state pools
are not verified by these authored fixtures.

## P3 implementation sequence

Items 1–6 below are exercised for the one-rank fixture. Item 7's independent
same-export rerun is pending on the current host; full off/on parity and multi-rank
acceptance require additional real experiments, not CPU contract mocks.

1. Recover the scope-specific tested verl SHA from `docs/integrations/verl-compatibility.json`.
   Inspect actual trainer/rollout/validation entry points at that revision.
   Pin a deliberately updated SHA only with an explicit compatibility record.
2. Add a narrow optional checkpoint-evaluation hook. Reuse `run_agent_budget`
   and `run_fixed`; do not reimplement metrics inside the trainer.
3. Freeze a held-out Base-only monitor pool before training. Enforce disjoint task
   IDs and inspect map-content leakage. Keep it separate from the Base/RL matched
   Section 3.3 pool.
4. Define a weight-version barrier: record trainer step, checkpoint hash/revision,
   pool hash, generation config and split for both validators. Do not silently
   compare stale workers against newly updated weights.
5. Preserve RNG, training mode and actor state. Keep extra evaluation rewards out
   of PPO/GiGPO advantages. Establish whether the integration is synchronous,
   checkpoint subprocess-based or otherwise; benchmark validation cost.
6. Emit namespaced train/validation metrics and health to JSON and optional W&B.
   Demonstrate offline W&B first; upload only if authorized.
7. Run a real short GPU training job. Compare standalone and checkpoint validation
   on identical frozen inputs/seeds, allowing only documented backend numerical
   differences. CPU mocks are contract tests, not integration evidence.

Acceptance: a pinned compatible upstream trainer, genuine checkpoint transitions,
matched evaluator artifacts, actual training curves, offline W&B records and a
working reproduction command. Avoid labeling a callback stub as completed P3.

## P4 prerequisites

CPU foundations may now proceed before P3 GPU acceptance: uniform unique ceil-decile BudgetMix, shared T per task occurrence,
budgets shown each turn, no infeasible-budget exclusion, PPO zero bootstrap and
GiGPO anchor including remaining budget. Process-reward mixing needs the unresolved
paper coefficients recovered from source.

## Paper migration still open

Original Base/RL revisions, final dataset and pool hashes, official DeepCoder
checker and error injector, multi-task figure-specific configs, NLL/LLR conventions,
reward weights, redistribution permissions and formal numerical agreement.
Track acquisition in `reproduction/asset-ledger.json`; do not fill gaps with toy
smoke metrics.
