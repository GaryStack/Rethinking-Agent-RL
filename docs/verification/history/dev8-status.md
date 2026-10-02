# Build status

Updated: 2026-10-01. Version: 0.1.0.dev8; development preview, not the full Research Preview milestone.

| Stage | State | Evidence / remaining work |
|---|---|---|
| P0 protocol/asset audit | initial audit implemented | Source ledger and candidate verl SHA; original configs, checkpoint and raw provenance remain unresolved |
| P1 CPU foundation | verified | Dependency-free package, Pass@(K,T), exact grid solver, offline demo/report, clean wheel install |
| P2 standalone diagnostics | toolkit implementation verified | HTTP/local HF, real FrozenLake and derived coding inference, frozen pools, step quality, recovery and replay audit; formal paper prompts/splits/checker/injector still pending |
| P3 verl dual validation | one-rank and same-export CLI verified; full parity pending | Actual 1.7B Ray/vLLM/FSDP updates, dual validation, raw replay and offline W&B; standalone records match; seeded actor RNG matches but one rollout/second-update state differs; full-trainer and multi-rank acceptance remain open |
| P4 mitigations | one-rank BudgetMix smoke verified | Two actual A100/Qwen3-1.7B REINFORCE++ updates; pre-repeat plans, completed groups, masks/rewards and actual offline W&B audited; PPO/GiGPO critic/resume and process rewards still open |
| P5 paper coverage | pending | Original four tasks, Terminal likelihood, mechanism analysis and figure/table evidence |
| P6 public release | pending | Release packaging/metadata, wider tutorials and reference artifacts |

Two local Qwen3-8B smoke runs executed genuine generation on an RTX 3090.
They use authored tiny fixtures, not formal paper datasets or Base/RL comparisons.
Coding recovery failures are retained, including strict-parser violations from
prose-prefixed model outputs. A completed smoke audit is not evidence of benchmark
performance or paper replication.

No model weights, private benchmark tests or upstream implementation are bundled.
No remote GitHub repository was created, and no results were uploaded.

Dev8 adds an installed stdlib-only `ard dynamics` command: raw dual-validation
audit, fixed-protocol checkpoint curves, gaps/incomplete-point handling and
descriptive adjacent tradeoff warnings. Actual exported dev7 raw runs now audit
portably without host-dependent symlink rebasing or rewriting their hashes.
The optional `--actor-seed` performs one-time initialization before native model
initialization, never at updates/capture. A fresh A100 suite completes fixed off/on
and BudgetMix two-update jobs, accepts fixed/BudgetMix audits and exact standalone
same-export comparison. All actor RNG fingerprints now match, but one second-step
rollout differs and input/gradient/parameter/optimizer divergence is detected.
Full parity remains explicitly false; dev7's negative evidence is preserved.
Final core suite: 232 passed, 3 opt-in namespace checks skipped; Ruff passes for
112 Python files. Reproduction protocols/data and public release are still pending.

The operator subsequently authorized remote GPU acceptance. Dev7 uses one idle
A100 with read-only mounted weights and a new isolated locked environment; no
original-host driver changes or GPU resets were made. Fixed-budget diagnostics
off/on and BudgetMix each completed two real Qwen3-1.7B optimizer updates.
Fixed-on and BudgetMix passed independent raw/tensor/dual-validation/W&B audits.
Standalone CLI on the exact Ray step-1 export also passed: both budget trajectory
and single-step action files match byte-for-byte, with weights independently rehashed.
The off/on pair has identical 128 training completions and identical non-RNG actor
fingerprints, but different Python/NumPy/Torch CPU/CUDA RNG fingerprints. This is
an explicit negative parity result, not complete trainer noninterference proof.
See [portable acceptance](../../integrations/remote-acceptance.md).
Final core suite: 207 passed, 3 opt-in namespace checks skipped; Ruff passes for
105 Python files. Fresh no-dependency dev7 wheel installation runs doctor,
BudgetMix plan/audit, readiness and the offline demo/replay audit.

Historical dev6: GPU acceptance was deferred; no new CUDA jobs were scheduled.
Dev6 adds `ard budgetmix plan/audit`, immutable pre-repeat assignment records,
budget-visible prompts, raw completed-group replay and budget histogram logging
through native validation. Local CPU verification calls the actual locked trainer
generation hook, native DataProto.repeat and Qwen3-1.7B tokenizer for two logical
updates; it loads no model weights and performs no optimizer updates.
The paper readiness tool turns the ledger into nine pending acquisition checklists,
not executable formal recipes. CI now includes planner/audit and readiness commands;
remote CI has not run because the repository remains local.
Final CPU suite: 190 passed, 3 opt-in host checks skipped. Fresh no-dependency wheel
installation passed planner/audit, readiness, doctor and the offline replay demo.

Dev5 adds opt-in actor-boundary parameter/buffer/optimizer/scheduler/batch/RNG
fingerprints and a dependency-free comparison tool. Actual tiny CPU AdamW off/on
updates match all observed components; this is not a Qwen/verl GPU test. The new
Ray capture path and diagnostics-off recipe have contract coverage but await GPU
execution. Current suite: 149 passed, 3 opt-in host checks skipped. Physical 0–4
all timed out allocating a 16-element CUDA tensor despite successful Torch import.
The old residual process initially survived SIGKILL. At 11:29 UTC it was absent,
but GPU device nodes had disappeared and NVML could not connect to the driver.
GPU recovery remains unconfirmed; no shared GPU reset was attempted by this project.

Dev3 adds a tested isolated Ray/vLLM environment, real budget-visible agent loop,
generated/observation token masks, completed-transfer weight-version tags and a
synchronous HF export/validation barrier. The actual offline W&B SDK history is
flushed in its owning controller before Ray shutdown. An earlier incomplete W&B
run is retained and rejected by the public acceptance script.

The 2026-09-29 run has 134 replayed rollout traces, nonzero gradient norms
127/336 and changed tensor counts 257/253. Held-out Pass@(K,T) and Q* remain zero.
These authored fixtures establish integration behavior, not model generalization.
On 2026-09-30 the default CPU suite passed 121 tests (3 opt-in checks skipped).
Current-host namespace checks timed out, and CUDA initialization failed/timed out
when attempting additional standalone GPU verification. These checks remain open;
the earlier completed GPU run and its independently rehashed evidence are retained.

Dev4 adds bounded Python/package/CUDA/namespace probes and exact audited
checkpoint-output comparison. The latter matched actual dev2 native-actor and
standalone 1.7B records. New Ray-export parity and full-trainer/multi-rank acceptance
remain pending while the current host's driver/namespace startup is unavailable.

Qwen3-1.7B now has a complete real two-environment smoke run, including budget
evaluation, fixed-state quality and clean/error recovery. Legacy output-format
failures and ineligible Base error-prefix anchors were discovered, not hidden.
An explicitly versioned action-only prompt and separately labeled independent BFS
witness error pool let the complete pipeline run without rewriting model outputs
or contaminating the Base monitor. Tiny coding fixtures passed; grid success
remained zero. None of these metrics are paper benchmark results.

See [verification](../../verification.md), [tutorial](../../model-evaluation.md) and
[next step](../../NEXT.md).
