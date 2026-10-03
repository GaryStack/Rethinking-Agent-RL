# Current status

## Unreleased usability work

The source now includes `ard traces import/audit/compare` for caller-attested
external logs, matched-task comparisons, token/time/cost coverage and CI gates.
See [the tutorial](../../tutorials/external-traces.md). The sample is synthetic.
Previously built dev10 release artifacts do not include these uncommitted changes.
A pinned Inspect log adapter and an executable local tool-loop case are now
available. Offline acceptance uses scripted model responses with actual tools.
Real-model/business case acceptance, additional adapters and resumable execution
remain open; this does not establish general enterprise readiness.


2026-10-02, `0.1.0.dev10`: local paper-companion/toolkit development preview.

Available: Pass@(K,T), deterministic tight-budget solver, fixed-state quality,
controlled error-prefix recovery, checkpoint dual validation, BudgetMix sampler
and pinned integration, stdlib dynamics, likelihood NLL/LLR/length control,
Base/RL/depth comparison reports, external scorer API, oracle/judge mixing and a
CPU reward-to-optimizer example. The locked native reward/advantage/actor optimizer chain is CPU-verified on a tiny
random model with scripted actions. A fresh authorized A100 also completed two
process-enabled Qwen3-1.7B one-rank Ray/vLLM/FSDP REINFORCE++ updates. Native reward
exports, terminal-only validation, real gradients and checkpoint changes were
independently audited; see [GPU process verification](../process-gpu.md).

Earlier real 1.7B Ray/vLLM/FSDP and BudgetMix tests are engineering smoke. The
seeded off/on pair differs in one rollout and second-update state; full trainer
noninterference is unverified. PPO/GiGPO training, multi-rank and resume remain open.
Formal numerical reproduction is unverified. BFCL/SQL/Terminal are external inputs,
not runnable bundled tasks. The new HF scorer passed tiny random-checkpoint alignment checks and one fixed response under local Qwen3-1.7B on CPU.

Current CPU regression: 284 tests pass, 3 optional namespace
tests skip. `ard toolkit-demo` creates ten audited agent-v2 evaluations and five
reports in one offline command. Likelihood CLI now propagates excluded events
into trajectory length control. See [toolkit verification](../toolkit-walkthrough.md).
Native process validation: [dev10 verification](../dev10.md).
Historical status is retained in [dev8 history](dev8-status.md).
No remote repository, push, PyPI release or online W&B upload was performed.

Public-source links and metadata versions are checked by `scripts/check_source_release.py`.
The Chinese entry point and changes are ready; authors/paper URL remain explicit
placeholders. A local Git bundle is prepared without publishing; see [release](../../release.md).
