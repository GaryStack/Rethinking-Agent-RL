# Current status

2026-10-03, `0.1.0.dev11`: local paper-companion / agent diagnostic preview.
The agreed first-release method and lightweight tool scope is implemented.

Available: Pass@(K,T), tight-budget evaluation, single-step/depth analysis,
controlled error-prefix recovery, checkpoint dual validation, likelihood/NLL/LLR,
BudgetMix, process-signal reward mixing and pinned training examples. Ordinary
JSONL imports and optional Inspect native-log imports now support paired version
reports, original-record inspection, resource coverage and CI regression gates.
See [method coverage](paper/coverage.md), [support matrix](docs/support-matrix.md),
[external logs](docs/tutorials/external-traces.md) and
[Inspect integration](docs/tutorials/inspect-agent.md).

Current core regression: **333 passed, 4 skipped**. Three skips require the opt-in
Linux namespace executor; one requires the optional Inspect SDK. The prior isolated
Inspect acceptance passed **27 tests**, including actual local tools and native
.eval/JSON ingestion with scripted model responses. Its live-model entry point
has not been accepted. See [Inspect evidence](docs/verification/inspect-adapter.md).

Real Qwen3-1.7B one-rank Ray/vLLM/FSDP process-enabled and BudgetMix smoke evidence
is retained. These engineering checks do not establish method effectiveness or
formal paper numerical reproduction. The seeded diagnostic off/on pair still
has a rollout and second-update state difference; full trainer noninterference,
PPO/GiGPO training, multi-rank and resume remain unverified. BFCL/SQL/Terminal
remain external analysis inputs rather than bundled runnable benchmarks.
See [process GPU evidence](docs/verification/process-gpu.md) and
[historical dev10 status](docs/verification/history/dev10-status.md).

The dev11 local release candidate includes the new trace and Inspect modules;
older dev10 artifacts remain historical. See [release instructions](docs/release.md)
and [dev11 checks](docs/verification/dev11.md). Author and paper metadata await
confirmation; no individual or link is invented. No destination remote is configured,
so public release and remote CI have not occurred. This is ready for local handoff,
not a claim of full enterprise acceptance or paper numerical replication.
