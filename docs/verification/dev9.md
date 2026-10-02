# Dev9 local preview verification — 2026-10-01

Incremental implementation of release handoff A→B→C→D. No data acquisition sweep,
new model training job, host driver change or remote publishing was performed.

## Code and docs

Added teacher-forced local HF scorer and stdlib NLL/LLR/length-control analysis;
`ard likelihood score/analyze`; audited `ard compare` reports for matched budgets,
fixed-state depth and recovery, including exact slack where certified; public
external scorer adapter; oracle/judge cache and explicit reward mixing; synthetic
REINFORCE reward-to-update example and experimental pinned grid reward_score hook.
Process trace replay checks the actual mixed reward against oracle step scores.

README, current STATUS/NEXT, paper coverage, input/support docs, four tutorials,
fixtures/static reports, source manifest, package metadata and CI were updated.
Dev8 status/next history remains intact. See `paper/coverage.md` for boundaries.

## Actual commands and outcomes

Validation ran first in `/tmp/ard-release-work`, with source import through `src`.
The independent interpreter is `/opt/anaconda3/bin/python`, not the project's venv.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /opt/anaconda3/bin/python -m pytest -q -p no:cacheprovider
ruff check .
ruff format --check src tests scripts examples
```

**252 passed, 3 skipped**, 7.15s in the staged final-code run. Skips are the three
explicit opt-in namespace-host tests, not successes. Ruff 0.11.13: all checks
passed; 130 Python files already formatted. The existing CPU contracts remain
covered; neither optional W&B nor namespace-host execution was rerun this round.

Optional HF verification used the pre-existing isolated Python environment
`/tmp/ard-ray-env-cXj8GA/bin/python` (Torch 2.9.0, Transformers 4.57.1):

```bash
PYTHONPATH=src /tmp/ard-ray-env-cXj8GA/bin/python scripts/verify_teacher_forcing_cpu.py
```

A newly initialized tiny local GPT2 checkpoint matches direct causal token logits
within 1e-6, preserves the action mask and rejects unstable boundaries/oversized
contexts. Separately, the scorer loaded user-supplied local Qwen3-1.7B weights on
CPU, rendered a short authored direction prompt and scored fixed `RIGHT`: status
ok, one response/action token. This was teacher forcing, not generation or training.
The private raw score is retained in `runs/dev9-verification/likelihood-1.7b.json`;
it is excluded from packages. The caller's model identity is recorded; no new full
weight rehash or paper benchmark provenance is claimed.

A new no-dependency wheel environment under `/tmp/ard-release-clean`, invoked
outside the source checkout with PYTHONPATH unset, passes installed doctor,
demo/replay audit, likelihood fixtures, BudgetMix plan/audit, process-training,
external scorer and dynamics-fixture commands. Optional dependencies report absent.
The process example with coefficient 0.5 ends at logit 0.07406655007232668; the
coefficient-0 contract leaves the logit unchanged. Real native process-enabled
VeRL optimizer execution remains unverified.

Wheel and source distribution payload inspection passes. No weights, runtime
traces, log/cache trees or known credential patterns are bundled. `dist/SHA256SUMS`
records final artifacts. Full source regression is also rerun after syncing to the
target repository; final result is recorded in the handoff update.

## Limits and next steps

Formal numerical reproduction stays false. Tool reward/length conventions do not
invent missing paper parameters. Native process reward is experimental; PPO/GiGPO
paths, multi-rank, resume and full trainer noninterference remain unverified.
Earlier seeded off/on rollout/state divergence is preserved. Metric-only dynamics
and error-prefix static examples explicitly lack raw training/model audit.
External single-step scorers do not automatically register full environments.
Citation authors/paper URL require confirmation before external publication.
