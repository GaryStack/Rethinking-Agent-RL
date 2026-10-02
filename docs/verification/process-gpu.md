# Real Qwen3-1.7B process-enabled GPU verification

2026-10-02. One authorized idle A100-SXM4-80GB, fresh non-inheriting Python
3.11.15 environment. All 221 frozen package versions matched the candidate
snapshot; the pinned VeRL commit `7dc46e834209948cf1cdd8a04d83f82b4a7efd24`
passed its 437-file lock. Driver initialization and bounded CUDA allocation passed.
Shared Qwen3-1.7B weights were read-only: no weight download/copy or driver changes.

The existing eight-state real Base-only monitor pool was independently replayed
and checked for task-ID/map-content holdout. The original Base tensor fingerprint
was recomputed and matched before reuse. This is not an oracle replacement pool.

## Executed recipe

Use the isolated pinned environment and an independently verified monitor pool:

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python scripts/run_verl_ray_smoke.py \
  --model-path YOUR_LOCAL_QWEN17 --verl-source YOUR_LOCKED_VERL \
  --monitor-config YOUR_MONITOR_CONFIG --pool YOUR_BASE_POOL \
  --output runs/process-ray --steps 2 --train-budget 2 --group-size 16 \
  --actor-seed 5422 --process-coefficient 0.5 --process-failure-policy error
python scripts/export_wandb_history.py --input YOUR_FINISHED_OFFLINE_WANDB_FILE --output runs/process-ray/wandb-history.jsonl
python scripts/audit_verl_ray_smoke.py --run runs/process-ray --base YOUR_LOCAL_QWEN17 --history runs/process-ray/trainer-history.jsonl --wandb-history runs/process-ray/wandb-history.jsonl --output runs/process-ray/verification.json
```

The trainer executed actual RayPPOTrainer, async vLLM multi-turn generation,
one-rank FSDP/native AdamW and REINFORCE++ on four authored tiny grid tasks.
Each update used 4 tasks × 16 completions with action budget 2. Coefficient 0.5
is a toolkit engineering choice, not a recovered paper setting. No training-state
capture or off/on parity comparison was requested in this run.

| Independent check | Observed result |
|---|---|
| Training completions | 64 per update, 128 total |
| Nonzero process contribution | 128/128; 117 were terminal failures |
| Native validation | 2 rows at steps 0,1,2; all terminal-only |
| Raw audited traces | 134; 10,599 environment tokens |
| Actor gradient norms | 76.0 and 348.0, finite and positive |
| Changed checkpoint tensors | 257 then 255 out of 311 common tensors |
| Dual-validation artifacts | Steps 0,1,2 complete, replayed and weights rehashed |
| Native/offline W&B history | Diagnostic points and gradients match |
| Native token-level-score exports | Match recomputed composition within absolute tolerance 1e-6 |

The native score audit joins by phase, update, task, decoded prompt and decoded
response with exact multiplicities. It rejects missing/duplicate trace slots,
configuration mismatch, altered scores/components, nonfinite fields and changed
generation identities. Existing raw token/action/mask replay and checkpoint
content checks remain active. The public auditor was rerun against the actual
completed GPU artifacts after adding this check; it passed without retraining or
rewriting the original verification result.

## Evidence and limits

Private evidence is retained under ignored `runs/process-gpu-20261002`: reports,
raw traces/diagnostic records, native and exported offline histories, source lock,
package verification, executed commands and hashes. Initial archive: 212 files,
SHA256 `50135f8039af3269b9490ea6d8b93221af0727778543fc92dcc3e250875a89ac`.
A separate integrated-audit result and source digest preserve the subsequent
public auditor check. Checkpoints remain on the authorized server; no weights,
optimizer state, binary W&B files, caches or credentials enter the public package.
GPU returned to idle after shutdown. The disk-ratio Ray warning occurred with
more than 420GB available; no object-spill or training failure occurred.

This establishes the narrow process-enabled engineering path. It does not
establish paper numerical reproduction, useful performance gains, formal reward
conventions, complete GRPO/PPO/GiGPO, multi-rank/resume or full trainer parity.
All native validation successes and monitor scores in this short fixture were
zero; reward/optimizer correctness does not imply improved validation performance.
The historical seeded off/on divergence remains unchanged and visible.

Default CPU regression after integrating the auditor: 278 passed, 3 opt-in
namespace tests skipped; Ruff lint/format checks pass for 136 Python files.
Negative audit cases cover native terminal-only substitution, validation shaping,
missing/duplicate slots, changed settings, nonfinite scores and identity tampering.
