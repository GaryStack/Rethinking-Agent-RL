# Checkpoint-bound training diagnostics

The shared evaluator performs immutable local HF checkpoint validation. This page
also documents the native `verl.workers.actor.dp_actor.DataParallelPPOActor` smoke
and its exact diagnostic off/on parity. Dev3 additionally exercises one-rank
RayPPOTrainer/vLLM/FSDP; follow the [Ray guide](ray-smoke.md) for that integration.
Multi-rank acceptance and paper GRPO/BudgetMix recipes remain pending.

## Reusable evaluation entry point

Install `.[hf]` and optionally `.[tracking]`. First collect a Base-only frozen
pool with `ard pool build --kind base_monitor`; use held-out validation tasks.
Do not mix the Base/RL Section 3.3 matched pool into training monitoring.

```bash
ard eval checkpoint --config validation.json --pool runs/base-pool \
  --checkpoint /path/to/exported-hf-checkpoint --checkpoint-step 100 \
  --samples 4 --output runs/validation-100 --dry-run
ard eval checkpoint --config validation.json --pool runs/base-pool \
  --checkpoint /path/to/exported-hf-checkpoint --checkpoint-step 100 \
  --samples 4 --output runs/validation-100 \
  --tracking-dir runs/tracking --wandb-offline
ard audit --input runs/validation-100 --pool runs/base-pool --verify-weights
```

Config must use `model.backend=transformers` and split `validation` or `smoke`.
Weight shard and tokenizer/config bytes are hashed, not just a checkpoint folder
name. Both validators use the same freshly loaded checkpoint. Generation inputs,
raw records, child manifests, pool/config identities and counts are retained.
Weights are rehashed afterwards; mutation aborts the job. Complete validation
is protected by a Python/NumPy/PyTorch CPU/selected-CUDA RNG guard, including model
construction. Evaluation rewards never enter training advantages. Code replay
requires explicit `--replay-code` and the existing namespace sandbox.

The optional `DiagnosticTracker` writes driver-only durable JSONL and W&B offline
history, rejects rank!=0 and conflicting or regressing steps, and keeps health and
completion counts visible beside scores. It never automatically uploads a run.
This is a synchronous checkpoint barrier, not asynchronous worker synchronization.
Hashing/reloading/exporting full weights is deliberately expensive; large-scale
trainer scheduling and overhead optimization remain follow-up work.

## Real native actor smoke

Dev3 also exercises the actual one-rank RayPPOTrainer/vLLM/FSDP chain. See the
[Ray training guide](ray-smoke.md) for the isolated environment, agent loop,
checkpoint barrier and raw-artifact acceptance. The native actor's exact off/on
parity below has not yet been established for the entire Ray trainer.

Use a clean upstream checkout at
`7dc46e834209948cf1cdd8a04d83f82b4a7efd24`, with its own required dependencies
installed, and this package's `.[actor-smoke]` extras. The script checks all 437
Python/YAML files against the packaged lock before importing upstream code.
Never point it at a dirty research checkout or assume the newest main is compatible.

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline PYTHONPATH=src \
python scripts/run_verl_actor_smoke.py \
  --model-path /path/to/Qwen3-1.7B --verl-source /path/to/clean-pinned-verl \
  --output runs/native-actor --steps 2 --group-size 4 \
  --algorithm reinforce --wandb-offline
python scripts/report_training_diagnostics.py \
  --input runs/native-actor/tracking/metrics.jsonl --output runs/native-actor/report.html
python scripts/export_wandb_history.py \
  --input runs/native-actor/tracking/wandb/OFFLINE_RUN/run-ID.wandb \
  --output runs/native-actor/offline-history.jsonl
python scripts/verify_training_artifacts.py \
  --input runs/native-actor --standalone runs/standalone-same-checkpoint \
  --output runs/native-actor/verification.json
```

This test uses LoRA r=4, exact generated token IDs, real terminal-success returns,
native upstream log-probability and optimizer update APIs, tiny held-out grid
validation and separate direction-symmetric one-step training corridors. The
REINFORCE choice is explicit; there is no automatic algorithm fallback. Default
`grpo` rejects an all-zero advantage batch. An unsuccessful task distribution is
not rescued with invented rewards. Failed longer left-goal pilot runs are retained.

Acceptance requires changed trainable weight hashes and nonzero gradients. The
first update is run twice from identical weights, AdamW state, training batch and
RNG states, with diagnostics off/on. Actual clipped gradients, actor metrics,
updated weights, optimizer state, batch and RNG must match. Checkpoint steps 0,
1 and 2 are evaluated independently using the same frozen Base-only pool.
`acceptance.json` is written only after all updates and checks complete.

These miniature tasks test plumbing, not improvement, convergence or reproduction.
Flat validation scores are legitimate; a failed acceptance remains a failed run.

## Prompt and intervention provenance

`grid_prompt_profile` / `coding_prompt_profile` may explicitly select
`action_only_v2`. Existing configs retain `legacy_v1` behavior and hashes. V2
prompts improve action-format instructions without silently repairing model output
or changing strict parsers; they are toolkit prompts, not paper-final prompts.

Some Base rollouts contain no geometric anchor for a distance-increasing wrong
action. `ard pool witness --config CONFIG --budget 8 --size 4 --output runs/witness`
builds independent BFS witness prefixes. The smoke script uses this **separate**
oracle-reference pool only for recovery when Base anchors are absent, and records
`independent_exact_geometry_witness_not_base`. It never inserts witness states
into the Base monitor or claims Base-distribution recovery evidence.
