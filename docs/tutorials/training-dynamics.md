# Training dynamics monitor

Freeze a Base-only pool once, with `--training-task-ids TRAIN_IDS.json` to check
task-ID disjointness. This does not prove content-level separation. Keep the pool,
decoding, sample counts and task split fixed across checkpoints.

```bash
ard eval checkpoint --config LOCAL_VALIDATION_CONFIG.json --pool runs/pool --checkpoint YOUR_LOCAL_CHECKPOINT --checkpoint-step 1 --samples 8 --output runs/checkpoint-1
```

For a paired training-series directory produced by the pinned integration:

```bash
ard dynamics --run YOUR_TRAINING_RUN --expected-steps 2 --output runs/dynamics
```

Recommended integration: export a checkpoint and run evaluation in a separate
process; this makes cost and checkpoint identity explicit. Every checkpoint runs
multi-turn and fixed-state validation, namespaced independently under validation.
The upper bound on single-step generations is states times samples; multi-turn
cost adds action budgets. Report missing checkpoints and protocol changes rather
than drawing fabricated continuous curves. Warnings are descriptive.

See [checkpoint integration](../integrations/checkpoint-validation.md),
[series schema and fixture instructions](../training-dynamics.md) and
[offline W&B](../integrations/ray-smoke.md). The tested pinned one-rank path retains
an off/on rollout/state difference; diagnostic noninterference, multi-rank and
resume are unverified. This preview does not require a new training run.

An authored metric-only preview is executable without models:
`python examples/dynamics_fixture.py --output runs/dynamics-fixture`. It does not
claim raw training audit; real series must use `ard dynamics`.
