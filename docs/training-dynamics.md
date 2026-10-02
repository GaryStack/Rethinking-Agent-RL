# Training dynamics without Torch, Ray or W&B

```bash
ard dynamics --run /path/to/completed/training-run \
  --expected-steps 2 --output runs/training-curves
```

Open `runs/training-curves/report.html`. This installed core command audits every
available `diagnostics/validation-N` from its complete raw budget and fixed-state
records and the original frozen `pool/`. It creates a JSON series, a static SVG
curve/report and a hashed artifact manifest. No model loading, inference, network,
Torch/Ray import or W&B account is needed. The script alias is
`scripts/report_training_dynamics.py` with the same arguments.

The full validation protocol must be constant: tasks, prompts, sampling,
environment, pool and split. Only checkpoint path/revision may change. Changing
the protocol makes the report fail, rather than draw incomparable curves.
`--expected-steps N` means checkpoint points 0 through N; absent or incomplete
points remain unavailable, never zero. Lines do not bridge missing points.

An adjacent interval is descriptively flagged when any Pass@(K,T) increases
while single-step Q*/test fraction decreases. These are observed finite-sample
changes, **not statistical significance, a causal training effect or evidence of
mechanistic degradation**. This report does not certify optimizer updates, resume
or paper reproduction. Q* and coding test fraction remain separately labeled;
they are not interchangeable scores. The metrics use a fixed [0,1] plot axis.

Use `--verify-weights` only where the original checkpoints still exist: this
independently rehashes them. Otherwise the report audits recorded checkpoint
identities and raw evaluation data and explicitly says `weights_rehashed=false`.
The dev7/dev8 exported evidence intentionally excludes model/optimizer weights.

## Portable archived evidence

Auditing preserves each original absolute model path as part of the recorded
configuration identity. A different host's symlinks must not silently rewrite
that path and change the config hash. Live inference still resolves model paths
normally. Recorded paths are not rebased, and original artifacts/hashes are never
rewritten to suit the local machine. Relative recorded model paths and external
task-file references are rejected. Weight verification remains a separate actual
filesystem check, not a claim based on stored hashes.

The tool was exercised on actual dev7 fixed/BudgetMix exported raw evidence.
Their tiny held-out Pass/Q* curves remain zero; no improvement or tradeoff is
invented. Unit tests use explicitly synthetic model/checkpoint fixtures and
independently verify raw replay and protocol-drift rejection.
