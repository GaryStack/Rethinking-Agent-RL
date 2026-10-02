# Explicit actor RNG initialization

The optional one-rank smoke switch `--actor-seed 5422` seeds Python, NumPy,
Torch CPU and local actor CUDA once, immediately before the unchanged native
worker's `init_model`. It accepts uint32 integer seeds, rejects repeat seeded
initialization and records `state-proofs/initialization.json` after native model
initialization completes. The run/config records contain the same protocol and
seed. Default smoke behavior remains unseeded/native.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_ray_acceptance_suite.py \
  --actor-seed 5422 --model-path /path/to/Qwen3-1.7B \
  --verl-source /path/to/clean/locked/verl --output /path/to/new/seeded-suite
```

This does **not** reset RNG before updates, checkpoint diagnostics or fingerprints.
It does not fix server/controller initialization, request scheduling, asynchronous
batching or backend numerical nondeterminism. No tolerance or ignored RNG field
is added to the comparator. `actor_rng_initialization_recorded` and
`observed_actor_rng_match` do not mean `full_trainer_parity_verified`.

A genuine CPU initialization regression with new tiny Linear models, AdamW,
input tensors and actual Python/NumPy/Torch random draws is available:

```bash
CUDA_VISIBLE_DEVICES= python scripts/verify_worker_seed_cpu.py --output runs/seed-cpu
```

Same-seed states/draws match, different seeds differ, and repeated state capture
does not change RNG. This CPU test is not GPU training evidence. The actual
seeded A100 off/on pilot does match all four RNG fingerprints, but differs in one
second-update rollout (`RIGHT,LEFT` versus `RIGHT,RIGHT` at the same first-action
seed). Second-update input/gradient/parameter/optimizer differences are retained.
The strict pair report is still `different`; backend/server noninterference
remains open. The earlier unseeded dev7 negative report is also preserved.
