# Observed actor-state capture and comparison

Dev5 adds an opt-in, read-only fingerprint path around the unchanged pinned
upstream actor update. It does not replace the optimizer or modify advantages.
Without `--capture-training-state`, the worker delegates directly to its native
parent after the existing nonzero/finite training-signal check.

## Verified scope

Dev8's seeded actual GPU pair matches all actor RNG fingerprints, but one
second-update rollout and subsequent training state differ. Explicit one-time
initialization is now implemented; complete parity is still not accepted. See
[initialization protocol and result](actor-initialization.md).

Dev7 executed the real A100/Qwen3-1.7B off/on pair. All 128 normalized training
completions and non-RNG actor fingerprints match, but all four RNG fingerprints
differ before/after both updates. The report remains `different`. See
[portable acceptance](remote-acceptance.md) for the negative result and required
future one-time initialization protocol. This is genuine GPU capture evidence,
not an accepted complete parity claim.

A real tiny CPU `Linear` model performs AdamW, clipping and StepLR updates twice
from identical weights and Python/NumPy/Torch RNG states. One branch also runs a
deterministic diagnostic computation. Parameter contents change and all observed
off/on state hashes match. Repeated capture leaves observed state unchanged.
This is optimizer instrumentation evidence, not Qwen, PPO, Ray or GPU acceptance.
The default stdlib suite additionally covers worker delegation, ordering and
immutable proof files using explicitly labeled mocks.

With Torch and NumPy available, reproduce the CPU check in a new directory:

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/verify_training_state_cpu.py --output runs/state-cpu
python scripts/compare_training_state.py --left runs/state-cpu/off \
  --right runs/state-cpu/on --output runs/state-cpu-comparison.json
```

The comparator itself needs only the installed core package; Torch is not imported.
It rejects self-comparison, missing components, noncontiguous updates, zero or
nonfinite gradient norms, unchanged parameters and mutated batch tensors.
Differences are reported by update and state field; a differing comparison exits 1.

## Repeat the real GPU pair

First pass the bounded runtime/CUDA preflight on an idle authorized GPU. Use the
isolated version freeze and clean verl SHA from `ray-smoke.md`. Keep the Base pool,
monitor configuration, model, training settings and seeds identical. Use two fresh
outputs, the same physical card, and run sequentially with offline W&B:

```bash
export CUDA_VISIBLE_DEVICES=YOUR_SINGLE_AUTHORIZED_PHYSICAL_GPU
export WANDB_MODE=offline HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/absolute/clean/verl:/absolute/this/repo/src
python scripts/run_verl_ray_smoke.py \
  --model-path /absolute/local/Qwen3-1.7B --verl-source /absolute/clean/verl \
  --monitor-config /absolute/monitor.json --pool /absolute/base-pool \
  --output runs/state-off --steps 2 --train-budget 2 --group-size 16 \
  --capture-training-state --diagnostics-off
python scripts/run_verl_ray_smoke.py \
  --model-path /absolute/local/Qwen3-1.7B --verl-source /absolute/clean/verl \
  --monitor-config /absolute/monitor.json --pool /absolute/base-pool \
  --output runs/state-on --steps 2 --train-budget 2 --group-size 16 \
  --capture-training-state
python scripts/compare_training_state.py --left runs/state-off/state-proofs \
  --right runs/state-on/state-proofs --output runs/state-comparison.json
```

`--diagnostics-off` skips our initial extra checkpoint diagnostic and checkpoint
sidecars, but retains native validation, exports, optimization and offline training
logging. Its output intentionally lacks dual-validation acceptance artifacts;
do not run the existing diagnostics-on acceptance script and expect it to pass.
`trainer-returned.json` records both switches for provenance. The generic proof
comparator does not independently certify that the two runs used opposite modes;
retain their launcher records and resolved configurations alongside the comparison.
`compare_ray_training_pair.py` additionally checks opposite modes, resolved
settings/source/environment/pool identity and every training completion. It
rejects missing slots and compares seeds/tokens/state, excluding only request IDs.

## Boundaries and cost

Capture requires one rank and CPU-offloaded parameters, optimizer state and input
tensors at both boundaries. It fails explicitly otherwise. Small device buffers
(at most 64 MiB each) may be copied; parameter/optimizer GPU copies are rejected.
CPU contiguous tensors are hashed in 1 MiB memoryview chunks; noncontiguous tensors
need a contiguous temporary. No weights or raw optimizer state are saved.
CUDA RNG capture requires an already initialized actor, not implicit CUDA startup.

Each immutable `state-proofs/update-XXXXXX.json` records hashes of parameters,
buffers, optimizer, scheduler, module training flags, batch tensors, Python RNG,
NumPy RNG, CPU Torch RNG and local actor CUDA RNG, plus actual gradient norms,
framework versions and capture time. Capturing costs a complete state scan twice
per update; enable it for small acceptance runs, not unattended large training.

The scope is the observed actor-worker boundary. Controller RNG, rollout-server
RNG, trajectory identity, distributed collectives and optimizer resume are NOT
covered. A match always leaves `full_trainer_parity_verified=false` and
`distributed_parity_verified=false`. Complete P3 requires additional actual
GPU/controller/rollout evidence and independent same-export output comparison.
