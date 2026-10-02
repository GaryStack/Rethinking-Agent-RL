# Portable one-GPU acceptance

Dev8 adds optional `--actor-seed 5422` to the suite. It initializes actor RNG once
before native model initialization and records the protocol. Actual seeded RNG
fingerprints match, but one rollout/second-update state still differs. See
[actor initialization](actor-initialization.md); the unseeded pilot described
below remains preserved rather than relabeled as a pass.

This is a real acceptance runner, not a simulated training demo. Run it on a
compute host with an explicitly authorized, idle GPU and a local Qwen3-1.7B
checkpoint. It does not handle SSH credentials, download weights, upload W&B
results, reset GPUs or modify an existing environment.

Use a new isolated Python 3.11 environment and install the complete
`requirements/verl-ray-candidate-py311-linux.txt` snapshot, then this project.
An existing environment that merely imports Torch is not sufficient: one observed
host had incompatible Transformers/Hugging Face Hub dependencies. Never repair
someone else's environment in place for this test.

The upstream source must pass the project's 437-file lock at
`7dc46e834209948cf1cdd8a04d83f82b4a7efd24`. Read-only shared model storage is
supported; no weight copying is required.

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python scripts/run_ray_acceptance_suite.py \
  --model-path /path/to/Qwen3-1.7B \
  --verl-source /path/to/clean/locked/verl \
  --output /path/to/new/acceptance-run
```

The runner performs a bounded CUDA allocation probe, actual Base-only monitor
collection with task-ID **and map-content** holdout checks, separate fixed-budget
diagnostics-off/on training, and BudgetMix training. Each training job performs
two real native optimizer updates with sixteen completions per task occurrence.
It exports actual offline W&B SDK history and independently audits checkpoint
content changes, rollout replay, diagnostics and BudgetMix assignments. Finally,
it compares observed off/on actor state and rollouts and runs the standalone CLI
on the exact step-1 export. Read `progress.json`, `logs/` and the final
`verification.json`; a nonzero comparison result remains a difference, not a pass.

All tasks here are authored small grid fixtures. The smoke algorithm is
REINFORCE++, not paper GRPO/PPO/GiGPO replication. BudgetMix's smoke maximum is
two actions; this does not establish performance at the paper's larger budgets.
Full controller RNG, rollout-server RNG, multi-rank and optimizer-resume parity
are not certified, even if observed actor fingerprints match.

An actual off/on pilot produced identical 128 training completions and matching
parameter/buffer/optimizer/scheduler/input/mode fingerprints, but different Python,
NumPy and Torch CPU/CUDA RNG fingerprints at both boundaries of both updates.
The report correctly returns `different`. The locked worker does not explicitly
initialize these RNGs in the inspected FSDP path; unpaired initialization is a
possible explanation, not an established diagnosis. A subsequent parity
experiment needs explicit, identical one-time initialization **before native
model/worker initialization**, with the initialization protocol recorded. Never
reset RNG immediately before capture, drop RNG fields or relax equality to make
this pilot pass. Preserve the original negative comparison.

## Failure handling

- Use a fresh output directory after each failed pilot; retain original logs.
- Ray needs short UNIX socket paths. The launcher creates a private `/tmp/ard-r-*`
  directory and records its exact location in `ray-runtime.json`. It retains Ray
  logs after shutdown and does not delete shared Ray directories.
- State capture supports the pinned native AdamW's OmegaConf `betas` container
  without modifying the optimizer. The resolved container contents and type are
  fingerprinted. A real CPU regression is available:
  `python scripts/verify_training_state_cpu.py --omegaconf-betas --output /path/to/new/cpu-proof`.
- Model/optimizer/input tensor capture still requires CPU offload, except bounded
  small model buffers. Unsupported/nonfinite states, zero advantages, missing
  traces and zero/invalid gradients fail closed.
- Output artifacts and W&B stay local/offline. Do not bundle checkpoints, caches,
  private research assets or login secrets in a public release.
