# Dev10 verification — 2026-10-02

Scope: complete the local paper-companion/toolkit preview, preserving deferred
formal reproduction and full trainer parity. No GPU training job, driver reset,
remote publication or upstream source modification was performed.

## Native process-signal chain

`scripts/verify_process_signal_cpu.py` verifies the 437-file upstream lock at
`7dc46e834209948cf1cdd8a04d83f82b4a7efd24`. It uses two explicitly scripted
FrozenLake action sequences and a freshly initialized tiny GPT2, not a research
checkpoint. It calls actual `GridAgentLoop`, `AgentLoopWorker` padding/reward
placement, `compute_advantage(reinforce_plus_plus)` and
`DataParallelPPOActor.update_policy` with AdamW.

```bash
CUDA_VISIBLE_DEVICES= GLOO_SOCKET_IFNAME=lo PYTHONPATH=src python scripts/verify_process_signal_cpu.py --verl-source YOUR_LOCKED_VERL --output runs/process-native-cpu
```

Actual run used the existing isolated `/tmp/ard-ray-env-cXj8GA/bin/python` and
read-only `/tmp/ard-verl-7dc46e-1MD5xz` source. The sole test-only upstream adaptation
is `dp_actor.get_device_id -> torch.device(cpu)`; reward/advantage/loss/optimizer
implementations were not replaced. Gloo binds loopback for a single CPU rank.

Both scripted trajectories fail, with terminal reward zero. Coefficient 0.5
produces rewards [0.2375, 0], nonzero masked advantages, gradient norm
1.7680033445358276 and 17 changed tiny-model tensors. The coefficient-0 control
has zero rewards, zero gradient and zero changed tensors. Validation stays
terminal-only with rewards [0,0]. Response/observation/padding masks are checked,
and no CUDA initialization occurs. Private evidence is retained at
`runs/dev10-verification/native-process/verification.json`.

This verifies a narrow native CPU reward-to-update chain. It does **not** verify
RayPPOTrainer.fit, vLLM generation, FSDP/GPU, research-model process training,
critic/GiGPO training, resume or paper reproduction. The real-model Ray launcher
now accepts paired `--process-coefficient` / `--process-failure-policy` flags and
records them, but its new process-enabled GPU mode remains experimental.

The first CPU pilot stopped at Gloo initialization because the sandbox could not
resolve/bind loopback; no optimizer update occurred. A fresh output with authorized
loopback access succeeded. The failed-pilot record is retained, not relabeled.
The local NVIDIA preflight also reports driver unavailable; no driver repair or
GPU model job was attempted. Both records are under ignored `runs/dev10-verification`.

## Correctness and release changes

- Enabled process signals require explicit `extra_info.ard_phase` (train or
  validation); only training receives mixing. Missing phase fails before rollout.
  Native validation and success metrics retain terminal semantics.
- Invalid process scores cannot hide behind a missing score with terminal fallback.
- Likelihood length controls exclude whole trajectories affected by supplied-event
  scoring failures/missing policy scores; absent trajectory identity disables the
  control. Completeness is scoped to supplied events, not unknown original records.
- Malformed/legacy score records fail with schema errors rather than accidental
  tracebacks or implicit format conversion.
- Package inspection now checks wheel code paths, hidden credential directories,
  ZIP links and member sizes before reading payloads. Example README provenance
  notes are included in the source distribution. CI covers new wheel commands.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /opt/anaconda3/bin/python -m pytest -q -p no:cacheprovider
ruff check .
ruff format --check src tests scripts examples
```

Staged suite: **268 passed, 3 skipped**, 7.41s. Skips are unchanged opt-in namespace
checks. Ruff 0.11.13: all checks passed, 132 Python files formatted. The independent
pytest interpreter is not the project `.venv`. New native CPU evidence is recorded
separately from this stdlib regression; dev9's real fixed-response 1.7B HF smoke
remains valid and was not needlessly repeated.

Version `0.1.0.dev10`: wheel and source archive are rebuilt; payload inspection,
fresh no-dependency wheel install, demo/audit, compare, likelihood fixtures,
BudgetMix plan/audit and external/process/dynamics examples are checked. The
source archive also rebuilds a wheel in isolation from the live checkout.
Target sync is followed by the same CPU/Ruff checks and byte-identity verification;
final commands/results are recorded in the implementation handoff.

Metadata authors/paper URL remain pending user confirmation. No GitHub remote,
push, PyPI upload or online W&B is created. Historical negative off/on evidence
and all scope limitations remain unchanged.
