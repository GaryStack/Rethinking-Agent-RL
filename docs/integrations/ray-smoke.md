# Verified one-rank Ray/vLLM smoke

This optional recipe exercises the **actual pinned upstream** `RayPPOTrainer`,
FSDP actor and async vLLM server with a custom multi-turn grid agent loop.
It is an integration fixture, not a paper algorithm or benchmark recipe.
Current acceptance status is recorded in `../verification.md`; import-only
preflight success never establishes training acceptance.
The 2026-09-29 Qwen3-1.7B run passed the raw-record, changed-tensor and offline
W&B audit. A separate public CLI rerun of these exports and multi-rank acceptance
remain pending; see the dated record for current-host infrastructure failures.
Dev5's opt-in state capture and diagnostics-off switches are described in
[training-state comparison](training-state-parity.md). These new switches are
not part of the already verified dev3 GPU run and await separate GPU acceptance.

Latest dev7: a remote A100 run separately exercised off/on capture and BudgetMix,
accepted fixed-on/BudgetMix raw/tensor/W&B evidence, and matched standalone CLI
records on the exact new Ray step-1 export. Off/on RNG fingerprints differ, so
complete parity is not accepted. The earlier pending statements above describe
dev3's original run only. See [portable acceptance](remote-acceptance.md).

## Isolation and source verification

Use a **new, non-inheriting** Python 3.11 Linux x86_64 environment, not an existing
research conda environment and not `venv --system-site-packages`. Install the
complete candidate snapshot in `requirements/verl-ray-candidate-py311-linux.txt`.
This is a version freeze, not a wheel-hash or cross-platform lock. The smoke
requires a locally available Qwen3-1.7B and one explicitly selected CUDA GPU.
Do not publish its weights. Set W&B to offline; this recipe does not upload.

Use a clean external verl checkout at
`7dc46e834209948cf1cdd8a04d83f82b4a7efd24`. Before imports, our source verifier checks
all 437 Python/YAML files and rejects additions or modifications. No upstream
source patches, dirty editable installations, or invented callback APIs are used.
Set `PYTHONPATH=/absolute/clean/verl:/absolute/this/repo/src` so optional subprocesses
also use these verified sources. The core package remains dependency-free.

First run the bounded CUDA check in
[runtime and comparison tools](runtime-and-comparison.md), explicitly selecting
the physical GPU. Idle NVML memory alone does not establish CUDA health.

The observed wheel set also needs Arrow loaded before vLLM actor deserialization.
The recipe uses Ray's actual `worker_process_setup_hook` to preload the pinned
Arrow module in workers; this is a documented environment workaround, not a
universal fix or a change to model outputs/training logic. The local object store
is explicitly bounded to 512 MiB instead of reserving a large fraction of host RAM.

## Prepare a genuinely Base-collected held-out pool

Copy `configs/examples/frozenlake_ray_monitor.json` and explicitly edit its model
path and Base revision. Its two authored validation maps are disjoint from the
four training corridors by both task IDs and map content.

```bash
export CUDA_VISIBLE_DEVICES=YOUR_SINGLE_PHYSICAL_GPU
export WANDB_MODE=offline HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ard eval budget --config /absolute/monitor.json --output runs/ray-base
ard pool build --input runs/ray-base --output runs/ray-base-pool \
  --base-model-id qwen17-base-monitor --size 8 --min-tasks 2
python scripts/run_verl_ray_smoke.py \
  --model-path /absolute/local/Qwen3-1.7B --verl-source /absolute/clean/verl \
  --monitor-config /absolute/monitor.json --pool runs/ray-base-pool \
  --output runs/ray-smoke --steps 2 --train-budget 2 --group-size 16 --preflight-only
# Remove --preflight-only to run. Choose a new output if a previous run exists.
```

If the Base collection cannot supply enough distinct states, collect more samples
or reduce the quota. Never replace this monitor with BFS/oracle-generated states.
Preflight checks actual imports and vLLM's architecture-inspection child but does
not create a trainer. A finished trainer only emits `trainer-returned.json`; that
file explicitly requires separate acceptance and is not itself a pass certificate.

## Actual integration points

`integrations/verl_grid.py` implements upstream `AgentLoopBase.run`: real generated
token IDs drive strict grid actions; generated token masks are 1 and observation
masks 0. Budget-visible feedback is appended through upstream chat templating.
Each request records sampling seeds, the server's actual weight version, prompt/
response IDs, masks and before/after states. Missing/mixed versions, token-cap
shortfalls and transport failures raise errors rather than becoming task failures.
`integrations/grid_trace.py` independently replays these traces on the CPU.

The locked legacy async FSDP worker drops `global_steps` when calling its rollout
weight transfer. `integrations/versioned_worker.py` narrowly subclasses that worker:
it awaits the **unchanged upstream weight transfer**, then awaits the real server's
version-tag RPC. Failed transfers never acquire a tag, and multi-rank use is
explicitly rejected. This is provenance attached to the completed barrier, not a
fallback inferred from requested trainer steps. Upstream files remain unchanged.

`integrations/ray_checkpoint.py` subclasses the real trainer's synchronous
`_save_checkpoint`, waits for the upstream commit marker and invokes the public
checkpoint-evaluation CLI in an isolated subprocess. Both multi-turn validation
and single-decision Q* reuse the existing evaluators and frozen Base pool. The
same-GPU recipe requires actor parameter offload and rejects async save and
unmerged LoRA. The sidecar finishes before upstream rollout weight updates.
Native `_validate` supplements the upstream driver's own logging with checkpoint-
aligned `diag/*` metrics, not a second W&B run. Export/barrier overhead is recorded.
The controller explicitly calls the owning W&B run's `finish()` before returning
to the driver; relying on object destructors across Ray shutdown dropped history
in an earlier pilot. Only SDK-exported complete history passes acceptance.

## Fail-closed acceptance

Export the finished offline `.wandb` file with `scripts/export_wandb_history.py`
(local SDK scanner, no sync), then run:

```bash
python scripts/audit_verl_ray_smoke.py --run runs/ray-smoke \
  --base /absolute/local/Qwen3-1.7B --history runs/ray-smoke/trainer-history.jsonl \
  --wandb-history runs/ray-smoke/offline-history.jsonl \
  --output runs/ray-smoke/acceptance.json
python scripts/report_training_diagnostics.py \
  --input runs/ray-smoke/tracking/metrics.jsonl --output runs/ray-smoke/report.html
```

Acceptance requires at least two completed steps, finite nonzero actual actor
gradients, changed **tensor bytes** rather than merely changed file metadata,
all rollout weight versions, actual multi-turn feedback and exact diagnostic
values in native/W&B history. Tensor identity uses a streaming, read-only subset
of the [official safetensors format](https://github.com/safetensors/safetensors#format),
ignoring shard layout and header metadata. Unsupported packed formats fail closed.

## Scope limits

One GPU uses upstream FSDP's one-rank NO_SHARD behavior: it is not multi-rank
distributed acceptance. The algorithm is explicit `reinforce_plus_plus`, terminal
success rewards and four tiny training maps. Zero-gradient/unchanged checkpoints
do not count as accepted training. HF-only exports do not verify optimizer resume.
The one-worker seed counter is smoke-only, not a distributed/resume sampler.
The upstream PPO actor uses the `reinforce_plus_plus` advantage estimator with no
critic; this is not a claim to reproduce the paper's PPO/GRPO configuration.
Training sampling is explicitly temperature/top-p 1/1; validation is .6/.9.
The training budget and group size are configurable. Small samples can produce
all-zero rewards: the smoke worker rejects nonfinite/all-zero generated-token
advantages **before** invoking the native optimizer, not just after checkpoint
auditing. Such failures remain negative evidence, not accepted training.
Full-trainer diagnostics off/on equivalence remains separate from the already
verified native actor off/on checks. Multi-rank synchronization, BudgetMix,
GiGPO and formal paper reproduction remain open.

After an independent CLI rerun, `scripts/compare_checkpoint_validation.py` audits
both validation directories and compares complete points plus both raw record
hashes. Its default mode needs neither model loading nor a GPU.
