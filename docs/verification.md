# Verification record

This public copy replaces personal model/interpreter paths with portable
placeholders. The unmodified record is retained privately; reported outcomes,
failed pilots and parity differences are unchanged.

## 2026-10-01: dev8 seeded actor and portable training curves

An explicit optional one-time RNG initialization runs before native actor
init_model, not update/capture. Real tiny CPU initialization/AdamW/input/random
draw regression passes for equal seeds, differs for unequal seeds, and capture
leaves state unchanged. Default native init/update behavior remains unchanged.

A new complete A100/Qwen3-1.7B suite with actor seed 5422 completes fixed off/on
and BudgetMix two-update jobs and accepts independent fixed/BudgetMix tensor,
gradient, raw trajectory, checkpoint dual validation and actual offline SDK W&B
audits. Same Ray step-1 export standalone raw outputs match again, with weights
rehash verified remotely. Original files are retained under
`runs/qwen17-seeded-dev8/acceptance-dev8-seeded` (weights/caches omitted).

All observed actor RNG hashes match before/after both updates; first-update actor
state matches. **Second-update comparison is still negative:** one completion in
`train-left-one` differs (`RIGHT,LEFT` vs `RIGHT,RIGHT`, first-action seed
1144144618), followed by different batch/gradient/parameters/optimizer. No cause
is established; server/controller/batching/backend reproducibility remains open.
The strict comparison exits 1. Old dev7 negative evidence is not replaced.

New installed `ard dynamics` audits raw checkpoint children and fixed protocol,
outputs JSON/hashed manifest/static SVG curves, preserves missing/incomplete
points and gives descriptive adjacent tradeoff warnings without significance or
causal claims. Real dev7/dev8 fixed and BudgetMix evidence has been audited locally
without optional ML packages. Curves are zero on these tiny held-out fixtures;
no fake learning improvement is displayed. Report weights_rehashed=false when
using exported evidence with no weights. Actual remote acceptance already rehashed
the original weights independently.

Cross-host audit now preserves recorded absolute model paths rather than changing
config hashes via local symlink resolution. Live inference still resolves paths
normally. Recorded relative model paths/external task files are rejected. The
pair comparer derives its original output namespace from recorded trainer metadata,
not the current export location. Original artifacts/hashes are not rewritten.
This allows stdlib-only re-audit/recompare of the actual dev7/dev8 exports.

Core suite: **232 passed, 3 opt-in namespace checks skipped**; Ruff covers 112
Python files. Synthetic regression fixtures are separately labeled, including
full raw replay, protocol-drift rejection, missing-point gaps and init guards.
Evidence archive SHA256:
`8100f8e00c5c520272a3d044d921af16a2e669fbbbe1b88d25c55d9f2fec2877`.
The final suite also passes in the actual locked Python 3.11 remote environment:
232 passed / 3 skipped; 112 files pass Ruff. Final wheel: 127,276 bytes, SHA256
`5e82fbc3fb7c1c1aef5e3b8bc831ae61f8531aab8397136f7134360c6ef4d0c6`.

See `training-dynamics.md` and `integrations/actor-initialization.md`.

## 2026-10-01: dev7 remote A100 acceptance

Explicit operator authorization enabled isolated remote testing. Mounted weights
and research files are read-only. The remote A100 passed a real CUDA tensor
allocation in a fresh Python 3.11 environment with the complete locked snapshot:
Torch 2.9.0+cu128, Ray 2.54.0, vLLM 0.11.2, Transformers 4.57.1, NumPy 1.26.4,
Arrow 21.0.0. Upstream VeRL passes the 437-file source lock. No original environment
or driver was modified, no shared GPU reset occurred, and W&B remained offline.

Artifacts: `runs/qwen17-remote-dev7/acceptance-v3/verification.json` and the
fixed-on/BudgetMix `acceptance.json`, `pair-comparison.json`, and
`standalone-comparison.json`. Model/optimizer weights and caches are intentionally
not copied into the evidence bundle. Actual weight hashing/auditing occurred on
the remote host, whose original checkpoints remain in its private work directory.

The real Base-only pool has eight held-out states, with disjoint task IDs and map
contents. Each of fixed-off, fixed-on and BudgetMix completes two real updates.
Fixed-on has 134 audited traces, 311 common tensors, changed counts 257/252 and
gradient norms 125.5/314. BudgetMix has 134 traces, changed counts 257/256 and norms
164/188. Each BudgetMix update has four groups × sixteen completions, histogram
{1:1,2:3}, mean 1.75; actual raw groups, planned occurrences, tokens/masks/rewards
and native/offline SDK W&B histograms agree. Both diagnostic-on jobs have three
checkpoint-bound dual-validation points independently recomputed and rehashed.

The standalone CLI on the exact fixed-on step-1 export matches both raw files:
budget trajectories SHA256 `76ac055af5569860391682b8752618411ad080324f516f83b7d84d1621c3a0d5`;
single-step actions SHA256 `7f0f2ca6318fbc0486df4a9089eb48259248298648a3d5646a90137d6f3a732b`.
Weights were rehashed independently. This closes the same-export CLI gap.

**Negative parity result retained:** off/on has eight training groups and 128
completions per run; all normalized training records and non-RNG actor-state
fingerprints match, but Python/NumPy/Torch CPU/CUDA RNG fingerprints differ
before and after both updates. Comparison exits 1 and `paired_observations_match`
is false. No tolerance, skipped field or artificial RNG reset makes it pass.
Full trainer/controller/server/distributed parity remains unverified.

Retained pilots: incompatible pre-existing HF Hub dependency; acceptance-v1 UNIX
socket path limit; acceptance-v2 unsupported OmegaConf AdamW betas. The project
now creates private short Ray runtime paths and fingerprints the resolved native
OmegaConf container without mutating state. Actual CPU AdamW regression passes
locally and remotely with that container. Each failure's original logs remain.

The GPU candidate reports package metadata `0.1.0.dev6`: these new runtime changes
were tested before the final dev7 version bump. Final dev7 adds documentation and
admission tests; this metadata is not retroactively rewritten in GPU artifacts.
Core suite: **207 passed, 3 opt-in namespace checks skipped**; Ruff passes for 105
Python files. Fresh non-inheriting no-dependency wheel install passes doctor,
BudgetMix plan/audit, readiness and the 1,152-trajectory synthetic demo/replay.
These authored-grid REINFORCE++ smoke tests are not GRPO/PPO/GiGPO or paper results.

Final wheel: 121,835 bytes; SHA256
`58da814d48f845beee554837841c08d67f27fe8d1d9e787b429d87824fb412c2`.
Evidence archive: SHA256
`09e8937bf3b10a6f104b29c9266a6e3a9486fcb693e5b56eb197325d6c7124fb`;
see the retained `EVIDENCE.md` for missing-weight/path-rebasing limitations.

See `integrations/remote-acceptance.md` for the runnable recipe and remaining scope.

## 2026-10-01: dev6 CPU BudgetMix, asset acquisition and packaging

The operator explicitly deferred GPU acceptance and authorized continued CPU
tool development. No CUDA inference/training, reset or driver repair was attempted.
The morning bounded checks on physical 0/1 returned cuInit=100; their reports
remain under `/tmp/ard-gpu{0,1}-health-20261001-064954.json`.

Implemented dependency-free unique ceil-decile sampling, immutable occurrence
plans/audits, bootstrap/anchor/degenerate-step contracts, an opt-in pinned raw-grid
pre-repeat trainer wrapper and completed BudgetMix group replay. Existing fixed-T
smoke commands remain default. BudgetMix generation metadata and histogram logging
are explicit; plan files are not executed rollouts. Guide: `integrations/budgetmix.md`.

Actual CPU execution in `/tmp/ard-dev6-budgetmix-cpu-final-20261001` imported the
verified clean upstream trainer, called its native `_get_gen_batch` and real
DataProto.repeat, and used the local Qwen3-1.7B tokenizer. Two logical update
plans each had three occurrences and four assigned completion slots per group;
duplicate task IDs stayed distinct and native uid grouping was retained. All
three groups per update were below the authored map's exact three-action minimum
and were NOT filtered. Histograms were {1:2,2:1} then {1:1,2:2}; means 4/3 and 5/3.
No model weights, optimizer updates, Ray cluster or CUDA initialization occurred.
Scope is actual CPU native hook/repeat/tokenizer, not full trainer/GPU acceptance.
Retained artifact: `runs/dev6-cpu-tools/native-budgetmix/verification.json`.

Suite: **190 passed, 3 explicit namespace checks skipped**; Ruff/check formatting
pass for 95 Python files. New tests cover unique probabilities for small Tmax,
large exact integer rounding, same-task distinct occurrences, global RNG neutrality,
complete groups, prompt rebuilding, unchanged validation, unsafe-input rejection,
one-time histogram metrics, zero-bootstrap, budget-aware anchors, degenerate step
signals, replayed infeasible groups and forged/rehashed plan metrics. Default suite
fixtures remain labeled, distinct from the optional actual native CPU check.

Targeted read-only research asset inventory was repeated on 2026-10-01. The final
PDF hash still matches. No source artifact was migrated or granted distribution
approval; all nine experiment checklists remain acquisition-pending. Original
published ledger is preserved; fresh inventory/checklist are in
`runs/dev6-cpu-tools/{assets,readiness}`. Clearing metadata cannot make the new
readiness tool issue a paper reproduction certificate.

Built final dev6 wheel: 118,089 bytes, SHA256
`f83306b5ad8e528c9d7d5ebebd15b502bcb4e91ad47ab50321005a61d0e06bae`.
A new non-inheriting venv `/tmp/ard-dev6-final-install-NefkWR` installed it with
`--no-index --no-deps`. From `/tmp` with PYTHONPATH cleared, installed doctor,
BudgetMix plan/audit with Tmax=24, fresh-ledger readiness, offline demo and complete
replay/hash/metric audit passed with no Torch/Ray/verl/vLLM/W&B installed. This
independent test caught a JSON integer-versus-string histogram key ordering bug;
it was fixed and covered by the Tmax=24 CLI regression test before final release.
CI configuration now exercises plan/audit and readiness; no remote CI run is claimed.

Pending: GPU BudgetMix updates and actual W&B history, standalone Ray-export
parity, full trainer off/on, distributed acceptance, PPO critic/GAE/save/resume,
GiGPO training, process reward integration and formal paper data/protocol migration.

## 2026-09-30: dev5 actor-state instrumentation and host checks

Actual CPU AdamW/StepLR updates with a tiny Linear model passed observed-state
off/on comparison in `/tmp/ard-dev5-state-cpu-20260930`; retained under
`runs/dev5-state-tools/cpu`. All ten components match; parameter contents changed,
gradient norms are nonzero and repeated capture leaves state unchanged. This is
not a Qwen/verl/GPU run. New Ray capture and diagnostics-off paths have contract
tests but have NOT executed on GPU. See `integrations/training-state-parity.md`.

Suite: **149 passed, 3 explicit namespace checks skipped**. Ruff and formatting
pass for 82 Python files. Fixtures are labeled and do not substitute for training.

At the operator's request, SIGKILL was sent to exact PID 3369205 and its remaining
thread 3372073. The zombie leader and running residual thread remained, with
SIGKILL pending; memory on physical 6/7 did not release. No parent shell was killed
and no GPU reset, driver restart or reboot was attempted on this shared machine.
Separate bounded real Torch checks on physical 0,1,2,3,4 all imported Torch
2.9.0+cu128, then timed out allocating a 16-element CUDA tensor. Each reached the
15-second timeout plus one-second reap deadline; none proved CUDA readiness.
Probe PIDs were absent on a later read-only check. Report retained as
`runs/dev5-state-tools/five-cards.json`; `usable_indices=[]`. Idle NVML memory is
not evidence of usability. GPU 5 reports an error and 6/7 remain occupied.
No further model/training starts are attempted until CUDA availability recovers.

Later read-only check at 11:29 UTC: both old PID/TID were absent, but all
`/dev/nvidia*` device nodes were also absent and `nvidia-smi` could not communicate
with the driver. Thus the process is no longer observed; successful memory release
or GPU recovery cannot be measured. This project did not execute any GPU reset,
module unload, driver restart or reboot. The cause of this host transition is
unknown; preserve it rather than inferring a successful CUDA recovery.

Built dev5 wheel: 110,086 bytes, SHA256
`90f423204371d36abbc562d9616d74dde186c6c7f6350f640cda95a98657125c`.
Installed with `--no-index --no-deps` into a fresh non-inheriting Python 3.12 venv
at `/tmp/ard-dev5-installed-epiOVz`. From `/tmp` with PYTHONPATH cleared, the installed
comparator matched actual CPU optimizer proofs without Torch/NumPy imports. Doctor
reported dev5 and no Torch/Ray/verl/vLLM/W&B; offline demo and full replay/hash/metric
audit passed. This confirms usable core packaging, not GPU integration acceptance.

## 2026-09-30: dev4 preflight and checkpoint comparison

Added bounded runtime probes and exact audited checkpoint output comparison.
Usage: `integrations/runtime-and-comparison.md`. The parent remains stdlib-only;
optional checks run in the selected child interpreter. Probe reports retain real
exit codes, output, timing, timeouts and reap status. The tools do not load models,
execute inference or start training.

Actual physical-0 check used `/tmp/ard-ray-env-cXj8GA/bin/python` (Python 3.11.15).
Python and pinned package metadata passed. Direct `cuInit(0)` timed out at 15
seconds and still had no exit code after the one-second reap deadline. The tensor
check was skipped. Namespace execution failed at its own ten-second startup
preflight. The tool returned 1 and saved `runs/dev4-tools/runtime-health.json`.
Its original dev3 stamp is retained because execution preceded the version bump.
A separate CPU-only final dev4 health report passed. No negative report was
promoted to CUDA or coding readiness.

The comparison tool first audits both inputs, rejects corrupt/incomplete points
and self-comparison, then compares complete point identities/metrics and both
raw record hashes. Different checkpoint steps cannot pass just because scores
or model weights match. Actual dev2 native-actor Qwen3-1.7B step-1 validation and
its independently executed standalone counterpart matched:

- Multi-turn SHA256: `c19761853f08afcb27a4961232b8137cdc8b07237de0c388a0aec2c779eb5505`.
- Single-decision SHA256: `8a9f39579a403915d5ce3c57befe5ae6f2eb23391fca68a30e7d419afd983755`.
- Complete diagnostic point: equal.

Report: `runs/dev4-tools/native17-comparison.json`. This is a new audit of existing
real inference artifacts, not new model execution or same-export Ray parity.
Weights were not rehashed by this comparison; the earlier dev2 full acceptance
did that. Evaluation output equality does not establish optimizer/state/RNG parity.

CPU suite: **134 passed, 3 opt-in host checks skipped**. Thirteen new checks cover
real child exits, bounded timeout/launch failures, invalid timeouts, immutable
CPU health reports and audited comparison fixtures with stale steps/corruption.
Those fixtures use explicit synthetic completions and are not training evidence.
Ruff and formatting checks pass for 78 Python files. No additional GPU training
or namespace acceptance is claimed while current-host startup remains unavailable.
Future GPU tests use physical 0/1. Full-Ray output parity, diagnostics off/on
optimizer/RNG equivalence, multi-rank synchronization and BudgetMix remain pending.

Built a 106,674-byte dev4 wheel, SHA256
`b89c85199346a08906e75b88e3f7cd506f808032fba0e68bf385145331b72299`.
Installed with `--no-index --no-deps` into a new non-inheriting venv outside the
source tree. With PYTHONPATH cleared, CPU preflight and the actual 1.7B artifact
comparison passed using the installed modules; all eight optional package
metadata entries were absent, confirming venv isolation. Offline demo and its
hash/replay/metric audit also passed. No model loading or GPU was required.

## 2026-09-30: dev3 one-rank Ray training audit

The actual GPU run completed on **2026-09-29**, using local Qwen3-1.7B and physical
RTX 3090 GPU 2. On 2026-09-30, the public acceptance script independently decoded
the recorded response tokens, replayed all 134 rollouts, recomputed all three
checkpoint dual-validation points and rehashed the actual exported weights.
`runs/qwen17-ray-dev3/verification-final.json` records `accepted=true` for that scope.

The clean upstream commit remains `7dc46e834209948cf1cdd8a04d83f82b4a7efd24`;
437 Python/YAML files are checked before optional imports. An isolated,
non-inheriting Python 3.11.15 venv was built from official PyPI packages. The
candidate version snapshot is `requirements/verl-ray-candidate-py311-linux.txt`.
Key versions: torch 2.9.0+cu128, vLLM 0.11.2, transformers 4.57.1, Ray 2.54.0,
pyarrow 21.0.0, pandas 2.2.3, peft 0.18.1 and W&B 0.25.1. The previous research
environment and upstream source were unchanged. Dependency inspection followed
the internet-research skill's primary-source routing; the resulting environment
is isolated rather than installed over the user's existing conda environment.

Executed command, from the staged checkout with these verified sources:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=2 WANDB_MODE=offline \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=/tmp/ard-verl-7dc46e-1MD5xz:/tmp/ard-p3-ray-stage-LUfvzY/src \
/tmp/ard-ray-env-cXj8GA/bin/python scripts/run_verl_ray_smoke.py \
  --model-path YOUR_LOCAL_QWEN17 \
  --verl-source /tmp/ard-verl-7dc46e-1MD5xz \
  --monitor-config /tmp/ard-verl-actor17-four-directions-dev2-20260929/validation-1/budget/config.json \
  --pool /tmp/ard-verl-actor17-four-directions-dev2-20260929/pool \
  --output /tmp/ard-ray17-dev3-retry10-20260929 \
  --steps 2 --train-budget 2 --group-size 16
```

The launch also directed Torch/Inductor/Triton/vLLM caches under `/tmp`, disabled
HTTP proxies, used eager vLLM and bounded the private Ray object store to 512 MiB.
Arrow is preloaded through Ray's real worker setup hook to address a native import
ordering failure observed with this wheel set. The actual FSDP worker performs
upstream weight transfer before the real server's version RPC is acknowledged.
Source files are neither patched nor bypassed.

The trainer uses the upstream PPO actor with `reinforce_plus_plus` advantages,
one epoch, no critic/KL and full BF16 weights with parameter/optimizer CPU offload.
One rank uses FSDP NO_SHARD. Four direction-symmetric authored training corridors
have T=2 and 16 samples per task; two held-out validation maps and a genuine frozen
Base pool have disjoint IDs and map contents. Train temperature/top-p are 1/1;
validation is .6/.9. No model outputs or terminal-success rewards were rewritten.

| Checkpoint | Actor gradient norm | Changed tensor contents | Training success | Export / barrier seconds |
|---|---:|---:|---:|---:|
| 1 | 127 | 257 / 311 | 1 / 64 | 7.21 / 46.44 |
| 2 | 336 | 253 / 311 | 10 / 64 | 7.62 / 46.59 |

Tensor comparison streams actual safetensors bytes and ignores header/shard-layout
differences. Base tensor fingerprint is
`05c8388d0452feab4382682d2feaa7c8b96f25eaf7338859e5ad8ad17f00bc72`;
the full checkpoint fingerprint is the dev2 Base identity recorded below.
Three selected k_proj tensors were also compared against the official safetensors
reader during development. Both actual updates contain nonzero advantages, and
the adapter rejects all-zero/nonfinite generated-token advantages before updating.

The 134 raw traces span actual server versions 0/1/2 and contain 10,599 environment
tokens. Generated tokens carry mask 1; appended feedback carries mask 0. The audit
checks initial prompt tokenization, decoded responses, mask/token continuity,
budget-visible histories, state transitions and terminal rewards independently.
For every checkpoint, Pass@(K,T) and fixed-state Q* remain zero, with 16 valid
single-decision samples, 72 actions and 144 completion tokens per diagnostic point.
No missing slots, infrastructure failures or invalid actions occur in these points.
Tiny training successes and flat held-out validation are retained without drawing
claims about paper results or generalization.

The original offline W&B binary contains three SDK history records with checkpoint
steps 0/1/2. Diagnostic values and actual actor gradients agree with the trainer's
native file logger. `report.html` renders local diagnostic curves. The controller
explicitly finishes the owning W&B run before Ray shutdown. Earlier retry9 had
three native JSONL points but only one persisted W&B point; the public audit rejects
it with `offline W&B steps incomplete`. It was not promoted to accepted evidence.

Failed pilots are kept under `runs/qwen17-ray-pilots-dev3`: batch-size mismatch,
native architecture/import-order crashes, unsupported load format, missing server
versions, a NumPy scalar rollout serialization failure, all-zero advantages and
incomplete W&B shutdown. The successful run uses the actual upstream APIs and
records these fixes in the guide rather than hiding failures.

Default CPU suite on 2026-09-30: **121 passed, 3 opt-in checks skipped**. New checks
cover checkpoint scheduling, strict server version transitions, raw multi-turn
replay, tensor-content identity, CLI import isolation and rejecting dropped/stale
offline W&B history. The current-host opt-in namespace suite was attempted twice
and failed three checks on executor launch timeouts; it is not counted as passing.
The earlier dev2 namespace results below describe that earlier environment.

Ruff check and formatting checks pass for all 72 Python files. Built dev3 wheel
`agent_rl_diagnostics-0.1.0.dev3-py3-none-any.whl` (104,479 bytes, SHA256
`eef6cfede412cde11c2b7e3153b95fe79cad04a1faccb1325e26965e0fe9ed75`). Installed
with `--no-index --no-deps` in a new venv outside the source tree. With PYTHONPATH
cleared, doctor, offline demo and audit passed; torch/Ray/vLLM/verl/W&B were absent.
The demo produced 1,152 trajectories and 3,092 actions with five artifact hashes,
environment replay and full metric recomputation verified.
That dependency-free installed wheel also audited the actual Ray step-1 dual
validation using its copied frozen pool: artifact hashes, state replay and metrics
passed without torch or a GPU. This raw-record audit omits optional weight
rehashing; the separate full training audit above already rehashed the exports.

An additional public CLI rerun of this actual step-1 export on 2026-09-30 failed
before inference at CUDA initialization. A bounded minimal driver `cuInit(0)` call
also timed out; physical 0 was tried after the operator authorized 0/1. NVML reported
those cards idle. No driver restart or GPU reset was performed. Standalone parity
for these Ray exports remains pending; dev2 native-actor parity remains its own
completed record. Full-Ray diagnostics off/on optimizer/RNG parity, optimizer
resume, multi-rank synchronization, GRPO/BudgetMix and paper reproduction remain
open. These limitations are explicit in `verification-final.json` and the matrix.

Original GPU run manifests retain their dev2 package stamp because execution
preceded the dev3 version bump; no original evidence was rewritten. Local artifact
copies exclude model exports and private Ray process logs. Full HF weights remain
at the original temporary run for optional rehashing and later inference. No remote
repository changes, W&B sync or data uploads were performed.

## 2026-09-29: dev2 real 1.7B verification

Local Qwen3-1.7B checkpoint (no download):
`YOUR_LOCAL_QWEN17`.
Full checkpoint fingerprint:
`8922df67227a992c8624e60442f0b455a7a7e5804de384861f931e0d77025b85`.
GPU: RTX 3090, CUDA device 2. Python 3.11.15, torch 2.9.1+cu128,
transformers 4.57.1, peft 0.18.1, wandb 0.25.1.
Research source and weights were read-only; no results were uploaded.

`/tmp/ard-qwen17-baseline-20260929`: real legacy-v1 inference; grid actions
were invalid prose/markdown and no eligible clean/error anchor existed. The
smoke failed rather than claiming full acceptance. Explicit action-only v2 still
left Base grid trajectories without eligible wrong-prefix anchors in another pilot.

Complete genuine inference command (run in the dev2 staging checkout):

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src \
YOUR_ISOLATED_ENV/bin/python scripts/run_local_model_smoke.py \
  --model-path YOUR_LOCAL_QWEN17 \
  --revision local-Qwen3-1.7B-operator-declared --dtype bfloat16 \
  --output /tmp/ard-qwen17-complete-v2-20260929 --prompt-profile action_only_v2
```

Artifacts: `acceptance.json`, `generation-checks.json`, `grid/` and `coding/`
raw trajectories/actions/continuations, frozen pools, replay audits and reports.
Generation checks exercised real model RNG/mode preservation and repeatability.
Grid R_single(Q*, gamma=.95)=0.18377297265624995; all tested grid Pass@(K,T)
were zero. Recovery uses a separately labeled independent geometric witness pool,
not the collected Base pool: clean/error success 0/0. Coding Pass@(1,2)=1,
single-step mean test fraction=1; two recovery pairs have clean/error success
.5/1. These are two authored toy coding tasks, not DeepCoder benchmark scores.
No small-sample gain or model superiority is inferred.

Current CPU suite: **87 passed, 3 host-only checks skipped**; Ruff checks pass.
Explicit namespace-enabled host suite: **90 passed**.
New coverage includes prompt-profile snapshot compatibility, independent witness
provenance, full shard identity, checkpoint binding, raw dual-validation audit,
changed-weight rejection, monotonic/idempotent rank-zero tracking, and local
curves with missing values omitted rather than zero-filled.

### Real pinned native verl actor acceptance

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=2 WANDB_MODE=offline PYTHONPATH=src \
YOUR_ISOLATED_ENV/bin/python scripts/run_verl_actor_smoke.py \
  --model-path YOUR_LOCAL_QWEN17 \
  --verl-source /tmp/ard-verl-7dc46e-1MD5xz \
  --output /tmp/ard-verl-actor17-four-directions-dev2-20260929 \
  --steps 2 --group-size 2 --algorithm reinforce --wandb-offline
```

Clean source commit `7dc46e834209948cf1cdd8a04d83f82b4a7efd24`; all 437
Python/YAML files checked before imports. Four direction-symmetric one-action
training corridors; held-out grid validation has disjoint task IDs and map content.
Terminal-success rewards only; generated response token IDs feed the native actor.
No reward rewriting, forced success or fake optimizer was used.

Both native updates changed trainable LoRA weight hashes. Step 1/2 gradient norms
were 1.701235294342041 / 0.00007205226575024426; nonzero-advantage episodes
were 1/2 out of 8. Rewards .125/.25 are tiny-run measurements, not evidence of
improvement. Frozen validation Pass@(K,T) and mean Q* were zero at all three steps;
each point has 72 actions and 144 completion tokens, no missing/infrastructure
failures. This negative validation result is retained.

All nine off/on diagnostic parity checks passed: updated weights, actual clipped
gradients, actor metrics, AdamW state, unchanged training batch, Python/NumPy/CPU/
CUDA RNG states. `acceptance.json` states `full_ray_trainer_tested=false` and
`paper_reproduction=false`. Full HF exports remain in the original temporary run,
not in repository artifacts; weights are not redistributed.

The actual W&B 0.25.1 offline binary was scanned on a temporary copy; exported
`offline-history.jsonl` contains three SDK history records, not just handwritten
JSON approximations. `report.html` draws local per-metric curves from tracking
JSONL. No W&B sync/upload was performed. Failed GRPO and REINFORCE left-corridor
pilots produced zero advantages, aborted before updates, and remain available.

Scope: **single-GPU native upstream actor**, not RayPPOTrainer, distributed/FSDP,
vLLM, BudgetMix, GiGPO or paper GRPO replication. The compatibility matrix records
the tested slice rather than claiming general upstream compatibility.

Independent public `ard eval checkpoint` reran exported step-1 weights with the
same config, pool and seeds. Both diagnostic points matched exactly, and SHA256
of **both raw multi-turn trajectories and fixed-state actions matched exactly**.
`scripts/verify_training_artifacts.py` rehashed actual exported weights, replayed
all three validation points, checked SDK W&B history values against tracker/core
metrics, checked nonzero native gradients and changed weights, and emitted
`verification.json` with status `verified`.

Dev2 wheel installed into a new dependency-free venv outside the source tree:
doctor, offline demo and audit passed with torch/Ray/vLLM/verl/W&B absent. This
installed wheel also audited actual 1.7B grid step and coding recovery records
(coding audit explicitly without sandbox replay). Optional inference/training
still requires the documented extras/upstream environment; the minimal wheel
does not pretend to supply them.

Local artifact copies are under `runs/qwen17-smoke-dev2`,
`runs/qwen17-native-actor-dev2`, `runs/qwen17-standalone-parity-dev2`, and
`runs/qwen17-pilots-dev2`. They are gitignored. Native exports/LoRA weight files
are excluded from these copies. Original temporary export paths are retained for
optional `--verify-weights`; without them, raw metric/replay audits remain possible.
The earlier complete inference run retains its original dev1 build stamp because
it was run during dev2 development; manifests were not rewritten retroactively.

The original 2026-09-28 dev1 record follows unchanged; its statements about no
training job describe that earlier day's work only.

Date: 2026-09-28. Package: 0.1.0.dev1. All verification was local; no remote
results upload, paid API call, model download or training job was performed.
Source research assets were read-only. Authored changes were developed in a
temporary checkout and copied to the new repository after checking for user edits.

## Tests and distribution

Ruff 0.11.13 (pinned development tool):
`ruff check .` and `ruff format --check src tests scripts`.

Python 3.12.4:
`PYTHONPATH=src python -m pytest -q`: **78 passed, 3 skipped**. The three skipped
checks require explicit namespace-enabled host execution. On that host,
`ARD_RUN_NAMESPACE_TESTS=1 PYTHONPATH=src python -m pytest -q`: **81 passed**.

Coverage includes estimator precision, missing jobs, duplicate slots and identities,
BFS/unreachable proofs, discounted versus undiscounted values, snapshot restoration,
v1/v2 contracts, immutable outputs, offline operation, complete prompt histories,
invalid action accounting, infrastructure failure handling, coding RUN/SUBMIT,
mocked HTTP payload/credentials/revision checks, reward-blind pool selection,
Base/RL contribution quotas, deduplication, training-task overlap checks,
fixed-action restoration, matched interventions, recovery, task-cluster intervals,
and replay detecting altered states even when artifact hashes are updated.

The three real namespace tests executed authored programs, checked actual stdin/
stdout grading and syntax failures, confirmed inaccessible home/research mounts and
network isolation, and verified wall-time and output bounds. Other coding unit tests
use an explicit authored-case lookup fixture and do not execute unisolated code.

A subprocess test blocks sockets and confirms torch, Ray, vLLM, verl, W&B and NumPy
are not loaded while the offline demo runs. HTTP contracts were mocked: a live HTTP
provider has not been tested.

Built the dev1 wheel using:
`python -m pip wheel --no-index --no-deps --no-build-isolation --wheel-dir /tmp/ard-p2-wheelhouse-dev1 .`.
Installed it into a new venv with `pip install --no-index --no-deps`. Outside the
source checkout, doctor, offline demo, audit and model-config dry-run all passed.
Doctor confirmed no optional training libraries installed. The demo generated
1,152 trajectories and 3,092 actions. Five artifact hashes, grid replay and full
summary recomputation passed. The new model dry-run did not load a model or call
an endpoint.

GitHub Actions is authored but has not run remotely. Python 3.11 is exercised by
real inference below, not by the full pytest suite; Python 3.13 remains a CI target.

## Real model acceptance

Two genuine inference passes used the existing local directory named Qwen3-8B,
an RTX 3090 (physical GPU 2), Python 3.11.15, Torch 2.9.1+cu128 and Transformers
4.57.1. Weights were not downloaded or copied. The final run used package dev1
and includes the new task-cluster CIs. Full weight hashing and original paper
checkpoint identity are not verified; revision is explicitly operator-declared.

Command (local executable/checkpoint paths normalized):

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src \
  /path/to/python-with-hf scripts/run_local_model_smoke.py \
  --model-path /path/to/existing/Qwen3-8B \
  --revision local-Qwen3-8B-operator-declared \
  --output /tmp/ard-p2-qwen3-smoke-dev1-20260928 --device cuda:0
```

Decoding: temperature 0.6, top_p 0.9, top_k 0, max_new_tokens 256, thinking disabled.
The generation checks confirmed identical text for repeated same-input/same-seed
calls, preserved CPU and CUDA RNG, and preserved the model's training mode.
They are saved in `generation-checks.json`.

| Authored smoke fixture | Budget run | Frozen pool | Single-step | Recovery |
|---|---|---|---|---|
| Two 4x4 grids, T=8 | 4 trajectories, 16 actions; Pass@(1,8)=Pass@(2,8)=0.5 | 8 states | 16 actions; R_single=0.758992890625 (Q*) | 5 pairs, 10 continuations; clean 0.8, error 0.2 |
| Two stdin coding tasks, T=2 | 2 trajectories, 4 actions; Pass@(1,2)=0 | 2 states | 4 actions; R_single=0.5 (test fraction) | 1 eligible pair, 2 continuations; clean 0, error 0 |

The coding error builder skipped the selected anchor with one remaining action.
Several coding outputs included prose before the action and were correctly rejected
by the strict toolkit parser. The failed recovery output also proposed RUN with
only one turn remaining. These are retained outcomes, not repaired model responses.
The smoke validates execution contracts, not a reliable estimate of coding ability.

Each pass saved configs, raw responses, all messages, snapshots, summaries, reports
and hashed manifests. Grid artifacts passed prompt/transition/Q* replay and metric
recomputation. Coding artifacts passed explicit sandbox replay and score/summary
recomputation, including independent reference-program grading. Base/RL matched
pool behavior has fixture tests but was not tested on a genuinely trained RL model.

Local final artifacts are preserved under `runs/qwen3-smoke-dev1/`, ignored by Git.
They include `acceptance.json`, generation checks and both environment pipelines.
Source temporary outputs remain intact as well. No private benchmarks or weights
are part of the repository.

## Foundation audit and remaining work

The P0 audit inspected nine paper experiment entries and four license candidates.
The final paper SHA-256 matched the construction plan. Source plot/CSV availability
does not establish raw numerical reproduction. The candidate verl SHA came from a
read-only upstream query, not from a tested trainer integration.

This P2 evidence uses derived toolkit prompts/checker/injectors and tiny authored
datasets. Formal paper checkpoints, dataset splits and frozen pools, official
DeepCoder checker, final coding-error perturbations, mechanism/NLL analyses, full
paper tables, verl/W&B dual validation, BudgetMix and process-reward training
remain open. Neither smoke success nor unit-test success closes those gaps.
