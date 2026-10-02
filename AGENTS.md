# Development guide

The core is a stdlib-only package under `src/agent_rl_diagnostics`.
`.[hf]` is an optional local inference extra, not a core dependency.
Install with `python -m pip install -e .`; check with `python -m pytest`.
The public entry point is `ard`, also available as `python -m agent_rl_diagnostics`.

Read `docs/protocols.md` before changing metrics, budgets or sampling.
Read `docs/reproduction/asset-ledger.json` before claiming paper reproduction.
`paper_v1` and `toolkit_v1` are distinct protocols. Synthetic demos are labeled.
Missing jobs and infrastructure failures are never silently converted to successes or zeros.
Keep torch, Ray, vLLM, verl and W&B optional; never import them in core modules.
Source research assets are read-only. Do not copy model weights, private traces or benchmark tests.
Record actual commands and artifacts when reporting a completed stage.

Read `docs/model-evaluation.md` for P2 parser, backend and sandbox limitations.
Keep v1 scripted trajectory compatibility when evolving v2 agent contracts.
Only execute candidate programs through the bounded namespace executor, never
through an unisolated fallback. Host integration tests require explicit
`ARD_RUN_NAMESPACE_TESTS=1`. Do not convert parse errors into infrastructure failures
or convert transport/executor launch failures into task failures.
P3 requires actual pinned verl integration and short-training evidence; callbacks
and W&B stubs alone do not complete it. Do not upload without authorization.

Dev2 checkpoint/actor scope is documented in `docs/integrations/checkpoint-validation.md`.
The native actor test is not RayPPOTrainer/vLLM/distributed acceptance. Keep upstream
source checks strict; never import a dirty local verl checkout. Explicit REINFORCE
smoke updates are not GRPO paper reproduction. Preserve zero-advantage failure
guards and failed pilot evidence. Witness error pools are independent oracle
references, never model-collected Base states. Optional `action_only_v2` prompts
must remain explicit; legacy configs and hashes must retain their behavior.

Dev3 has scope-specific real one-rank RayPPOTrainer/vLLM/FSDP evidence; see
`docs/integrations/ray-smoke.md` and `docs/verification.md`. Its isolated Python
3.11 environment uses the complete candidate version freeze. Keep Arrow's worker
setup hook, completed-transfer version tagging, controller-owned W&B finish and
the pre-optimizer zero-advantage guard. Do not infer multi-rank, optimizer-resume
or full-trainer diagnostic-on/off parity from this one-rank acceptance.
Keep failed pilots and infrastructure failures visible. The operator expanded
testing/use to physical CUDA 0/1 on 2026-10-01; first check occupancy and bounded
CUDA allocation. The operator subsequently deferred GPU acceptance while the
driver is unavailable. Do not use other cards or reset shared GPUs/restart the driver.

Dev4 bounded runtime checks and checkpoint comparison are documented in
`docs/integrations/runtime-and-comparison.md`. Do not infer CUDA readiness from
package metadata or idle NVML, or training-state parity from matching evaluation
outputs. Preserve negative probe reports and use the shared artifact audits.

Dev5 actor state fingerprints are opt-in and one-rank/CPU-offload only. Read
docs/integrations/training-state-parity.md. Tiny CPU AdamW evidence is not GPU,
Qwen, PPO or complete trainer parity. Both full-trainer and distributed parity
flags must stay false in observed-state comparison reports. Default actor updates
still delegate directly to the native parent after the training-signal guard.

Dev6 may progress CPU P4 implementation before pending P3 GPU acceptance, per the
operator's 2026-10-01 instruction. Read docs/integrations/budgetmix.md. Plans are
not rollouts. Preserve unique ceil-decile sampling, pre-repeat occurrence identity,
budget-visible prompts, unchanged native validation and infeasible budgets. The
actual CPU hook check is not actor/critic training or optimizer resume acceptance.

Dev7 adds portable remote acceptance. The operator authorized the remote host's
single idle A100 on 2026-10-01; this does not authorize repairing the original
host's driver or using other people's GPUs. Use a new isolated environment,
read-only mounted weights, offline W&B, private short Ray socket directories and
fresh outputs after failures. Read docs/integrations/remote-acceptance.md. Never
put SSH credentials in code, logs or documentation. Comparison reports certify
only observed actor boundaries/rollouts, not controller/server/distributed parity.

Dev8 explicitly seeds the one-rank actor once before native init_model only when
requested. Never seed at update/capture boundaries to manufacture parity. The
real seeded pair matches RNG but diverges in one rollout and second-update state;
keep that negative evidence. `ard dynamics` audits raw children and fixed protocol
before plotting. Preserve missing points as unavailable and descriptive warnings
as noncausal/non-significant. Recorded config audit paths are not live-inference
paths: preserve original absolute identities without local symlink rebasing;
never claim weight rehashing when original weights are absent.
