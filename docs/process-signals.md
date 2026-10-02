# Process signals

`training.process_signals.reward_for_episode` consumes explicit terminal reward,
step inputs and a scorer callback. `oracle_score` checks scorer units and [0,1]
range. `CachedJudge` keys scores by exact step content plus explicit judge identity
(revision, prompt and scoring rule). The caller owns persistence; failures (`None`)
are not cached. Judge exceptions propagate.

The toolkit convention is `terminal + coefficient * mean(step scores)`.
`coefficient` and `failure_policy` are required. This is not an asserted paper
coefficient or recovered paper reward convention. `error` rejects unavailable
scores; `terminal_only` skips the entire process component and reports fallback.
Invalid/nonfinite scores always fail. Never mix diagnostic validation scores into
training unless this explicit reward path is enabled.

```bash
python examples/process_signal_training.py --coefficient 0.5
python examples/process_signal_training.py --coefficient 0
```

This CPU example implements four REINFORCE gradient-ascent updates of a Bernoulli
policy. A cached external judge supplies process scores to the reward used in each
actual parameter update. With zero terminal reward, coefficient 0 leaves the logit
unchanged; a positive coefficient changes it. It is synthetic integration evidence,
not PPO/GRPO/GiGPO training acceptance.

The pinned `GridAgentLoop` also exposes an opt-in path: add
`ard_process_signal: {enabled: true, coefficient: YOUR_VALUE, failure_policy: error}`
to the trainer configuration. Audited grid Q* values compose the reward written
into native `AgentLoopOutput.reward_score`; raw terminal reward remains separate
in the trace. Without this config the native reward is unchanged. The complete reward placement → advantage → native actor optimizer chain
has a CPU check using scripted actions and a tiny random HF model. It also passed
two actual Qwen3-1.7B one-rank Ray/vLLM/FSDP REINFORCE++ updates on an authorized
A100. All 128 training rows had nonzero process contributions; six native
validation rows retained terminal-only rewards. The public
`audit_verl_ray_smoke.py` joins replayed traces to native score exports, while
checking real gradients, checkpoint changes and dual validation. See the
[exact GPU verification scope](verification/process-gpu.md).
PPO critic/GAE shaping, GiGPO step advantages and full GRPO integration are deferred.


## Keep native validation unchanged

When process signals are enabled, every row must provide `extra_info.ard_phase`
as `train` or `validation`. Missing/unknown phase fails before rollout. Only train
rows receive the mixed reward; validation reports terminal success. Do not infer
phase from decoding temperature. Invalid step scores are rejected even if another
score is missing and `terminal_only` was requested.

The pinned Ray launcher exposes a runnable path (requires an available authorized
GPU and the locked environment described in `integrations/ray-smoke.md`):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_verl_ray_smoke.py --model-path YOUR_LOCAL_MODEL --verl-source YOUR_LOCKED_VERL --monitor-config YOUR_MONITOR_CONFIG --pool YOUR_POOL --output runs/process-ray --steps 2 --actor-seed 5422 --process-coefficient 0.5 --process-failure-policy error
```

Both process flags are required together. Coefficient 0.5 is an illustrative
toolkit choice, not a recovered paper value. The launcher tags training/validation
rows and records settings in the resolved config and returned manifest.

CPU-only locked-native check, without model weights or a Ray cluster:

```bash
CUDA_VISIBLE_DEVICES= GLOO_SOCKET_IFNAME=lo python scripts/verify_process_signal_cpu.py --verl-source YOUR_LOCKED_VERL --output runs/process-native-cpu
```

This calls actual `GridAgentLoop`, upstream padding/reward assembly, REINFORCE++
and `DataParallelPPOActor.update_policy`. Only actor device resolution is patched
to CPU. Two fixed failed trajectories have terminal reward 0; process rewards
[0.2375, 0] produce nonzero advantages and update 17 tiny-model tensors, while
coefficient 0 produces no parameter change. Validation rewards remain [0,0].
The check requires Gloo loopback access. It does not certify real LLM rollouts,
Ray orchestration, FSDP/GPU, critic training or numerical reproduction.
