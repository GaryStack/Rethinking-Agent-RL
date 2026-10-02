# Model evaluation and frozen-state workflows

The public core stays dependency-free. Both adapters receive complete messages
and return raw text plus request/usage metadata. Heavy imports occur only when the
local Transformers adapter is constructed; HTTP imports occur on generation calls.

## Backends

HTTP: edit `configs/examples/frozenlake_agent_http.json`. `base_url` ends in `/v1`;
the adapter posts to `/chat/completions`. Credentials belong in an environment
variable referenced by `api_key_env`, never in configs or URLs. A required response
revision is checked against `X-Model-Revision` or response `model_revision`.
Ordinary endpoints rarely attest weight revisions: by default checkpoint identity
is operator-declared and seed support is explicitly `unknown`.

`send_top_k` is a provider extension. Enable it only if your endpoint supports it;
otherwise the manifest records `top_k_control: not_sent`. Unsupported seeds are
not represented as reproducible generation. Teacher-forced logprobs are not
implemented: a chat endpoint is not an NLL-scoring adapter.

Local Transformers model object (replace the HTTP `model` object):

```json
{
  "backend": "transformers",
  "model_id": "my-base",
  "revision": "my-immutable-checkpoint-id",
  "path": "/path/to/existing/checkpoint",
  "device": "cuda:0",
  "dtype": "float16",
  "chat_template_kwargs": {"enable_thinking": false}
}
```

The adapter uses the checkpoint's chat template, temperature/top_p, explicit top_k
(default 0), token cap and per-slot Torch RNG. It restores CPU/device RNG and
train/eval mode. Reproducibility across hardware/dependency versions needs separate
checks. Config/tokenizer/index hashes are recorded; full weights are not hashed
automatically. Operator-declared revisions are not verified weight identities.

Implementation references: [generation API](https://huggingface.co/docs/transformers/main_classes/text_generation)
and [chat templates](https://huggingface.co/docs/transformers/chat_templating).

## Budgets and pools

Each T gets new budget-visible rollouts. Change `budgets` for a tight-budget sweep;
retain infeasible grid budgets to expose structural ceilings. Choose n >= max(K).
Configs save expanded tasks so artifacts do not depend on later task-file changes.

`pool build --kind base_monitor` only accepts the declared Base collector.
`--kind matched --base-model-id BASE --rl-model-id RL --input BASE_RUN RL_RUN`
uses equal role contributions and distinct complete inputs. This does not verify
whether an operator-labeled RL checkpoint really underwent RL training. Selection
balances task/depth with a deterministic seed; rewards are not selection criteria.
The pool remains unchanged when another checkpoint is evaluated.

Pass `--training-task-ids /path/to/train_ids.json` for a disjoint ID check. Without
it, held-out status is `not_verified`. Map-content leakage is not checked even if
IDs are disjoint. Smoke pools are not the paper's original 512-state held-out
monitor or Section 3.3 pool.

`eval step` samples G actions independently from each snapshot, then averages
actions within states and states equally. `eval recovery` continues paired
clean/error states with matched remaining budgets, weighting intervention pairs
equally. CIs resample whole task clusters, retaining shared-task dependence and
the state/pair-weighted estimator. They omit some Monte Carlo uncertainty and
are not informative evidence with only one or two tasks.

## Derived coding environment

`configs/examples/deepcoder_agent_http.json` has two authored toy tasks. Use other
datasets only with appropriate authorization. Fields are `question`, `public_tests`
and `hidden_tests`, with input/output strings. Each case runs in a fresh process;
the checker compares whitespace-separated tokens. No function-call, custom judge
or official LiveCodeBench checker is supplied.

Actions must be exactly RUN or SUBMIT followed by a fenced Python program; a bare
fenced program is an implicit SUBMIT. Surrounding prose is invalid and costs an
action. SUBMIT always terminates, including failures. RUN scratch execution has
empty stdin and reveals only its own stdout/stderr. Test fractions are analysis
records, not hidden-test feedback in the prompt. Strict parsing can dominate
results when models emit prose: inspect raw actions before interpreting zeros
as lack of programming skill.

The executor requires Linux bubblewrap, util-linux `prlimit` and `/usr/bin/python3`.
It fails closed when namespaces are unavailable. Programs have private PID/network/
tmp namespaces, no home/research-directory mounts, read-only runtime mounts, no
capabilities and bounded CPU/memory/output/wall time. NPROC applies after namespace
setup to work on shared servers. This is not an audited hostile multi-tenant
service: no cgroups or seccomp policy is supplied. Do not expose it directly to
untrusted internet users. Program errors/timeouts are task outcomes; sandbox
launch failures are infrastructure failures, not incorrect answers.

Coding error pools require independent reference programs:

```bash
ard pool errors --input runs/coding-pool --output runs/coding-errors --references configs/examples/coding_smoke_references.json
ard eval recovery --config configs/examples/deepcoder_agent_http.json --pool runs/coding-errors --output runs/coding-recovery
ard audit --input runs/coding-recovery --pool runs/coding-errors --replay-code
```

References identify `code`, `source` (dataset_reference or independent_reference)
and `revision`, and must pass configured tests. The toolkit introduces a syntax
error; it does not select references from the tested checkpoint's successful
outputs. This is not the paper's Qwen3-1.7B injector. Grid interventions increase
exact shortest distance and keep nonterminal states; optional recoverability
filtering is recorded explicitly.

## Audit and failure policy

`ard audit` checks hashes, identities, schemas and metric recomputation. Grids also
replay prompts, transitions and Q*. Coding requires `--replay-code` to rerun
programs explicitly; otherwise the CLI reports `code_replay_not_requested`.
Coding pool loading validates hashes and snapshot structure, not full historical
execution. Hashes demonstrate consistency, not an untrusted provider's authenticity.

Default `failure_policy: fail` aborts infrastructure-failed jobs without publishing
a completed directory. `record` preserves missing completions as null estimates;
it does not convert transport failures into task failures. Results and credentials
remain local: no automatic W&B upload or external write happens.

## Next stage

Immutable checkpoint dual validation and optional offline W&B are now implemented.
See [the native actor integration guide](integrations/checkpoint-validation.md).
Full Ray/vLLM integration, distributed BudgetMix sampling and process rewards
remain unimplemented. See `NEXT.md`.
