# Teacher-forced likelihood inputs

`ard likelihood score` accepts a JSON array. Every event has `event_id`, `task_id`,
`trajectory_id`, `source` (base/rl), boolean `success`, exact rendered `prompt`
and fixed `response`. Optional `action_span: [start,end]` is a half-open character
range relative to the response. Render prompts once with the chosen chat template;
use those exact strings under both checkpoints. Do not score sampled logprobs.

```bash
ard likelihood score --input examples/likelihood/events.json --model YOUR_LOCAL_BASE --model-id BASE_REVISION --device cpu --max-length 4096 --failure-policy record --output runs/base-scores
ard likelihood score --input examples/likelihood/events.json --model YOUR_LOCAL_RL --model-id RL_REVISION --device cpu --max-length 4096 --failure-policy record --output runs/rl-scores
ard likelihood analyze --base runs/base-scores/scores.json --rl runs/rl-scores/scores.json --token-caliper 0.2 --step-caliper 1 --output runs/likelihood
```

HF is optional (`.[hf]`). It uses the raw temperature-1 distribution, shifted causal
logits, no top-p and no truncation. Prompt+response must preserve a token prefix;
a token crossing the response boundary is rejected. Long contexts and scoring
errors are explicit. `record` stores failures; `error` stops before publishing.
A missing action span yields null action NLL, never zero. Overlapping tokens are
included and the boolean mask is saved; action NLL conditions on all preceding
response text, including reasoning. This differs from independently scoring an
extracted action without its reasoning prefix.

LLR is log P_RL minus log P_Base on the identical response and prompt. Inputs,
token IDs, offsets and action masks must align; different tokenization is rejected.
Reports include response NLL sum/mean, Base action NLL sum/mean and response LLR
sum/mean. Length control aggregates events per trajectory and greedily matches
within task and outcome using explicit token/step calipers; input/matched/unmatched
counts are saved. This toolkit control is descriptive, not a recovered paper preset.
Incomplete scoring yields no complete full estimate; retained points are labeled
as observed pairs. Synthetic score fixtures run without HF:

```bash
ard likelihood analyze --base examples/likelihood/base-scores.json --rl examples/likelihood/rl-scores.json --token-caliper 0.2 --step-caliper 1 --output runs/likelihood-demo
```

Fixture logprobs are authored deterministic values, not model measurements.
No historical paper figure values are represented as recomputed evidence.

Length control excludes every trajectory with a supplied event that failed or is
missing under either policy. If a failure lacks trajectory identity, length control
is unavailable. Completeness describes only the union of supplied events, not an
independent proof that every original trajectory event was supplied.
