# External scorer API

```bash
python examples/external_scorer.py
```

`evaluate_external_step(states, generate=..., score=..., scorer=..., model_id=...,
samples=8)` is a public Python adapter. Each state has unique `state_id`, `task_id`,
nonnegative `prefix_depth`, positive `remaining_budget` and your own immutable
payload (messages/snapshot recommended). `generate(state,sample)` returns your
model response; `score(state,response)` returns a finite [0,1] value. Both callbacks
receive independent copies. Exceptions stop evaluation, rather than invent failures
or zeros. Use bounded namespace execution for any candidate programs.

The adapter returns raw records and the existing fixed-state summary: action means,
equal state weighting and task-cluster CI. Freeze the same states for both models.
The authored toy exact-match scorer needs no core modification. It is an external
single-step scorer interface, not automatic registration of a new built-in rollout,
error injector or exact budget solver. Reports for different scorer IDs cannot be
combined on one axis. Callback users own persistence/provenance; these external
results are not accepted by `ard audit` without a built-in replay adapter.
