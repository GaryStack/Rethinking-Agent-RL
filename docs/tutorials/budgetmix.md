# BudgetMix plug-in

```bash
ard budgetmix plan --task-ids configs/examples/budgetmix_task_ids.json --tmax 24 --group-size 8 --seed 5422 --output runs/mix
ard budgetmix audit --input runs/mix
```

Task IDs are a JSON array; repeated IDs are different occurrences. Budgets are
sampled uniformly from the unique set ceil(k*Tmax/10), k=1..10. One draw is shared
by a task occurrence's rollout group; display the remaining budget each turn and
retain infeasible assignments. A plan is not a rollout or training result.

For an existing trainer, use `training.budgetmix.plan_occurrences` before rollout
repetition, then `audit_groups` on completed groups. The optional pinned VeRL
`budgetmix_trainer_class` performs raw-grid prompt replacement at the pre-repeat
hook and leaves native validation budgets fixed. See
[adapter and verified recipe](../integrations/budgetmix.md).

Two one-rank REINFORCE++ updates have engineering smoke evidence. PPO bootstrap
and GiGPO budget-aware anchors have CPU contracts only; critic training,
GiGPO integration and resume are deferred. No complete algorithm combination is
implied by sampler correctness.
