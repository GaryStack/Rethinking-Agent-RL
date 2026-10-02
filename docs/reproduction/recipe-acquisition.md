# From local research assets to executable paper recipes

GPU acceptance is deferred; source/configuration/provenance review is not.
Use `report_paper_readiness.py` and the read-only `audit_local_assets.py` without
copying weights, private traces or benchmark tests into this repository.
Keep the initial published ledger intact; retain fresh inventories under `runs/`.

For each of the nine experiment IDs, acquire a manifest containing:

- Paper section/figure, exact final protocol, environment and action-count convention.
- Base/RL revision and tokenizer hashes; training checkpoint step when applicable.
- Dataset source/revision, split hash, task IDs, redistribution decision and checker.
- Full prompt, sampling settings, seeds, expected task/sample slots and budget mode.
- Raw rollout/action/token-score/state-pool hashes and the aggregation command.
- Figure/table command, dependencies, expected outputs and numerical tolerances.

Then verify acquisition in this order: section 3.1/3.2 raw budget-conditioned
records; section 3.3 equal Base/RL pool and G=16; frozen Base 512x8 monitor; recovery
injector identity; likelihood spans/length matching; process reward weights;
BudgetMix formal configurations and intervention checkpoints. Some acquisition
tasks need the author to identify final assets; do not silently infer them from
old scripts or plotting constants.

The existing toolkit FrozenLake, derived coding checker and BudgetMix smoke are
working examples, not substitutions for those formal assets. `paper_v1` remains
rejected by evaluators until its full environment protocol is recovered. There is
no executable full-paper reproduction recipe yet; the acquisition checklist is
explicitly preparatory, not a completed recipe.

Publishing a CPU development preview can precede GPU completion if limitations
and licenses are clear. Creating/pushing a remote, redistributing data/weights or
uploading W&B remains a separate operator decision. Remote CI, release artifacts,
public documentation and independent fresh-machine trials remain release gates.
