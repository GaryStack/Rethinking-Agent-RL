# Research asset audit

The original research workspace remains a read-only source. The ledger records
relative paths, hashes, availability and missing fields; it does not publish the
source artifacts. Reproduce the inventory with:

```bash
python scripts/audit_local_assets.py --source /path/to/research-workspace --output /tmp/new-audit
```

`asset-map.json` maps paper sections to candidate entry points. `asset-ledger.json`
is the initial targeted inspection, not a verification of paper numerical results.
Plot scripts or derived CSVs alone remain `summary_only`; a raw-recomputable claim
requires rollouts, configurations, revisions and a successfully executed aggregation.

Observed migration issues:

- Old BudgetMix uses weighted slack above a structural minimum and hidden budgets;
  the final paper uses uniform unique ceil-deciles, visible budget and no infeasible-T exclusion.
- The old fixed monitor defaults to G=4; final dynamics is 512x8 and Section 3.3 is G=16.
- Mutation/DiD error builders differ from the final random-action/code-perturbation protocol.
- Historical recovery pilots differ in model identity and cannot stand in for final evidence.
- A BFCL annotated feasible path is generally a minimum-budget upper bound; it is not an optimality proof.
- Figure sources may contain only summary numbers; action NLL sum/per-token and final LLR spans require source recovery.

Remaining acquisition fields: formal Base/RL checkpoint revisions, dataset split
hashes, state pools/seeds, original configurations, judge prompts/cache and reward
mixing weights, Table 1 rounding, raw scoring records and distribution permissions.
The tool foundation can proceed while these fields remain unresolved.

Generate an actionable, immutable acquisition checklist (CPU only):

```bash
python scripts/report_paper_readiness.py --ledger docs/reproduction/asset-ledger.json --output runs/paper-readiness
```

The current nine experiments remain pending. The report records the ledger hash
and keeps `paper_reproduction_verified=false` even if someone edits metadata to
claim completion. Final paper commands require independent raw-data verification,
not simply clearing `missing_fields`. See [recipe acquisition](recipe-acquisition.md).

Licensing inventory is in `license-inventory.json`; observed upstream terms do not
authorize redistribution of associated benchmark data or model weights. No upstream
or local research implementation code has been migrated into this foundation.
