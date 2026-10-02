# Contributing

Install the dev extra and run `python -m pytest` and `ruff check .`.
Keep changes small and attach an executable example for new public interfaces.

For a metric change, document its denominator, aggregation, budget convention,
missing-data behavior, scorer and evidence type. Add hand-calculated or independent
reference checks where a numerical error could change a research conclusion.

For a new environment, define success, the agent action unit, snapshot restoration
and available budget certificates. A known successful trajectory is an upper bound
on minimum budget, not an optimality proof.

Keep optional training dependencies outside the core. Every training integration
must identify an upstream commit and its actual tested backend/hardware.
Synthetic examples, paper-result recomputation and new training runs have distinct
evidence records. Describe verification limits in the PR.

GPU acceptance may be deferred while implementing CPU tools; never merge that
status into a training acceptance claim. For BudgetMix changes, test duplicate task
occurrences, unique small-Tmax probabilities, visible budgets, complete group
denominators and infeasible choices. Exercise `ard budgetmix plan/audit` from an
installed wheel outside the source tree. Paper readiness reports are acquisition
checklists, not certificates based on metadata. Do not clear source-ledger gaps
without actual provenance and independent numerical verification.
