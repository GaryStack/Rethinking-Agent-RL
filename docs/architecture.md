# Foundation architecture

The core is implemented from scratch with the standard library. No research source
file or upstream trainer was copied. Modules follow the construction plan:

```text
config -> evaluator -> environment/scripted policy -> trajectory records
                                  |
trajectory records -> metrics -> JSON/CSV -> static HTML
                                  |
                       manifest + content hashes
```

`contracts/schema.py` validates trajectory identities, budget accounting, terminal
state chains and explicit infrastructure outcomes. `metrics/pass_at_kt.py` is the
single implementation of the estimator and bootstrap. `environments/frozenlake.py`
provides deterministic simulation, snapshots, BFS and finite-horizon DP.

`evaluators/budget.py` owns independent per-T sample slots and structured artifacts.
It uses per-slot local RNGs, leaving Python's global RNG unchanged. The CLI calls
this evaluator; it has no separate metric implementation. Reports read its summary.

The first config format is JSON (`ard-budget-config-v1`) to avoid a YAML runtime
dependency. Scientific job identity excludes the output path. Completed output
directories are immutable; results are built in a sibling temporary directory and
published when all files are written. Runtime timestamps and wall time may differ
between otherwise identical runs.

Audit validates SHA-256, record schema and model/job identity, replays grid transitions
and Q* scores, and recomputes the summary. Hashes detect alteration relative to a manifest; they do not certify an
untrusted manifest's scientific authenticity.

The P2 agent path uses `agent_config.py`, `contracts/agent.py` and lazy chat adapters
in `models/chat.py`. `TextGrid` and derived `DeepCoder` share initial_messages,
apply, snapshot and restore without forcing programs into integer grid fields.
The old v1 path remains unchanged for offline demos.

`evaluators/agent.py` owns complete history/budget rollout. `pools/frozen.py` creates
reward-blind frozen inputs; `pools/errors.py` creates paired intervention states.
`evaluators/fixed.py` restores before each candidate and `evaluators/recovery.py`
restores before each continuation. Both store raw records independently of summaries.
The existing Pass@(K,T) implementation is reused; `metrics/cluster.py` supplies
task-cluster CIs for state/pair-weighted metrics. HTML reads stored summaries only.

`evaluators/audit.py` binds records to their original pools and recomputes summaries.
Grids replay transitions/inputs/Q*. Coding execution is an explicit optional replay,
with no unisolated fallback. Pool loading validates coding snapshot structure and
hashes, not full historical execution. HTTP revision/seed support and local full-
weight hash limitations are reported rather than inferred.

The CLI has no metric fork, automatic remote logging or training side effect.
Training adapters will remain optional and must reuse these evaluators. A real verl
checkpoint hook, synchronization and W&B evidence are P3, not placeholder APIs.
