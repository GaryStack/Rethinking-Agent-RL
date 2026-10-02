# Tight-budget evaluation

The CPU demo scans K/T with independent budget-visible rerolls. For neural
inference, edit `budgets: [4, 6, 8]`, `ks: [1, 2]` and `samples_per_task: 2` in
the HTTP example. All K must be at most the sample count.

```bash
ard eval budget --config configs/examples/frozenlake_agent_http.json --output runs/base-budget
ard audit --input runs/base-budget
ard compare --input runs/base-budget runs/rl-budget --labels Base RL --output runs/budget-comparison
```

`runs/rl-budget` is produced with your RL config and the identical tasks/K/T grid.
Outputs include Pass@(K,T), task bootstrap intervals and structural ceilings.
FrozenLake uses exact deterministic BFS, including unreachable tasks in the
ceiling. Slack is T minus the exact minimum; `ard compare` adds a separate slack
panel when certificates are available. Unreachable/no-certificate tasks have no
finite slack and remain in the unconditional budget metric. No exact conclusion
is assigned to coding or externally supplied BFCL feasible paths.

Report JSON accompanies HTML. Unavailable metrics stay null. External task files
use `tasks_file` instead of `tasks`; see [inputs](../inputs.md).
