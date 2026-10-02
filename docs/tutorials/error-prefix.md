# Error-prefix evaluation

Collect a frozen pool through the quickstart. Both models must see the same
histories, snapshots and remaining budgets. Ordinary and error states are separate
reports, each with depth-stratified single-step quality.

```bash
ard pool errors --input runs/pool --output runs/errors
ard eval step --config configs/examples/frozenlake_agent_http.json --pool runs/errors --output runs/error-step --samples 8
ard eval recovery --config configs/examples/frozenlake_agent_http.json --pool runs/errors --output runs/recovery --samples 2
ard audit --input runs/recovery --pool runs/errors
ard report --input runs/recovery --pool runs/errors --output runs/recovery-report
```

Repeat step/recovery using your RL config and compare with `ard compare --input
BASE_RUN RL_RUN --labels Base RL --pool runs/errors --output runs/error-comparison`.
Compare only the same evaluator kind and scorer; Q* and coding test fractions have
different units. Continuation reports use equally weighted intervention pairs.
Successful action lengths are conditional on success.

Grid interventions consume one matched action and increase shortest distance.
Infeasible error states are retained unless `--require-recoverable` is explicitly
requested. Coding uses tested independent references and syntax mutations, not
the paper's model-generated injector. Natural error histories can be supplied as
external frozen states with the [external scorer API](../extensions.md); automatic
natural-error discovery is not implemented. Unknown recoverability is not proof
of recovery. See [protocols](../protocols.md).
