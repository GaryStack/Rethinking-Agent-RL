# Quickstart

Install Python 3.11+ and `python -m pip install -e .`. Run `ard demo --offline
--output runs/demo`, then `ard audit --input runs/demo`. Choose fresh output
folders each time: completed results are immutable.

For your model, edit `model_id`, `revision` and `base_url` in
`configs/examples/frozenlake_agent_http.json`. A compatible chat server is user supplied.

```bash
ard eval budget --config configs/examples/frozenlake_agent_http.json --output runs/base --dry-run
ard eval budget --config configs/examples/frozenlake_agent_http.json --output runs/base
ard pool build --input runs/base --output runs/pool --kind base_monitor --base-model-id YOUR_MODEL_ID --size 8 --min-tasks 2
ard eval step --config configs/examples/frozenlake_agent_http.json --pool runs/pool --output runs/base-step --samples 8
ard audit --input runs/base-step --pool runs/pool
```

For RL comparison, edit a separate config with the RL checkpoint, keeping tasks,
decoding, budgets and sampling counts fixed. Use the same immutable pool for both.

```bash
ard eval step --config YOUR_RL_CONFIG.json --pool runs/pool --output runs/rl-step --samples 8
ard compare --input runs/base-step runs/rl-step --labels Base RL --pool runs/pool --output runs/step-comparison
```

A Base-only monitor is not the paper's equal-contribution matched pool. For that
comparison, collect both models' rollouts and use `ard pool build --kind matched
--input BASE_RUN RL_RUN --base-model-id BASE_ID --rl-model-id RL_ID ...`.
Pools fail rather than duplicate inputs to meet an oversized quota.

Local checkpoint: install `.[hf]`, then use `backend: transformers`, local `path`,
`device`, `dtype`, `model_id` and explicit `revision` in the model config. Assets
are user supplied; no automatic downloads or remote code. See
[model evaluation](model-evaluation.md) and [support matrix](support-matrix.md).
# Offline walkthrough

After installing the core, run `ard toolkit-demo --output runs/toolkit-walkthrough`
and open `runs/toolkit-walkthrough/index.html`. It creates and audits actual
agent-v2 budget, frozen-state, error/recovery and Base-monitor artifacts using
explicitly scripted CPU policies. See the [walkthrough tutorial](tutorials/offline-toolkit.md)
for outputs and scope; it does not train models or reproduce paper numbers.
