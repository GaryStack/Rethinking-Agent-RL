# Bring Inspect evaluations into Agent RL Diagnostics

Import native Inspect `.eval` or JSON evaluation logs, then use the existing
matched-task report and CI gate. Ordinary JSONL analysis remains stdlib-only.
This optional adapter is tested with **inspect-ai 0.3.275** through the public
[log reader API](https://inspect.aisi.org.uk/reference/inspect_ai.log.html).

## Install and import your existing evaluation

From a source checkout:

```bash
python -m pip install -e '.[inspect]'
ard traces inspect --input your-run.eval --design design.json --scorer match --label base --output runs/inspect-base
ard traces audit --input runs/inspect-base
```

Declare the expected task IDs, sample slots, dataset/scorer/protocol revisions and
budget condition in [the design JSON](../../examples/external-traces/design.json)
before looking at results; do not infer the evaluation schedule from successful
records. Replace the example's IDs and conditions with your real schedule. Imports
require one Inspect task/model log at a time. Repeat for a second version:

```bash
ard traces inspect --input candidate-run.eval --design design.json --scorer match --label candidate --output runs/inspect-candidate
ard traces compare --base runs/inspect-base --candidate runs/inspect-candidate --min-delta -0.02 --output runs/inspect-comparison
```

Select a scorer by its exact log name. Binary `C/I`, boolean and numeric `0/1`
outcomes are supported; partial credit and composite scores require explicit
preprocessing. Never select the first scorer automatically. Unscored `U`, absent
scores and invalidated samples stay missing. Evaluation-level sample errors stay
infrastructure failures; a tool error that the agent handles is part of its actual
trajectory and terminal score, not an evaluation crash.

Inspect epoch 1 maps to sample slot 0. Numeric sample IDs become strings; collisions
with string IDs are rejected. Configured epochs must agree with the design.
Interrupted logs can be imported; absent slots make full metrics unavailable.

Token totals sum sample `model_usage`, including any grader calls; `role_usage`
is not added again. Cost is imported only if every recorded model usage entry has
`total_cost` and the declared cost unit is USD. Unknown usage remains unknown.
Sample `total_time` supplies latency. A resource-limit failure with an explicit
negative score is classified as budget exhaustion; operator/custom interruptions
are not automatically turned into budget failures.

The normalized source rows retain the complete SDK-exported samples (messages,
tool events, scores and store), evaluation specification, selected scorer, SDK
version and original file SHA256. `traces audit` rechecks the exported JSONL and
metrics; it does not re-read or authenticate the original `.eval`, nor replay an
environment. Keep the native logs if you need to revisit the original evidence.
Dataset/scorer IDs and budget semantics remain caller declarations: matching names
alone cannot prove that prompts, grading and conditions are equivalent.

## Run a small tool-agent case

```bash
python examples/inspect_inventory.py --output runs/inventory-case
```

Open `runs/inventory-case/comparison/report.html`. This executes Inspect's model /
tool loop and an authored parameterized SQLite inventory lookup. A controlled
first-request fault affects the pen query. The base tool exposes the error; the
candidate tool retries once internally. The two tasks share the same inputs,
targets, scorer and three-assistant-turn limit. Internal retry requests are
recorded in each sample's `store.inventory_calls`; they consume work even though
they do not add another assistant turn.

By default, **model responses are scripted**, while tools and native logging really
execute. The expected synthetic success rates are 0.5 and 1.0. Scripted mode reports
zero LLM tokens and unknown cost; these are not model cost or efficacy benchmarks.
The example is a complete integration case, not a company dataset or a measured
model improvement. No model, network service or GPU is used after installation.

For a real model supported by Inspect, configure its credentials or local serving
endpoint using that provider's normal environment settings, then run:

```bash
python examples/inspect_inventory.py --model <inspect-provider/model> --output runs/inventory-live
```

This performs actual inference and may incur provider charges. Inspect logs and
the report retain the selected model and `live_model` case mode. The model may
already retry successfully with the base tool; improvement is not assumed or
required. The live-model path is provided but is not covered by the offline
acceptance run. With two authored cases, results cannot establish general
performance. The example only executes fixed parameterized SQL, not generated
programs or arbitrary model-written queries.

For CI, install `.[dev,inspect]` and run `python -m pytest tests/test_inspect_logs.py`.
The optional native test checks real tool messages, internal retry counts, original
file hashes, JSON/.eval ingestion and both summary outcomes. The separate Inspect
CI job runs this test on Python 3.12; core users do not need the optional extra.
