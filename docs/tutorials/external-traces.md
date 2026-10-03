# Analyze your own agent logs

This stdlib-only workflow imports existing JSONL, checks its structure and recorded
file integrity, compares versions on matched tasks, and opens their original
records next to each task's result. It does not run an agent or contact a service.
Use `traces audit` for these recorded artifacts; `ard audit` remains the separate
built-in environment workflow. Imported outcomes are caller-attested, never
certified as environment-replayed or independently true.

## Try the complete workflow

From a source checkout after `python -m pip install -e .`:

```bash
ard traces import --input examples/external-traces/base.jsonl --design examples/external-traces/design.json --label base --output runs/logs-base
ard traces import --input examples/external-traces/candidate.jsonl --design examples/external-traces/design.json --label candidate --output runs/logs-candidate
ard traces audit --input runs/logs-base
ard traces compare --base runs/logs-base --candidate runs/logs-candidate --min-delta 0 --output runs/logs-comparison
```

Open `runs/logs-comparison/report.html`. This **authored synthetic fixture** shows
success rising from 0.5 to 1.0, with recorded cost rising from USD 0.0018 to 0.0026.
The second task's traces show the extra retry. It demonstrates report navigation,
not a measured method improvement or controlled recovery experiment.

## Replace the example with your data

Provide one JSON object per attempt, with no blank lines. Required fields:

| Field | Meaning |
|---|---|
| `task_id` | Nonempty string from the declared evaluation design |
| `sample_id` | Integer slot, from zero through `samples_per_task - 1` |
| `status` | `completed`, `task_failure`, `budget_exhausted`, `infrastructure_failure`, or `missing` |
| `success` | Boolean for completed attempts; false for task failure/budget exhaustion; null or absent for infrastructure failure/missing |

Optional `tokens` (nonnegative integer), `latency_seconds`, and `cost`
(nonnegative finite numbers) describe each attempt's total usage. Count all calls,
including internal retries. Omit unknown values; never fill them with zero.
Latency totals are summed attempt durations, not concurrent job wall-clock time.
Use the design's declared cost unit consistently. Zero is allowed only when known.
Store tool calls, messages and arbitrary metadata in additional fields: the entire
original JSON object is preserved and displayed. Do not include secrets or private
content in reports you intend to share: reports copy the original records locally.

A mapping file can select nested fields without modifying source logs:

```json
{"task_id": "metadata.case_id", "success": "result.passed", "tokens": "usage.total_tokens"}
```

Pass it as `--mapping mapping.json`. Unmapped fields use canonical names; dotted
paths address nested objects, not array indexes. Values are not coerced: string
`"false"` and integer `1` are not boolean outcomes. Provider-specific status values
must be explicitly converted to the documented status vocabulary before import.

The design JSON declares expected task IDs and slots, dataset revision, protocol,
scorer, budget unit/limit, and cost unit (see the example). Declare these **before
examining outcomes**, from your actual evaluation schedule. Both versions must
have the same design. Identities are caller declarations, not proof of identical
prompts, tests, or grader behavior; version these IDs when those change. Use separate
imports for different budget conditions. Labels identify model/agent versions.

## Interpret results and use CI

Success rate averages attempts within each task, then tasks equally. An absent
slot or infrastructure failure makes the full rate and paired delta unavailable.
Task failures and exhausted budgets count as completed unsuccessful attempts.
No observed-only replacement silently fills an incomplete evaluation.

Resource `total` requires every expected slot to have a value. `known_total` and
coverage are shown separately, including consumption on failed attempts. No cost
per success or cost efficiency is inferred from incomplete resource records.

The paired difference is candidate minus base; the 95% percentile bootstrap
resamples tasks. It captures between-task variation, not all sampling or judge
uncertainty. One or two tasks cannot establish broad method effectiveness.

`--min-delta -0.02` permits a point-estimate drop of up to two percentage points.
Without a threshold, the command produces a descriptive report only. Exit codes:

- `0`: report completed; requested gate passed, or no gate requested.
- `1`: requested gate failed or evaluation was incomplete; report is still saved.
- `2`: invalid input, incompatible designs, corrupted artifact, or output error.

The gate is a declared engineering tolerance, not a statistical significance test.
`comparison.json` contains machine-readable metrics, gate and source manifest
hashes; `evidence.json` contains the source rows. Import directories keep the exact
source JSONL, normalized records, recomputable summary and hashes. Existing output
directories are never overwritten. Keep imported directories for re-auditing.

## Capability boundaries

Historical logs support recorded outcome/resource comparison. They do not establish
Pass@K sampling assumptions, budget-conditioned rerollouts, optimal actions, exact
minimum budgets, or counterfactual recovery. Those require additional controlled
execution or environment capabilities. This initial adapter targets small offline
batches; additional framework exporters, concurrent execution and resumable jobs remain
future work. A pinned [Inspect native-log adapter](inspect-agent.md) is available.
No training dependencies are needed.
