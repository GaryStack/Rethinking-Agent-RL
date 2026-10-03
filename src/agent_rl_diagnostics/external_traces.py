"""Offline, caller-attested trace imports. Never execute or replay imported content."""

import html
import json
from collections import Counter
from statistics import fmean

from .artifacts.io import canonical_hash, file_hash, new_output_directory, read_json, write_json
from .contracts.schema import SchemaError, finite, integer
from .metrics.pass_at_kt import paired_bootstrap_delta

FIELDS = {"task_id", "sample_id", "status", "success", "tokens", "latency_seconds", "cost"}
RESOURCES = ("tokens", "latency_seconds", "cost")
STATUSES = {"completed", "task_failure", "budget_exhausted", "infrastructure_failure", "missing"}
LIMITS = (
    "Recorded outcomes are caller-attested, not environment-replayed. "
    "Hashes check integrity, not truth. No counterfactual recovery, Pass@K, "
    "optimal-action or minimum-budget claim. CI resamples tasks only."
)


def validate_design(design):
    required = {
        "task_ids",
        "samples_per_task",
        "protocol_id",
        "scorer_id",
        "dataset_id",
        "budget_unit",
        "budget_limit",
        "cost_unit",
    }
    if not isinstance(design, dict) or set(design) != required:
        raise SchemaError(f"design requires exactly {sorted(required)}")
    tasks = design["task_ids"]
    if (
        not isinstance(tasks, list)
        or not tasks
        or any(not isinstance(t, str) or not t.strip() for t in tasks)
        or len(set(tasks)) != len(tasks)
    ):
        raise SchemaError("task_ids must be nonempty unique strings")
    integer(design["samples_per_task"], "samples_per_task", 1)
    for key in ("protocol_id", "scorer_id", "dataset_id", "budget_unit", "cost_unit"):
        if not isinstance(design[key], str) or not design[key].strip():
            raise SchemaError(f"{key} must be a nonempty string")
    if finite(design["budget_limit"], "budget_limit") <= 0:
        raise SchemaError("budget_limit must be positive")
    return {**design, "task_ids": sorted(tasks)}


def normalize(rows, mapping, design):
    if (
        not isinstance(mapping, dict)
        or set(mapping) - FIELDS
        or any(not isinstance(v, str) or not v for v in mapping.values())
    ):
        raise SchemaError("mapping must map supported field names to dotted source paths")
    records, seen = [], set()
    for line, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise SchemaError(f"row {line} must be an object")
        record = {}
        for field in FIELDS:
            value = row
            for component in mapping.get(field, field).split("."):
                value = value.get(component) if isinstance(value, dict) else None
            record[field] = value
        task, sample = record["task_id"], record["sample_id"]
        if not isinstance(task, str) or task not in design["task_ids"]:
            raise SchemaError(f"row {line}: task outside declared design")
        integer(sample, "sample_id")
        if sample >= design["samples_per_task"] or (task, sample) in seen:
            raise SchemaError(f"row {line}: duplicate or unexpected sample slot")
        seen.add((task, sample))
        status, success = record["status"], record["success"]
        if not isinstance(status, str) or status not in STATUSES:
            raise SchemaError(f"row {line}: unsupported status")
        if status == "completed":
            if not isinstance(success, bool):
                raise SchemaError("completed requires boolean success")
        elif status in {"task_failure", "budget_exhausted"}:
            if success is not False:
                raise SchemaError("task failure / budget exhaustion requires success=false")
        elif success is not None:
            raise SchemaError("missing / infrastructure failure requires success=null or absent")
        for field in RESOURCES:
            value = record[field]
            if value is not None:
                if finite(value, field) < 0:
                    raise SchemaError(f"{field} must be nonnegative")
                if field == "tokens":
                    integer(value, field)
        record["source_line"] = line
        records.append(record)
    return sorted(records, key=lambda r: (r["task_id"], r["sample_id"]))


def summarize(records, design):
    expected = len(design["task_ids"]) * design["samples_per_task"]
    counts = Counter(r["status"] for r in records)
    absent = expected - len(records)
    by_task = {}
    for task in design["task_ids"]:
        rows = [r for r in records if r["task_id"] == task]
        valid = [r for r in rows if r["success"] is not None]
        complete = len(valid) == design["samples_per_task"]
        by_task[task] = {
            "success_rate": fmean(r["success"] for r in valid) if complete else None,
            "observed_attempts": len(valid),
            "expected_attempts": design["samples_per_task"],
        }
    rates = [v["success_rate"] for v in by_task.values()]
    resources = {}
    for field in RESOURCES:
        values = [r[field] for r in records if r[field] is not None]
        resources[field] = {
            "known_total": sum(values) if values else None,
            "total": sum(values) if len(values) == expected else None,
            "known_slots": len(values),
            "expected_slots": expected,
        }
    return {
        "success_rate": fmean(rates) if all(v is not None for v in rates) else None,
        "per_task": by_task,
        "status_counts": dict(counts),
        "absent_slots": absent,
        "resources": resources,
        "expected_slots": expected,
    }


def _read_rows(path):
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line, text in enumerate(stream, 1):
            if not text.strip():
                raise SchemaError(f"blank JSONL row {line}; remove it to preserve line identities")
            rows.append(json.loads(text, parse_constant=lambda x: _bad_number(x)))
    return rows


def _bad_number(value):
    raise SchemaError(f"non-finite JSON constant: {value}")


def import_traces(source, design, mapping, label, output):
    design = validate_design(design)
    if not isinstance(label, str) or not label.strip():
        raise SchemaError("label must be a nonempty string")
    # Copy first so the parsed bytes and hash always describe the same source snapshot.
    with new_output_directory(output) as staging:
        (staging / "source.jsonl").write_bytes(source.read_bytes())
        rows = _read_rows(staging / "source.jsonl")
        records = normalize(rows, mapping, design)
        summary = summarize(records, design)
        write_json(staging / "records.json", records)
        write_json(staging / "summary.json", summary)
        manifest = {
            "schema_version": "ard-external-traces-v1",
            "label": label,
            "design": design,
            "mapping": mapping,
            "evidence_kind": "external_recorded",
            "environment_replayed": False,
            "artifacts": {
                name: file_hash(staging / name)
                for name in ("source.jsonl", "records.json", "summary.json")
            },
        }
        write_json(staging / "manifest.json", manifest)
    return {"output": str(output), "summary": summary, "limitations": LIMITS}


def audit_traces(directory):
    manifest = read_json(directory / "manifest.json")
    if (
        not isinstance(manifest, dict)
        or set(manifest)
        != {
            "schema_version",
            "label",
            "design",
            "mapping",
            "evidence_kind",
            "environment_replayed",
            "artifacts",
        }
        or manifest.get("schema_version") != "ard-external-traces-v1"
        or manifest.get("evidence_kind") != "external_recorded"
        or manifest.get("environment_replayed") is not False
        or not isinstance(manifest.get("label"), str)
        or not manifest["label"].strip()
    ):
        raise SchemaError("unsupported external trace manifest")
    expected = {"source.jsonl", "records.json", "summary.json"}
    if not isinstance(manifest.get("artifacts"), dict) or set(manifest["artifacts"]) != expected:
        raise SchemaError("external trace artifact set mismatch")
    for name, digest in manifest["artifacts"].items():
        path = directory / name
        if path.is_symlink() or file_hash(path) != digest:
            raise SchemaError(f"artifact hash mismatch: {name}")
    design = validate_design(manifest["design"])
    records = normalize(_read_rows(directory / "source.jsonl"), manifest["mapping"], design)
    summary = summarize(records, design)
    if canonical_hash(records) != canonical_hash(
        read_json(directory / "records.json")
    ) or canonical_hash(summary) != canonical_hash(read_json(directory / "summary.json")):
        raise SchemaError("external normalization or metric recomputation mismatch")
    return manifest, records, summary


def compare_traces(base, candidate, output, *, min_delta=None, seed=0):
    integer(seed, "seed")
    bm, br, bs = audit_traces(base)
    cm, cr, cs = audit_traces(candidate)
    if validate_design(bm["design"]) != validate_design(cm["design"]):
        raise SchemaError(
            "comparison requires identical declared dataset, tasks, sampling, "
            "scorer, protocol, budget and cost unit"
        )
    if min_delta is not None:
        finite(min_delta, "min_delta")
        if not -1 <= min_delta <= 1:
            raise SchemaError("min_delta must be in [-1,1]")
    complete = bs["success_rate"] is not None and cs["success_rate"] is not None
    delta = (
        paired_bootstrap_delta(
            {t: r["success_rate"] for t, r in bs["per_task"].items()},
            {t: r["success_rate"] for t, r in cs["per_task"].items()},
            seed=seed,
        )
        if complete
        else None
    )
    tasks = []
    for task in sorted(bs["per_task"]):
        b, c = bs["per_task"][task]["success_rate"], cs["per_task"][task]["success_rate"]
        tasks.append(
            {
                "task_id": task,
                "base": b,
                "candidate": c,
                "delta": c - b if b is not None and c is not None else None,
            }
        )
    gate = (
        "not_requested"
        if min_delta is None
        else "incomplete"
        if not complete
        else "pass"
        if delta["mean"] >= min_delta
        else "fail"
    )
    report = {
        "schema_version": "ard-external-comparison-v1",
        "evidence_kind": "external_recorded",
        "labels": [bm["label"], cm["label"]],
        "design": bm["design"],
        "base": bs,
        "candidate": cs,
        "paired_success_delta": delta,
        "bootstrap_seed": seed,
        "tasks": tasks,
        "gate": {
            "status": gate,
            "min_delta": min_delta,
            "rule": "point_estimate_candidate_minus_base",
        },
        "limitations": LIMITS,
        "sources": {
            "base_manifest_sha256": file_hash(base / "manifest.json"),
            "candidate_manifest_sha256": file_hash(candidate / "manifest.json"),
        },
    }
    with new_output_directory(output) as staging:
        write_json(staging / "comparison.json", report)
        # Self-contained offline evidence for each version; rendered as escaped text only.
        evidence = {
            "base": _read_rows(base / "source.jsonl"),
            "candidate": _read_rows(candidate / "source.jsonl"),
        }
        write_json(staging / "evidence.json", evidence)
        (staging / "report.html").write_text(_render(report, br, cr, evidence), encoding="utf-8")
    return {
        "output": str(output),
        "report": str(output / "report.html"),
        "gate": report["gate"],
        "paired_success_delta": delta,
    }


def _render(report, base_records, candidate_records, evidence):
    def esc(value):
        return html.escape(str(value))

    def display(value):
        return "unavailable" if value is None else esc(round(value, 6))

    rows = []
    for metric in ("success_rate", *RESOURCES):
        values = []
        for side in ("base", "candidate"):
            summary = report[side]
            if metric == "success_rate":
                values.append(display(summary[metric]))
            else:
                resource = summary["resources"][metric]
                values.append(
                    f"{display(resource['total'])} (known subtotal "
                    f"{display(resource['known_total'])}; "
                    f"{resource['known_slots']}/{resource['expected_slots']} slots)"
                )
        rows.append(f"<tr><td>{esc(metric)}</td><td>{values[0]}</td><td>{values[1]}</td></tr>")
    overview = (
        "<table><tr><th>Metric</th><th>Base</th><th>Candidate</th></tr>"
        + "".join(rows)
        + "</table>"
    )
    delta = report["paired_success_delta"]
    overview += (
        f"<p>Paired success change: {display(delta['mean'])}; "
        f"95% task-bootstrap CI [{display(delta['ci_low'])}, "
        f"{display(delta['ci_high'])}]</p>"
        if delta
        else "<p>Paired success change: unavailable (incomplete evaluation).</p>"
    )
    overview += f"<p>Cost unit: {esc(report['design']['cost_unit'])}; latency: summed seconds.</p>"
    for side in ("base", "candidate"):
        summary = report[side]
        overview += (
            f"<p>{esc(side)}: absent slots={summary['absent_slots']}; "
            f"statuses={esc(summary['status_counts'])}</p>"
        )
    overview += f"<p>Regression gate: {esc(report['gate']['status'])}</p>"
    sections = []
    for task in report["tasks"]:
        traces = []
        for label, records in (("base", base_records), ("candidate", candidate_records)):
            for row in records:
                if row["task_id"] == task["task_id"]:
                    raw = evidence[label][row["source_line"] - 1]
                    traces.append(
                        f"<details><summary>{esc(label)} sample {row['sample_id']}: "
                        f"{esc(row['status'])}; success={esc(row['success'])}; "
                        f"source line {row['source_line']}</summary><pre>"
                        f"{esc(json.dumps(raw, ensure_ascii=False, indent=2))}"
                        "</pre></details>"
                    )
        sections.append(
            f"<section><h2>{esc(task['task_id'])}</h2><p>"
            f"Base: {esc(task['base'])}; candidate: {esc(task['candidate'])}; "
            f"delta: {esc(task['delta'])}</p>{''.join(traces)}</section>"
        )
    return (
        '<!doctype html><meta charset="utf-8"><title>External agent comparison</title>'
        "<style>body{max-width:1000px;margin:2em auto;font-family:system-ui}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}section{border-top:1px solid #ddd}"
        "</style><h1>External agent comparison</h1>"
        f"<p>{esc(report['labels'])}</p><p>{esc(LIMITS)}</p>"
        "<p>None means unavailable, never zero. Resource totals include all recorded attempts, "
        "including failures. Gate uses the point estimate, not statistical significance.</p>"
        + overview
        + "<details><summary>Full metrics and provenance</summary>"
        + f"<pre>{esc(json.dumps(report, indent=2, ensure_ascii=False))}</pre></details>"
        + "".join(sections)
    )
