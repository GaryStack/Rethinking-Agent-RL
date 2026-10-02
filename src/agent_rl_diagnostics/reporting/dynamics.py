"""Audited checkpoint series; descriptive warnings, not causal/statistical claims."""

import copy
import json
import re
from html import escape
from pathlib import Path

from ..artifacts.io import canonical_hash, file_hash, new_output_directory, read_json, write_json
from ..contracts.schema import SchemaError, finite, integer
from ..integrations.checkpoint import audit_checkpoint_validation


def analyze_points(points, *, expected_steps):
    integer(expected_steps, "expected_steps", 1)
    by_step = {}
    pools, splits, keys = set(), set(), set()
    for point in points:
        step = integer(point["checkpoint_step"], "checkpoint_step")
        if step > expected_steps or step in by_step:
            raise SchemaError("unexpected or duplicate checkpoint step")
        by_step[step] = point
        pools.add(point["pool_hash"])
        splits.add(point["split"])
        keys.update(point["metrics"])
    if not points or len(pools) != 1 or len(splits) != 1:
        raise SchemaError("nonempty series requires one fixed pool and split")
    multi = sorted(k for k in keys if re.fullmatch(r"diag/[^/]+/multi/pass_k\d+_t\d+", k))
    single = sorted(
        k for k in keys if re.fullmatch(r"diag/[^/]+/step/(q_star|test_fraction)_mean", k)
    )
    if not multi or len(single) != 1:
        raise SchemaError("one environment-specific scorer and Pass@(K,T) required")
    environment = single[0].split("/")[1]
    if any(k.split("/")[1] != environment for k in multi):
        raise SchemaError("cannot mix environments or scorer units")
    columns = [*multi, single[0]]
    rows, warnings = [], []
    for step in range(expected_steps + 1):
        point = by_step.get(step)
        healthy = point is not None and point["metrics"].get("diag/health/incomplete") == 0
        values = {}
        for key in columns:
            value = point["metrics"].get(key) if healthy else None
            if value is not None:
                finite(value, key)
                if not 0 <= value <= 1:
                    raise SchemaError("diagnostic score outside [0,1]")
            values[key] = value
        complete = healthy and all(v is not None for v in values.values())
        if not complete:
            warnings.append({"step": step, "kind": "missing_or_incomplete_validation"})
        rows.append(
            {"step": step, "status": "complete" if complete else "unavailable", "values": values}
        )
    tradeoffs = []
    for previous, current in zip(rows, rows[1:]):
        if previous["status"] != "complete" or current["status"] != "complete":
            continue
        delta_single = current["values"][single[0]] - previous["values"][single[0]]
        for key in multi:
            delta_multi = current["values"][key] - previous["values"][key]
            if delta_multi > 0 and delta_single < 0:
                tradeoffs.append(
                    {
                        "from_step": previous["step"],
                        "to_step": current["step"],
                        "multi_metric": key,
                        "multi_delta": delta_multi,
                        "single_delta": delta_single,
                    }
                )
    return {
        "schema_version": "ard-training-dynamics-v1",
        "status": "complete" if not warnings else "incomplete",
        "pool_hash": next(iter(pools)),
        "split": next(iter(splits)),
        "expected_checkpoint_points": expected_steps + 1,
        "observed_checkpoint_points": len(points),
        "single_metric": single[0],
        "multi_metrics": multi,
        "series": rows,
        "warnings": warnings,
        "observed_adjacent_tradeoffs": tradeoffs,
        "statistical_significance_established": False,
        "causal_training_effect_established": False,
        "optimizer_updates_verified": False,
        "paper_reproduction": False,
    }


def render_dynamics(report):
    columns = [*report["multi_metrics"], report["single_metric"]]
    colors = ["#2563eb", "#059669", "#7c3aed", "#d97706", "#db2777"]
    paths, legend = [], []
    end = len(report["series"]) - 1
    for index, key in enumerate(columns):
        color = colors[index % len(colors)]
        segments = []
        for row in report["series"]:
            value = row["values"][key]
            if value is None:
                segments.append(None)
            else:
                segments.append((55 + row["step"] / end * 660, 255 - value * 220))
        command, drawing = "M", []
        for point in segments:
            if point is None:
                command = "M"
                continue
            x, y = point
            drawing.append(f"{command}{x:.2f},{y:.2f}")
            command = "L"
            paths.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>')
        paths.append(
            f'<path d="{" ".join(drawing)}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        legend.append(f'<li style="color:{color}">{escape(key)}</li>')
    cells = "".join(
        "<tr><td>"
        + str(row["step"])
        + "</td><td>"
        + escape(row["status"])
        + "</td>"
        + "".join(
            "<td>"
            + ("unavailable" if row["values"][key] is None else str(row["values"][key]))
            + "</td>"
            for key in columns
        )
        + "</tr>"
        for row in report["series"]
    )
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Training dynamics</title><style>body{font:15px/1.6 system-ui;max-width:1100px;margin:32px auto;padding:16px}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}svg{width:100%;height:auto}pre{white-space:pre-wrap}</style><h1>Audited training diagnostics</h1><p>Descriptive checkpoint curves, not causal or significance evidence. Missing values remain unavailable; lines do not bridge missing points. Pass@(K,T) and Q*/test fraction are different quantities.</p><svg viewBox="0 0 760 290" role="img" aria-label="Checkpoint metrics on a fixed zero-to-one axis"><path d="M55 35V255H715" fill="none" stroke="#333"/><text x="20" y="40">1</text><text x="20" y="260">0</text><text x="55" y="280">step 0</text><text x="640" y="280">step '
        + str(end)
        + "</text>"
        + "".join(paths)
        + "</svg><ul>"
        + "".join(legend)
        + '</ul><div style="overflow:auto"><table><tr><th>step</th><th>status</th>'
        + "".join("<th>" + escape(k) + "</th>" for k in columns)
        + "</tr>"
        + cells
        + "</table></div><h2>Warnings and observed adjacent tradeoffs</h2><pre>"
        + escape(
            json.dumps(
                {
                    "warnings": report["warnings"],
                    "tradeoffs": report["observed_adjacent_tradeoffs"],
                },
                indent=2,
            )
        )
        + '</pre><a href="summary.json">JSON report</a></html>'
    )


def report_dynamics(run: Path, output: Path, *, expected_steps: int, verify_weights=False):
    points, identities, configs = [], [], []
    root = run / "diagnostics"
    for directory in sorted(root.glob("validation-*")):
        if directory.is_symlink() or not directory.is_dir():
            raise SchemaError("validation must be an ordinary directory")
        point = audit_checkpoint_validation(directory, run / "pool", verify_weights=verify_weights)
        if directory.name != f"validation-{point['checkpoint_step']}":
            raise SchemaError("validation directory and checkpoint step disagree")
        raw = copy.deepcopy(read_json(directory / "budget/config.json"))
        raw.pop("output", None)
        raw["model"].pop("path", None)
        raw["model"].pop("revision", None)
        configs.append(canonical_hash(raw))
        points.append(point)
        identities.append(
            {
                "step": point["checkpoint_step"],
                "checkpoint_id": point["checkpoint_id"],
                "event_id": point["event_id"],
            }
        )
    if len(set(configs)) != 1:
        raise SchemaError("validation protocol/config changed beyond checkpoint identity")
    report = analyze_points(points, expected_steps=expected_steps)
    report.update(
        raw_validation_audited=True,
        weights_rehashed=verify_weights,
        checkpoint_identities=identities,
        evidence_kind=raw["evidence_kind"],
        protocol_id=raw["protocol_id"],
    )
    with new_output_directory(output) as staging:
        write_json(staging / "summary.json", report)
        (staging / "report.html").write_text(render_dynamics(report), encoding="utf-8")
        write_json(
            staging / "manifest.json",
            {
                "schema_version": "ard-dynamics-report-manifest-v1",
                "kind": "training_dynamics_report",
                "source_run": str(run.resolve()),
                "checkpoint_identities": identities,
                "artifacts": {
                    name: file_hash(staging / name) for name in ("summary.json", "report.html")
                },
            },
        )
    return report
