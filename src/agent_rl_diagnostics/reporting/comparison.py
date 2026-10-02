"""One static report entry for audited Base/RL budget, state and recovery results."""

import html
import json
from collections import defaultdict
from statistics import fmean

from ..contracts.schema import SchemaError


def compare_summaries(summaries, *, labels):
    if len(summaries) != len(labels) or not summaries or len(set(labels)) != len(labels):
        raise SchemaError("one distinct label required per run")
    first = summaries[0]
    for row in summaries[1:]:
        for key in (
            "kind",
            "environment",
            "scorer",
            "pool_hash",
            "evidence_kind",
            "protocol_id",
            "budget_mode",
            "samples_per_state",
            "samples_per_condition",
            "certificates",
        ):
            if row.get(key) != first.get(key):
                raise SchemaError(f"comparison mixes {key}")
        if first.get("kind") in {"step_quality", "recovery"}:
            slots = "per_state" if first["kind"] == "step_quality" else "per_pair"
            identity = "state_id" if slots == "per_state" else "pair_id"
            if {x[identity] for x in row[slots]} != {x[identity] for x in first[slots]}:
                raise SchemaError("comparison requires identical frozen inputs")
        elif "per_task" in first:

            def identity(x):
                return (x["task_id"], x["total_budget"], x["k"])

            if {identity(x) for x in row["per_task"]} != {identity(x) for x in first["per_task"]}:
                raise SchemaError("budget comparison requires matched tasks and K/T grid")
    series = []
    for label, summary in zip(labels, summaries, strict=True):
        if summary.get("kind") == "step_quality":
            by_depth = defaultdict(list)
            for state in summary["per_state"]:
                by_depth[state["depth"]].append(state["score_mean"])
            for depth, values in sorted(by_depth.items()):
                series.append(
                    {
                        "label": label,
                        "metric": summary["scorer"],
                        "x": depth,
                        "x_unit": "prefix_depth",
                        "value": fmean(values) if None not in values else None,
                        "states": len(values),
                    }
                )
        elif summary.get("kind") == "recovery":
            for index, key in enumerate(
                ("clean_success", "error_success", "paired_delta_error_minus_clean")
            ):
                series.append(
                    {
                        "label": label,
                        "metric": key,
                        "x": index,
                        "x_unit": "condition",
                        "value": summary[key],
                    }
                )
        else:
            models = {cell["model_id"] for cell in summary["cells"]}
            for cell in summary["cells"]:
                series.append(
                    {
                        "label": label if len(models) == 1 else f"{label}/{cell['model_id']}",
                        "metric": f"Pass@{cell['k']}",
                        "x": cell["total_budget"],
                        "x_unit": "action_budget",
                        "value": cell["pass_at_k"],
                        "structural_ceiling": cell.get("structural_success_ceiling"),
                    }
                )
            slack_groups = defaultdict(list)
            for task in summary["per_task"]:
                certificate = summary.get("certificates", {}).get(task["task_id"], {})
                minimum = certificate.get("value")
                if minimum is None or certificate.get("kind") != "exact_minimum":
                    continue
                key = (task["model_id"], task["k"], task["total_budget"] - minimum)
                value = (
                    task["pass_at_k"]
                    if not task.get("missing_slots") and not task.get("infrastructure_failures")
                    else None
                )
                slack_groups[key].append(value)
            for (model, k, slack), values in sorted(slack_groups.items()):
                series.append(
                    {
                        "label": label if len(models) == 1 else f"{label}/{model}",
                        "metric": f"Pass@{k} by slack",
                        "x": slack,
                        "x_unit": "T_minus_exact_minimum",
                        "value": fmean(values) if None not in values else None,
                        "tasks": len(values),
                    }
                )
    return {
        "schema_version": "ard-comparison-v1",
        "series": series,
        "scorer": first.get("scorer"),
        "pool_hash": first.get("pool_hash"),
        "interpretation": "descriptive_comparison_no_significance_claim",
    }


def render_analysis(report):
    """Portable HTML with separate SVG panels per metric; null points stay unavailable."""
    rows = report.get("series", [])
    panels = []
    for metric in sorted({x["metric"] for x in rows}):
        points = [x for x in rows if x["metric"] == metric and x["value"] is not None]
        if not points:
            panels.append(f"<p>{html.escape(metric)}: unavailable</p>")
            continue
        xmin, xmax = min(x["x"] for x in points), max(x["x"] for x in points)
        ymin, ymax = min(x["value"] for x in points), max(x["value"] for x in points)
        labels = sorted({x["label"] for x in points})
        dots = []
        for point in points:
            x = 50 + 450 * (point["x"] - xmin) / max(1, xmax - xmin)
            y = 190 - 150 * (point["value"] - ymin) / max(1e-9, ymax - ymin)
            color = ("#247ba0", "#e45756", "#555555")[labels.index(point["label"]) % 3]
            title = html.escape(f"{point['label']}: x={point['x']}, value={point['value']}")
            dots.append(
                f'<circle cx="{x}" cy="{y}" r="5" fill="{color}"><title>{title}</title></circle>'
            )
        panels.append(
            f'<h2>{html.escape(metric)}</h2><svg viewBox="0 0 600 240" '
            f'role="img" aria-label="{html.escape(metric)}"><path d="M50 30V190H520" '
            f'fill="none" stroke="black"/><text x="5" y="30">{ymax:.3g}</text>'
            f'<text x="5" y="190">{ymin:.3g}</text><text x="50" y="220">{xmin}</text>'
            f'<text x="500" y="220">{xmax}</text>{"".join(dots)}</svg>'
            f"<p>{html.escape(points[0]['x_unit'])}; {html.escape(', '.join(labels))}. "
            "Hover each point for its label and value.</p>"
        )
    data = html.escape(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return (
        '<!doctype html><meta charset="utf-8"><title>Diagnostic comparison</title>'
        "<style>body{max-width:960px;margin:2rem auto;font-family:system-ui}svg{width:100%;"
        "max-width:700px}pre{white-space:pre-wrap}</style><h1>Diagnostic comparison</h1>"
        "<p>Descriptive results. Missing values are unavailable; panels have separate units.</p>"
        + "".join(panels)
        + f"<details><summary>Full data and exclusions</summary><pre>{data}</pre></details>"
    )
