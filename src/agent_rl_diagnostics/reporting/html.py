from __future__ import annotations

from html import escape
from typing import Any

COLORS = ("#1665d8", "#d15a22", "#009578", "#9c52c0")


def _number(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "unavailable"
    return f"{100 * value:.1f}%" if percent else f"{value:.3f}"


def _plot(cells: list[dict[str, Any]], k: int) -> str:
    selected = [c for c in cells if c["k"] == k]
    budgets = sorted({c["total_budget"] for c in selected})
    models = list(dict.fromkeys(c["model_id"] for c in selected))
    left, top, width, height = 58, 22, 470, 170
    span = max(budgets) - min(budgets) or 1

    def x(t: int) -> float:
        return left + (t - min(budgets)) / span * width

    def y(v: float) -> float:
        return top + (1 - v) * height

    parts = [f'<svg viewBox="0 0 620 268" role="img" aria-label="Pass at K={k} by budget">']
    for value in (0, 0.25, 0.5, 0.75, 1):
        parts.append(
            f'<line x1="{left}" y1="{y(value)}" x2="{left + width}" y2="{y(value)}" stroke="#e2e8f0"/>'
        )
        parts.append(
            f'<text x="48" y="{y(value) + 4}" text-anchor="end">{int(100 * value)}%</text>'
        )
    for t in budgets:
        parts.append(f'<text x="{x(t)}" y="212" text-anchor="middle">{t}</text>')
    parts.append('<text x="290" y="235" text-anchor="middle">T (agent actions)</text>')
    ceilings = {c["total_budget"]: c["structural_success_ceiling"] for c in selected}
    coords = " ".join(f"{x(t)},{y(ceilings[t])}" for t in budgets)
    parts.append(
        f'<polyline points="{coords}" fill="none" stroke="#64748b" stroke-dasharray="5 4"/>'
    )
    for index, model in enumerate(models):
        color = COLORS[index % len(COLORS)]
        rows = sorted(
            (c for c in selected if c["model_id"] == model), key=lambda c: c["total_budget"]
        )
        segment = []
        for row in rows:
            value = row["pass_at_k"]
            if value is None:
                if segment:
                    parts.append(
                        f'<polyline points="{" ".join(segment)}" fill="none" stroke="{color}" stroke-width="2"/>'
                    )
                segment = []
                continue
            px, py = x(row["total_budget"]), y(value)
            segment.append(f"{px},{py}")
            tooltip = escape(f"{model}: T={row['total_budget']}, Pass@{k}={value:.4f}")
            parts.append(
                f'<circle cx="{px}" cy="{py}" r="3.5" fill="{color}"><title>{tooltip}</title></circle>'
            )
        if segment:
            parts.append(
                f'<polyline points="{" ".join(segment)}" fill="none" stroke="{color}" stroke-width="2"/>'
            )
    parts.append("</svg>")
    legend = " ".join(
        f'<span style="color:{COLORS[i % len(COLORS)]}">{escape(m)}</span>'
        for i, m in enumerate(models)
    )
    return f'<div class="chart"><h3>Pass@(K={k}, T)</h3>{"".join(parts)}<p>{legend} · dashed: structural ceiling</p></div>'


def render_report(summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    cells = summary["cells"]
    charts = "".join(_plot(cells, k) for k in sorted({c["k"] for c in cells}))
    table_rows = []
    for row in cells:
        ci = (
            "unavailable"
            if row["ci_low"] is None
            else (
                f"[{_number(row['ci_low'], percent=True)}, {_number(row['ci_high'], percent=True)}]"
            )
        )
        table_rows.append(
            "<tr>"
            + "".join(
                f"<td>{escape(str(v))}</td>"
                for v in (
                    row["model_id"],
                    row["total_budget"],
                    row["k"],
                    _number(row["pass_at_k"], percent=True),
                    ci,
                    _number(row["structural_success_ceiling"], percent=True),
                    _number(row["feasible_subset_pass_at_k"], percent=True),
                    row["valid_samples"],
                    row["missing_slots"],
                    row["infrastructure_failures"],
                    row["status"],
                )
            )
            + "</tr>"
        )
    cert_rows = "".join(
        f"<tr><td>{escape(task)}</td><td>{escape(cert['kind'])}</td>"
        f"<td>{'unreachable' if cert['unreachable'] else cert['value']}</td>"
        f"<td>{escape(cert['solver_version'])}</td></tr>"
        for task, cert in summary["certificates"].items()
    )
    run_id = escape(str(manifest["run_id"]))
    config_hash = escape(str(summary["config_hash"]))
    cost = manifest.get("cost", {})
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agent RL Diagnostics — {run_id}</title><style>
body{{font:15px/1.6 system-ui,sans-serif;color:#182b42;background:#f6f8fb;margin:0}}
main{{max-width:1180px;margin:32px auto;padding:0 24px}} h1,h2,h3{{line-height:1.25}}
.notice{{background:#fff2ce;padding:16px;border-left:4px solid #d59b24}}
.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}}
.chart{{background:white;padding:18px;border-radius:10px}} svg{{width:100%;font:12px system-ui}}
table{{border-collapse:collapse;background:white;width:100%;font-size:13px}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid #e2e8f0}}
.scroll{{overflow-x:auto}} code{{overflow-wrap:anywhere}} a{{color:#1665d8}}
</style></head><body><main>
<h1>Agent RL Diagnostics</h1>
<p class="notice"><strong>Synthetic demonstration.</strong> These are actual rollouts of scripted random and shortest-path policies on toy grids. They are not LLM results or paper reproduction.</p>
<h2>Budget stress test</h2><p>Each T is sampled independently. A budget is supplied before every action. Scores weight tasks equally, including structurally unreachable tasks.</p>
<div class="charts">{charts}</div>
<h2>Metrics and health</h2><p>CI: task-cluster percentile bootstrap; measures between-task uncertainty only. Toy CIs are not evidence of generalization. Missing cells remain unavailable.</p>
<div class="scroll"><table><thead><tr><th>Policy</th><th>T</th><th>K</th><th>Pass@(K,T)</th><th>95% CI</th><th>Ceiling</th><th>Feasible subset</th><th>Samples</th><th>Missing</th><th>Infra failures</th><th>Status</th></tr></thead><tbody>{"".join(table_rows)}</tbody></table></div>
<h2>Structural budget certificates</h2><p>Deterministic moves; one action per turn; holes terminate in failure; goal terminates in success. Feasible-subset scores supplement the full-task result.</p>
<table><thead><tr><th>Task</th><th>Proof kind</th><th>Minimum actions</th><th>Solver</th></tr></thead><tbody>{cert_rows}</tbody></table>
<h2>Provenance and cost</h2><p>Run: <code>{run_id}</code><br>Config hash: <code>{config_hash}</code><br>Actions: {escape(str(cost.get("actions", "unavailable")))}; trajectories: {escape(str(cost.get("trajectories", "unavailable")))}.</p>
<p><a href="summary.json">JSON metrics</a></p>
<p>The run directory also stores config.json, trajectories.jsonl, metrics.csv and manifest.json with SHA-256 hashes. Frozen-state quality, recovery and training-dynamic reports will be added in subsequent stages.</p>
</main></body></html>"""
