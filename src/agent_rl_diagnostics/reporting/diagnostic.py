from __future__ import annotations

import json
from html import escape
from typing import Any


def render_diagnostic_report(
    summary: dict[str, Any], manifest: dict[str, Any], *, manifest_href: str = "manifest.json"
) -> str:
    kind = escape(str(manifest["kind"]))
    environment = escape(str(manifest["environment"]))
    checkpoint = escape(str(manifest.get("checkpoint_id", "not applicable")))
    evidence = escape(str(manifest["evidence_kind"]))
    overview = {
        key: value
        for key, value in summary.items()
        if key not in {"cells", "per_task", "per_state", "per_pair", "certificates"}
    }
    rows = summary.get("cells", summary.get("per_state", summary.get("per_pair", [])))
    if manifest["kind"] == "checkpoint_validation":
        rows = [
            {"metric": key, "value": value} for key, value in sorted(summary["metrics"].items())
        ]
    headers = list(rows[0]) if rows else []

    def cell(value: Any) -> str:
        return escape(
            json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        )

    table = (
        "<table><thead><tr>"
        + "".join(f"<th>{escape(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
    )
    table += "".join(
        "<tr>" + "".join(f"<td>{cell(row.get(h))}</td>" for h in headers) + "</tr>" for row in rows
    )
    table += "</tbody></table>"
    notice = (
        "Synthetic/test demonstration."
        if manifest["evidence_kind"] == "demo_synthetic"
        else "Recorded model evaluation. Raw model inputs and outputs are archived in this run."
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Agent RL Diagnostics: {kind}</title>
<style>body{{font:15px/1.6 system-ui;background:#f6f8fb;color:#182b42}}main{{max-width:1200px;margin:32px auto;padding:0 24px}}table{{border-collapse:collapse;background:white}}th,td{{text-align:left;vertical-align:top;padding:8px;border-bottom:1px solid #ddd}}.scroll{{overflow:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:16px}}.notice{{background:#fff2ce;padding:16px}}</style></head><body><main>
<h1>{kind}</h1><p class="notice">{notice}</p><p>Environment: {environment}<br>Checkpoint: {checkpoint}<br>Evidence: {evidence}</p>
<h2>Overview</h2><pre>{escape(json.dumps(overview, indent=2, sort_keys=True))}</pre>
<h2>Measurements and health</h2><div class="scroll">{table}</div>
<h2>Provenance</h2><p>Definitions and execution profile are stored in the source run's config and manifest. Single-step scorer units stay separate by environment. Missing values remain null.</p><p><a href="summary.json">JSON metrics</a> · <a href="{escape(manifest_href, quote=True)}">Source run manifest</a></p>
</main></body></html>"""
