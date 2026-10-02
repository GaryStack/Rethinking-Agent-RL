#!/usr/bin/env python3
"""Render local diagnostic JSONL into small independent per-metric SVG charts."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path


def render(points: list[dict]) -> str:
    if not points:
        raise ValueError("no diagnostic points")
    if any(a["checkpoint_step"] >= b["checkpoint_step"] for a, b in zip(points, points[1:])):
        raise ValueError("checkpoint steps must be strictly increasing")
    keys = sorted({key for point in points for key in point["metrics"]})
    sections = []
    for key in keys:
        if not ("/multi/pass_" in key or "/step/q_star" in key or key.startswith("train/")):
            continue
        pairs = [(p["checkpoint_step"], p["metrics"].get(key)) for p in points]
        pairs = [(x, y) for x, y in pairs if isinstance(y, (int, float)) and math.isfinite(y)]
        if not pairs:
            continue
        xs, ys = zip(*pairs)
        lo, hi = min(0, min(ys)), max(1, max(ys))
        span = max(xs) - min(xs) or 1
        coords = [
            (40 + 420 * (x - min(xs)) / span, 160 - 130 * (y - lo) / (hi - lo)) for x, y in pairs
        ]
        line = " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
        dots = "".join(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3"/>' for x, y in coords)
        values = ", ".join(f"step {x}: {y:.6g}" for x, y in pairs)
        sections.append(
            f'<section><h3>{html.escape(key)}</h3><svg viewBox="0 0 500 190" role="img" aria-label="{html.escape(values)}"><path d="M40 20V160H470" stroke="#888" fill="none"/><polyline points="{line}" stroke="#2563eb" fill="none"/>{dots}</svg><p>{html.escape(values)}</p></section>'
        )
    return (
        '<!doctype html><meta charset="utf-8"><title>Local training diagnostics</title><style>body{font-family:system-ui;max-width:1000px;margin:40px auto}section{display:inline-block;width:47%;vertical-align:top;margin:1%}svg{width:100%}h3{font-size:14px;overflow-wrap:anywhere}</style><h1>Local checkpoint diagnostics</h1><p>Small smoke run, not paper reproduction or evidence of benchmark improvement. Metrics use independent axes. Missing values are omitted, not zero-filled. No network resources.</p>'
        + "".join(sections)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    points = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    document = render(points)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(document)
    print(
        json.dumps(
            {"output": str(args.output), "checkpoint_steps": [p["checkpoint_step"] for p in points]}
        )
    )


if __name__ == "__main__":
    main()
