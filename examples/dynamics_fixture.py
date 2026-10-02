"""Authored metric-only fixture; use ard dynamics for audited raw training series."""

import argparse
from pathlib import Path

from agent_rl_diagnostics.artifacts.io import new_output_directory, write_json
from agent_rl_diagnostics.reporting.dynamics import analyze_points, render_dynamics


def generate(output):
    points = [
        {
            "checkpoint_step": step,
            "pool_hash": "authored-fixture",
            "split": "smoke",
            "metrics": {
                "diag/grid/multi/pass_k1_t8": multi,
                "diag/grid/step/q_star_mean": single,
                "diag/health/incomplete": 0,
            },
        }
        for step, multi, single in ((0, 0.5, 0.5), (1, 0.75, 0.25))
    ]
    report = analyze_points(points, expected_steps=1)
    report["evidence_kind"] = "demo_synthetic"
    report["raw_validation_audited"] = False
    with new_output_directory(output) as root:
        write_json(root / "summary.json", report)
        (root / "report.html").write_text(render_dynamics(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    generate(parser.parse_args().output)
