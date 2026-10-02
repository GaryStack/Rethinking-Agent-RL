#!/usr/bin/env python3
"""Replay one complete BudgetMix update; never treat plans as executed rollouts."""

import argparse
import json
from pathlib import Path

from agent_rl_diagnostics.artifacts.io import read_json
from agent_rl_diagnostics.integrations.budgetmix_audit import audit_budgetmix_traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--update", required=True, type=int)
    parser.add_argument("--group-size", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.update < 1 or args.output.exists() or args.output.is_symlink():
        parser.error("positive update and a new output file required")
    candidates = [read_json(path) for path in sorted(args.traces.glob("*.json"))]
    selected = [
        trace for trace in candidates if trace.get("budgetmix", {}).get("update") == args.update
    ]
    report = {
        "scope": "completed_raw_grid_group_replay",
        "update": args.update,
        "metrics": audit_budgetmix_traces(selected, group_size=args.group_size),
        "other_traces_not_in_scope": len(candidates) - len(selected),
        "optimizer_training_verified": False,
        "paper_reproduction": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
