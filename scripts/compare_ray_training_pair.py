#!/usr/bin/env python3
"""Compare actual off/on actor state and grid training trajectories, not full parity."""

import argparse
import json
from pathlib import Path

from agent_rl_diagnostics.integrations.ray_pair import compare_ray_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("new output required")
    report = compare_ray_pair(args.left.resolve(), args.right.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(report))
    return 0 if report["status"] == "matched_observed_actor_and_rollouts" else 1


if __name__ == "__main__":
    raise SystemExit(main())
