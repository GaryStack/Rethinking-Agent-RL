#!/usr/bin/env python3
"""Compare observed actor state proofs; does not certify complete trainer parity."""

import argparse
import json
from pathlib import Path

from agent_rl_diagnostics.integrations.training_state import compare_training_states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output exists")
    report = compare_training_states(args.left, args.right)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report))
    return 0 if not report["differences"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
