#!/usr/bin/env python3
"""Audit and compare two independently produced checkpoint diagnostic directories."""

import argparse
import json
from pathlib import Path

from agent_rl_diagnostics.integrations.comparison import compare_checkpoint_validations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--verify-weights", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output exists; choose a new file")
    report = compare_checkpoint_validations(
        args.left, args.right, args.pool, verify_weights=args.verify_weights
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "matched" else 1


if __name__ == "__main__":
    raise SystemExit(main())
