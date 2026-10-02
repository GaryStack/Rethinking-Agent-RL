#!/usr/bin/env python3
"""Build no-dependency checkpoint curves by auditing raw dual-validation records."""

import argparse
import json
from pathlib import Path

from agent_rl_diagnostics.reporting.dynamics import report_dynamics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-steps", required=True, type=int)
    parser.add_argument("--verify-weights", action="store_true")
    args = parser.parse_args()
    report = report_dynamics(
        args.run,
        args.output,
        expected_steps=args.expected_steps,
        verify_weights=args.verify_weights,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output),
                "raw_validation_audited": True,
                "weights_rehashed": args.verify_weights,
                "tradeoffs": len(report["observed_adjacent_tradeoffs"]),
            }
        )
    )


if __name__ == "__main__":
    main()
