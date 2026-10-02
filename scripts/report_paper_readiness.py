#!/usr/bin/env python3
"""Generate an immutable acquisition checklist from the existing asset ledger."""

import argparse
import json
from pathlib import Path

from agent_rl_diagnostics.artifacts.io import file_hash, new_output_directory, read_json, write_json
from agent_rl_diagnostics.training.readiness import paper_asset_readiness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = paper_asset_readiness(read_json(args.ledger))
    report["ledger_sha256"] = file_hash(args.ledger)
    with new_output_directory(args.output) as root:
        write_json(root / "readiness.json", report)
        lines = [
            "# Paper asset acquisition checklist",
            "",
            "Metadata inspection only; not a paper reproduction certificate.",
            "",
        ]
        for entry in report["experiments"]:
            lines.extend([f"## {entry['experiment_id']} ({entry['paper_location']})", ""])
            lines.extend(f"- [ ] {task}" for task in entry["acquisition_tasks"])
            lines.append("")
        (root / "checklist.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "pending_experiments": report["pending_experiments"],
                "paper_reproduction_verified": False,
            }
        )
    )


if __name__ == "__main__":
    main()
