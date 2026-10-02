#!/usr/bin/env python3
"""Targeted read-only inventory. Presence/hashes do not certify reproduction."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_rl_diagnostics.artifacts.io import file_hash, new_output_directory, read_json, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def observe(source: Path, relative: str) -> dict[str, Any]:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"expected relative source path: {relative}")
    result: dict[str, Any] = {"relative_path": relative}
    try:
        target = source / path
        stat = target.stat()
        if not target.is_file():
            result["status"] = "not_a_regular_file"
        elif stat.st_size > 100 * 1024 * 1024:
            result.update(status="not_hashed_size_limit", bytes=stat.st_size)
        else:
            result.update(status="observed", bytes=stat.st_size, sha256=file_hash(target))
    except OSError as exc:
        result.update(status="unavailable", error_type=type(exc).__name__, errno=exc.errno)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.source.is_dir():
        parser.error("source must be a readable research-workspace directory")
    source_map = read_json(PROJECT_ROOT / "docs/reproduction/asset-map.json")
    entries = []
    for entry in source_map["entries"]:
        plots = [observe(args.source, path) for path in entry["plot_candidates"]]
        code = [observe(args.source, path) for path in entry["code_candidates"]]
        entries.append(
            {
                "experiment_id": entry["experiment_id"],
                "paper_location": entry["paper_location"],
                "protocol_id": "paper_v1",
                "plot_artifacts": plots,
                "code_artifacts": code,
                "raw_artifacts": [],
                "verification_status": "not_reproduced",
                "evidence_kind": "summary_only"
                if any(p["status"] == "observed" for p in plots)
                else "missing",
                "missing_fields": entry["missing_fields"],
                "redistribution_status": "unverified",
                "verification_command": None,
            }
        )
    paper = observe(args.source, source_map["paper"]["path"])
    expected = source_map["paper"]["expected_sha256"]
    paper.update(expected_sha256=expected, matches_final_pdf=paper.get("sha256") == expected)
    license_entries = []
    for entry in source_map["license_candidates"]:
        license_entries.append(
            {
                **entry,
                "observation": observe(args.source, entry["path"]),
                "migrated_code": False,
                "distribution_review": "pending",
            }
        )
    with new_output_directory(args.output.resolve()) as staging:
        write_json(
            staging / "asset-ledger.json",
            {
                "schema_version": "ard-asset-ledger-v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source_label": "original_research_workspace",
                "inspection": "targeted_read_only_hash_inventory",
                "paper": paper,
                "entries": entries,
                "warning": "Candidate files and summary scripts are not verified raw experiment provenance.",
            },
        )
        write_json(
            staging / "license-inventory.json",
            {
                "schema_version": "ard-license-inventory-v1",
                "entries": license_entries,
                "new_repository_original_code": "MIT",
                "third_party_code_vendored": False,
            },
        )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "experiments": len(entries),
                "final_pdf_hash_matches": paper["matches_final_pdf"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
