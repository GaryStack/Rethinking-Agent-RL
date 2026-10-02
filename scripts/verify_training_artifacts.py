#!/usr/bin/env python3
"""Audit a real native-actor smoke, actual W&B export and standalone parity."""

import argparse
import json
import math
from pathlib import Path

from agent_rl_diagnostics.artifacts.io import file_hash, read_json, write_json
from agent_rl_diagnostics.contracts.schema import SchemaError
from agent_rl_diagnostics.integrations.checkpoint import audit_checkpoint_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--standalone", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    acceptance = read_json(args.input / "acceptance.json")
    updates = acceptance["updates"]
    if not updates or not all(acceptance["diagnostic_parity"].values()):
        raise SchemaError("missing real updates or failed diagnostic parity")
    for update in updates:
        norms = update["native_actor_metrics"]["actor/grad_norm"]
        if update["trainable_before"] == update["trainable_after"] or not any(
            math.isfinite(x) and x > 0 for x in norms
        ):
            raise SchemaError("no real parameter/gradient transition")
    points = [
        json.loads(line)
        for line in (args.input / "tracking/metrics.jsonl").read_text().splitlines()
    ]
    history = [
        json.loads(line) for line in (args.input / "offline-history.jsonl").read_text().splitlines()
    ]
    if len(points) != len(updates) + 1 or len(history) != len(points):
        raise SchemaError("incomplete tracking history")
    for step, (point, row) in enumerate(zip(points, history, strict=True)):
        core = audit_checkpoint_validation(
            args.input / f"validation-{step}", args.input / "pool", verify_weights=True
        )
        if (
            point["checkpoint_step"] != step
            or row["checkpoint_step"] != step
            or row["checkpoint_id"] != core["checkpoint_id"]
        ):
            raise SchemaError("checkpoint history mismatch")
        if any(point["metrics"].get(k) != v for k, v in core["metrics"].items()) or any(
            row.get(k) != v for k, v in point["metrics"].items() if v is not None
        ):
            raise SchemaError("W&B/core metrics mismatch")
    offline = audit_checkpoint_validation(args.standalone, args.input / "pool", verify_weights=True)
    online = read_json(args.input / f"validation-{offline['checkpoint_step']}" / "metrics.json")
    if online != offline:
        raise SchemaError("standalone and driver metrics differ")
    equal = {}
    for relative in ("budget/trajectories.jsonl", "step/actions.jsonl"):
        equal[relative] = file_hash(args.standalone / relative) == file_hash(
            args.input / f"validation-{offline['checkpoint_step']}" / relative
        )
    if not all(equal.values()):
        raise SchemaError("standalone and driver raw records differ")
    evidence = {
        "status": "verified",
        "scope": acceptance["scope"],
        "algorithm": acceptance["algorithm"],
        "real_updates": len(updates),
        "wandb_sdk_history_rows": len(history),
        "standalone_metrics_equal": True,
        "standalone_raw_record_hashes_equal": equal,
        "diagnostic_parity": acceptance["diagnostic_parity"],
        "paper_reproduction": False,
        "full_ray_trainer_tested": False,
    }
    write_json(args.output, evidence)
    print(json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
