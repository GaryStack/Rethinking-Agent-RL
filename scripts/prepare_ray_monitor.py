#!/usr/bin/env python3
"""Collect a real Base-only held-out grid pool with tensor-content provenance.

No oracle replacement or automatic quota duplication; mounts may remain read-only.
"""

import argparse
import json
from pathlib import Path

from agent_rl_diagnostics.agent_config import parse_agent_config
from agent_rl_diagnostics.artifacts.io import read_json, verify_artifacts, write_json
from agent_rl_diagnostics.integrations.grid_fixture import TRAIN_GRID_TASKS, check_grid_holdout
from agent_rl_diagnostics.integrations.tensor_identity import tensor_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or args.size < 2:
        parser.error("new output and size>=2 required")
    raw = read_json(args.template)
    raw["model"].update(
        path=str(args.model_path.resolve()),
        device=args.device,
        revision="pending_tensor_content_verification",
    )
    config = parse_agent_config(raw, base=args.template.parent)
    check_grid_holdout(config.tasks)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "configuration_only",
                    "model_loading": False,
                    "inference_executed": False,
                    "pool_size": args.size,
                }
            )
        )
        return
    from agent_rl_diagnostics.evaluators.agent import audit_agent_budget, run_agent_budget
    from agent_rl_diagnostics.pools.frozen import build_pool, load_pool

    identity = tensor_identity(args.model_path)
    raw["model"]["revision"] = "tensor-content:" + identity["fingerprint"]
    config = parse_agent_config(raw, base=args.template.parent)
    args.output.mkdir(parents=True)
    write_json(args.output / "base-tensor-identity.json", identity)
    write_json(args.output / "monitor-config.json", config.raw)
    budget = args.output / "budget"
    run_agent_budget(config, budget)
    audit_agent_budget(budget, verify_artifacts(budget))
    build_pool(
        [budget],
        args.output / "pool",
        kind="base_monitor",
        size=args.size,
        seed=config.raw["seed"],
        base_model_id=config.model_id,
        max_per_task=args.size,
        min_tasks=2,
        training_task_ids={task["task_id"] for task in TRAIN_GRID_TASKS},
    )
    manifest, states, _ = load_pool(args.output / "pool")
    report = {
        "scope": "actual_base_collected_heldout_grid_monitor",
        "pool_hash": manifest["pool_hash"],
        "states": len(states),
        "base_tensor_fingerprint": identity["fingerprint"],
        "task_ids_disjoint": True,
        "map_contents_disjoint": True,
        "paper_reproduction": False,
        "oracle_states_used": False,
    }
    write_json(args.output / "verification.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
