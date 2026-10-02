"""Immutable, CPU-only BudgetMix plans: assignments are not executed rollouts."""

from pathlib import Path

from ..artifacts.io import file_hash, new_output_directory, read_json, verify_artifacts, write_json
from ..contracts.schema import SchemaError
from .budgetmix import audit_groups, plan_occurrences


def create_plan(task_ids, *, tmax, seed, update, group_size, output: Path):
    assignments = plan_occurrences(task_ids, tmax=tmax, seed=seed, update=update)
    # This repeats assignment metadata only, not inference or model samples.
    summary = audit_groups(
        [r for r in assignments for _ in range(group_size)], group_size=group_size
    )
    plan = {
        "schema_version": "ard-budgetmix-plan-v1",
        "task_ids": task_ids,
        "tmax": tmax,
        "seed": seed,
        "update": update,
        "group_size": group_size,
        "assignments": assignments,
        "rollouts_executed": False,
        "paper_reproduction": False,
    }
    with new_output_directory(output) as root:
        write_json(root / "plan.json", plan)
        write_json(root / "metrics.json", summary)
        write_json(
            root / "manifest.json",
            {
                "schema_version": "ard-manifest-v1",
                "kind": "budgetmix_plan",
                "artifacts": {
                    name: file_hash(root / name) for name in ("plan.json", "metrics.json")
                },
            },
        )
    return {"output": str(output), "rollouts_executed": False, **summary}


def audit_plan(directory: Path):
    manifest = verify_artifacts(directory)
    if manifest.get("kind") != "budgetmix_plan" or set(manifest["artifacts"]) != {
        "plan.json",
        "metrics.json",
    }:
        raise SchemaError("not a complete BudgetMix plan")
    plan = read_json(directory / "plan.json")
    if (
        plan.get("schema_version") != "ard-budgetmix-plan-v1"
        or plan.get("rollouts_executed") is not False
        or plan.get("paper_reproduction") is not False
    ):
        raise SchemaError("invalid/unexecuted plan scope")
    expected = plan_occurrences(
        plan["task_ids"], tmax=plan["tmax"], seed=plan["seed"], update=plan["update"]
    )
    if expected != plan["assignments"]:
        raise SchemaError("plan differs from occurrence sampling")
    summary = audit_groups(
        [r for r in expected for _ in range(plan["group_size"])], group_size=plan["group_size"]
    )
    # JSON object keys become strings; compare the serialized metric identity.
    from ..artifacts.io import canonical_hash

    if canonical_hash(summary) != canonical_hash(read_json(directory / "metrics.json")):
        raise SchemaError("BudgetMix plan metrics mismatch")
    return {"status": "verified_plan", "rollouts_executed": False, **summary}
