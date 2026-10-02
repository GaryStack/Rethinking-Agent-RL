"""Unique ceil-decile BudgetMix contracts, independently of a trainer/backend."""

import math
import random
from collections import Counter

from ..artifacts.io import canonical_hash
from ..contracts.schema import SchemaError, integer, text


def budget_candidates(tmax: int) -> tuple[int, ...]:
    integer(tmax, "tmax", 1)
    # Integer arithmetic avoids float rounding, including for very large budgets.
    return tuple(sorted({(k * tmax + 9) // 10 for k in range(1, 11)}))


def plan_occurrences(task_ids, *, tmax: int, seed: int, update: int):
    """Sample once per pre-repeat row, not per task ID or generated completion.

    Each choice uses a private PRNG seeded by the logical occurrence. Global RNG
    state is untouched. Repeated task IDs receive distinct occurrence identities.
    No minimum-budget input exists: infeasible choices are intentionally retained.
    """
    candidates = budget_candidates(tmax)
    integer(seed, "seed")
    integer(update, "update", 1)
    if not isinstance(task_ids, (list, tuple)) or not task_ids:
        raise SchemaError("nonempty ordered task_ids required")
    records = []
    for row, task_id in enumerate(task_ids):
        text(task_id, "task_id")
        identity = canonical_hash(["ard-budgetmix-v1", seed, update, row, task_id])
        records.append(
            {
                "task_id": task_id,
                "task_occurrence_id": identity,
                "row": row,
                "budget": random.Random(int(identity, 16)).choice(candidates),
                "tmax": tmax,
                "seed": seed,
                "update": update,
            }
        )
    return records


def validate_assignment(record):
    if not isinstance(record, dict):
        raise SchemaError("assignment must be an object")
    task_id = text(record.get("task_id"), "task_id")
    occurrence = text(record.get("task_occurrence_id"), "task_occurrence_id")
    row = integer(record.get("row"), "row")
    tmax = integer(record.get("tmax"), "tmax", 1)
    seed = integer(record.get("seed"), "seed")
    update = integer(record.get("update"), "update", 1)
    budget = integer(record.get("budget"), "budget", 1)
    expected_id = canonical_hash(["ard-budgetmix-v1", seed, update, row, task_id])
    expected_budget = random.Random(int(expected_id, 16)).choice(budget_candidates(tmax))
    if occurrence != expected_id or budget != expected_budget:
        raise SchemaError("assignment identity or deterministic budget mismatch")


def audit_groups(records, *, group_size: int):
    """Audit explicit repeated rollout assignments; never reconstruct absent IDs."""
    integer(group_size, "group_size", 1)
    if not isinstance(records, list) or not records:
        raise SchemaError("nonempty rollout assignments required")
    groups = {}
    for record in records:
        validate_assignment(record)
        occurrence = record["task_occurrence_id"]
        if occurrence in groups and groups[occurrence][0] != record:
            raise SchemaError("task occurrence has inconsistent rollout budgets/metadata")
        groups.setdefault(occurrence, [record, 0])[1] += 1
    if any(count != group_size for _, count in groups.values()):
        raise SchemaError("incomplete or duplicate task occurrence group")
    unique = [record for record, _ in groups.values()]
    if len({r["update"] for r in unique}) != 1 or len({r["seed"] for r in unique}) != 1:
        raise SchemaError("mixed update or sampler seeds")
    if sorted(r["row"] for r in unique) != list(range(len(unique))):
        raise SchemaError("missing/duplicate pre-repeat occurrence rows")
    if len({r["tmax"] for r in unique}) != 1:
        raise SchemaError("mixed maximum budgets")
    return {
        "groups": len(groups),
        "completions": len(records),
        "group_size": group_size,
        "budget_mean": sum(r["budget"] for r in unique) / len(unique),
        "budget_histogram": {
            str(budget): count
            for budget, count in sorted(Counter(r["budget"] for r in unique).items())
        },
    }


def terminal_bootstrap(next_value: float, termination_reason: str) -> float:
    """PPO action-level boundary contract; not an upstream GAE replacement."""
    if type(next_value) not in (int, float) or not math.isfinite(next_value):
        raise SchemaError("finite next value required")
    if termination_reason in ("success", "environment_terminal", "budget_exhausted"):
        return 0.0
    if termination_reason == "continuing":
        return float(next_value)
    raise SchemaError("unknown/infrastructure termination cannot produce a training target")


def gigpo_anchor(task_occurrence_id: str, snapshot: dict, remaining_budget: int) -> str:
    text(task_occurrence_id, "task_occurrence_id")
    integer(remaining_budget, "remaining_budget", 1)
    if not isinstance(snapshot, dict) or not snapshot:
        raise SchemaError("nonempty restorable state snapshot required")
    return canonical_hash([task_occurrence_id, snapshot, remaining_budget])


def normalize_step_rewards(rewards):
    """Population-standardized GiGPO step signal; singleton/constant => zero.

    Caller groups by budget-aware anchor. Episode advantages are not modified.
    """
    if not isinstance(rewards, (list, tuple)) or not rewards:
        raise SchemaError("nonempty finite step rewards required")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in rewards):
        raise SchemaError("nonempty finite step rewards required")
    mean = math.fsum(rewards) / len(rewards)
    variance = math.fsum((v - mean) ** 2 for v in rewards) / len(rewards)
    if variance == 0:
        return [0.0] * len(rewards)
    if not math.isfinite(variance):
        raise SchemaError("step variance overflow")
    return [(v - mean) / math.sqrt(variance) for v in rewards]
