"""Task-weighted Pass@(K,T), with explicit missing-data accounting."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from ..contracts.schema import SchemaError, Trajectory, finite, integer


def pass_at_k(n: int, c: int, k: int) -> float:
    """1 - choose(n-c,k)/choose(n,k), stable without huge binomial integers."""
    integer(n, "n", 1)
    integer(c, "c")
    integer(k, "k", 1)
    if c > n or k > n:
        raise SchemaError("require 0 <= c <= n and 1 <= k <= n")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return -math.expm1(sum(math.log1p(-c / (n - i)) for i in range(k)))


def _quantile(values: Sequence[float], p: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lo = math.floor(index)
    hi = math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def bootstrap_mean(
    task_values: Sequence[float],
    *,
    seed: int = 0,
    replicates: int = 1000,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Percentile CI, resampling tasks. Does not model within-task sampling error."""
    if not task_values:
        raise SchemaError("bootstrap requires at least one task")
    integer(seed, "seed")
    integer(replicates, "replicates", 1)
    finite(confidence, "confidence")
    if not 0 < confidence < 1:
        raise SchemaError("confidence must be in (0,1)")
    values = [finite(v, "task value") for v in task_values]
    rng = random.Random(seed)
    draws = [fmean(rng.choices(values, k=len(values))) for _ in range(replicates)]
    tail = (1 - confidence) / 2
    return {
        "mean": fmean(values),
        "ci_low": _quantile(draws, tail),
        "ci_high": _quantile(draws, 1 - tail),
        "task_count": len(values),
        "replicates": replicates,
        "confidence": confidence,
    }


def paired_bootstrap_delta(
    reference: Mapping[str, float],
    candidate: Mapping[str, float],
    **kwargs: Any,
) -> dict[str, float | int]:
    if not reference or reference.keys() != candidate.keys():
        raise SchemaError("paired bootstrap requires the exact same nonempty task IDs")
    return bootstrap_mean(
        [finite(candidate[t], t) - finite(reference[t], t) for t in sorted(reference)],
        **kwargs,
    )


def summarize_budget(
    trajectories: Iterable[Trajectory],
    *,
    models: Sequence[str],
    tasks: Sequence[str],
    budgets: Sequence[int],
    ks: Sequence[int],
    expected_samples: int,
    seed: int = 0,
    replicates: int = 1000,
) -> dict[str, Any]:
    """Strict full-task estimate; partial observed-task mean is explicitly separate."""
    integer(expected_samples, "expected_samples", 1)
    for label, values in (("models", models), ("tasks", tasks), ("budgets", budgets), ("ks", ks)):
        if not values or len(set(values)) != len(values):
            raise SchemaError(f"{label} must be nonempty and unique")
    for v in (*budgets, *ks):
        integer(v, "budget/K", 1)
    if max(ks) > expected_samples:
        raise SchemaError("expected_samples must be >= max(K)")
    groups: dict[tuple[str, str, int], list[Trajectory]] = defaultdict(list)
    seen: set[tuple[str, str, int, int]] = set()
    identities: set[tuple[str, str, str, str, str]] = set()
    model_checkpoints: dict[str, str] = {}
    for record in trajectories:
        if record.model_id not in models or record.task_id not in tasks:
            raise SchemaError("trajectory contains an unexpected model or task")
        if record.total_budget not in budgets or record.sample_id >= expected_samples:
            raise SchemaError("trajectory contains an unexpected budget or sample slot")
        slot = (record.model_id, record.task_id, record.total_budget, record.sample_id)
        if slot in seen:
            raise SchemaError(f"duplicate sample slot: {slot}")
        seen.add(slot)
        identities.add(
            (
                record.run_id,
                record.protocol_id,
                record.budget_mode,
                record.schema_version,
                record.evidence_kind,
            )
        )
        previous = model_checkpoints.setdefault(record.model_id, record.checkpoint_id)
        if previous != record.checkpoint_id:
            raise SchemaError("one model_id cannot combine multiple checkpoint versions")
        groups[(record.model_id, record.task_id, record.total_budget)].append(record)
    if len(identities) > 1:
        raise SchemaError("cannot aggregate mixed run/protocol/budget-mode/evidence identities")

    cells: list[dict[str, Any]] = []
    per_task: list[dict[str, Any]] = []
    for model in models:
        for budget in budgets:
            rows = []
            for task in tasks:
                records = groups[(model, task, budget)]
                valid = [r for r in records if r.status == "completed"]
                rows.append(
                    {
                        "model_id": model,
                        "task_id": task,
                        "total_budget": budget,
                        "n": len(valid),
                        "c": sum(r.success is True for r in valid),
                        "expected_samples": expected_samples,
                        "missing_slots": expected_samples - len(records),
                        "infrastructure_failures": len(records) - len(valid),
                    }
                )
            for k in ks:
                eligible = [r for r in rows if r["n"] >= k]
                scores = [pass_at_k(r["n"], r["c"], k) for r in eligible]
                complete = all(r["n"] == expected_samples for r in rows)
                estimate = (
                    bootstrap_mean(scores, seed=seed, replicates=replicates) if complete else None
                )
                cells.append(
                    {
                        "model_id": model,
                        "total_budget": budget,
                        "k": k,
                        "status": "complete" if complete else "incomplete",
                        "pass_at_k": estimate["mean"] if estimate else None,
                        "ci_low": estimate["ci_low"] if estimate else None,
                        "ci_high": estimate["ci_high"] if estimate else None,
                        "observed_task_mean": fmean(scores) if scores else None,
                        "expected_tasks": len(tasks),
                        "eligible_tasks": len(eligible),
                        "valid_samples": sum(r["n"] for r in rows),
                        "missing_slots": sum(r["missing_slots"] for r in rows),
                        "infrastructure_failures": sum(r["infrastructure_failures"] for r in rows),
                    }
                )
                per_task.extend(
                    {
                        **r,
                        "k": k,
                        "pass_at_k": pass_at_k(r["n"], r["c"], k) if r["n"] >= k else None,
                    }
                    for r in rows
                )
    return {
        "aggregation": "equal_task_weight",
        "ci_method": "task_cluster_percentile_bootstrap",
        "ci_scope": "between_task_uncertainty_only",
        "cells": cells,
        "per_task": per_task,
    }
