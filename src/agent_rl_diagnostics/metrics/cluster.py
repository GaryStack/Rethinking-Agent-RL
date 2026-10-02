"""Task-cluster uncertainty without changing state/pair-weighted estimators."""

from __future__ import annotations

import random
from statistics import fmean
from typing import Any

from ..contracts.schema import integer


def cluster_ci(
    rows: list[dict[str, Any]], field: str, *, replicates: int, seed: int
) -> dict[str, Any]:
    integer(replicates, "replicates", 1)
    groups: dict[str, list[float]] = {}
    for row in rows:
        if row[field] is None:
            return {"low": None, "high": None, "status": "incomplete"}
        groups.setdefault(row["task_id"], []).append(row[field])
    if not groups:
        return {"low": None, "high": None, "status": "empty"}
    clusters = list(groups.values())
    rng = random.Random(seed)
    values = sorted(
        fmean(value for _ in clusters for value in rng.choice(clusters)) for _ in range(replicates)
    )

    def percentile(p: float) -> float:
        index = p * (len(values) - 1)
        lo = int(index)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)

    return {
        "low": percentile(0.025),
        "high": percentile(0.975),
        "confidence": 0.95,
        "method": "percentile_task_cluster",
        "weighting": "equal_states_or_pairs_within_resampled_tasks",
        "task_clusters": len(clusters),
        "replicates": replicates,
        "seed": seed,
        "status": "available",
    }
