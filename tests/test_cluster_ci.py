import random

from agent_rl_diagnostics.metrics.cluster import cluster_ci


def test_cluster_ci_is_seeded_and_leaves_global_rng_unchanged():
    rows = [
        {"task_id": "a", "value": 1},
        {"task_id": "a", "value": 0},
        {"task_id": "b", "value": -1},
    ]
    before = random.getstate()
    first = cluster_ci(rows, "value", replicates=100, seed=7)
    assert first == cluster_ci(rows, "value", replicates=100, seed=7)
    assert first["low"] == -1 and first["high"] == 0.5
    assert first["task_clusters"] == 2 and random.getstate() == before


def test_cluster_ci_missing_is_unavailable():
    ci = cluster_ci([{"task_id": "a", "value": None}], "value", replicates=10, seed=1)
    assert ci["low"] is None and ci["status"] == "incomplete"
