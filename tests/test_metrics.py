import math
from dataclasses import replace

import pytest

from agent_rl_diagnostics.contracts import SchemaError, StepRecord, Trajectory
from agent_rl_diagnostics.metrics import paired_bootstrap_delta, pass_at_k, summarize_budget


@pytest.mark.parametrize("n,c,k", [(4, 1, 2), (12, 5, 8), (20, 0, 4), (20, 20, 4)])
def test_pass_at_k_matches_exact_combinatorial_definition(n, c, k):
    assert pass_at_k(n, c, k) == pytest.approx(1 - math.comb(n - c, k) / math.comb(n, k))


@pytest.mark.parametrize("n,c,k", [(0, 0, 1), (2, 3, 1), (2, 1, 3), (4, -1, 1), (True, 0, 1)])
def test_invalid_counts_rejected(n, c, k):
    with pytest.raises(SchemaError):
        pass_at_k(n, c, k)


def test_small_nonzero_probability_retains_precision():
    assert pass_at_k(10**12, 1, 1) == pytest.approx(1e-12, rel=1e-12, abs=0)


def record(task="a", sample=0, success=True):
    step = StepRecord(0, 1, 0, 2, 1, float(success), True, success)
    return Trajectory(
        "run",
        "model",
        "checkpoint-1",
        task,
        f"{task}:{sample}",
        sample,
        1,
        "budget_aware_rerollout",
        sample,
        "completed",
        success,
        "success" if success else "environment_terminal",
        (step,),
        "demo_synthetic",
        "toolkit_v1",
    )


def summarize(records, tasks=("a", "b"), expected=2):
    return summarize_budget(
        records,
        models=["model"],
        tasks=tasks,
        budgets=[1],
        ks=[1, 2],
        expected_samples=expected,
        replicates=20,
    )


def test_equal_task_weight_and_task_ci():
    output = summarize(
        [record("a", 0), record("a", 1), record("b", 0, False), record("b", 1, False)]
    )
    assert output["cells"][0]["pass_at_k"] == 0.5
    assert output["cells"][1]["pass_at_k"] == 0.5
    assert output["aggregation"] == "equal_task_weight"


def test_missing_task_is_not_zero_and_not_silently_dropped():
    cell = summarize([record("a", 0), record("a", 1)])["cells"][0]
    assert cell["status"] == "incomplete"
    assert cell["pass_at_k"] is None
    assert cell["observed_task_mean"] == 1.0
    assert cell["eligible_tasks"] == 1
    assert cell["missing_slots"] == 2


def test_infrastructure_failure_has_separate_accounting():
    failed = replace(
        record("a", 1),
        status="infrastructure_failure",
        success=None,
        termination_reason="worker_crash",
        steps=(),
    )
    cell = summarize([record("a", 0), failed], tasks=("a",))["cells"][0]
    assert cell["pass_at_k"] is None
    assert cell["missing_slots"] == 0
    assert cell["infrastructure_failures"] == 1
    assert cell["valid_samples"] == 1


def test_duplicate_slots_rejected():
    with pytest.raises(SchemaError, match="duplicate"):
        summarize([record(), record()])


def test_checkpoint_versions_cannot_be_combined():
    with pytest.raises(SchemaError, match="checkpoint"):
        summarize([record(sample=0), replace(record(sample=1), checkpoint_id="checkpoint-2")])


def test_protocols_cannot_be_combined():
    with pytest.raises(SchemaError, match="mixed"):
        summarize([record(sample=0), replace(record(sample=1), protocol_id="paper_v1")])


def test_paired_ci_resamples_task_differences_not_independent_models():
    output = paired_bootstrap_delta({"a": 0.2, "b": 0.8}, {"a": 0.3, "b": 0.9}, replicates=20)
    assert output["mean"] == pytest.approx(0.1)
    assert output["ci_low"] == pytest.approx(0.1)
    assert output["ci_high"] == pytest.approx(0.1)
    with pytest.raises(SchemaError, match="same"):
        paired_bootstrap_delta({"a": 0}, {"b": 1})


def test_budget_aware_curves_are_allowed_to_be_nonmonotonic():
    early = record(sample=0)
    steps = (StepRecord(0, 2, 0, 0, 0, 0, False, False), StepRecord(1, 1, 0, 0, 0, 0, False, False))
    late = replace(
        early, total_budget=2, success=False, termination_reason="budget_exhausted", steps=steps
    )
    result = summarize_budget(
        [early, late],
        models=["model"],
        tasks=["a"],
        budgets=[1, 2],
        ks=[1],
        expected_samples=1,
        replicates=10,
    )
    assert [c["pass_at_k"] for c in result["cells"]] == [1, 0]
