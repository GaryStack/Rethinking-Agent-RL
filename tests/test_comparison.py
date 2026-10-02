import pytest

from agent_rl_diagnostics.contracts.schema import SchemaError
from agent_rl_diagnostics.reporting.comparison import compare_summaries, render_analysis


def summary(scorer="q", score=0.5):
    return {
        "kind": "step_quality",
        "scorer": scorer,
        "pool_hash": "p",
        "per_state": [{"state_id": "s", "depth": 1, "score_mean": score}],
    }


def test_comparison_separates_units_and_missing():
    with pytest.raises(SchemaError):
        compare_summaries([summary(), summary("other")], labels=["Base", "RL"])
    report = compare_summaries([summary(), summary(score=None)], labels=["Base", "RL"])
    assert report["series"][1]["value"] is None
    assert "unavailable" in render_analysis(report)
    with pytest.raises(SchemaError):
        compare_summaries([summary(), summary()], labels=["same", "same"])
