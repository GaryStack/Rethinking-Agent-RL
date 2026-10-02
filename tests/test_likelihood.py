import copy
import math

import pytest

from agent_rl_diagnostics.analysis.likelihood import length_control, pair_scores, score_record
from agent_rl_diagnostics.contracts.schema import SchemaError


def record(source="base", *, model="base", event_id="e", trajectory="t", logs=None):
    event = {
        "event_id": event_id,
        "task_id": "task",
        "trajectory_id": trajectory,
        "source": source,
        "success": True,
        "prompt": "input:",
        "response": "a b",
        "action_span": [2, 3],
    }
    return score_record(
        event,
        model_id=model,
        prompt_ids=[0],
        response_ids=[1, 2],
        logprobs=logs or [-math.log(2), -math.log(4)],
        offsets=[(0, 1), (2, 3)],
    )


def test_nll_math_and_llr_sign():
    x = record()
    y = record(model="rl", logs=[-math.log(2), -math.log(2)])
    assert x["action_nll_sum"] == pytest.approx(math.log(4))
    assert x["response_nll_sum"] == pytest.approx(math.log(8))
    result = pair_scores([x], [y])
    assert result["paired"][0]["llr_sum"] == pytest.approx(math.log(2))
    assert result["full_estimate_available"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("prompt_ids", [7]),
        ("response_ids", [3, 4]),
        ("input_hash", "different"),
        ("action_mask", [True, False]),
    ],
)
def test_pair_rejects_incomparable_tokens(field, value):
    x, y = record(), record(model="rl")
    y[field] = value
    with pytest.raises(SchemaError):
        pair_scores([x], [y])


def test_failures_and_duplicates():
    x = record()
    result = pair_scores([x], [{"event_id": "e", "status": "scoring_failure"}])
    assert not result["paired"] and not result["full_estimate_available"]
    assert pair_scores([x], [])["excluded"][0]["reason"] == "missing_policy_score"
    with pytest.raises(SchemaError):
        pair_scores([x, x], [x])


def test_no_action_and_invalid_logprobs():
    event = {
        "event_id": "x",
        "task_id": "t",
        "trajectory_id": "tr",
        "source": "base",
        "success": False,
        "prompt": "p",
        "response": "a",
    }
    row = score_record(
        event, model_id="b", prompt_ids=[0], response_ids=[1], logprobs=[-1], offsets=[(0, 1)]
    )
    assert row["action_nll_sum"] is None
    for value in (float("nan"), float("inf"), 0.1):
        with pytest.raises(SchemaError):
            score_record(
                event,
                model_id="b",
                prompt_ids=[0],
                response_ids=[1],
                logprobs=[value],
                offsets=[(0, 1)],
            )


def test_length_control_reports_filtering():
    base = [record(), record("rl", event_id="e2", trajectory="tr2")]
    rows = pair_scores(base, copy.deepcopy(base))["paired"]
    result = length_control(rows, token_caliper=0, step_caliper=0)
    assert result["matched_pairs"] == 1
    assert result["unmatched_trajectories"] == 0
    assert length_control(rows[:1], token_caliper=0, step_caliper=0)["matched_pairs"] == 0


def test_action_totals_are_recomputed_from_mask():
    x, y = record(), record(model="rl")
    x["action_nll_sum"] = 999
    result = pair_scores([x], [y])
    assert result["paired"][0]["base_action_nll_sum"] == pytest.approx(math.log(4))


def test_length_control_excludes_whole_incompletely_scored_trajectory():
    base = [
        record(event_id="b1"),
        record(event_id="b2"),
        record("rl", event_id="r1", trajectory="rl-trajectory"),
    ]
    result = pair_scores(base, [base[0], base[2]])
    control = length_control(
        result["paired"], token_caliper=1, step_caliper=10, excluded_events=result["excluded"]
    )
    assert control["matched_pairs"] == 0
    assert control["excluded_incomplete_trajectories"] == 1
    assert control["eligible_trajectories"] == 1


def test_unidentified_scoring_failures_disable_length_control():
    row = pair_scores([record()], [record()])["paired"]
    control = length_control(
        row, token_caliper=1, step_caliper=1, excluded_events=[{"event_id": "missing-identity"}]
    )
    assert not control["available"]
    assert control["reason"] == "excluded_events_lack_trajectory_identity"


def test_cli_length_control_preserves_missing_event_exclusions(tmp_path, capsys):
    from agent_rl_diagnostics.artifacts.io import read_json, write_json
    from agent_rl_diagnostics.cli import main

    rows = [
        record(event_id="b1"),
        record(event_id="b2"),
        record("rl", event_id="r1", trajectory="rl-trajectory"),
    ]
    write_json(tmp_path / "base.json", rows)
    write_json(tmp_path / "rl.json", [rows[0], rows[2]])
    assert (
        main(
            [
                "likelihood",
                "analyze",
                "--base",
                str(tmp_path / "base.json"),
                "--rl",
                str(tmp_path / "rl.json"),
                "--token-caliper",
                "1",
                "--step-caliper",
                "10",
                "--output",
                str(tmp_path / "report"),
            ]
        )
        == 0
    )
    control = read_json(tmp_path / "report" / "summary.json")["length_control"]
    assert control["matched_pairs"] == 0
    assert control["excluded_incomplete_trajectories"] == 1


@pytest.mark.parametrize(
    "bad", [None, {}, [{"status": "ok"}], [{"event_id": "e", "status": "legacy"}]]
)
def test_malformed_policy_scores_fail_with_schema_error(bad):
    with pytest.raises(SchemaError):
        pair_scores(bad, [])
