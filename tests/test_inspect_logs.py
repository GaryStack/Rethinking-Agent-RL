import copy
import sys

import pytest

from agent_rl_diagnostics.external_traces import normalize, summarize
from agent_rl_diagnostics.integrations.inspect_logs import convert_inspect, import_inspect


@pytest.fixture
def design():
    return dict(
        task_ids=["a"],
        samples_per_task=2,
        protocol_id="tools-v1",
        scorer_id="match-v1",
        dataset_id="cases-v1",
        budget_unit="assistant_turns",
        budget_limit=3,
        cost_unit="USD",
    )


def sample(epoch=1, value="C", **extra):
    return {
        "id": "a",
        "epoch": epoch,
        "scores": {"match": {"value": value}},
        "model_usage": {
            "agent": {"total_tokens": 7, "total_cost": 0.01},
            "grader": {"total_tokens": 3, "total_cost": 0.02},
        },
        "total_time": 0.4,
        "messages": [{"role": "tool", "content": "7"}],
        **extra,
    }


def convert(samples, design, **extra):
    data = {
        "status": "success",
        "eval": {"model": "mockllm/test", "config": {"epochs": 2}, "scorers": [{"name": "match"}]},
        "samples": samples,
    }
    data.update(extra)
    return convert_inspect(
        data, scorer="match", design=design, source_sha256="source-digest", inspect_version="tested"
    )


def test_epoch_usage_and_sample_retention(design):
    samples = [sample(), sample(2, "I")]
    before = copy.deepcopy(samples)
    rows = convert(samples, design)
    assert [r["sample_id"] for r in rows] == [0, 1]
    assert [r["success"] for r in rows] == [True, False]
    assert rows[0]["tokens"] == 10 and rows[0]["cost"] == 0.03
    assert rows[0]["latency_seconds"] == 0.4
    assert rows[0]["inspect_sample"] == samples[0]
    assert rows[0]["inspect_source"]["sha256"] == "source-digest"
    assert samples == before
    summary = summarize(normalize(rows, {}, design), design)
    assert summary["success_rate"] == 0.5


@pytest.mark.parametrize(
    "extra,status",
    [
        ({"error": {"message": "transport"}}, "infrastructure_failure"),
        ({"invalidation": {"reason": "stale"}}, "missing"),
        ({"scores": {}}, "missing"),
        ({"scores": {"match": {"value": "U"}}}, "missing"),
        ({"scores": {"match": {"value": None}}}, "missing"),
    ],
)
def test_unavailable_outcomes_preserved(design, extra, status):
    row = convert([sample(**extra)], design)[0]
    assert row["status"] == status and row["success"] is None
    assert summarize(normalize([row], {}, design), design)["success_rate"] is None


@pytest.mark.parametrize("value", ["C", "I", True, False, 0, 1, 0.0, 1.0])
def test_binary_scores(design, value):
    assert isinstance(convert([sample(value=value)], design)[0]["success"], bool)


@pytest.mark.parametrize("value", ["P", "N", "false", 0.5, {"accuracy": 1}, [1]])
def test_reject_ambiguous_scores(design, value):
    with pytest.raises(ValueError, match="binary"):
        convert([sample(value=value)], design)


def test_budget_limit_and_operator_interrupt(design):
    assert (
        convert([sample(value="I", limit={"type": "turn"})], design)[0]["status"]
        == "budget_exhausted"
    )
    assert (
        convert([sample(value="I", limit={"type": "operator"})], design)[0]["status"] == "completed"
    )
    assert convert([sample(limit={"type": "turn"})], design)[0]["success"] is True


def test_unknown_usage_stays_unknown(design):
    row = convert([sample(model_usage={})], design)[0]
    assert row["tokens"] is None and row["cost"] is None
    row = convert([sample(model_usage={"m": {"total_tokens": 10}})], design)[0]
    assert row["cost"] is None
    row = convert([sample()], {**design, "cost_unit": "EUR"})[0]
    assert row["cost"] is None


def test_reject_duplicates_and_unexpected_epochs(design):
    with pytest.raises(ValueError, match="duplicate"):
        convert([sample(), sample()], design)
    with pytest.raises(ValueError):
        convert([sample(3)], design)
    with pytest.raises(ValueError, match="epochs"):
        convert([sample()], {**design, "samples_per_task": 1})


def test_no_scorer_silent_fallback(design):
    with pytest.raises(ValueError, match="not found"):
        convert([sample(scores={"other": {"value": "C"}})], design, eval={"config": {"epochs": 2}})


def test_integer_string_collision_and_running_log(design):
    with pytest.raises(ValueError, match="collide"):
        convert([sample(id=1), sample(id="1", epoch=2)], {**design, "task_ids": ["1"]})
    with pytest.raises(ValueError, match="status"):
        convert([sample()], design, status="started")


def test_missing_optional_dependency_and_bad_suffix(tmp_path, monkeypatch, design):
    with pytest.raises(ValueError, match="local"):
        import_inspect(
            tmp_path / "log.zip", scorer="match", design=design, label="x", output=tmp_path / "out"
        )
    monkeypatch.setitem(sys.modules, "inspect_ai", None)
    with pytest.raises(ImportError, match="optional extra"):
        import_inspect(
            tmp_path / "log.eval", scorer="match", design=design, label="x", output=tmp_path / "out"
        )
    assert not (tmp_path / "out").exists()


def test_native_inspect_tool_case(tmp_path, monkeypatch):
    import importlib.util
    import io
    import json
    from pathlib import Path

    pytest.importorskip("inspect_ai", reason="optional Inspect native runtime")
    from inspect_ai.log import read_eval_log

    from agent_rl_diagnostics.artifacts.io import file_hash, read_json, write_json
    from agent_rl_diagnostics.cli import main
    from agent_rl_diagnostics.external_traces import audit_traces

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "inspect-data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "inspect-cache"))
    example = Path(__file__).resolve().parents[1] / "examples" / "inspect_inventory.py"
    spec = importlib.util.spec_from_file_location("inventory_example", example)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "case"
    result = module.run(output)
    assert result["mode"] == "scripted_model_real_tools"
    for label, rate, requests in [("base", 0.5, 1), ("retry", 1.0, 2)]:
        _, _, summary = audit_traces(output / label)
        assert summary["success_rate"] == rate
        assert summary["resources"]["cost"]["total"] is None
        rows = [
            json.loads(line) for line in (output / label / "source.jsonl").read_text().splitlines()
        ]
        pen = next(r for r in rows if r["task_id"] == "pen")
        assert pen["inspect_sample"]["store"]["inventory_calls"] == requests
        assert any(m["role"] == "tool" for m in pen["inspect_sample"]["messages"])
        native = next((output / f"native-{label}").glob("*.eval"))
        assert pen["inspect_source"]["sha256"] == file_hash(native)
    native = next((output / "native-retry").glob("*.eval"))
    log = read_eval_log(io.BytesIO(native.read_bytes()), format="eval", resolve_attachments=True)
    json_log = tmp_path / "native.json"
    write_json(json_log, log.model_dump(mode="json", exclude_none=True))
    imported = tmp_path / "json-import"
    assert (
        main(
            [
                "traces",
                "inspect",
                "--input",
                str(json_log),
                "--design",
                str(output / "design.json"),
                "--scorer",
                "match",
                "--label",
                "json",
                "--output",
                str(imported),
            ]
        )
        == 0
    )
    assert audit_traces(imported)[2]["success_rate"] == 1.0
    assert (
        read_json(output / "comparison" / "comparison.json")["paired_success_delta"]["mean"] == 0.5
    )
