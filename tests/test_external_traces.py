import json

import pytest

from agent_rl_diagnostics.artifacts.io import file_hash, read_json, write_json
from agent_rl_diagnostics.cli import main
from agent_rl_diagnostics.external_traces import (
    audit_traces,
    compare_traces,
    import_traces,
    normalize,
    summarize,
    validate_design,
)


@pytest.fixture
def design():
    return dict(
        task_ids=["a", "b"],
        samples_per_task=1,
        protocol_id="tools-v1",
        scorer_id="exact-v1",
        dataset_id="heldout-v1",
        budget_unit="tool_calls",
        budget_limit=5,
        cost_unit="USD",
    )


def record(task="a", success=True, **kwargs):
    return dict(task_id=task, sample_id=0, status="completed", success=success, **kwargs)


def imported(tmp_path, name, rows, design, mapping=None):
    source = tmp_path / f"{name}.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    output = tmp_path / name
    import_traces(source, design, mapping or {}, name, output)
    return output


def test_mapping_and_source_roundtrip(tmp_path, design):
    rows = [
        {
            "item": {"id": "a"},
            "sample_id": 0,
            "status": "completed",
            "success": True,
            "trace": [{"tool": "lookup", "output": "<script>alert(1)</script>"}],
        },
        record("b"),
    ]
    rows[1]["item"] = {"id": "b"}
    output = imported(tmp_path, "mapped", rows, design, {"task_id": "item.id"})
    _, records, summary = audit_traces(output)
    assert records[0]["source_line"] == 1
    assert summary["success_rate"] == 1
    assert summary["resources"]["cost"]["total"] is None
    assert json.loads((output / "source.jsonl").read_text().splitlines()[0]) == rows[0]
    compare_traces(output, output, tmp_path / "comparison")
    html = (tmp_path / "comparison/report.html").read_text()
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "source line 1" in html


@pytest.mark.parametrize("status", ["missing", "infrastructure_failure"])
def test_unavailable_not_failure(status, design):
    rows = [record("a"), dict(task_id="b", sample_id=0, status=status, success=None)]
    summary = summarize(normalize(rows, {}, design), design)
    assert summary["success_rate"] is None
    assert summary["status_counts"][status] == 1


def test_missing_slots_and_partial_cost(design):
    summary = summarize(normalize([record(cost=0.2)], {}, design), design)
    assert summary["success_rate"] is None and summary["absent_slots"] == 1
    assert summary["resources"]["cost"]["known_total"] == 0.2
    assert summary["resources"]["cost"]["total"] is None


@pytest.mark.parametrize("status", ["task_failure", "budget_exhausted"])
def test_task_failures_are_completed_unsuccessful(status, design):
    rows = [record(), dict(task_id="b", sample_id=0, status=status, success=False)]
    assert summarize(normalize(rows, {}, design), design)["success_rate"] == 0.5


@pytest.mark.parametrize(
    "change",
    [
        dict(success=1),
        dict(success="false"),
        dict(tokens=True),
        dict(tokens=-1),
        dict(cost=float("nan")),
        dict(latency_seconds=-1),
        dict(sample_id=1),
        dict(task_id="unexpected"),
        dict(status="unknown"),
        dict(status="infrastructure_failure"),
        dict(status="task_failure"),
    ],
)
def test_reject_bad_rows(change, design):
    row = record()
    row.update(change)
    with pytest.raises(ValueError):
        normalize([row], {}, design)


def test_duplicate_and_invalid_design(design):
    with pytest.raises(ValueError, match="duplicate"):
        normalize([record(), record()], {}, design)
    with pytest.raises(ValueError):
        validate_design({**design, "task_ids": ["a", "a"]})
    with pytest.raises(ValueError):
        normalize([record()], {"typo": "success"}, design)


def test_tamper_and_recompute(tmp_path, design):
    output = imported(tmp_path, "run", [record(), record("b")], design)
    summary = read_json(output / "summary.json")
    summary["success_rate"] = 0
    write_json(output / "summary.json", summary)
    with pytest.raises(ValueError, match="hash"):
        audit_traces(output)
    manifest = read_json(output / "manifest.json")
    manifest["artifacts"]["summary.json"] = file_hash(output / "summary.json")
    write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="recomputation"):
        audit_traces(output)


def test_paired_comparison_and_protocol_mismatch(tmp_path, design):
    base = imported(tmp_path, "base", [record(success=False), record("b")], design)
    candidate = imported(tmp_path, "candidate", [record(), record("b")], design)
    result = compare_traces(base, candidate, tmp_path / "result", min_delta=0.1)
    assert result["gate"]["status"] == "pass"
    assert result["paired_success_delta"]["mean"] == 0.5
    assert result["paired_success_delta"]["task_count"] == 2
    wrong = imported(tmp_path, "wrong", [record(), record("b")], {**design, "scorer_id": "v2"})
    with pytest.raises(ValueError, match="identical"):
        compare_traces(base, wrong, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_cli_gates_and_audit(tmp_path, design, capsys):
    base = imported(tmp_path, "base", [record(), record("b")], design)
    worse = imported(tmp_path, "worse", [record(success=False), record("b")], design)
    missing = imported(tmp_path, "missing", [record()], design)
    assert main(["traces", "audit", "--input", str(base)]) == 0
    for run in (worse, missing):
        output = tmp_path / f"compare-{run.name}"
        assert (
            main(
                [
                    "traces",
                    "compare",
                    "--base",
                    str(base),
                    "--candidate",
                    str(run),
                    "--output",
                    str(output),
                    "--min-delta",
                    "-0.1",
                ]
            )
            == 1
        )
        assert (output / "report.html").exists()
    assert (
        main(
            [
                "traces",
                "compare",
                "--base",
                str(base),
                "--candidate",
                str(worse),
                "--output",
                str(tmp_path / "invalid"),
                "--min-delta",
                "nan",
            ]
        )
        == 2
    )


def test_cli_import_and_no_overwrite(tmp_path, design):
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(record()) + "\n")
    spec = tmp_path / "design.json"
    write_json(spec, design)
    args = [
        "traces",
        "import",
        "--input",
        str(source),
        "--design",
        str(spec),
        "--label",
        "v1",
        "--output",
        str(tmp_path / "out"),
    ]
    assert main(args) == 0
    assert main(args) == 2


def test_empty_input_is_explicitly_incomplete(tmp_path, design):
    run = imported(tmp_path, "empty", [], design)
    _, _, summary = audit_traces(run)
    assert summary["absent_slots"] == 2
    assert summary["success_rate"] is None
