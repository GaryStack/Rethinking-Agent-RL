"""Authored reporting fixtures, not GPU or paper-training evidence."""

import copy

import pytest

from agent_rl_diagnostics.contracts.schema import SchemaError
from agent_rl_diagnostics.reporting.dynamics import analyze_points, render_dynamics


def point(step, multi=0.5, single=0.5):
    return {
        "checkpoint_step": step,
        "pool_hash": "fixture_pool",
        "split": "validation",
        "metrics": {
            "diag/grid/multi/pass_k1_t8": multi,
            "diag/grid/step/q_star_mean": single,
            "diag/health/incomplete": 0,
        },
    }


def test_tradeoff_is_descriptive_and_no_causal_claim():
    report = analyze_points([point(0), point(1, 0.75, 0.25)], expected_steps=1)
    assert report["status"] == "complete"
    assert len(report["observed_adjacent_tradeoffs"]) == 1
    assert report["observed_adjacent_tradeoffs"][0]["single_delta"] == -0.25
    assert not report["statistical_significance_established"]
    assert not report["causal_training_effect_established"]
    assert not report["optimizer_updates_verified"]


def test_missing_point_is_not_zero_or_bridged():
    report = analyze_points([point(0), point(2, 0.75, 0.25)], expected_steps=2)
    assert report["status"] == "incomplete"
    assert all(v is None for v in report["series"][1]["values"].values())
    assert not report["observed_adjacent_tradeoffs"]
    html = render_dynamics(report)
    assert "unavailable" in html and "<svg" in html
    assert "M55.00,145.00 M715.00,90.00" in html


def test_incomplete_evaluation_values_are_not_promoted_to_full_estimate():
    bad = point(1, 0.9, 0.1)
    bad["metrics"]["diag/health/incomplete"] = 1
    report = analyze_points([point(0), bad], expected_steps=1)
    assert report["series"][1]["status"] == "unavailable"
    assert not report["observed_adjacent_tradeoffs"]


@pytest.mark.parametrize("mutation", ["duplicate", "pool", "split", "scorer", "nan", "range"])
def test_invalid_or_incomparable_series_rejected(mutation):
    first, second = point(0), copy.deepcopy(point(1))
    if mutation == "duplicate":
        second["checkpoint_step"] = 0
    if mutation == "pool":
        second["pool_hash"] = "other"
    if mutation == "split":
        second["split"] = "test"
    if mutation == "scorer":
        second["metrics"]["diag/coding/step/test_fraction_mean"] = 0.4
    if mutation == "nan":
        second["metrics"]["diag/grid/step/q_star_mean"] = float("nan")
    if mutation == "range":
        second["metrics"]["diag/grid/multi/pass_k1_t8"] = 1.1
    with pytest.raises(SchemaError):
        analyze_points([first, second], expected_steps=1)


@pytest.mark.parametrize("changed_protocol", [False, True])
def test_full_report_audits_raw_synthetic_children_and_rejects_protocol_drift(
    tmp_path, monkeypatch, changed_protocol
):
    from contextlib import nullcontext

    from agent_rl_diagnostics.agent_config import parse_agent_config
    from agent_rl_diagnostics.artifacts.io import file_hash, read_json
    from agent_rl_diagnostics.cli import main
    from agent_rl_diagnostics.evaluators.agent import run_agent_budget
    from agent_rl_diagnostics.integrations.checkpoint import run_checkpoint_validation
    from agent_rl_diagnostics.pools.frozen import build_pool
    from agent_rl_diagnostics.reporting.dynamics import report_dynamics
    from tests.agent_helpers import FixtureModel, agent_config
    from tests.test_checkpoint_integration import fake_checkpoint

    config = agent_config(tmp_path)
    run_agent_budget(config, tmp_path / "base", FixtureModel())
    build_pool(
        [tmp_path / "base"],
        tmp_path / "pool",
        kind="base_monitor",
        size=3,
        seed=7,
        base_model_id="base",
        max_per_task=3,
    )
    checkpoint = fake_checkpoint(tmp_path / "weights")

    def create(spec, sampling):
        model = FixtureModel()
        model.metadata = {"kind": "test_fixture", "revision": spec["revision"]}
        return model

    monkeypatch.setattr("agent_rl_diagnostics.integrations.checkpoint.create_model", create)
    monkeypatch.setattr(
        "agent_rl_diagnostics.integrations.checkpoint.preserve_runtime_rng", lambda _: nullcontext()
    )
    for step in (0, 1):
        raw = copy.deepcopy(config.raw)
        raw["split"] = "smoke"
        raw["model"] = {
            "backend": "transformers",
            "model_id": "fixture",
            "revision": "placeholder",
            "path": str(checkpoint),
        }
        if changed_protocol and step == 1:
            raw["decoding"]["temperature"] = 0.7
        bound = parse_agent_config(raw, base=tmp_path)
        run_checkpoint_validation(
            bound,
            tmp_path / "pool",
            checkpoint,
            tmp_path / f"diagnostics/validation-{step}",
            checkpoint_step=step,
            samples=2,
        )
    if changed_protocol:
        with pytest.raises(SchemaError, match="protocol/config changed"):
            report_dynamics(tmp_path, tmp_path / "report", expected_steps=1)
        assert not (tmp_path / "report").exists()
    else:
        assert (
            main(
                [
                    "dynamics",
                    "--run",
                    str(tmp_path),
                    "--expected-steps",
                    "1",
                    "--output",
                    str(tmp_path / "report"),
                ]
            )
            == 0
        )
        summary = read_json(tmp_path / "report/summary.json")
        assert summary["raw_validation_audited"] and summary["status"] == "complete"
        assert summary["evidence_kind"] == "demo_synthetic"
        manifest = read_json(tmp_path / "report/manifest.json")
        assert manifest["artifacts"]["summary.json"] == file_hash(tmp_path / "report/summary.json")
