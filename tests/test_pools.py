import json

import pytest

from agent_rl_diagnostics.artifacts.io import canonical_hash, file_hash, write_json
from agent_rl_diagnostics.cli import main
from agent_rl_diagnostics.contracts import SchemaError
from agent_rl_diagnostics.evaluators.agent import run_agent_budget
from agent_rl_diagnostics.evaluators.fixed import run_fixed, summarize_fixed
from agent_rl_diagnostics.evaluators.recovery import run_recovery
from agent_rl_diagnostics.pools.errors import build_error_pool
from agent_rl_diagnostics.pools.frozen import build_pool, load_pool

from .agent_helpers import FixtureModel, agent_config


def pool_fixture(tmp_path):
    config = agent_config(tmp_path)
    source, pool = tmp_path / "source", tmp_path / "pool"
    run_agent_budget(config, source, FixtureModel())
    build_pool(
        [source], pool, kind="base_monitor", size=3, seed=7, base_model_id="base", max_per_task=3
    )
    return config, source, pool


def test_pool_reward_blind_dedup_and_validation(tmp_path):
    _, _, pool = pool_fixture(tmp_path)
    manifest, states, _ = load_pool(pool)
    assert len(states) == 3
    assert manifest["selection"]["candidate_count"] == 6
    assert manifest["selection"]["reward_blind"] is True
    assert manifest["selection"]["heldout_task_id_check"] == "not_verified"
    assert all("local_score" not in str(state) for state in states)
    assert main(["pool", "audit", "--input", str(pool)]) == 0


def test_pool_does_not_duplicate_states_to_fill_target(tmp_path):
    _, source, _ = pool_fixture(tmp_path)
    with pytest.raises(SchemaError, match="insufficient distinct"):
        build_pool(
            [source],
            tmp_path / "oversize",
            kind="base_monitor",
            size=4,
            seed=7,
            base_model_id="base",
            max_per_task=4,
        )


def test_pool_task_split_check_rejects_training_overlap(tmp_path):
    _, source, _ = pool_fixture(tmp_path)
    with pytest.raises(SchemaError, match="overlap"):
        build_pool(
            [source],
            tmp_path / "leak",
            kind="base_monitor",
            size=1,
            seed=7,
            base_model_id="base",
            training_task_ids={"corridor"},
        )


def test_matched_pool_equal_contributions_and_no_duplicate_inputs(tmp_path):
    _, source, _ = pool_fixture(tmp_path)
    rl_config = agent_config(tmp_path, model_id="rl")
    rl_source = tmp_path / "rl-source"
    run_agent_budget(rl_config, rl_source, FixtureModel())
    destination = tmp_path / "matched"
    result = build_pool(
        [source, rl_source],
        destination,
        kind="matched",
        size=2,
        seed=7,
        base_model_id="base",
        rl_model_id="rl",
    )
    assert result["role_counts"] == {"base": 1, "rl": 1}
    assert len({s["state_id"] for s in load_pool(destination)[1]}) == 2


def test_fixed_action_starts_from_same_snapshot_each_sample(tmp_path):
    config, _, pool = pool_fixture(tmp_path)
    output = tmp_path / "step"
    result = run_fixed(config, pool, output, samples=2, model=FixtureModel())
    assert result["actions"] == 6
    assert result["r_single"] == pytest.approx((0.95**2 + 0.95 + 1) / 3)
    assert main(["audit", "--input", str(output), "--pool", str(pool)]) == 0
    assert main(["audit", "--input", str(output)]) == 2


def test_fixed_missing_slots_do_not_become_zero(tmp_path):
    config, _, pool = pool_fixture(tmp_path)
    output = tmp_path / "step"
    run_fixed(config, pool, output, samples=2, model=FixtureModel())
    records = [json.loads(line) for line in (output / "actions.jsonl").read_text().splitlines()]
    states = load_pool(pool)[1]
    summary = summarize_fixed(
        records[:-1], states, samples=2, scorer="frozenlake_q_star_gamma_0.95_v1", model_id="base"
    )
    assert summary["r_single"] is None and not summary["complete"]


def test_distance_errors_use_same_cost_and_recovery_budget(tmp_path):
    config, _, pool = pool_fixture(tmp_path)
    errors = tmp_path / "errors"
    result = build_error_pool(pool, errors, seed=7, require_recoverable=True)
    assert result["pairs"] == 1
    manifest, states, _ = load_pool(errors)
    assert {s["remaining_budget"] for s in states} == {3}
    assert {s["prefix_depth"] for s in states} == {2}
    output = tmp_path / "recovery"
    result = run_recovery(config, errors, output, samples=2, model=FixtureModel())
    assert result["clean_success"] == 1 and result["error_success"] == 1
    assert main(["audit", "--input", str(output), "--pool", str(errors)]) == 0


def test_changed_pool_input_detected_even_if_hashes_are_updated(tmp_path):
    _, _, pool = pool_fixture(tmp_path)
    manifest, states, tasks = load_pool(pool)
    states[0]["snapshot"]["state"] = 0 if states[0]["snapshot"]["state"] != 0 else 1
    from agent_rl_diagnostics.pools.frozen import assign_state_id

    assign_state_id(states[0])
    path = pool / "states.jsonl"
    path.write_text("\n".join(json.dumps(s) for s in states) + "\n")
    manifest["artifacts"]["states.jsonl"] = file_hash(path)
    manifest["pool_hash"] = canonical_hash(
        {
            "environment": manifest["environment"],
            "states": states,
            "tasks": list(tasks.values()),
            "selection": manifest["selection"],
            "pairs": None,
        }
    )
    write_json(pool / "manifest.json", manifest)
    with pytest.raises(SchemaError, match="replayed prefix"):
        load_pool(pool)


def test_coding_pool_independent_errors_and_recovery(tmp_path, monkeypatch):
    from agent_rl_diagnostics.agent_config import AgentConfig

    from .agent_helpers import CODE, FixtureExecutor

    monkeypatch.setattr(AgentConfig, "executor", lambda self: FixtureExecutor())
    config = agent_config(tmp_path, coding=True)
    source, pool, errors, recovery = [
        tmp_path / name
        for name in ("coding-source", "coding-pool", "coding-errors", "coding-recovery")
    ]
    run_agent_budget(config, source, FixtureModel("RUN\n```python\n" + CODE + "\n```"))
    build_pool(
        [source], pool, kind="base_monitor", size=5, seed=7, base_model_id="base", max_per_task=5
    )
    references = tmp_path / "references.json"
    write_json(
        references,
        {"sum": {"code": CODE, "source": "independent_reference", "revision": "fixture-v1"}},
    )
    result = build_error_pool(pool, errors, references=references, executor=FixtureExecutor())
    assert result["pairs"] == 4
    result = run_recovery(
        config, errors, recovery, model=FixtureModel("SUBMIT\n```python\n" + CODE + "\n```")
    )
    assert result["error_success"] == result["clean_success"] == 1
    assert main(["audit", "--input", str(recovery), "--pool", str(errors), "--replay-code"]) == 0


def test_early_v1_recovery_summary_without_additive_ci_fields(tmp_path):
    config, _, pool = pool_fixture(tmp_path)
    errors, output = tmp_path / "errors", tmp_path / "recovery"
    build_error_pool(pool, errors, seed=7, require_recoverable=True)
    run_recovery(config, errors, output, model=FixtureModel())
    summary = json.loads((output / "summary.json").read_text())
    summary.pop("paired_delta_ci")
    for row in summary["per_pair"]:
        row.pop("task_id")
    write_json(output / "summary.json", summary)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["artifacts"]["summary.json"] = file_hash(output / "summary.json")
    write_json(output / "manifest.json", manifest)
    assert main(["audit", "--input", str(output), "--pool", str(errors)]) == 0
