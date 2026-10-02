import copy
import json
from dataclasses import replace

import pytest

from agent_rl_diagnostics.agent_config import parse_agent_config
from agent_rl_diagnostics.artifacts.io import read_json, verify_artifacts
from agent_rl_diagnostics.cli import main
from agent_rl_diagnostics.contracts import SchemaError
from agent_rl_diagnostics.contracts.agent import AgentTrajectory
from agent_rl_diagnostics.environments.deepcoder import DeepCoder, parse_code_action
from agent_rl_diagnostics.environments.text_grid import parse_grid_action
from agent_rl_diagnostics.evaluators.agent import audit_agent_budget, rollout, run_agent_budget
from agent_rl_diagnostics.models import BackendError

from .agent_helpers import CODE, FixtureExecutor, FixtureModel, agent_config


def test_model_rollout_keeps_complete_inputs_and_budget(tmp_path):
    config = agent_config(tmp_path)
    model = FixtureModel()
    trajectory = rollout(config, config.tasks[0], model, run_id="run", budget=5, sample_id=0)
    assert trajectory.success
    assert len(trajectory.steps) == 3
    assert [step.remaining_budget for step in trajectory.steps] == [5, 4, 3]
    assert [len(step.messages) for step in trajectory.steps] == [2, 4, 6]
    assert all(
        "Remaining action budget" in step.messages[-1]["content"] for step in trajectory.steps
    )
    assert AgentTrajectory.from_dict(json.loads(json.dumps(trajectory.to_dict()))) == trajectory
    assert "Q*" not in str(model.calls)


def test_invalid_action_consumes_budget_and_scores_zero(tmp_path):
    config = agent_config(tmp_path)
    trajectory = rollout(
        config, config.tasks[0], FixtureModel("RIGHT or LEFT"), run_id="run", budget=2, sample_id=0
    )
    assert not trajectory.success
    assert trajectory.termination_reason == "budget_exhausted"
    assert len(trajectory.steps) == 2
    assert all(
        step.local_score == 0 and step.action["kind"] == "invalid" for step in trajectory.steps
    )
    assert trajectory.steps[-1].snapshot_after["action_count"] == 2


def test_grid_parser_does_not_extract_arbitrary_reasoning():
    assert parse_grid_action("Let's choose RIGHT.")["kind"] == "invalid"
    assert parse_grid_action("<think>LEFT is bad</think>RIGHT")["direction"] == "RIGHT"


def test_agent_budget_recomputes_and_has_derived_identity(tmp_path):
    config = agent_config(tmp_path)
    output = tmp_path / "budget"
    result = run_agent_budget(config, output, FixtureModel())
    manifest = verify_artifacts(output)
    assert result["status"] == "complete"
    assert manifest["paper_reproduction"] is False
    assert audit_agent_budget(output, manifest)["cells"][0]["pass_at_k"] == 1
    assert main(["audit", "--input", str(output)]) == 0


def test_model_failure_aborts_without_publishing_fake_results(tmp_path):
    class FailingModel(FixtureModel):
        def generate(self, messages, *, seed):
            raise BackendError("fixture failed")

    config = agent_config(tmp_path)
    output = tmp_path / "failed"
    with pytest.raises(BackendError):
        run_agent_budget(config, output, FailingModel())
    assert not output.exists()
    raw = {**config.raw, "failure_policy": "record"}
    config = parse_agent_config(raw, base=tmp_path)
    output = tmp_path / "incomplete"
    result = run_agent_budget(config, output, FailingModel())
    assert result["status"] == "incomplete"
    summary = read_json(output / "summary.json")
    assert summary["cells"][0]["pass_at_k"] is None
    assert summary["cells"][0]["infrastructure_failures"] == 2
    assert main(["audit", "--input", str(output)]) == 0


def test_hole_is_task_failure_not_infrastructure_failure(tmp_path):
    raw = agent_config(tmp_path).raw
    raw["tasks"] = [{"task_id": "hole", "desc": ["SHG"]}]
    config = parse_agent_config(raw, base=tmp_path)
    trajectory = rollout(
        config, config.tasks[0], FixtureModel(), run_id="run", budget=5, sample_id=0
    )
    assert trajectory.status == "completed" and trajectory.success is False
    assert trajectory.termination_reason == "environment_terminal"


def test_coding_run_scored_but_does_not_submit_or_leak_hidden_tests(tmp_path):
    config = agent_config(tmp_path, coding=True)
    env = DeepCoder(config.tasks[0], FixtureExecutor())
    assert "17 29" not in str(env.initial_messages(3))
    result = env.apply("RUN\n```python\n" + CODE + "\n```", 3)
    assert result["local_score"] == 1
    assert result["scorer"] == "derived_stdin_test_fraction_v1"
    assert result["reward"] == 0 and not result["terminated"]
    assert "46" not in result["observation"]
    snapshot = env.snapshot()
    submitted = env.apply("SUBMIT\n```python\n" + CODE + "\n```", 2)
    assert submitted["success"] and submitted["terminated"]
    fresh = DeepCoder(config.tasks[0], FixtureExecutor())
    fresh.restore(snapshot)
    assert fresh.code == CODE and fresh.action_count == 1 and not fresh.terminal


def test_failed_submit_is_terminal(tmp_path):
    env = DeepCoder(agent_config(tmp_path, coding=True).tasks[0], FixtureExecutor())
    outcome = env.apply("SUBMIT\n```python\nprint(0)\n```", 2)
    assert outcome["terminated"] and not outcome["success"]
    with pytest.raises(SchemaError, match="terminal"):
        env.apply("RIGHT", 1)


def test_bare_fenced_code_is_explicit_implicit_submission():
    assert parse_code_action("```python\nprint(1)\n```")["kind"] == "submit"
    assert parse_code_action("RUN\n```python\nprint(1)\n```")["kind"] == "run"
    assert parse_code_action("print(1)")["kind"] == "invalid"


def test_snapshot_chain_cannot_be_changed(tmp_path):
    config = agent_config(tmp_path)
    trajectory = rollout(
        config, config.tasks[0], FixtureModel(), run_id="run", budget=5, sample_id=0
    )
    bad = copy.deepcopy(trajectory.steps[1].snapshot_before)
    bad["state"] = 0
    with pytest.raises(SchemaError, match="snapshot chain"):
        replace(
            trajectory,
            steps=(
                trajectory.steps[0],
                replace(trajectory.steps[1], snapshot_before=bad),
                trajectory.steps[2],
            ),
        )
