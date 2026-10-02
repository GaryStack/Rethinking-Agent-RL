import importlib.util
from pathlib import Path

import pytest

from agent_rl_diagnostics.contracts.schema import SchemaError
from agent_rl_diagnostics.training.process_signals import CachedJudge, mix_rewards, oracle_score


def test_mixing_and_failure_policy():
    assert mix_rewards(1, [0.2, 0.8], coefficient=0.3, failure_policy="error")["reward"] == 1.15
    assert mix_rewards(1, [None], coefficient=1, failure_policy="terminal_only")["reward"] == 1
    with pytest.raises(SchemaError):
        mix_rewards(1, [None], coefficient=1, failure_policy="error")
    with pytest.raises(SchemaError):
        mix_rewards(1, [2], coefficient=1, failure_policy="terminal_only")


def test_judge_cache_identity_and_failure():
    calls = []
    cache = {}
    judge = CachedJudge(lambda step: calls.append(step) or 0.5, identity="v1", cache=cache)
    assert judge.score({"response": "A"}) == judge.score({"response": "A"}) == 0.5
    assert len(calls) == 1
    CachedJudge(lambda step: 1, identity="v2", cache=cache).score({"response": "A"})
    assert len(cache) == 2
    with pytest.raises(SchemaError):
        oracle_score({"local_score": 1, "scorer": "other"}, scorer="expected")


def test_process_signal_changes_training_update():
    path = Path(__file__).parents[1] / "examples/process_signal_training.py"
    spec = importlib.util.spec_from_file_location("process_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    off, on = module.train(0), module.train(0.5)
    assert off["final_logit"] == 0
    assert on["final_logit"] > 0
    assert any(x["gradient"] > 0 for x in on["history"])


def test_grid_process_reward_consumes_audited_oracle():
    from agent_rl_diagnostics.training.process_signals import grid_process_reward
    from tests.test_grid_trace import fixture

    trace = fixture()
    result = grid_process_reward(trace, coefficient=0.5, failure_policy="error")
    assert result["reward"] > trace["terminal_reward"]
    assert trace["terminal_reward"] == 1
    trace["response_mask"] = [1, 1, 1]
    with pytest.raises(SchemaError):
        grid_process_reward(trace, coefficient=0.5, failure_policy="error")


def test_process_trace_detects_reward_tampering():
    from agent_rl_diagnostics.integrations.grid_trace import audit_grid_trace
    from agent_rl_diagnostics.training.process_signals import grid_process_reward
    from tests.test_grid_trace import fixture

    trace = fixture()
    trace["process_signal"] = grid_process_reward(trace, coefficient=0.5, failure_policy="error")
    assert audit_grid_trace(trace)["terminal_reward"] == 1.0
    trace["process_signal"]["reward"] = 10
    with pytest.raises(SchemaError):
        audit_grid_trace(trace)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 2, -0.1])
def test_invalid_scores_cannot_hide_behind_missing_fallback(bad):
    with pytest.raises(SchemaError):
        mix_rewards(0, [None, bad], coefficient=1, failure_policy="terminal_only")


def test_process_phase_and_explicit_parameters():
    from agent_rl_diagnostics.training.process_signals import (
        process_signal_settings,
        process_training_phase,
    )

    assert process_training_phase({"ard_phase": "validation"}) == "validation"
    for info in ({}, {"ard_phase": "test"}):
        with pytest.raises(SchemaError):
            process_training_phase(info)
    assert process_signal_settings(0, "error")["coefficient"] == 0
    for coefficient in (-1, float("nan"), True):
        with pytest.raises(SchemaError):
            process_signal_settings(coefficient, "error")
