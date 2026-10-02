import copy

import pytest

from agent_rl_diagnostics.contracts.schema import SchemaError
from agent_rl_diagnostics.environments.text_grid import TextGrid
from agent_rl_diagnostics.integrations.grid_trace import audit_grid_trace


def fixture():
    task = {"task_id": "authored-mask-fixture", "desc": ["SFG"]}
    env = TextGrid(task, prompt_profile="action_only_v2")
    trace = {
        "schema_version": "ard-verl-grid-rollout-v1",
        "task": task,
        "budget": 2,
        "weight_version": 7,
        "initial_messages": env.initial_messages(2),
        "initial_prompt_ids": [100],
        "steps": [],
        "response_ids": [2, 99, 2],
        "response_mask": [1, 0, 1],
        "success": True,
        "terminal_reward": 1.0,
        "termination_reason": "success",
    }
    for turn, prompt in enumerate(([100], [100, 2, 99])):
        before = env.snapshot()
        outcome = env.apply("RIGHT", 2 - turn)
        step = {
            "turn": turn,
            "remaining_budget": 2 - turn,
            "weight_version": 7,
            "prompt_ids": prompt,
            "snapshot_before": before,
            "snapshot_after": env.snapshot(),
            "response": "RIGHT",
            "response_ids": [2],
            "outcome": outcome,
        }
        if turn == 0:
            step.update(
                feedback_ids=[99],
                feedback_messages=[
                    {
                        "role": "user",
                        "content": env.budget_observation(2, 1, outcome["observation"]),
                    }
                ],
            )
        trace["steps"].append(step)
    return trace


def test_training_trace_replays_budget_and_environment_loss_mask():
    assert audit_grid_trace(fixture()) == {
        "actions": 2,
        "generated_tokens": 2,
        "environment_tokens": 1,
        "terminal_reward": 1.0,
        "weight_version": 7,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("response_mask", [1, 1, 1]),
        ("terminal_reward", 0.0),
        ("weight_version", 8),
        ("termination_reason", "budget_exhausted"),
    ],
)
def test_training_trace_rejects_rehashed_mask_reward_version_tampering(field, value):
    trace = copy.deepcopy(fixture())
    trace[field] = value
    with pytest.raises(SchemaError):
        audit_grid_trace(trace)


def test_process_signal_cannot_be_applied_to_validation():
    from agent_rl_diagnostics.training.process_signals import grid_process_reward

    trace = fixture()
    trace["process_signal"] = grid_process_reward(trace, coefficient=0.5, failure_policy="error")
    trace["ard_phase"] = "validation"
    with pytest.raises(SchemaError, match="validation"):
        audit_grid_trace(trace)
