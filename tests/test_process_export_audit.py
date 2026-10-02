import copy

import pytest

from agent_rl_diagnostics.contracts.schema import SchemaError
from agent_rl_diagnostics.integrations.process_audit import audit_process_exports
from agent_rl_diagnostics.training.process_signals import grid_process_reward
from tests.test_grid_trace import fixture


def inputs():
    traces, exports = [], []
    for phase, version in (("train", 0), ("validation", 0), ("validation", 1)):
        trace = fixture()
        trace["ard_phase"] = phase
        trace["weight_version"] = version
        for step in trace["steps"]:
            step["weight_version"] = version
        reward = trace["terminal_reward"]
        if phase == "train":
            trace["process_signal"] = grid_process_reward(
                trace, coefficient=0.5, failure_policy="error"
            )
            reward = trace["process_signal"]["reward"]
        checkpoint_step = version + (phase == "train")
        traces.append(trace)
        exports.append(
            (
                phase,
                checkpoint_step,
                {
                    "step": checkpoint_step,
                    "gts": trace["task"]["task_id"],
                    "input": str(trace["initial_prompt_ids"]),
                    "output": str(trace["response_ids"]),
                    "score": reward,
                    "composed_reward": reward,
                    "grid_success": trace["terminal_reward"],
                    "process_component": reward - trace["terminal_reward"],
                    "grid_actions": 2,
                    "action_budget": 2,
                },
            )
        )
    return traces, exports


def audit(traces, exports):
    return audit_process_exports(
        traces,
        exports,
        settings={"enabled": True, "coefficient": 0.5, "failure_policy": "error"},
        steps=1,
        group_size=1,
        train_task_ids={"authored-mask-fixture"},
        validation_task_ids={"authored-mask-fixture"},
        decode=str,
    )


def test_real_export_join_preserves_reward_and_validation_units():
    result = audit(*inputs())
    assert result["training_rows"] == result["nonzero_process_training_rows"] == 1
    assert result["validation_rows"] == 2
    assert result["validation_terminal_only"]


@pytest.mark.parametrize(
    "mutation",
    [
        "native-terminal",
        "validation-process",
        "missing",
        "duplicate",
        "settings",
        "nan",
        "identity",
    ],
)
def test_export_audit_rejects_rewards_and_identity_tampering(mutation):
    traces, exports = copy.deepcopy(inputs())
    if mutation == "native-terminal":
        exports[0][2]["score"] = exports[0][2]["composed_reward"] = 1.0
    elif mutation == "validation-process":
        exports[1][2]["score"] = exports[1][2]["composed_reward"] = 1.5
        exports[1][2]["process_component"] = 0.5
    elif mutation == "missing":
        traces.pop()
        exports.pop()
    elif mutation == "duplicate":
        traces.append(copy.deepcopy(traces[0]))
        exports.append(copy.deepcopy(exports[0]))
    elif mutation == "settings":
        traces[0]["process_signal"] = grid_process_reward(
            {k: v for k, v in traces[0].items() if k != "process_signal"},
            coefficient=0.3,
            failure_policy="error",
        )
    elif mutation == "nan":
        exports[0][2]["score"] = float("nan")
    else:
        exports[0][2]["output"] = "different tokens"
    with pytest.raises(SchemaError):
        audit(traces, exports)
