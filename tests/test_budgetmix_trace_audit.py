"""Scripted traces only: replay/denominator contracts, not neural model runs."""

import copy

import pytest

from agent_rl_diagnostics.contracts.schema import SchemaError
from agent_rl_diagnostics.environments.text_grid import TextGrid
from agent_rl_diagnostics.integrations.budgetmix_audit import audit_budgetmix_traces
from agent_rl_diagnostics.training.budgetmix import plan_occurrences


def traces():
    records = plan_occurrences(["authored-infeasible"], tmax=1, seed=5422, update=1)
    assignment = records[0]
    task = {"task_id": assignment["task_id"], "desc": ["SFFG"]}
    result = []
    for sample in range(2):
        env = TextGrid(task, prompt_profile="action_only_v2")
        before = env.snapshot()
        outcome = env.apply("RIGHT", 1)
        result.append(
            {
                "schema_version": "ard-verl-grid-rollout-v1",
                "request_id": str(sample),
                "task": task,
                "budget": 1,
                "budgetmix": copy.deepcopy(assignment),
                "weight_version": 0,
                "initial_messages": env.initial_messages(1),
                "initial_prompt_ids": [100],
                "response_ids": [2],
                "response_mask": [1],
                "success": False,
                "terminal_reward": 0.0,
                "termination_reason": "budget_exhausted",
                "steps": [
                    {
                        "turn": 0,
                        "remaining_budget": 1,
                        "weight_version": 0,
                        "prompt_ids": [100],
                        "response_ids": [2],
                        "response": "RIGHT",
                        "snapshot_before": before,
                        "snapshot_after": env.snapshot(),
                        "outcome": outcome,
                    }
                ],
            }
        )
        # Initial messages must precede the action, not describe the resulting state.
        result[-1]["initial_messages"] = TextGrid(
            task, prompt_profile="action_only_v2"
        ).initial_messages(1)
    return result


def test_infeasible_rollouts_remain_completed_zero_reward_groups():
    report = audit_budgetmix_traces(traces(), group_size=2)
    assert report["budget_terminated_fraction"] == report["zero_reward_group_fraction"] == 1
    assert report["completions"] == report["actions"] == 2


@pytest.mark.parametrize("mode", ["missing", "duplicate", "version", "budget", "mask"])
def test_failed_incomplete_or_tampered_groups_never_pass(mode):
    values = traces()
    if mode == "missing":
        values.pop()
    if mode == "duplicate":
        values[1]["request_id"] = values[0]["request_id"]
    if mode == "version":
        values[1]["weight_version"] = 1
        values[1]["steps"][0]["weight_version"] = 1
    if mode == "budget":
        values[0]["budgetmix"]["budget"] = 2
    if mode == "mask":
        values[0]["response_mask"] = [0]
    with pytest.raises(SchemaError):
        audit_budgetmix_traces(values, group_size=2)
