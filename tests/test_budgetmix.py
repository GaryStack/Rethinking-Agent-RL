import copy
import json
import random

import pytest

from agent_rl_diagnostics.artifacts.io import file_hash, read_json, write_json
from agent_rl_diagnostics.cli import main
from agent_rl_diagnostics.contracts.schema import SchemaError
from agent_rl_diagnostics.training.budgetmix import (
    audit_groups,
    budget_candidates,
    gigpo_anchor,
    normalize_step_rewards,
    plan_occurrences,
    terminal_bootstrap,
)


@pytest.mark.parametrize(
    "tmax,expected",
    [
        (1, (1,)),
        (3, (1, 2, 3)),
        (10, tuple(range(1, 11))),
        (24, (3, 5, 8, 10, 12, 15, 17, 20, 22, 24)),
        (70, tuple(range(7, 71, 7))),
    ],
)
def test_unique_ceil_deciles(tmax, expected):
    assert budget_candidates(tmax) == expected


@pytest.mark.parametrize("value", [0, -1, True, 2.5, "10"])
def test_invalid_max_budget_rejected(value):
    with pytest.raises(SchemaError):
        budget_candidates(value)


def test_huge_budget_uses_exact_integer_rounding():
    n = 10**30 + 1
    assert budget_candidates(n)[0] == (n + 9) // 10
    assert budget_candidates(n)[-1] == n


def test_occurrence_separation_repeatability_and_global_rng_preserved():
    before = random.getstate()
    records = plan_occurrences(["same", "same", "other"], tmax=3, seed=42, update=1)
    assert records == plan_occurrences(["same", "same", "other"], tmax=3, seed=42, update=1)
    assert random.getstate() == before
    assert len({r["task_occurrence_id"] for r in records}) == 3
    assert records != plan_occurrences(["same", "same", "other"], tmax=3, seed=42, update=2)
    assert audit_groups([r for r in records for _ in range(4)], group_size=4)["groups"] == 3


def test_small_tmax_uniform_over_unique_candidates_not_weighted_deciles():
    records = plan_occurrences(["same"] * 6000, tmax=3, seed=5422, update=1)
    counts = [sum(r["budget"] == t for r in records) for t in (1, 2, 3)]
    assert all(1800 < n < 2200 for n in counts)


@pytest.mark.parametrize(
    "mutation", ["budget", "identity", "missing", "duplicate", "row", "mixed_update"]
)
def test_group_audit_rejects_assignment_tampering(mutation):
    original = plan_occurrences(["a", "a"], tmax=3, seed=42, update=1)
    records = [copy.deepcopy(r) for r in original for _ in range(2)]
    if mutation == "budget":
        records[0]["budget"] = 4
    if mutation == "identity":
        records[0]["task_occurrence_id"] = "wrong"
    if mutation == "missing":
        records.pop()
    if mutation == "duplicate":
        records.append(copy.deepcopy(records[0]))
    if mutation == "row":
        records[0]["row"] = 9
    if mutation == "mixed_update":
        records = original[:1] * 2 + plan_occurrences(["a", "a"], tmax=3, seed=42, update=2)[1:] * 2
    with pytest.raises(SchemaError):
        audit_groups(records, group_size=2)


@pytest.mark.parametrize("reason", ["success", "environment_terminal", "budget_exhausted"])
def test_ppo_terminal_bootstrap_zero(reason):
    assert terminal_bootstrap(12.5, reason) == 0


def test_bootstrap_preserves_nonterminal_and_rejects_infrastructure():
    assert terminal_bootstrap(12.5, "continuing") == 12.5
    with pytest.raises(SchemaError):
        terminal_bootstrap(12.5, "infrastructure_failure")
    with pytest.raises(SchemaError):
        terminal_bootstrap(float("nan"), "budget_exhausted")


def test_gigpo_anchor_contains_occurrence_state_and_remaining_budget():
    anchor = gigpo_anchor("occurrence", {"position": 1}, 2)
    assert anchor != gigpo_anchor("other", {"position": 1}, 2)
    assert anchor != gigpo_anchor("occurrence", {"position": 2}, 2)
    assert anchor != gigpo_anchor("occurrence", {"position": 1}, 1)


def test_degenerate_step_groups_zero_without_touching_episode_signal():
    assert normalize_step_rewards([3]) == [0]
    assert normalize_step_rewards([3, 3]) == [0, 0]
    assert normalize_step_rewards([1, 3]) == [-1, 1]
    with pytest.raises(SchemaError):
        normalize_step_rewards([float("inf")])


def test_cli_plan_audit_immutable_and_recomputes_rehashed_tampering(tmp_path, capsys):
    ids, output = tmp_path / "ids.json", tmp_path / "plan"
    write_json(ids, ["same", "same", "other"])
    command = ["budgetmix", "plan", "--task-ids", str(ids), "--tmax", "24", "--output", str(output)]
    assert main(command) == 0
    assert json.loads(capsys.readouterr().out)["rollouts_executed"] is False
    assert main(["budgetmix", "audit", "--input", str(output)]) == 0
    assert main(command) == 2
    plan = read_json(output / "plan.json")
    plan["assignments"][0]["budget"] = 4
    write_json(output / "plan.json", plan)
    manifest = read_json(output / "manifest.json")
    manifest["artifacts"]["plan.json"] = file_hash(output / "plan.json")
    write_json(output / "manifest.json", manifest)
    assert main(["budgetmix", "audit", "--input", str(output)]) == 2
