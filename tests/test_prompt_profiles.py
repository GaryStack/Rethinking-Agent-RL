import copy

import pytest

from agent_rl_diagnostics.agent_config import parse_agent_config
from agent_rl_diagnostics.contracts.schema import SchemaError
from agent_rl_diagnostics.environments.deepcoder import DeepCoder
from agent_rl_diagnostics.environments.text_grid import TextGrid, parse_grid_action

from .agent_helpers import FixtureExecutor, agent_config


def test_grid_profiles_are_versioned_and_do_not_repair_outputs():
    task = {"task_id": "grid", "desc": ["SFFG"]}
    old, new = TextGrid(task), TextGrid(task, prompt_profile="action_only_v2")
    assert "prompt_profile" not in old.snapshot()
    assert "ACTUAL current position" in new.observation()
    with pytest.raises(SchemaError, match="profile"):
        old.restore(new.snapshot())
    assert parse_grid_action("**RIGHT** because it is optimal")["kind"] == "invalid"
    assert parse_grid_action("RIGHT")["kind"] == "move"


def test_coding_last_turn_requires_submission_without_changing_parser(tmp_path):
    config = agent_config(tmp_path, coding=True)
    old = DeepCoder(config.tasks[0], FixtureExecutor())
    new = DeepCoder(config.tasks[0], FixtureExecutor(), prompt_profile="action_only_v2")
    assert "last action" in new.initial_messages(1)[1]["content"]
    assert "last action" not in old.initial_messages(1)[1]["content"]
    with pytest.raises(SchemaError, match="profile"):
        new.restore(old.snapshot())


def test_prompt_profile_is_part_of_scientific_config_identity(tmp_path):
    old = agent_config(tmp_path)
    raw = copy.deepcopy(old.raw)
    raw["grid_prompt_profile"] = "action_only_v2"
    new = parse_agent_config(raw, base=tmp_path)
    assert new.config_hash != old.config_hash
    raw["coding_prompt_profile"] = "action_only_v2"
    with pytest.raises(SchemaError, match="coding only"):
        parse_agent_config(raw, base=tmp_path)
