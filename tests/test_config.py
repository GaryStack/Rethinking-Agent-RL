import json
from importlib.resources import files

import pytest

from agent_rl_diagnostics.config import parse_config
from agent_rl_diagnostics.contracts import SchemaError


def config():
    return json.loads(files("agent_rl_diagnostics").joinpath("resources/demo.json").read_text())


@pytest.mark.parametrize(
    "field,value",
    [
        ("protocol_id", "paper_v1"),
        ("evidence_kind", "raw_recomputable"),
        ("samples_per_task", 7),
        ("seed", True),
        ("budgets", [1, 1]),
        ("budget_mode", "budget_blind_prefix"),
        ("bootstrap_replicates", 0),
    ],
)
def test_invalid_scientific_configuration_rejected(tmp_path, field, value):
    raw = config()
    raw[field] = value
    with pytest.raises(SchemaError):
        parse_config(raw, base_directory=tmp_path)


def test_duplicate_maps_under_different_names_rejected(tmp_path):
    raw = config()
    raw["tasks"].append({"task_id": "duplicate", "desc": raw["tasks"][0]["desc"]})
    with pytest.raises(SchemaError, match="map contents"):
        parse_config(raw, base_directory=tmp_path)


def test_output_path_excluded_from_scientific_identity(tmp_path):
    one = config()
    two = {**one, "output": "elsewhere"}
    assert (
        parse_config(one, base_directory=tmp_path).config_hash
        == parse_config(two, base_directory=tmp_path).config_hash
    )


def test_malformed_policy_is_a_schema_error(tmp_path):
    raw = config()
    raw["models"][0]["policy"] = []
    with pytest.raises(SchemaError, match="policy"):
        parse_config(raw, base_directory=tmp_path)
