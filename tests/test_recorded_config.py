"""Cross-host audit identity must not depend on this host's symlink layout."""

from pathlib import Path

import pytest

from agent_rl_diagnostics.agent_config import parse_agent_config
from agent_rl_diagnostics.artifacts.io import canonical_hash, read_json
from agent_rl_diagnostics.contracts.schema import SchemaError


def raw():
    data = read_json(Path(__file__).parents[1] / "configs/examples/frozenlake_ray_monitor.json")
    data["model"]["path"] = "/recorded/server/model"
    return data


def test_recorded_absolute_path_survives_local_symlink_normalization(monkeypatch):
    data = raw()
    original = Path.resolve

    def changed(path, *args, **kwargs):
        if str(path) == data["model"]["path"]:
            return Path("/different/mount/model")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", changed)
    live = parse_agent_config(data, base=Path.cwd())
    recorded = parse_agent_config(data, base=Path.cwd(), recorded=True)
    assert live.model["path"] == "/different/mount/model"
    assert recorded.model["path"] == data["model"]["path"]
    assert live.config_hash != recorded.config_hash
    assert recorded.config_hash == canonical_hash(recorded.raw)


def test_recorded_relative_model_paths_rejected():
    data = raw()
    data["model"]["path"] = "../weights"
    with pytest.raises(SchemaError, match="absolute"):
        parse_agent_config(data, base=Path.cwd(), recorded=True)
