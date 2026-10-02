"""CPU contract fixtures for fail-closed smoke auditing, NOT training evidence."""

import json
import runpy
from pathlib import Path

import pytest

from agent_rl_diagnostics.artifacts.io import write_json
from agent_rl_diagnostics.contracts.schema import SchemaError


def test_incomplete_trainer_cannot_be_accepted(tmp_path):
    api = runpy.run_path(str(Path(__file__).parents[1] / "scripts/audit_verl_ray_smoke.py"))
    write_json(tmp_path / "trainer-returned.json", {"steps_requested": 1})
    with pytest.raises(SchemaError, match="two completed"):
        api["audit"](tmp_path, tmp_path, tmp_path / "absent", tmp_path / "absent")


@pytest.mark.parametrize("checkpoints", [[0], [0, 0, 0]])
def test_dropped_or_stale_offline_history_cannot_be_accepted(tmp_path, checkpoints):
    api = runpy.run_path(str(Path(__file__).parents[1] / "scripts/audit_verl_ray_smoke.py"))
    write_json(tmp_path / "trainer-returned.json", {"steps_requested": 2})
    native = tmp_path / "native.jsonl"
    history = tmp_path / "offline.jsonl"
    native.write_text("".join(json.dumps({"step": step}) + "\n" for step in range(3)))
    history.write_text(
        "".join(json.dumps({"checkpoint_step": step}) + "\n" for step in checkpoints)
    )
    with pytest.raises(SchemaError, match="offline W&B"):
        api["audit"](tmp_path, tmp_path, native, history)


@pytest.mark.parametrize("mapping", [[], {"weight_map": []}, {"weight_map": {"x": []}}])
def test_malformed_tensor_index_fails_closed(tmp_path, mapping):
    from agent_rl_diagnostics.integrations.tensor_identity import tensor_identity

    write_json(tmp_path / "model.safetensors.index.json", mapping)
    with pytest.raises(SchemaError):
        tensor_identity(tmp_path)
