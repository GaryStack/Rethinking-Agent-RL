"""Independently audited authored fixtures, not model/training evidence."""

import copy
from contextlib import nullcontext

import pytest

from agent_rl_diagnostics.agent_config import parse_agent_config
from agent_rl_diagnostics.contracts.schema import SchemaError
from agent_rl_diagnostics.evaluators.agent import run_agent_budget
from agent_rl_diagnostics.integrations.checkpoint import run_checkpoint_validation
from agent_rl_diagnostics.integrations.comparison import compare_checkpoint_validations
from agent_rl_diagnostics.pools.frozen import build_pool

from .agent_helpers import FixtureModel, agent_config
from .test_checkpoint_integration import fake_checkpoint


@pytest.fixture
def validations(tmp_path, monkeypatch):
    config = agent_config(tmp_path)
    source, pool = tmp_path / "source", tmp_path / "pool"
    run_agent_budget(config, source, FixtureModel())
    build_pool([source], pool, kind="base_monitor", size=3, seed=7, base_model_id="base")
    root = fake_checkpoint(tmp_path / "checkpoint")
    raw = copy.deepcopy(config.raw)
    raw["split"] = "smoke"
    raw["model"] = {
        "backend": "transformers",
        "model_id": "fixture",
        "revision": "placeholder",
        "path": str(root),
    }
    config = parse_agent_config(raw, base=tmp_path)

    def create(spec, sampling):
        model = FixtureModel()
        model.metadata = {"kind": "test_fixture", "revision": spec["revision"]}
        return model

    monkeypatch.setattr("agent_rl_diagnostics.integrations.checkpoint.create_model", create)
    monkeypatch.setattr(
        "agent_rl_diagnostics.integrations.checkpoint.preserve_runtime_rng",
        lambda device: nullcontext(),
    )

    def produce(name, step=1):
        directory = tmp_path / name
        run_checkpoint_validation(config, pool, root, directory, checkpoint_step=step, samples=2)
        return directory

    return pool, produce


def test_independent_validations_match(validations):
    pool, produce = validations
    report = compare_checkpoint_validations(produce("left"), produce("right"), pool)
    assert report["status"] == "matched" and all(report["raw_records_equal"].values())
    assert not report["training_state_parity_verified"] and not report["weights_rehashed"]


def test_checkpoint_step_difference_reported_even_for_same_weights(validations):
    pool, produce = validations
    report = compare_checkpoint_validations(produce("left"), produce("right", step=2), pool)
    assert report["status"] == "different"
    assert report["point_mismatches"]["checkpoint_step"] == {"left": 1, "right": 2}


def test_corrupt_raw_artifact_cannot_be_compared(validations):
    pool, produce = validations
    left, right = produce("left"), produce("right")
    with (right / "step/actions.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises((SchemaError, ValueError)):
        compare_checkpoint_validations(left, right, pool)


def test_comparing_directory_with_itself_is_rejected(tmp_path):
    with pytest.raises(SchemaError, match="distinct"):
        compare_checkpoint_validations(tmp_path, tmp_path, tmp_path)
