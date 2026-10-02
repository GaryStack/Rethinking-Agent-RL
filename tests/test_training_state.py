"""Fingerprint proof contracts only; these authored hashes are not training evidence."""

import copy

import pytest

from agent_rl_diagnostics.artifacts.io import write_json
from agent_rl_diagnostics.contracts.schema import SchemaError
from agent_rl_diagnostics.integrations.training_state import STATE_KEYS, compare_training_states


def proof():
    before = {key: "a" * 64 for key in STATE_KEYS}
    after = dict(before, parameters="b" * 64, optimizer="c" * 64)
    return {
        "schema_version": "ard-training-state-v1",
        "update": 1,
        "scope": "single_rank_actor_worker_boundary",
        "framework_versions": {"torch": "fixture", "numpy": "fixture"},
        "before": before,
        "after": after,
        "actor_grad_norms": [1.0],
    }


def pair(tmp_path, first, second):
    for name, value in (("left", first), ("right", second)):
        directory = tmp_path / name
        directory.mkdir()
        write_json(directory / "update-000001.json", value)
    return tmp_path / "left", tmp_path / "right"


def test_matching_observed_states_does_not_claim_complete_parity(tmp_path):
    report = compare_training_states(*pair(tmp_path, proof(), proof()))
    assert report["status"] == "matched_observed_actor_states" and report["cuda_rng_observed"]
    assert not report["full_trainer_parity_verified"] and not report["distributed_parity_verified"]


@pytest.mark.parametrize(
    "field",
    ["parameters", "optimizer", "scheduler", "torch_cpu_rng", "torch_cuda_rng", "training_modes"],
)
def test_hidden_actor_state_difference_reported(tmp_path, field):
    left, right = proof(), proof()
    right["after"][field] = "d" * 64
    report = compare_training_states(*pair(tmp_path, left, right))
    assert report["status"] == "different"
    assert f"after/{field}" in report["differences"][0]["fields"]


@pytest.mark.parametrize(
    "mutation",
    ["missing_cuda", "unchanged", "mutated_batch", "zero_grad", "missing_field", "noncontiguous"],
)
def test_incomplete_or_invalid_proof_cannot_pass(tmp_path, mutation):
    left = proof()
    right = copy.deepcopy(left)
    if mutation == "missing_cuda":
        right["before"]["torch_cuda_rng"] = None
    if mutation == "unchanged":
        right["after"]["parameters"] = right["before"]["parameters"]
    if mutation == "mutated_batch":
        right["after"]["batch_tensors"] = "e" * 64
    if mutation == "zero_grad":
        right["actor_grad_norms"] = [0.0]
    if mutation == "missing_field":
        del right["before"]["optimizer"]
    if mutation == "noncontiguous":
        right["update"] = 2
    with pytest.raises(SchemaError):
        compare_training_states(*pair(tmp_path, left, right))
