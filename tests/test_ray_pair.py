"""Fail-closed comparison contracts only, not actual trainer acceptance."""

import pytest

from agent_rl_diagnostics.artifacts.io import read_json, write_json
from agent_rl_diagnostics.contracts.schema import SchemaError
from agent_rl_diagnostics.environments.text_grid import TextGrid
from agent_rl_diagnostics.integrations.grid_fixture import TRAIN_GRID_TASKS
from agent_rl_diagnostics.integrations.ray_pair import compare_ray_pair
from tests.test_training_state import proof


def records(tmp_path, *, left_mode=False, right_mode=True):
    paths = [tmp_path / "left", tmp_path / "right"]
    for root, mode in zip(paths, (left_mode, right_mode), strict=True):
        root.mkdir()
        write_json(
            root / "trainer-returned.json",
            {
                "checkpoint_diagnostics_enabled": mode,
                "actor_state_capture_enabled": True,
                "scope": "actual_single_rank_RayPPOTrainer_vllm_FSDP",
                "steps_requested": 2,
                "samples_per_training_task": 4,
                "train_action_budget": 2,
                "algorithm": "reinforce_plus_plus",
                "budgetmix_enabled": False,
            },
        )
    return paths


def test_self_comparison_rejected(tmp_path):
    with pytest.raises(SchemaError, match="distinct"):
        compare_ray_pair(tmp_path, tmp_path)


@pytest.mark.parametrize("left,right", [(True, True), (False, False), (0, 1)])
def test_same_or_implicit_modes_cannot_claim_off_on(tmp_path, left, right):
    with pytest.raises(SchemaError):
        compare_ray_pair(*records(tmp_path, left_mode=left, right_mode=right))


def test_resolved_scientific_config_difference_rejected_before_state_comparison(tmp_path):
    first, second = records(tmp_path)
    for root, temperature in zip((first, second), (1.0, 0.6), strict=True):
        write_json(
            root / "verl-config.json",
            {
                "trainer": {
                    "experiment_name": root.name,
                    "default_local_dir": str(root / "checkpoints"),
                },
                "actor_rollout_ref": {"rollout": {"temperature": temperature}},
            },
        )
    with pytest.raises(SchemaError, match="resolved configurations"):
        compare_ray_pair(first, second)


def fixture_complete_pair(tmp_path, monkeypatch):
    from agent_rl_diagnostics.integrations import ray_pair

    # Explicit pool fixture; never used as real replay/model evidence.
    monkeypatch.setattr(ray_pair, "load_pool", lambda _: ({"fixture": True}, [], {}))
    roots = records(tmp_path)
    for root in roots:
        write_json(
            root / "verl-config.json",
            {
                "trainer": {
                    "experiment_name": root.name,
                    "default_local_dir": str(root / "checkpoints"),
                },
                "actor_rollout_ref": {"rollout": {"trace": {"experiment_name": root.name}}},
            },
        )
        for name in ("source-lock.json", "environment.json", "monitor-config.json"):
            write_json(root / name, {"fixture": True})
        (root / "state-proofs").mkdir()
        destination = root / "rollouts/grid-traces"
        destination.mkdir(parents=True)
        for version in (0, 1):
            state = proof()
            state["update"] = version + 1
            write_json(root / f"state-proofs/update-{version + 1:06d}.json", state)
            for task, direction in zip(
                TRAIN_GRID_TASKS, ("RIGHT", "LEFT", "DOWN", "UP"), strict=True
            ):
                for sample in range(4):
                    env = TextGrid(task, prompt_profile="action_only_v2")
                    before = env.snapshot()
                    initial = env.initial_messages(2)
                    outcome = env.apply(direction, 2)
                    name = f"{version}-{task['task_id']}-{sample}"
                    trace = {
                        "schema_version": "ard-verl-grid-rollout-v1",
                        "task": task,
                        "budget": 2,
                        "weight_version": version,
                        "request_id": root.name + name,
                        "initial_messages": initial,
                        "initial_prompt_ids": [100],
                        "response_ids": [2],
                        "response_mask": [1],
                        "success": True,
                        "terminal_reward": 1.0,
                        "termination_reason": "success",
                        "steps": [
                            {
                                "turn": 0,
                                "remaining_budget": 2,
                                "weight_version": version,
                                "prompt_ids": [100],
                                "response_ids": [2],
                                "response": direction,
                                "snapshot_before": before,
                                "snapshot_after": env.snapshot(),
                                "outcome": outcome,
                                "seed": sample,
                            }
                        ],
                    }
                    write_json(destination / f"{name}.json", trace)
    return roots


def test_matched_observations_do_not_claim_full_controller_server_parity(tmp_path, monkeypatch):
    report = compare_ray_pair(*fixture_complete_pair(tmp_path, monkeypatch))
    assert report["status"] == "matched_observed_actor_and_rollouts"
    assert report["training_completions_per_run"] == 32
    assert not report["full_trainer_parity_verified"] and not report["controller_rng_verified"]
    assert not report["rollout_server_rng_verified"]


def test_unrelated_trace_name_is_not_silently_normalized(tmp_path, monkeypatch):
    left, right = fixture_complete_pair(tmp_path, monkeypatch)
    config = read_json(right / "verl-config.json")
    config["actor_rollout_ref"]["rollout"]["trace"]["experiment_name"] = "unrelated"
    write_json(right / "verl-config.json", config)
    with pytest.raises(SchemaError, match="owning trainer"):
        compare_ray_pair(left, right)


def test_exported_config_namespace_does_not_depend_on_current_host_path(tmp_path, monkeypatch):
    left, right = fixture_complete_pair(tmp_path, monkeypatch)
    for root in (left, right):
        config = read_json(root / "verl-config.json")
        config["trainer"]["default_local_dir"] = f"/original/server/{root.name}/checkpoints"
        write_json(root / "verl-config.json", config)
    report = compare_ray_pair(left, right)
    assert report["status"] == "matched_observed_actor_and_rollouts"
    assert report["observed_actor_rng_match"]


def test_seeded_run_requires_initialization_record_and_resolved_protocol(tmp_path, monkeypatch):
    left, right = fixture_complete_pair(tmp_path, monkeypatch)
    for root in (left, right):
        returned = read_json(root / "trainer-returned.json")
        returned["actor_initialization_seed"] = 5422
        write_json(root / "trainer-returned.json", returned)
        write_json(
            root / "state-proofs/initialization.json",
            {
                "schema_version": "ard-worker-initialization-v1",
                "protocol": "one_time_before_native_init_model",
                "seed": 5422,
                "cuda_device": 0,
                "rng_domains": ["python", "numpy", "torch_cpu", "torch_cuda_local"],
                "full_trainer_parity_verified": False,
            },
        )
    with pytest.raises(SchemaError, match="initialization protocol"):
        compare_ray_pair(left, right)
    for root in (left, right):
        config = read_json(root / "verl-config.json")
        config["ard_actor_initialization"] = {
            "protocol": "one_time_before_native_init_model",
            "seed": 5422,
        }
        write_json(root / "verl-config.json", config)
    assert compare_ray_pair(left, right)["actor_rng_initialization_recorded"]


@pytest.mark.parametrize("mode", ["missing", "seed", "optimizer"])
def test_hidden_missing_or_different_observations_detected(tmp_path, monkeypatch, mode):
    left, right = fixture_complete_pair(tmp_path, monkeypatch)
    path = next((right / "rollouts/grid-traces").glob("*.json"))
    if mode == "missing":
        path.unlink()  # remove only this test's authored temporary fixture
        with pytest.raises(SchemaError, match="completion slots"):
            compare_ray_pair(left, right)
        return
    if mode == "seed":
        trace = read_json(path)
        trace["steps"][0]["seed"] = 999
        write_json(path, trace)
    else:
        state = read_json(right / "state-proofs/update-000001.json")
        state["after"]["optimizer"] = "e" * 64
        write_json(right / "state-proofs/update-000001.json", state)
    assert compare_ray_pair(left, right)["status"] == "different"
