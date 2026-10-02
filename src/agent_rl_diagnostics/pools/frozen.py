from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from typing import Any

from ..agent_config import load_agent_config
from ..artifacts.io import (
    canonical_hash,
    file_hash,
    new_output_directory,
    read_json,
    verify_artifacts,
    write_json,
)
from ..contracts.agent import AgentTrajectory
from ..contracts.schema import SchemaError, integer, text
from ..environments.deepcoder import DeepCoder, parse_code_action
from ..environments.text_grid import TextGrid
from ..evaluators.agent import audit_agent_budget, read_records, write_records
from ..models.chat import validate_messages


def state_content(state: dict[str, Any]) -> dict[str, Any]:
    return {
        key: state[key]
        for key in (
            "environment",
            "task_id",
            "task_hash",
            "snapshot",
            "messages",
            "remaining_budget",
            "prefix_depth",
        )
    }


def assign_state_id(state: dict[str, Any]) -> dict[str, Any]:
    state["state_id"] = canonical_hash(state_content(state))
    return state


def validate_state(state: dict[str, Any], task: dict[str, Any], environment: str) -> None:
    if not isinstance(state, dict) or state.get("schema_version") != "ard-frozen-state-v1":
        raise SchemaError("unsupported frozen state schema")
    if (
        state.get("task_id") != task["task_id"]
        or state.get("task_hash") != canonical_hash(task)
        or state.get("environment") != environment
    ):
        raise SchemaError("frozen state task/environment mismatch")
    integer(state.get("remaining_budget"), "remaining_budget", 1)
    integer(state.get("prefix_depth"), "prefix_depth")
    validate_messages(state.get("messages", []))
    snapshot = state.get("snapshot")
    if not isinstance(snapshot, dict) or snapshot.get("action_count") != state["prefix_depth"]:
        raise SchemaError("snapshot action_count/prefix_depth mismatch")
    if state.get("state_id") != canonical_hash(state_content(state)):
        raise SchemaError("frozen state content hash mismatch")
    # The collector's chat protocol has one initial user message and two messages per action.
    if len(state["messages"]) != 2 + 2 * state["prefix_depth"]:
        raise SchemaError("frozen state history length/prefix mismatch")
    total = state["remaining_budget"] + state["prefix_depth"]
    expected = f"Total action budget: {total}. Remaining action budget including this action: {state['remaining_budget']}.\n"
    if not state["messages"][-1]["content"].startswith(expected):
        raise SchemaError("frozen prompt does not show the declared total/remaining budget")
    if environment == "frozenlake_deterministic":
        env = TextGrid(task, prompt_profile=snapshot.get("prompt_profile", "legacy_v1"))
        if state["messages"][:2] != env.initial_messages(total):
            raise SchemaError("pool initial prompt differs from the grid protocol")
        for index in range(state["prefix_depth"]):
            action_message = state["messages"][2 + 2 * index]
            observation_message = state["messages"][3 + 2 * index]
            if action_message["role"] != "assistant":
                raise SchemaError("prefix action message has wrong role")
            result = env.apply(action_message["content"], total - index)
            feedback = {
                "role": "user",
                "content": env.budget_observation(total, total - index - 1, result["observation"]),
            }
            if result["terminated"] or observation_message != feedback:
                raise SchemaError("frozen prefix cannot be replayed to a nonterminal state")
        if env.snapshot() != state["snapshot"]:
            raise SchemaError("frozen snapshot differs from replayed prefix")
    else:
        profile = snapshot.get("prompt_profile", "legacy_v1")
        # This path checks prompt/code structure only; it does not execute code.
        env = DeepCoder(task, None, prompt_profile=profile)
        if state["messages"][:2] != env.initial_messages(total):
            raise SchemaError("coding pool initial prompt differs from its profile")
        current_code = None
        for index in range(state["prefix_depth"]):
            message = state["messages"][2 + 2 * index]
            feedback = state["messages"][3 + 2 * index]
            if message["role"] != "assistant" or feedback["role"] != "user":
                raise SchemaError("invalid coding prefix message roles")
            action = parse_code_action(message["content"])
            if action["kind"] == "submit":
                raise SchemaError("coding prefix continues after a final submission")
            if action["kind"] == "run":
                current_code = action["code"]
        if current_code != snapshot.get("current_code"):
            raise SchemaError("coding snapshot code differs from prefix")
        if (
            snapshot.get("version") != "ard-code-snapshot-v1"
            or snapshot.get("task_hash") != canonical_hash(task)
            or snapshot.get("terminal") is not False
        ):
            raise SchemaError("invalid/noncontinuable coding snapshot")


def write_pool(
    directory: Path,
    *,
    states: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    environment: str,
    evidence_kind: str,
    selection: dict[str, Any],
    kind: str = "state_pool",
    pairs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    write_json(directory / "tasks.json", tasks)
    write_records(directory / "states.jsonl", states)
    names = ["tasks.json", "states.jsonl"]
    if pairs is not None:
        write_records(directory / "pairs.jsonl", pairs)
        names.append("pairs.jsonl")
    pool_hash = canonical_hash(
        {
            "environment": environment,
            "states": states,
            "tasks": tasks,
            "selection": selection,
            "pairs": pairs,
        }
    )
    manifest = {
        "schema_version": "ard-manifest-v1",
        "kind": kind,
        "status": "complete",
        "environment": environment,
        "evidence_kind": evidence_kind,
        "protocol_id": "toolkit_v1",
        "pool_hash": pool_hash,
        "selection": selection,
        "state_count": len(states),
        "tasks_count": len({s["task_id"] for s in states}),
        "artifacts": {name: file_hash(directory / name) for name in names},
    }
    write_json(directory / "manifest.json", manifest)
    return manifest


def load_pool(
    directory: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    manifest = verify_artifacts(directory)
    if (
        manifest.get("kind") not in {"state_pool", "error_pool"}
        or manifest.get("status") != "complete"
    ):
        raise SchemaError("not a complete state/error pool")
    tasks_list = read_json(directory / "tasks.json")
    if not isinstance(tasks_list, list) or any(not isinstance(t, dict) for t in tasks_list):
        raise SchemaError("pool tasks must be an array of objects")
    if any(not isinstance(task.get("task_id"), str) for task in tasks_list):
        raise SchemaError("pool tasks need string IDs")
    tasks = {task["task_id"]: task for task in tasks_list}
    if len(tasks) != len(tasks_list):
        raise SchemaError("duplicate pool task IDs")
    states = read_records(directory / "states.jsonl")
    if not states or len({state.get("state_id") for state in states}) != len(states):
        raise SchemaError("pool is empty or contains duplicate inputs")
    for state in states:
        if state.get("task_id") not in tasks:
            raise SchemaError("pool references an unknown task")
        validate_state(state, tasks[state["task_id"]], manifest["environment"])
    pairs = read_records(directory / "pairs.jsonl") if manifest["kind"] == "error_pool" else None
    expected = canonical_hash(
        {
            "environment": manifest["environment"],
            "states": states,
            "tasks": tasks_list,
            "selection": manifest["selection"],
            "pairs": pairs,
        }
    )
    if expected != manifest.get("pool_hash") or len(states) != manifest.get("state_count"):
        raise SchemaError("pool hash/count mismatch")
    if pairs is not None:
        by_id = {state["state_id"]: state for state in states}
        pair_ids = set()
        used_states = set()
        for pair in pairs:
            text(pair.get("pair_id"), "pair_id")
            if pair["pair_id"] in pair_ids:
                raise SchemaError("duplicate intervention pair")
            pair_ids.add(pair["pair_id"])
            clean, error = (
                by_id.get(pair.get("clean_state_id")),
                by_id.get(pair.get("error_state_id")),
            )
            if (
                clean is None
                or error is None
                or clean["task_id"] != error["task_id"]
                or pair.get("task_id") != clean["task_id"]
                or clean["remaining_budget"] != error["remaining_budget"]
                or clean["prefix_depth"] != error["prefix_depth"]
            ):
                raise SchemaError("error pair task/budget mismatch")
            if clean["state_id"] == error["state_id"]:
                raise SchemaError("clean/error inputs cannot be identical")
            used_states.update((clean["state_id"], error["state_id"]))
        if used_states != set(by_id):
            raise SchemaError("error pool contains unpaired states")
    return manifest, states, tasks


def build_pool(
    inputs: list[Path],
    output: Path,
    *,
    kind: str,
    size: int,
    seed: int,
    base_model_id: str,
    rl_model_id: str | None = None,
    max_per_task: int = 8,
    min_tasks: int = 1,
    training_task_ids: set[str] | None = None,
) -> dict[str, Any]:
    for name, value in (("size", size), ("max_per_task", max_per_task), ("min_tasks", min_tasks)):
        integer(value, name, 1)
    integer(seed, "seed")
    if kind not in {"base_monitor", "matched"} or (
        kind == "matched" and (not rl_model_id or size % 2)
    ):
        raise SchemaError("matched pool requires RL model ID and an even size")
    if rl_model_id == base_model_id:
        raise SchemaError("Base and RL IDs must be different")
    candidates: list[dict[str, Any]] = []
    tasks: dict[str, dict[str, Any]] = {}
    environments, evidence, checkpoints = set(), set(), {}
    source_hashes = []
    for directory in inputs:
        manifest = verify_artifacts(directory)
        if manifest.get("kind") != "agent_budget" or manifest.get("status") != "complete":
            raise SchemaError("pool requires complete agent-budget source runs")
        config = load_agent_config(directory / "config.json")
        role = (
            "base"
            if config.model_id == base_model_id
            else "rl"
            if config.model_id == rl_model_id
            else None
        )
        if role is None or (kind == "base_monitor" and role != "base"):
            raise SchemaError("source run policy does not match the requested pool roles")
        audit_agent_budget(directory, manifest)
        previous = checkpoints.setdefault(config.model_id, config.checkpoint_id)
        if previous != config.checkpoint_id:
            raise SchemaError("collector model ID combines different revisions")
        environments.add(config.environment)
        evidence.add(config.raw["evidence_kind"])
        source_hashes.append(file_hash(directory / "manifest.json"))
        for task in config.tasks:
            if task["task_id"] in tasks and tasks[task["task_id"]] != task:
                raise SchemaError("same task ID refers to different task content")
            tasks[task["task_id"]] = task
        for value in read_records(directory / "trajectories.jsonl"):
            trajectory = AgentTrajectory.from_dict(value)
            for step in trajectory.steps:
                state = {
                    "schema_version": "ard-frozen-state-v1",
                    "environment": config.environment,
                    "task_id": trajectory.task_id,
                    "task_hash": canonical_hash(tasks[trajectory.task_id]),
                    "snapshot": step.snapshot_before,
                    "messages": step.messages,
                    "remaining_budget": step.remaining_budget,
                    "prefix_depth": step.turn_index,
                    "source": {
                        "model_id": config.model_id,
                        "checkpoint_id": config.checkpoint_id,
                        "run_id": trajectory.run_id,
                        "sample_id": trajectory.sample_id,
                        "role": role,
                    },
                }
                candidates.append(assign_state_id(state))
    if len(environments) != 1 or len(evidence) != 1:
        raise SchemaError("pool cannot combine environments or evidence profiles")
    environment = next(iter(environments))
    if training_task_ids is not None and set(tasks) & training_task_ids:
        raise SchemaError("source task IDs overlap training tasks")
    roles = ["base"] if kind == "base_monitor" else ["base", "rl"]
    unique = {(state["source"]["role"], state["state_id"]): state for state in candidates}
    available = list(unique.values())
    selected, used = [], set()
    task_counts: Counter[str] = Counter()
    depth_counts: Counter[int] = Counter()
    role_counts: Counter[str] = Counter()
    for index in range(size):
        role = roles[index % len(roles)]
        eligible = [
            s
            for s in available
            if s["source"]["role"] == role
            and s["state_id"] not in used
            and task_counts[s["task_id"]] < max_per_task
        ]
        if not eligible:
            raise SchemaError(
                "insufficient distinct states under role/task constraints; collect more data"
            )
        state = min(
            eligible,
            key=lambda s: (
                task_counts[s["task_id"]],
                depth_counts[s["prefix_depth"]],
                canonical_hash([seed, s["state_id"], role]),
            ),
        )
        selected.append(copy.deepcopy(state))
        used.add(state["state_id"])
        task_counts[state["task_id"]] += 1
        depth_counts[state["prefix_depth"]] += 1
        role_counts[role] += 1
    if len(task_counts) < min_tasks:
        raise SchemaError("selected pool has insufficient task coverage")
    selected.sort(key=lambda s: s["state_id"])
    selection = {
        "kind": kind,
        "size": size,
        "seed": seed,
        "max_per_task": max_per_task,
        "min_tasks": min_tasks,
        "reward_blind": True,
        "candidate_count": len(candidates),
        "role_counts": dict(role_counts),
        "source_manifest_hashes": source_hashes,
        "collector_checkpoints": checkpoints,
        "heldout_task_id_check": "disjoint" if training_task_ids is not None else "not_verified",
        "training_map_content_check": "not_verified",
    }
    with new_output_directory(output) as staging:
        manifest = write_pool(
            staging,
            states=selected,
            tasks=list(tasks.values()),
            environment=environment,
            evidence_kind=next(iter(evidence)),
            selection=selection,
        )
    load_pool(output)
    return {
        "output": str(output),
        "pool_hash": manifest["pool_hash"],
        "states": len(selected),
        "kind": kind,
        "role_counts": dict(role_counts),
    }
