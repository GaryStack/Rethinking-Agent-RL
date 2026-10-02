from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

from ..artifacts.io import canonical_hash, file_hash, new_output_directory, read_json
from ..contracts.schema import SchemaError, integer
from ..environments.deepcoder import DeepCoder
from ..environments.execution import NamespaceExecutor
from ..environments.frozenlake import ACTIONS, GridTask
from ..environments.text_grid import TextGrid
from ..evaluators.agent import paired_seed
from .frozen import assign_state_id, load_pool, write_pool


def _after_action(
    anchor: dict[str, Any], task: dict[str, Any], response: str, env: TextGrid | DeepCoder
) -> dict[str, Any]:
    env.restore(anchor["snapshot"])
    outcome = env.apply(response, anchor["remaining_budget"])
    if outcome["terminated"]:
        raise SchemaError("intervention must leave a nonterminal continuation state")
    state = copy.deepcopy(anchor)
    state["prefix_depth"] += 1
    state["remaining_budget"] -= 1
    state["snapshot"] = env.snapshot()
    total = state["prefix_depth"] + state["remaining_budget"]
    state["messages"] += [
        {"role": "assistant", "content": response},
        {
            "role": "user",
            "content": env.budget_observation(
                total, state["remaining_budget"], outcome["observation"]
            ),
        },
    ]
    state["task_hash"] = canonical_hash(task)
    return assign_state_id(state)


def build_error_pool(
    source: Path,
    output: Path,
    *,
    seed: int = 5422,
    require_recoverable: bool = False,
    references: Path | None = None,
    executor: NamespaceExecutor | None = None,
) -> dict[str, Any]:
    integer(seed, "seed")
    manifest, anchors, tasks = load_pool(source)
    if manifest["kind"] != "state_pool":
        raise SchemaError("error construction requires a regular frozen state pool")
    environment = manifest["environment"]
    reference_map = None
    if environment == "deepcoder_stdin":
        if references is None:
            raise SchemaError("coding errors require explicit independent reference programs")
        reference_map = read_json(references)
        if not isinstance(reference_map, dict):
            raise SchemaError("references must map task IDs to code/source/revision records")
        executor = executor or NamespaceExecutor()
        executor.preflight()
    states, pairs, used = [], [], set()
    skipped = 0
    for anchor in anchors:
        if anchor["remaining_budget"] < 2:
            skipped += 1
            continue
        task = tasks[anchor["task_id"]]
        severity: dict[str, Any]
        if environment == "frozenlake_deterministic":
            grid = GridTask(task["task_id"], tuple(task["desc"]))
            current = anchor["snapshot"]["state"]
            distance = grid.shortest_distance(current)
            if distance is None:
                skipped += 1
                continue
            clean_actions, error_actions = [], []
            for index, name in enumerate(ACTIONS):
                nxt = grid.move(current, index)
                after_distance = grid.shortest_distance(nxt)
                if grid.tile(nxt) in "HG" or after_distance is None:
                    continue
                if after_distance == distance - 1:
                    clean_actions.append((name, after_distance))
                if after_distance > distance and (
                    not require_recoverable or after_distance <= anchor["remaining_budget"] - 1
                ):
                    error_actions.append((name, after_distance))
            if not clean_actions or not error_actions:
                skipped += 1
                continue
            rng = random.Random(paired_seed(seed, anchor["state_id"]))
            clean_response, clean_distance = rng.choice(clean_actions)
            error_response, error_distance = rng.choice(error_actions)
            profile = anchor["snapshot"].get("prompt_profile", "legacy_v1")
            clean = _after_action(
                anchor, task, clean_response, TextGrid(task, prompt_profile=profile)
            )
            error = _after_action(
                anchor, task, error_response, TextGrid(task, prompt_profile=profile)
            )
            severity = {
                "kind": "shortest_distance_increase",
                "anchor_distance": distance,
                "clean_distance": clean_distance,
                "error_distance": error_distance,
                "error_recoverable": error_distance <= error["remaining_budget"],
                "clean_recoverable": clean_distance <= clean["remaining_budget"],
                "recoverability_proof": "exact_deterministic_bfs",
            }
            injector = "toolkit_distance_increase_v1"
        else:
            reference = reference_map.get(anchor["task_id"])
            if (
                not isinstance(reference, dict)
                or set(reference) != {"code", "source", "revision"}
                or reference["source"] not in {"dataset_reference", "independent_reference"}
                or not all(isinstance(v, str) and v for v in reference.values())
            ):
                raise SchemaError(
                    "reference must identify independent/dataset source, revision and code"
                )
            env = DeepCoder(task, executor)
            if not env.grade(reference["code"])["passed"]:
                raise SchemaError("independent reference does not pass the declared tests")
            mutated = reference["code"] + "\nif True print('ard-syntax-error')"
            clean_response = "RUN\n```python\n" + reference["code"] + "\n```"
            error_response = "RUN\n```python\n" + mutated + "\n```"
            profile = anchor["snapshot"].get("prompt_profile", "legacy_v1")
            clean = _after_action(
                anchor, task, clean_response, DeepCoder(task, executor, prompt_profile=profile)
            )
            error = _after_action(
                anchor, task, error_response, DeepCoder(task, executor, prompt_profile=profile)
            )
            severity = {
                "kind": "syntax_mutation",
                "reference_test_fraction": 1.0,
                "mutation_test_fraction": 0.0,
                "error_recoverable": True,
                "clean_recoverable": True,
                "recoverability_proof": "tested_reference_submission",
                "reference_source": reference["source"],
                "reference_revision": reference["revision"],
            }
            injector = "toolkit_independent_reference_syntax_v1"
        if clean["state_id"] in used or error["state_id"] in used:
            skipped += 1
            continue
        pair_id = canonical_hash(
            [anchor["state_id"], injector, clean["state_id"], error["state_id"]]
        )
        states.extend([clean, error])
        used.update([clean["state_id"], error["state_id"]])
        pairs.append(
            {
                "pair_id": pair_id,
                "anchor_state_id": anchor["state_id"],
                "task_id": anchor["task_id"],
                "clean_state_id": clean["state_id"],
                "error_state_id": error["state_id"],
                "injector": injector,
                "matched_action_cost": 1,
                "severity": severity,
            }
        )
    if not pairs:
        raise SchemaError(
            "no eligible clean/error pairs; collect deeper prefixes or longer budgets"
        )
    selection = {
        "source_pool_hash": manifest["pool_hash"],
        "seed": seed,
        "require_recoverable": require_recoverable,
        "skipped_anchors": skipped,
        "selection_uses_evaluated_model_outcomes": False,
        "reference_file_hash": file_hash(references) if references else None,
    }
    with new_output_directory(output) as staging:
        result = write_pool(
            staging,
            states=states,
            tasks=list(tasks.values()),
            environment=environment,
            evidence_kind=manifest["evidence_kind"],
            selection=selection,
            kind="error_pool",
            pairs=pairs,
        )
    load_pool(output)
    return {
        "output": str(output),
        "pool_hash": result["pool_hash"],
        "pairs": len(pairs),
        "skipped_anchors": skipped,
    }
