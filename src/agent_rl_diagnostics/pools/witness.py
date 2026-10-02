"""Independent exact-geometry prefixes for controlled recovery probes, not Base pools."""

from __future__ import annotations

from collections import deque
from pathlib import Path

from ..agent_config import AgentConfig
from ..artifacts.io import canonical_hash, new_output_directory
from ..contracts.schema import SchemaError, integer
from ..environments.frozenlake import ACTIONS, GridTask
from ..environments.text_grid import TextGrid
from .frozen import assign_state_id, load_pool, write_pool


def build_witness_pool(config: AgentConfig, output: Path, *, budget: int, size: int = 4) -> dict:
    integer(budget, "budget", 2)
    integer(size, "size", 1)
    if config.environment != "frozenlake_deterministic":
        raise SchemaError("exact witness anchors are available only for deterministic grids")
    candidates = []
    for task in config.tasks:
        grid = GridTask(task["task_id"], tuple(task["desc"]))
        env = TextGrid(task, prompt_profile=config.raw.get("grid_prompt_profile", "legacy_v1"))
        start = env.snapshot()["state"]
        paths = {start: []}
        queue = deque([start])
        while queue:
            state = queue.popleft()
            for action in range(4):
                nxt = grid.move(state, action)
                if nxt not in paths and grid.tile(nxt) not in "HG":
                    paths[nxt] = paths[state] + [action]
                    queue.append(nxt)
        for position, path in paths.items():
            remaining = budget - len(path)
            distance = grid.shortest_distance(position)
            if remaining < 2 or distance is None:
                continue
            after = [
                grid.shortest_distance(grid.move(position, a))
                for a in range(4)
                if grid.tile(grid.move(position, a)) not in "HG"
            ]
            if not any(d == distance - 1 for d in after) or not any(
                d is not None and d > distance for d in after
            ):
                continue
            env = TextGrid(task, prompt_profile=config.raw.get("grid_prompt_profile", "legacy_v1"))
            messages = env.initial_messages(budget)
            for index, action in enumerate(path):
                response = ACTIONS[action]
                outcome = env.apply(response, budget - index)
                messages += [
                    {"role": "assistant", "content": response},
                    {
                        "role": "user",
                        "content": env.budget_observation(
                            budget, budget - index - 1, outcome["observation"]
                        ),
                    },
                ]
            candidates.append(
                assign_state_id(
                    {
                        "schema_version": "ard-frozen-state-v1",
                        "environment": config.environment,
                        "task_id": task["task_id"],
                        "task_hash": canonical_hash(task),
                        "snapshot": env.snapshot(),
                        "messages": messages,
                        "remaining_budget": remaining,
                        "prefix_depth": len(path),
                        "source": {
                            "kind": "independent_exact_geometry_witness",
                            "solver_version": "ard-grid-bfs-v1",
                            "witness_actions": path,
                        },
                    }
                )
            )
    candidates.sort(key=lambda s: (s["prefix_depth"], s["task_id"], s["state_id"]))
    selected = []
    while candidates and len(selected) < size:
        counts = {
            t["task_id"]: sum(s["task_id"] == t["task_id"] for s in selected) for t in config.tasks
        }
        chosen = min(
            candidates, key=lambda s: (counts[s["task_id"]], s["prefix_depth"], s["state_id"])
        )
        candidates.remove(chosen)
        selected.append(chosen)
    if len(selected) != size:
        raise SchemaError("insufficient geometric witness anchors")
    with new_output_directory(output) as staging:
        manifest = write_pool(
            staging,
            states=selected,
            tasks=list(config.tasks),
            environment=config.environment,
            evidence_kind=config.raw["evidence_kind"],
            selection={
                "kind": "oracle_reference",
                "budget": budget,
                "geometry_eligibility": "nonterminal_clean_and_distance_increasing_actions",
                "selection_uses_evaluated_model_outcomes": False,
                "not_base_collected": True,
                "heldout_task_id_check": "not_verified",
            },
        )
    load_pool(output)
    return {
        "pool_hash": manifest["pool_hash"],
        "states": len(selected),
        "source": "independent_exact_geometry_witness",
    }
