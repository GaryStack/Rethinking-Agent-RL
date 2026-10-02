from __future__ import annotations

import re
from typing import Any

from ..contracts.schema import SchemaError, integer
from .frozenlake import ACTIONS, GridEnvironment, GridTask, finite_horizon_q

PROMPT_VERSION = "ard-grid-text-v1"
SCORER = "frozenlake_q_star_gamma_0.95_v1"


def parse_grid_action(response: str) -> dict[str, Any]:
    # Explicit strict final-answer protocol; do not silently pick a direction in reasoning.
    final = response.rsplit("</think>", 1)[-1].strip()
    match = re.fullmatch(r"(LEFT|DOWN|RIGHT|UP)[.!]?", final, flags=re.IGNORECASE)
    return {"kind": "move", "direction": match.group(1).upper()} if match else {"kind": "invalid"}


class TextGrid:
    name = "frozenlake_deterministic"
    scorer = SCORER

    def __init__(self, task: dict[str, Any], *, prompt_profile: str = "legacy_v1") -> None:
        if prompt_profile not in {"legacy_v1", "action_only_v2"}:
            raise SchemaError("unsupported grid prompt profile")
        self.prompt_profile = prompt_profile
        self.task = dict(task)
        self.grid = GridTask(task["task_id"], tuple(task["desc"]))
        self.env = GridEnvironment(self.grid)

    def snapshot(self) -> dict[str, Any]:
        value = self.env.export_snapshot()
        if self.prompt_profile != "legacy_v1":
            value["prompt_profile"] = self.prompt_profile
        return value

    def restore(self, snapshot: dict[str, Any]) -> None:
        if snapshot.get("prompt_profile", "legacy_v1") != self.prompt_profile:
            raise SchemaError("grid snapshot/prompt profile mismatch")
        self.env.restore_snapshot({k: v for k, v in snapshot.items() if k != "prompt_profile"})
        if self.grid.tile(self.env.state) in "HG":
            raise SchemaError("cannot generate from a terminal grid snapshot")

    def observation(self) -> str:
        rows = []
        for i, row in enumerate(self.grid.desc):
            cells = list(row)
            if i == self.env.state // len(row):
                cells[self.env.state % len(row)] = "@"
            rows.append(" ".join(cells))
        coordinates = (
            f"Your ACTUAL current position is row {self.env.state // len(self.grid.desc[0])}, column {self.env.state % len(self.grid.desc[0])} (zero-indexed). Only @ shows your current position.\n"
            if self.prompt_profile == "action_only_v2"
            else ""
        )
        return coordinates + "Current grid:\n" + "\n".join(rows)

    def initial_messages(self, total_budget: int) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "You control a deterministic FrozenLake agent. @ is your position, S the start, F safe ice, H a terminal failure, G a terminal success. Each move, including a boundary move or invalid answer, costs one action. Return exactly LEFT, DOWN, RIGHT, or UP. Reach G within the action budget."
                + (
                    " Choose ONLY your NEXT move from the ACTUAL current grid, not a complete plan. Output exactly ONE unformatted word: LEFT or DOWN or RIGHT or UP. Do not explain or use markdown. Example valid reply: RIGHT. Example invalid reply: **RIGHT** followed by an explanation."
                    if self.prompt_profile == "action_only_v2"
                    else ""
                ),
            },
            {
                "role": "user",
                "content": self.budget_observation(total_budget, total_budget, self.observation()),
            },
        ]

    @staticmethod
    def budget_observation(total_budget: int, remaining: int, observation: str) -> str:
        return f"Total action budget: {total_budget}. Remaining action budget including this action: {remaining}.\n{observation}"

    def apply(self, response: str, remaining: int) -> dict[str, Any]:
        integer(remaining, "remaining", 1)
        if self.grid.tile(self.env.state) in "HG":
            raise SchemaError("cannot act from a terminal grid state")
        action = parse_grid_action(response)
        if action["kind"] == "invalid":
            self.env.action_count += 1
            return {
                "action": action,
                "observation": "Invalid action. Return exactly one direction.\n"
                + self.observation(),
                "reward": 0.0,
                "terminated": False,
                "success": False,
                "local_score": 0.0,
                "scorer": SCORER,
            }
        index = ACTIONS.index(action["direction"])
        score = finite_horizon_q(self.grid, remaining, gamma=0.95)[remaining][self.env.state][index]
        _, reward, terminal, success = self.env.step(index)
        return {
            "action": action,
            "observation": self.observation(),
            "reward": reward,
            "terminated": terminal,
            "success": success,
            "local_score": score,
            "scorer": SCORER,
        }
