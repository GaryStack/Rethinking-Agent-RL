"""Small FrozenLake-compatible grid; explicit success DP and discounted Q*."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from ..artifacts.io import canonical_hash
from ..contracts.schema import BudgetCertificate, SchemaError, integer

ACTIONS = ("LEFT", "DOWN", "RIGHT", "UP")
DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))
SOLVER_VERSION = "ard-grid-bfs-v1"


@dataclass(frozen=True)
class GridTask:
    task_id: str
    desc: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.task_id or not self.desc or not self.desc[0]:
            raise SchemaError("grid needs a task_id and a nonempty map")
        if any(not isinstance(row, str) or len(row) != len(self.desc[0]) for row in self.desc):
            raise SchemaError("map must be rectangular")
        tiles = "".join(self.desc)
        if any(c not in "SFHG" for c in tiles) or tiles.count("S") != 1 or tiles.count("G") != 1:
            raise SchemaError("map must have exactly one S and G, and only SFHG tiles")

    @property
    def start(self) -> int:
        return "".join(self.desc).index("S")

    def tile(self, state: int) -> str:
        integer(state, "state")
        if state >= self.size:
            raise SchemaError("state is outside the map")
        return "".join(self.desc)[state]

    @property
    def size(self) -> int:
        return len(self.desc) * len(self.desc[0])

    @property
    def map_hash(self) -> str:
        return canonical_hash({"desc": self.desc, "dynamics": "grid-v1"})

    def move(self, state: int, action: int) -> int:
        self.tile(state)
        integer(action, "action")
        if action >= len(ACTIONS):
            raise SchemaError("action must be 0..3 (LEFT, DOWN, RIGHT, UP)")
        row, col = divmod(state, len(self.desc[0]))
        dr, dc = DELTAS[action]
        row = max(0, min(len(self.desc) - 1, row + dr))
        col = max(0, min(len(self.desc[0]) - 1, col + dc))
        return row * len(self.desc[0]) + col

    def shortest_distance(self, state: int | None = None) -> int | None:
        state = self.start if state is None else state
        if self.tile(state) == "H":
            return None
        queue = deque([(state, 0)])
        seen = {state}
        while queue:
            current, distance = queue.popleft()
            if self.tile(current) == "G":
                return distance
            for action in range(4):
                nxt = self.move(current, action)
                if nxt not in seen and self.tile(nxt) != "H":
                    seen.add(nxt)
                    queue.append((nxt, distance + 1))
        return None

    def certificate(self, state: int | None = None) -> BudgetCertificate:
        state = self.start if state is None else state
        distance = self.shortest_distance(state)
        return BudgetCertificate(
            kind="exact_minimum",
            value=distance,
            state_hash=canonical_hash({"map_hash": self.map_hash, "state": state}),
            solver_version=SOLVER_VERSION,
            unreachable=distance is None,
            assumptions=(
                "deterministic transitions",
                "one move per agent action",
                "holes terminal failure",
                "goal terminal success",
            ),
        )


@lru_cache(maxsize=256)
def finite_horizon_q(
    task: GridTask,
    horizon: int,
    *,
    gamma: float = 0.95,
    slippery: bool = False,
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    """Q[h][state][action]. gamma=1 gives optimal finite-horizon success probability."""
    integer(horizon, "horizon")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not 0 <= gamma <= 1:
        raise SchemaError("gamma must be in [0,1]")
    if type(slippery) is not bool:
        raise SchemaError("slippery must be boolean")
    table = [tuple((0.0,) * 4 for _ in range(task.size))]
    for h in range(1, horizon + 1):
        layer = []
        for state in range(task.size):
            scores = []
            for action in range(4):
                if task.tile(state) in "HG":
                    scores.append(0.0)
                    continue
                actual = ((action - 1) % 4, action, (action + 1) % 4) if slippery else (action,)
                value = 0.0
                for move in actual:
                    nxt = task.move(state, move)
                    tile = task.tile(nxt)
                    value += (
                        1.0
                        if tile == "G"
                        else 0.0
                        if tile == "H"
                        else gamma * max(table[h - 1][nxt])
                    ) / len(actual)
                scores.append(value)
            layer.append(tuple(scores))
        table.append(tuple(layer))
    return tuple(table)


class GridEnvironment:
    """Deterministic simulator. Stochastic dynamics are currently solver-only."""

    def __init__(self, task: GridTask) -> None:
        self.task = task
        self.state = task.start
        self.action_count = 0

    def step(self, action: int) -> tuple[int, float, bool, bool]:
        if self.task.tile(self.state) in "HG":
            raise SchemaError("cannot step a terminal state")
        self.state = self.task.move(self.state, action)
        self.action_count += 1
        tile = self.task.tile(self.state)
        return self.state, float(tile == "G"), tile in "HG", tile == "G"

    def export_snapshot(self) -> dict[str, Any]:
        return {
            "version": "ard-grid-snapshot-v1",
            "map_hash": self.task.map_hash,
            "state": self.state,
            "action_count": self.action_count,
        }

    def restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        if (
            snapshot.get("version") != "ard-grid-snapshot-v1"
            or snapshot.get("map_hash") != self.task.map_hash
        ):
            raise SchemaError("snapshot map/version mismatch")
        state = integer(snapshot.get("state"), "snapshot.state")
        self.task.tile(state)
        count = integer(snapshot.get("action_count"), "snapshot.action_count")
        self.state, self.action_count = state, count


def scripted_action(
    policy: str,
    task: GridTask,
    state: int,
    remaining_budget: int,
    rng: random.Random,
) -> int:
    """Demo policies only. remaining_budget is supplied before every action."""
    integer(remaining_budget, "remaining_budget", 1)
    if policy == "random":
        return rng.randrange(4)
    if policy == "shortest_path":
        distances = [task.shortest_distance(task.move(state, a)) for a in range(4)]
        finite_distances = [d for d in distances if d is not None]
        if finite_distances:
            best = min(finite_distances)
            return rng.choice([a for a, d in enumerate(distances) if d == best])
        return rng.randrange(4)
    raise SchemaError(f"unsupported scripted policy: {policy}")
