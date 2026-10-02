"""Strict v1 trajectory contracts. No heavy framework imports."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

SCHEMA_VERSION = "ard-v1"
BUDGET_MODES = {"budget_aware_rerollout", "budget_blind_prefix"}
EVIDENCE_KINDS = {"demo_synthetic", "raw_recomputable", "summary_only", "missing"}


class SchemaError(ValueError):
    """A record violates an explicit scientific or data invariant."""


def integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SchemaError(f"{name} must be an integer >= {minimum}")
    return value


def text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{name} must be a nonempty string")
    return value


def finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SchemaError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class BudgetCertificate:
    kind: str
    value: int | None
    state_hash: str
    solver_version: str
    assumptions: tuple[str, ...]
    unreachable: bool = False
    action_unit: str = "agent_action"

    def __post_init__(self) -> None:
        text(self.kind, "certificate.kind")
        if self.kind not in {
            "exact_minimum",
            "proven_lower_bound",
            "witness_upper_bound",
            "annotation_only",
            "unknown",
        }:
            raise SchemaError("invalid certificate kind")
        if type(self.unreachable) is not bool:
            raise SchemaError("unreachable must be boolean")
        if self.value is not None:
            integer(self.value, "certificate.value")
        if self.kind == "exact_minimum" and (self.value is None) != self.unreachable:
            raise SchemaError("exact certificate needs a finite value or an unreachable proof")
        if self.unreachable and self.kind != "exact_minimum":
            raise SchemaError("unreachable proof requires exact_minimum")
        text(self.state_hash, "state_hash")
        text(self.solver_version, "solver_version")


@dataclass(frozen=True)
class StepRecord:
    turn_index: int
    remaining_budget: int
    state: int
    action: int
    next_state: int
    reward: float
    terminated: bool
    success: bool
    local_score: float | None = None
    scorer: str | None = None

    def __post_init__(self) -> None:
        integer(self.turn_index, "turn_index")
        integer(self.remaining_budget, "remaining_budget", 1)
        integer(self.state, "state")
        integer(self.action, "action")
        integer(self.next_state, "next_state")
        finite(self.reward, "reward")
        if type(self.terminated) is not bool or type(self.success) is not bool:
            raise SchemaError("terminated and success must be boolean")
        if self.success and not self.terminated:
            raise SchemaError("FrozenLake success is terminal")
        if (self.local_score is None) != (self.scorer is None):
            raise SchemaError("local_score and scorer must be supplied together")
        if self.local_score is not None:
            finite(self.local_score, "local_score")
            text(self.scorer, "scorer")


@dataclass(frozen=True)
class Trajectory:
    run_id: str
    model_id: str
    checkpoint_id: str
    task_id: str
    task_occurrence_id: str
    sample_id: int
    total_budget: int
    budget_mode: str
    seed: int
    status: str
    success: bool | None
    termination_reason: str
    steps: tuple[StepRecord, ...]
    evidence_kind: str
    protocol_id: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "model_id",
            "checkpoint_id",
            "task_id",
            "task_occurrence_id",
            "termination_reason",
            "protocol_id",
        ):
            text(getattr(self, name), name)
        integer(self.sample_id, "sample_id")
        integer(self.total_budget, "total_budget", 1)
        integer(self.seed, "seed")
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError(f"schema_version must be {SCHEMA_VERSION}")
        for name in ("budget_mode", "evidence_kind", "status"):
            text(getattr(self, name), name)
        if self.budget_mode not in BUDGET_MODES:
            raise SchemaError("invalid budget_mode")
        if self.evidence_kind not in EVIDENCE_KINDS:
            raise SchemaError("invalid evidence_kind")
        if self.status not in {"completed", "infrastructure_failure"}:
            raise SchemaError("invalid trajectory status")
        if self.status == "completed" and type(self.success) is not bool:
            raise SchemaError("completed trajectory requires boolean success")
        if self.status == "infrastructure_failure" and self.success is not None:
            raise SchemaError("infrastructure failure must have success=null")
        if not isinstance(self.steps, tuple) or any(
            not isinstance(step, StepRecord) for step in self.steps
        ):
            raise SchemaError("steps must be a tuple of StepRecords")
        if len(self.steps) > self.total_budget:
            raise SchemaError("trajectory exceeds action budget")
        for i, step in enumerate(self.steps):
            if step.turn_index != i or step.remaining_budget != self.total_budget - i:
                raise SchemaError("turn index/remaining budget is inconsistent")
            if i and self.steps[i - 1].next_state != step.state:
                raise SchemaError("state chain is broken")
            if i < len(self.steps) - 1 and step.terminated:
                raise SchemaError("trajectory continues after a terminal action")
        if self.status == "completed":
            if not self.steps:
                raise SchemaError("completed trajectory requires at least one action")
            last = self.steps[-1]
            if self.success != last.success:
                raise SchemaError("trajectory success disagrees with final action")
            expected = (
                "success"
                if last.success
                else "environment_terminal"
                if last.terminated
                else "budget_exhausted"
            )
            if self.termination_reason != expected:
                raise SchemaError("termination_reason disagrees with final action")
            if not last.terminated and len(self.steps) != self.total_budget:
                raise SchemaError("nonterminal completed rollout must exhaust its budget")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Trajectory:
        if not isinstance(value, Mapping):
            raise SchemaError("trajectory must be an object")
        data = dict(value)
        steps = data.get("steps")
        if not isinstance(steps, list):
            raise SchemaError("steps must be an array")
        try:
            data["steps"] = tuple(StepRecord(**step) for step in steps)
            return cls(**data)
        except TypeError as exc:
            raise SchemaError(f"invalid trajectory fields: {exc}") from exc
