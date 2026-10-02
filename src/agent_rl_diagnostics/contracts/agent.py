"""Environment-neutral records; v1 grid-only records remain supported."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..artifacts.io import canonical_hash
from ..models.chat import validate_messages
from .schema import EVIDENCE_KINDS, SchemaError, finite, integer, text

AGENT_SCHEMA_VERSION = "ard-agent-v2"


@dataclass(frozen=True)
class AgentStep:
    turn_index: int
    remaining_budget: int
    messages: list[dict[str, str]]
    response: str
    generation: dict[str, Any]
    snapshot_before: dict[str, Any]
    snapshot_after: dict[str, Any]
    action: dict[str, Any]
    observation: str
    reward: float
    terminated: bool
    success: bool
    local_score: float
    scorer: str

    def __post_init__(self) -> None:
        integer(self.turn_index, "turn_index")
        integer(self.remaining_budget, "remaining_budget", 1)
        validate_messages(self.messages)
        for name in ("response", "observation"):
            if not isinstance(getattr(self, name), str):
                raise SchemaError(f"{name} must be text")
        for name in ("generation", "snapshot_before", "snapshot_after", "action"):
            if not isinstance(getattr(self, name), dict):
                raise SchemaError(f"{name} must be an object")
        finite(self.reward, "reward")
        finite(self.local_score, "local_score")
        text(self.scorer, "scorer")
        if not 0 <= self.local_score <= 1:
            raise SchemaError("local score must be in [0,1]")
        if type(self.terminated) is not bool or type(self.success) is not bool:
            raise SchemaError("terminal/success flags must be boolean")
        if self.success and not self.terminated:
            raise SchemaError("success must be terminal under these environments")


@dataclass(frozen=True)
class AgentTrajectory:
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
    steps: tuple[AgentStep, ...]
    evidence_kind: str
    protocol_id: str
    environment: str
    initial_snapshot: dict[str, Any]
    initial_messages: list[dict[str, str]]
    prefix_depth: int = 0
    schema_version: str = AGENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "model_id",
            "checkpoint_id",
            "task_id",
            "task_occurrence_id",
            "termination_reason",
            "environment",
        ):
            text(getattr(self, name), name)
        for name in ("sample_id", "seed", "prefix_depth"):
            integer(getattr(self, name), name)
        integer(self.total_budget, "total_budget", 1)
        if self.schema_version != AGENT_SCHEMA_VERSION or self.protocol_id != "toolkit_v1":
            raise SchemaError("unsupported agent schema/protocol")
        if self.budget_mode != "budget_aware_rerollout":
            raise SchemaError("agent runner requires budget-aware rerollout")
        text(self.evidence_kind, "evidence_kind")
        if self.evidence_kind not in EVIDENCE_KINDS:
            raise SchemaError("invalid evidence kind")
        validate_messages(self.initial_messages)
        if self.status not in {"completed", "infrastructure_failure"}:
            raise SchemaError("invalid trajectory status")
        if (self.status == "completed" and type(self.success) is not bool) or (
            self.status == "infrastructure_failure" and self.success is not None
        ):
            raise SchemaError(
                "success must be boolean for completion and null for infrastructure failure"
            )
        if len(self.steps) > self.total_budget:
            raise SchemaError("trajectory exceeds budget")
        previous = self.initial_snapshot
        for index, step in enumerate(self.steps):
            if (
                not isinstance(step, AgentStep)
                or step.turn_index != index
                or step.remaining_budget != self.total_budget - index
            ):
                raise SchemaError("step indexing/remaining budget mismatch")
            if canonical_hash(previous) != canonical_hash(step.snapshot_before):
                raise SchemaError("snapshot chain mismatch")
            if index == 0 and step.messages != self.initial_messages:
                raise SchemaError("initial generation input differs from frozen input")
            if index < len(self.steps) - 1 and step.terminated:
                raise SchemaError("trajectory continues after terminal action")
            previous = step.snapshot_after
        if self.status == "completed":
            if not self.steps or self.steps[-1].success != self.success:
                raise SchemaError("completion disagrees with final step")
            last = self.steps[-1]
            reason = (
                "success"
                if last.success
                else "environment_terminal"
                if last.terminated
                else "budget_exhausted"
            )
            if self.termination_reason != reason or (
                not last.terminated and len(self.steps) != self.total_budget
            ):
                raise SchemaError("completion termination/budget mismatch")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["steps"] = list(value["steps"])
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AgentTrajectory:
        try:
            if not isinstance(value, dict) or not isinstance(value.get("steps"), list):
                raise SchemaError("agent trajectory requires an object and steps array")
            data = dict(value)
            data["steps"] = tuple(AgentStep(**step) for step in data["steps"])
            return cls(**data)
        except TypeError as exc:
            raise SchemaError("invalid agent trajectory fields") from exc
