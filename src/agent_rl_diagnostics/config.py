"""Versioned stdlib JSON configuration; relative paths belong to the config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts.io import canonical_hash, read_json
from .contracts.schema import SchemaError, integer, text
from .environments.frozenlake import GridTask


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    policy: str

    @property
    def checkpoint_id(self) -> str:
        return f"scripted:{self.policy}:v1"


@dataclass(frozen=True)
class BudgetConfig:
    tasks: tuple[GridTask, ...]
    models: tuple[ModelSpec, ...]
    budgets: tuple[int, ...]
    ks: tuple[int, ...]
    samples_per_task: int
    seed: int
    bootstrap_replicates: int
    output: Path | None
    raw: dict[str, Any]

    @property
    def config_hash(self) -> str:
        # Output location does not change the scientific job identity.
        return canonical_hash({k: v for k, v in self.raw.items() if k != "output"})

    @property
    def estimated_action_upper_bound(self) -> int:
        return len(self.tasks) * len(self.models) * self.samples_per_task * sum(self.budgets)


def parse_config(value: Any, *, base_directory: Path) -> BudgetConfig:
    if not isinstance(value, dict):
        raise SchemaError("config must be a JSON object")
    allowed = {
        "schema_version",
        "protocol_id",
        "evidence_kind",
        "environment",
        "budget_mode",
        "tasks",
        "models",
        "budgets",
        "ks",
        "samples_per_task",
        "seed",
        "bootstrap_replicates",
        "output",
    }
    if set(value) - allowed:
        raise SchemaError(f"unknown config keys: {sorted(set(value) - allowed)}")
    if value.get("schema_version") != "ard-budget-config-v1":
        raise SchemaError("config.schema_version must be ard-budget-config-v1")
    if value.get("protocol_id") != "toolkit_v1" or value.get("evidence_kind") != "demo_synthetic":
        raise SchemaError("scripted demo runner requires toolkit_v1 / demo_synthetic")
    if value.get("environment") != "frozenlake_deterministic":
        raise SchemaError("this runner supports frozenlake_deterministic")
    if value.get("budget_mode") != "budget_aware_rerollout":
        raise SchemaError("runner independently samples each T; requires budget_aware_rerollout")
    for name in ("tasks", "models", "budgets", "ks"):
        if not isinstance(value.get(name), list) or not value[name]:
            raise SchemaError(f"{name} must be a nonempty array")
    tasks = []
    for item in value["tasks"]:
        if not isinstance(item, dict) or set(item) != {"task_id", "desc"}:
            raise SchemaError("task needs exactly task_id and desc")
        text(item["task_id"], "task_id")
        if not isinstance(item["desc"], list) or not all(isinstance(r, str) for r in item["desc"]):
            raise SchemaError("desc must be an array of strings")
        tasks.append(GridTask(item["task_id"], tuple(item["desc"])))
    models = []
    for item in value["models"]:
        if not isinstance(item, dict) or set(item) != {"model_id", "policy"}:
            raise SchemaError("model needs exactly model_id and policy")
        text(item["model_id"], "model_id")
        text(item["policy"], "policy")
        if item["policy"] not in {"random", "shortest_path"}:
            raise SchemaError("demo policy must be random or shortest_path")
        models.append(ModelSpec(item["model_id"], item["policy"]))
    for name, sequence in (
        ("task IDs", [t.task_id for t in tasks]),
        ("map contents", [t.map_hash for t in tasks]),
        ("model IDs", [m.model_id for m in models]),
        ("budgets", value["budgets"]),
        ("ks", value["ks"]),
    ):
        try:
            unique = len(set(sequence)) == len(sequence)
        except TypeError as exc:
            raise SchemaError(f"invalid values in {name}") from exc
        if not unique:
            raise SchemaError(f"{name} must be unique")
    budgets = tuple(integer(x, "budget", 1) for x in value["budgets"])
    ks = tuple(integer(x, "K", 1) for x in value["ks"])
    samples = integer(value.get("samples_per_task"), "samples_per_task", 1)
    if samples < max(ks):
        raise SchemaError("samples_per_task must be >= max(K)")
    seed = integer(value.get("seed", 0), "seed")
    replicates = integer(value.get("bootstrap_replicates", 1000), "bootstrap_replicates", 1)
    output = value.get("output")
    if output is not None:
        text(output, "output")
        output = (base_directory / output).resolve()
    return BudgetConfig(
        tuple(tasks),
        tuple(models),
        budgets,
        ks,
        samples,
        seed,
        replicates,
        output,
        dict(value),
    )


def load_config(path: Path) -> BudgetConfig:
    path = path.resolve()
    return parse_config(read_json(path), base_directory=path.parent)
