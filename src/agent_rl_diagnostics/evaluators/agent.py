from __future__ import annotations

import copy
import json
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from .. import __version__
from ..agent_config import AgentConfig, load_recorded_agent_config
from ..artifacts.io import canonical_hash, file_hash, new_output_directory, read_json, write_json
from ..contracts.agent import AgentStep, AgentTrajectory
from ..contracts.schema import SchemaError
from ..environments.deepcoder import DeepCoder
from ..environments.execution import ExecutorError, NamespaceExecutor
from ..environments.frozenlake import GridTask
from ..environments.text_grid import TextGrid
from ..metrics import summarize_budget
from ..models import BackendError, ChatModel, create_model
from ..reporting.diagnostic import render_diagnostic_report


def paired_seed(*values: Any) -> int:
    return int(canonical_hash(values)[:16], 16) % (2**31)


def make_environment(
    config: AgentConfig, task: dict[str, Any], executor: NamespaceExecutor | None = None
) -> TextGrid | DeepCoder:
    if config.environment == "frozenlake_deterministic":
        return TextGrid(task, prompt_profile=config.raw.get("grid_prompt_profile", "legacy_v1"))
    return DeepCoder(
        task,
        executor or config.executor(),
        prompt_profile=config.raw.get("coding_prompt_profile", "legacy_v1"),
    )


def rollout(
    config: AgentConfig,
    task: dict[str, Any],
    model: ChatModel,
    *,
    run_id: str,
    budget: int,
    sample_id: int,
    executor: NamespaceExecutor | None = None,
    frozen: dict[str, Any] | None = None,
) -> AgentTrajectory:
    env = make_environment(config, task, executor)
    prefix_depth = 0
    messages = env.initial_messages(budget)
    if frozen is not None:
        env.restore(frozen["snapshot"])
        messages = copy.deepcopy(frozen["messages"])
        prefix_depth = frozen["prefix_depth"]
        if budget != frozen["remaining_budget"]:
            raise SchemaError("continuation budget differs from frozen state")
    initial_snapshot = env.snapshot()
    initial_messages = copy.deepcopy(messages)
    seed = paired_seed(
        config.raw["seed"],
        task["task_id"],
        budget,
        sample_id,
        frozen["state_id"] if frozen else "initial",
    )
    steps: list[AgentStep] = []
    status, success, reason = "completed", False, "budget_exhausted"
    for index in range(budget):
        before = env.snapshot()
        remaining = budget - index
        try:
            generated = model.generate(messages, seed=paired_seed(seed, index))
            outcome = env.apply(generated.text, remaining)
        except (BackendError, ExecutorError):
            if config.raw.get("failure_policy", "fail") == "fail":
                raise
            status, success, reason = (
                "infrastructure_failure",
                None,
                "generation_or_executor_failure",
            )
            break
        steps.append(
            AgentStep(
                index,
                remaining,
                copy.deepcopy(messages),
                generated.text,
                generated.metadata,
                before,
                env.snapshot(),
                outcome["action"],
                outcome["observation"],
                outcome["reward"],
                outcome["terminated"],
                outcome["success"],
                outcome["local_score"],
                outcome["scorer"],
            )
        )
        messages.append({"role": "assistant", "content": generated.text})
        messages.append(
            {
                "role": "user",
                "content": env.budget_observation(
                    prefix_depth + budget,
                    remaining - 1,
                    outcome["observation"],
                ),
            }
        )
        if outcome["terminated"]:
            success = outcome["success"]
            reason = "success" if success else "environment_terminal"
            break
    return AgentTrajectory(
        run_id,
        config.model_id,
        config.checkpoint_id,
        task["task_id"],
        f"{task['task_id']}:t{budget}:s{sample_id}",
        sample_id,
        budget,
        "budget_aware_rerollout",
        seed,
        status,
        success,
        reason,
        tuple(steps),
        config.raw["evidence_kind"],
        "toolkit_v1",
        config.environment,
        initial_snapshot,
        initial_messages,
        prefix_depth,
    )


def summarize_agent_budget(config: AgentConfig, records: list[AgentTrajectory]) -> dict[str, Any]:
    summary = summarize_budget(
        records,
        models=[config.model_id],
        tasks=[t["task_id"] for t in config.tasks],
        budgets=config.raw["budgets"],
        ks=config.raw["ks"],
        expected_samples=config.raw["samples_per_task"],
        seed=config.raw["seed"],
        replicates=config.raw["bootstrap_replicates"],
    )
    certificates: dict[str, Any] = {}
    if config.environment == "frozenlake_deterministic":
        for task in config.tasks:
            certificate = asdict(GridTask(task["task_id"], tuple(task["desc"])).certificate())
            certificate["assumptions"] = list(certificate["assumptions"])
            certificates[task["task_id"]] = certificate
    for cell in summary["cells"]:
        feasible = {
            task_id
            for task_id, certificate in certificates.items()
            if certificate["value"] is not None and certificate["value"] <= cell["total_budget"]
        }
        cell["structural_success_ceiling"] = (
            len(feasible) / len(config.tasks) if certificates else None
        )
        cell["exact_min_coverage"] = len(certificates) / len(config.tasks)
        cell["feasible_tasks"] = len(feasible) if certificates else None
        rows = [
            row
            for row in summary["per_task"]
            if row["k"] == cell["k"]
            and row["total_budget"] == cell["total_budget"]
            and row["task_id"] in feasible
        ]
        cell["feasible_subset_pass_at_k"] = (
            fmean(row["pass_at_k"] for row in rows)
            if (rows and all(row["n"] == config.raw["samples_per_task"] for row in rows))
            else None
        )
    summary.update(
        schema_version="ard-agent-budget-summary-v1",
        kind="agent_budget",
        environment=config.environment,
        config_hash=config.config_hash,
        evidence_kind=config.raw["evidence_kind"],
        protocol_id="toolkit_v1",
        certificates=certificates,
        budget_mode="budget_aware_rerollout",
    )
    return summary


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def read_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def finalize_artifacts(
    directory: Path,
    *,
    config: AgentConfig,
    kind: str,
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    record_name: str,
    model_metadata: dict[str, Any],
    run_id: str,
    tick: float,
    pool_hash: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    write_json(directory / "config.json", config.raw)
    write_records(directory / record_name, records)
    write_json(directory / "summary.json", summary)
    manifest = {
        "schema_version": "ard-manifest-v1",
        "kind": kind,
        "run_id": run_id,
        "status": "complete"
        if summary.get(
            "complete", all(cell["status"] == "complete" for cell in summary.get("cells", []))
        )
        else "incomplete",
        "protocol_id": "toolkit_v1",
        "evidence_kind": config.raw["evidence_kind"],
        "paper_reproduction": False,
        "environment": config.environment,
        "config_hash": config.config_hash,
        "dataset_hash": canonical_hash(config.tasks),
        "state_pool_hash": pool_hash,
        "model": model_metadata,
        "checkpoint_id": config.checkpoint_id,
        "decoding": asdict(config.sampling),
        "dependencies": {"package": __version__, "python": platform.python_version()},
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "cost": {"wall_seconds": time.perf_counter() - tick, "records": len(records)},
        "sampling": {
            "seed": config.raw["seed"],
            "seed_policy": "checkpoint_independent_slot_seed",
            "seed_support": model_metadata.get("seed_support", "unknown"),
        },
        **(extra or {}),
    }
    (directory / "report.html").write_text(
        render_diagnostic_report(summary, manifest), encoding="utf-8"
    )
    names = ("config.json", record_name, "summary.json", "report.html")
    manifest["artifacts"] = {name: file_hash(directory / name) for name in names}
    write_json(directory / "manifest.json", manifest)
    return manifest


def run_agent_budget(
    config: AgentConfig, output: Path, model: ChatModel | None = None
) -> dict[str, Any]:
    tick = time.perf_counter()
    run_id = f"agent-budget-{config.config_hash[:16]}"
    with new_output_directory(output) as staging:
        backend = model or create_model(config.model, config.sampling)
        executor = config.executor() if config.environment == "deepcoder_stdin" else None
        if executor:
            executor.preflight()
        records = [
            rollout(
                config,
                task,
                backend,
                run_id=run_id,
                budget=budget,
                sample_id=sample,
                executor=executor,
            )
            for task in config.tasks
            for budget in config.raw["budgets"]
            for sample in range(config.raw["samples_per_task"])
        ]
        summary = summarize_agent_budget(config, records)
        manifest = finalize_artifacts(
            staging,
            config=config,
            kind="agent_budget",
            summary=summary,
            records=[r.to_dict() for r in records],
            record_name="trajectories.jsonl",
            model_metadata=backend.metadata,
            run_id=run_id,
            tick=tick,
            extra={"action_count": sum(len(r.steps) for r in records)},
        )
    return {
        "output": str(output),
        "report": str(output / "report.html"),
        "run_id": run_id,
        "status": manifest["status"],
        "trajectories": len(records),
        "actions": manifest["action_count"],
    }


def replay_trajectory(
    config: AgentConfig,
    record: AgentTrajectory,
    task: dict[str, Any],
    executor: NamespaceExecutor | None = None,
) -> None:
    env = make_environment(config, task, executor)
    if record.prefix_depth:
        env.restore(record.initial_snapshot)
    elif (
        env.snapshot() != record.initial_snapshot
        or env.initial_messages(record.total_budget) != record.initial_messages
    ):
        raise SchemaError("initial snapshot/prompt does not match environment")
    messages = copy.deepcopy(record.initial_messages)
    for step in record.steps:
        if step.messages != messages or step.snapshot_before != env.snapshot():
            raise SchemaError("trajectory history/snapshot replay mismatch")
        if step.generation.get("input_messages_hash") != canonical_hash(step.messages):
            raise SchemaError("generation input hash mismatch")
        outcome = env.apply(step.response, step.remaining_budget)
        if env.snapshot() != step.snapshot_after or any(
            outcome[key] != getattr(step, key)
            for key in (
                "action",
                "observation",
                "reward",
                "terminated",
                "success",
                "local_score",
                "scorer",
            )
        ):
            raise SchemaError("environment/action/score replay mismatch")
        messages += [
            {"role": "assistant", "content": step.response},
            {
                "role": "user",
                "content": env.budget_observation(
                    record.prefix_depth + record.total_budget,
                    step.remaining_budget - 1,
                    step.observation,
                ),
            },
        ]


def audit_agent_budget(
    directory: Path, manifest: dict[str, Any], *, replay_code: bool = False
) -> dict[str, Any]:
    config = load_recorded_agent_config(directory / "config.json")
    records = [
        AgentTrajectory.from_dict(value) for value in read_records(directory / "trajectories.jsonl")
    ]
    if (
        manifest.get("config_hash") != config.config_hash
        or manifest.get("checkpoint_id") != config.checkpoint_id
        or manifest.get("dataset_hash") != canonical_hash(config.tasks)
        or manifest.get("environment") != config.environment
        or manifest.get("evidence_kind") != config.raw["evidence_kind"]
        or manifest.get("protocol_id") != config.raw["protocol_id"]
    ):
        raise SchemaError("agent manifest/config/checkpoint mismatch")
    expected_run = f"agent-budget-{config.config_hash[:16]}"
    tasks = {t["task_id"]: t for t in config.tasks}
    executor = (
        config.executor() if config.environment == "deepcoder_stdin" and replay_code else None
    )
    for record in records:
        if (
            record.model_id != config.model_id
            or record.checkpoint_id != config.checkpoint_id
            or record.run_id != expected_run
        ):
            raise SchemaError("agent trajectory identity mismatch")
        if (
            record.prefix_depth
            or record.task_id not in tasks
            or record.environment != config.environment
            or record.evidence_kind != config.raw["evidence_kind"]
        ):
            raise SchemaError("agent trajectory task/environment/evidence mismatch")
        if config.environment == "frozenlake_deterministic" or replay_code:
            replay_trajectory(config, record, tasks[record.task_id], executor)
    summary = read_json(directory / "summary.json")
    if summary != summarize_agent_budget(config, records):
        raise SchemaError("agent metrics differ from trajectory recomputation")
    return summary
