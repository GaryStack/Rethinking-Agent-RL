from __future__ import annotations

import copy
import time
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any

from ..agent_config import AgentConfig
from ..artifacts.io import canonical_hash, new_output_directory
from ..contracts.agent import AgentStep
from ..contracts.schema import SchemaError, integer
from ..environments.execution import ExecutorError
from ..metrics.cluster import cluster_ci
from ..models import BackendError, ChatModel, create_model
from ..pools.frozen import load_pool
from .agent import finalize_artifacts, make_environment, paired_seed


def check_pool_config(
    config: AgentConfig,
    manifest: dict[str, Any],
    states: list[dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
) -> None:
    if (
        manifest["environment"] != config.environment
        or manifest["evidence_kind"] != config.raw["evidence_kind"]
    ):
        raise SchemaError("pool and evaluator environment/evidence differ")
    config_tasks = {task["task_id"]: task for task in config.tasks}
    for state in states:
        if config_tasks.get(state["task_id"]) != tasks[state["task_id"]]:
            raise SchemaError("pool task differs from configured dataset")


def summarize_fixed(
    records: list[dict[str, Any]],
    states: list[dict[str, Any]],
    *,
    samples: int,
    scorer: str,
    model_id: str,
    bootstrap_replicates: int = 1000,
    bootstrap_seed: int = 5422,
) -> dict[str, Any]:
    by_state = {state["state_id"]: [] for state in states}
    seen = set()
    for record in records:
        key = (record.get("state_id"), record.get("sample_id"))
        if (
            key in seen
            or key[0] not in by_state
            or type(key[1]) is not int
            or not 0 <= key[1] < samples
        ):
            raise SchemaError("duplicate or unexpected fixed-state sample slot")
        seen.add(key)
        if record.get("model_id") != model_id:
            raise SchemaError("mixed models in fixed-state result")
        by_state[key[0]].append(record)
    rows = []
    for state in states:
        samples_here = by_state[state["state_id"]]
        valid = [row for row in samples_here if row["status"] == "completed"]
        scores = [row["step"]["local_score"] for row in valid]
        if any(row["step"]["scorer"] != scorer for row in valid):
            raise SchemaError("mixed scorer units")
        rows.append(
            {
                "state_id": state["state_id"],
                "task_id": state["task_id"],
                "depth": state["prefix_depth"],
                "remaining_budget": state["remaining_budget"],
                "n": len(valid),
                "expected_samples": samples,
                "missing_slots": samples - len(samples_here),
                "infra_failures": len(samples_here) - len(valid),
                "score_mean": fmean(scores) if len(valid) == samples else None,
                "observed_score_mean": fmean(scores) if scores else None,
            }
        )
    complete = all(row["n"] == samples for row in rows)
    return {
        "schema_version": "ard-step-summary-v1",
        "kind": "step_quality",
        "scorer": scorer,
        "model_id": model_id,
        "aggregation": "actions_then_equal_states",
        "complete": complete,
        "r_single": fmean(row["score_mean"] for row in rows) if complete else None,
        "r_single_ci": cluster_ci(
            rows, "score_mean", replicates=bootstrap_replicates, seed=bootstrap_seed
        ),
        "state_count": len(states),
        "samples_per_state": samples,
        "valid_samples": sum(row["n"] for row in rows),
        "per_state": rows,
    }


def run_fixed(
    config: AgentConfig,
    pool: Path,
    output: Path,
    *,
    samples: int = 8,
    model: ChatModel | None = None,
) -> dict[str, Any]:
    integer(samples, "samples", 1)
    manifest, states, tasks = load_pool(pool)
    check_pool_config(config, manifest, states, tasks)
    tick = time.perf_counter()
    job_hash = canonical_hash([config.config_hash, manifest["pool_hash"], samples, "step_v1"])
    run_id = f"step-{job_hash[:16]}"
    with new_output_directory(output) as staging:
        backend = model or create_model(config.model, config.sampling)
        executor = config.executor() if config.environment == "deepcoder_stdin" else None
        if executor:
            executor.preflight()
        records = []
        scorer = make_environment(config, tasks[states[0]["task_id"]], executor).scorer
        for state in states:
            for sample in range(samples):
                env = make_environment(config, tasks[state["task_id"]], executor)
                env.restore(state["snapshot"])
                seed = paired_seed(config.raw["seed"], state["state_id"], sample)
                record = {
                    "run_id": run_id,
                    "state_id": state["state_id"],
                    "sample_id": sample,
                    "model_id": config.model_id,
                    "checkpoint_id": config.checkpoint_id,
                    "seed": seed,
                    "pool_hash": manifest["pool_hash"],
                    "status": "completed",
                    "step": None,
                }
                try:
                    generated = backend.generate(state["messages"], seed=seed)
                    outcome = env.apply(generated.text, state["remaining_budget"])
                    step = AgentStep(
                        0,
                        state["remaining_budget"],
                        copy.deepcopy(state["messages"]),
                        generated.text,
                        generated.metadata,
                        state["snapshot"],
                        env.snapshot(),
                        outcome["action"],
                        outcome["observation"],
                        outcome["reward"],
                        outcome["terminated"],
                        outcome["success"],
                        outcome["local_score"],
                        outcome["scorer"],
                    )
                    record["step"] = asdict(step)
                except (BackendError, ExecutorError):
                    if config.raw.get("failure_policy", "fail") == "fail":
                        raise
                    record["status"] = "infrastructure_failure"
                records.append(record)
        summary = summarize_fixed(
            records,
            states,
            samples=samples,
            scorer=scorer,
            model_id=config.model_id,
            bootstrap_replicates=config.raw["bootstrap_replicates"],
            bootstrap_seed=config.raw["seed"],
        )
        summary.update(
            pool_hash=manifest["pool_hash"],
            config_hash=config.config_hash,
            evidence_kind=config.raw["evidence_kind"],
            environment=config.environment,
        )
        finalize_artifacts(
            staging,
            config=config,
            kind="step_quality",
            summary=summary,
            records=records,
            record_name="actions.jsonl",
            model_metadata=backend.metadata,
            run_id=run_id,
            tick=tick,
            pool_hash=manifest["pool_hash"],
            extra={"samples_per_state": samples, "state_ids": [s["state_id"] for s in states]},
        )
    return {
        "output": str(output),
        "run_id": run_id,
        "r_single": summary["r_single"],
        "scorer": scorer,
        "states": len(states),
        "actions": len(records),
    }
