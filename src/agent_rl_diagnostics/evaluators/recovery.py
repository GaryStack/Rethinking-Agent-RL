from __future__ import annotations

import time
from pathlib import Path
from statistics import fmean
from typing import Any

from ..agent_config import AgentConfig
from ..artifacts.io import canonical_hash, new_output_directory
from ..contracts.agent import AgentTrajectory
from ..contracts.schema import SchemaError, integer
from ..metrics.cluster import cluster_ci
from ..models import ChatModel, create_model
from ..pools.frozen import load_pool
from .agent import finalize_artifacts, read_records, rollout
from .fixed import check_pool_config


def summarize_recovery(
    records: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    *,
    samples: int,
    model_id: str,
    bootstrap_replicates: int = 1000,
    bootstrap_seed: int = 5422,
) -> dict[str, Any]:
    by_condition = {
        (pair["pair_id"], condition): [] for pair in pairs for condition in ("clean", "error")
    }
    seen = set()
    for record in records:
        key = (record.get("pair_id"), record.get("condition"))
        trajectory = AgentTrajectory.from_dict(record["trajectory"])
        slot = (*key, trajectory.sample_id)
        if (
            key not in by_condition
            or slot in seen
            or trajectory.sample_id >= samples
            or trajectory.model_id != model_id
        ):
            raise SchemaError("duplicate or unexpected recovery sample slot/model")
        seen.add(slot)
        by_condition[key].append(trajectory)
    rows, successful_steps = [], {"clean": [], "error": []}
    for pair in pairs:
        row: dict[str, Any] = {
            "pair_id": pair["pair_id"],
            "task_id": pair["task_id"],
            "error_recoverable": pair["severity"]["error_recoverable"],
        }
        for condition in ("clean", "error"):
            values = by_condition[(pair["pair_id"], condition)]
            valid = [r for r in values if r.status == "completed"]
            row[condition + "_n"] = len(valid)
            row[condition + "_success"] = (
                sum(r.success is True for r in valid) / samples if len(valid) == samples else None
            )
            row[condition + "_infra_failures"] = len(values) - len(valid)
            row[condition + "_missing_slots"] = samples - len(values)
            successful_steps[condition].extend(len(r.steps) for r in valid if r.success)
        row["paired_delta_error_minus_clean"] = (
            row["error_success"] - row["clean_success"]
            if (row["error_success"] is not None and row["clean_success"] is not None)
            else None
        )
        rows.append(row)
    complete = all(row["clean_n"] == samples and row["error_n"] == samples for row in rows)
    return {
        "schema_version": "ard-recovery-summary-v1",
        "kind": "recovery",
        "complete": complete,
        "pair_count": len(pairs),
        "samples_per_condition": samples,
        "aggregation": "equal_intervention_pairs",
        "model_id": model_id,
        "clean_success": fmean(r["clean_success"] for r in rows) if complete else None,
        "error_success": fmean(r["error_success"] for r in rows) if complete else None,
        "paired_delta_error_minus_clean": fmean(r["paired_delta_error_minus_clean"] for r in rows)
        if complete
        else None,
        "error_recoverable_fraction": fmean(r["error_recoverable"] for r in rows),
        "paired_delta_ci": cluster_ci(
            rows,
            "paired_delta_error_minus_clean",
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "successful_continuation_actions": {
            c: fmean(v) if v else None for c, v in successful_steps.items()
        },
        "per_pair": rows,
    }


def run_recovery(
    config: AgentConfig,
    pool: Path,
    output: Path,
    *,
    samples: int = 1,
    model: ChatModel | None = None,
) -> dict[str, Any]:
    integer(samples, "samples", 1)
    manifest, states, tasks = load_pool(pool)
    if manifest["kind"] != "error_pool":
        raise SchemaError("recovery evaluation requires an error pool")
    check_pool_config(config, manifest, states, tasks)
    pairs = read_records(pool / "pairs.jsonl")
    by_id = {state["state_id"]: state for state in states}
    tick = time.perf_counter()
    job_hash = canonical_hash([config.config_hash, manifest["pool_hash"], samples, "recovery_v1"])
    run_id = f"recovery-{job_hash[:16]}"
    with new_output_directory(output) as staging:
        backend = model or create_model(config.model, config.sampling)
        executor = config.executor() if config.environment == "deepcoder_stdin" else None
        if executor:
            executor.preflight()
        records = []
        for pair in pairs:
            for condition in ("clean", "error"):
                state = by_id[pair[condition + "_state_id"]]
                for sample in range(samples):
                    trajectory = rollout(
                        config,
                        tasks[state["task_id"]],
                        backend,
                        run_id=run_id,
                        budget=state["remaining_budget"],
                        sample_id=sample,
                        executor=executor,
                        frozen=state,
                    )
                    records.append(
                        {
                            "pair_id": pair["pair_id"],
                            "condition": condition,
                            "state_id": state["state_id"],
                            "trajectory": trajectory.to_dict(),
                        }
                    )
        summary = summarize_recovery(
            records,
            pairs,
            samples=samples,
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
            kind="recovery",
            summary=summary,
            records=records,
            record_name="continuations.jsonl",
            model_metadata=backend.metadata,
            run_id=run_id,
            tick=tick,
            pool_hash=manifest["pool_hash"],
            extra={"samples_per_condition": samples},
        )
    return {
        "output": str(output),
        "run_id": run_id,
        "pairs": len(pairs),
        "clean_success": summary["clean_success"],
        "error_success": summary["error_success"],
    }
