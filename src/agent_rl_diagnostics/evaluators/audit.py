from __future__ import annotations

from pathlib import Path
from typing import Any

from ..agent_config import load_recorded_agent_config
from ..artifacts.io import canonical_hash, read_json
from ..contracts.agent import AgentStep, AgentTrajectory
from ..contracts.schema import SchemaError
from ..pools.frozen import load_pool
from .agent import make_environment, read_records, replay_trajectory
from .fixed import check_pool_config, summarize_fixed
from .recovery import summarize_recovery


def audit_fixed_or_recovery(
    directory: Path, manifest: dict[str, Any], pool: Path, *, replay_code: bool = False
) -> dict[str, Any]:
    config = load_recorded_agent_config(directory / "config.json")
    pool_manifest, states, tasks = load_pool(pool)
    check_pool_config(config, pool_manifest, states, tasks)
    if (
        manifest.get("config_hash") != config.config_hash
        or manifest.get("checkpoint_id") != config.checkpoint_id
        or manifest.get("state_pool_hash") != pool_manifest["pool_hash"]
        or manifest.get("dataset_hash") != canonical_hash(config.tasks)
        or manifest.get("environment") != config.environment
        or manifest.get("evidence_kind") != config.raw["evidence_kind"]
        or manifest.get("protocol_id") != config.raw["protocol_id"]
    ):
        raise SchemaError("evaluator config/checkpoint/pool identity mismatch")
    by_id = {state["state_id"]: state for state in states}
    executor = (
        config.executor() if config.environment == "deepcoder_stdin" and replay_code else None
    )
    if manifest["kind"] == "step_quality":
        records = read_records(directory / "actions.jsonl")
        samples = manifest["samples_per_state"]
        expected_run = (
            "step-"
            + canonical_hash([config.config_hash, pool_manifest["pool_hash"], samples, "step_v1"])[
                :16
            ]
        )
        for record in records:
            state = by_id.get(record.get("state_id"))
            if (
                state is None
                or record.get("run_id") != expected_run
                or record.get("checkpoint_id") != config.checkpoint_id
                or record.get("pool_hash") != pool_manifest["pool_hash"]
            ):
                raise SchemaError("fixed-state action identity mismatch")
            if record.get("status") == "infrastructure_failure":
                if record.get("step") is not None:
                    raise SchemaError("failed action cannot contain a completed step")
                continue
            if record.get("status") != "completed":
                raise SchemaError("unsupported fixed action status")
            step = AgentStep(**record["step"])
            if (
                step.turn_index != 0
                or step.remaining_budget != state["remaining_budget"]
                or step.snapshot_before != state["snapshot"]
                or step.messages != state["messages"]
                or step.generation.get("input_messages_hash") != canonical_hash(state["messages"])
            ):
                raise SchemaError("action was not evaluated from the frozen input")
            if config.environment == "frozenlake_deterministic" or replay_code:
                env = make_environment(config, tasks[state["task_id"]], executor)
                env.restore(state["snapshot"])
                result = env.apply(step.response, step.remaining_budget)
                if env.snapshot() != step.snapshot_after or any(
                    result[key] != getattr(step, key)
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
                    raise SchemaError("fixed action environment/score replay mismatch")
        scorer = (
            "frozenlake_q_star_gamma_0.95_v1"
            if config.environment == "frozenlake_deterministic"
            else "derived_stdin_test_fraction_v1"
        )
        expected = summarize_fixed(
            records,
            states,
            samples=samples,
            scorer=scorer,
            model_id=config.model_id,
            bootstrap_replicates=config.raw["bootstrap_replicates"],
            bootstrap_seed=config.raw["seed"],
        )
    else:
        if pool_manifest["kind"] != "error_pool":
            raise SchemaError("recovery audit requires the original error pool")
        pairs = read_records(pool / "pairs.jsonl")
        by_pair = {pair["pair_id"]: pair for pair in pairs}
        records = read_records(directory / "continuations.jsonl")
        samples = manifest["samples_per_condition"]
        expected_run = (
            "recovery-"
            + canonical_hash(
                [config.config_hash, pool_manifest["pool_hash"], samples, "recovery_v1"]
            )[:16]
        )
        for record in records:
            pair = by_pair.get(record.get("pair_id"))
            condition = record.get("condition")
            if (
                pair is None
                or condition not in {"clean", "error"}
                or record.get("state_id") != pair[condition + "_state_id"]
            ):
                raise SchemaError("recovery pair/condition identity mismatch")
            state = by_id[record["state_id"]]
            trajectory = AgentTrajectory.from_dict(record["trajectory"])
            if (
                trajectory.run_id != expected_run
                or trajectory.checkpoint_id != config.checkpoint_id
                or trajectory.environment != config.environment
                or trajectory.evidence_kind != config.raw["evidence_kind"]
                or trajectory.task_id != state["task_id"]
                or trajectory.initial_messages != state["messages"]
                or trajectory.initial_snapshot != state["snapshot"]
                or trajectory.prefix_depth != state["prefix_depth"]
                or trajectory.total_budget != state["remaining_budget"]
            ):
                raise SchemaError("continuation did not start from the matched frozen state")
            if config.environment == "frozenlake_deterministic" or replay_code:
                replay_trajectory(config, trajectory, tasks[state["task_id"]], executor)
        expected = summarize_recovery(
            records,
            pairs,
            samples=samples,
            model_id=config.model_id,
            bootstrap_replicates=config.raw["bootstrap_replicates"],
            bootstrap_seed=config.raw["seed"],
        )
    # Early v1 summaries predate additive CI fields. Recompute all point/health
    # measurements; validate CIs when present, do not invent them in old artifacts.
    stored = read_json(directory / "summary.json")
    for optional in ("r_single_ci", "paired_delta_ci"):
        if optional not in stored:
            expected.pop(optional, None)
    if "paired_delta_ci" not in stored:
        old_rows = {row["pair_id"]: row for row in stored.get("per_pair", [])}
        for row in expected.get("per_pair", []):
            if "task_id" not in old_rows.get(row["pair_id"], {}):
                row.pop("task_id", None)
    expected.update(
        pool_hash=pool_manifest["pool_hash"],
        config_hash=config.config_hash,
        evidence_kind=config.raw["evidence_kind"],
        environment=config.environment,
    )
    summary = read_json(directory / "summary.json")
    if manifest.get("run_id") != expected_run or summary != expected:
        raise SchemaError("fixed/recovery run identity or summary recomputation mismatch")
    return summary
