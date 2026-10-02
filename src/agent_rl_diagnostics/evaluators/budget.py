from __future__ import annotations

import csv
import json
import platform
import random
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Iterator

from .. import __version__
from ..artifacts.io import canonical_hash, file_hash, git_identity, new_output_directory, write_json
from ..config import BudgetConfig
from ..contracts import SCHEMA_VERSION, StepRecord, Trajectory
from ..environments.frozenlake import GridEnvironment, finite_horizon_q, scripted_action
from ..metrics import paired_bootstrap_delta, summarize_budget
from ..reporting.html import render_report


def generate_trajectories(config: BudgetConfig, run_id: str) -> Iterator[Trajectory]:
    for model in config.models:
        for task in config.tasks:
            q_table = finite_horizon_q(task, max(config.budgets), gamma=0.95)
            for budget in config.budgets:
                for sample_id in range(config.samples_per_task):
                    seed = int(
                        canonical_hash(
                            [config.seed, model.model_id, task.task_id, budget, sample_id]
                        )[:16],
                        16,
                    )
                    rng = random.Random(seed)
                    env = GridEnvironment(task)
                    steps = []
                    for turn in range(budget):
                        remaining = budget - turn
                        state = env.state
                        action = scripted_action(model.policy, task, state, remaining, rng)
                        nxt, reward, terminal, success = env.step(action)
                        steps.append(
                            StepRecord(
                                turn,
                                remaining,
                                state,
                                action,
                                nxt,
                                reward,
                                terminal,
                                success,
                                q_table[remaining][state][action],
                                "frozenlake_q_star_gamma_0.95_v1",
                            )
                        )
                        if terminal:
                            break
                    last = steps[-1]
                    reason = (
                        "success"
                        if last.success
                        else "environment_terminal"
                        if last.terminated
                        else "budget_exhausted"
                    )
                    yield Trajectory(
                        run_id,
                        model.model_id,
                        model.checkpoint_id,
                        task.task_id,
                        f"{task.task_id}:t{budget}:s{sample_id}",
                        sample_id,
                        budget,
                        "budget_aware_rerollout",
                        seed,
                        "completed",
                        last.success,
                        reason,
                        tuple(steps),
                        "demo_synthetic",
                        "toolkit_v1",
                    )


def build_summary(config: BudgetConfig, trajectories: list[Trajectory]) -> dict[str, Any]:
    summary = summarize_budget(
        trajectories,
        models=[m.model_id for m in config.models],
        tasks=[t.task_id for t in config.tasks],
        budgets=config.budgets,
        ks=config.ks,
        expected_samples=config.samples_per_task,
        seed=config.seed,
        replicates=config.bootstrap_replicates,
    )
    certificates = {task.task_id: asdict(task.certificate()) for task in config.tasks}
    for certificate in certificates.values():
        certificate["assumptions"] = list(certificate["assumptions"])
    by_task = summary["per_task"]
    for cell in summary["cells"]:
        feasible = {
            task_id
            for task_id, cert in certificates.items()
            if cert["value"] is not None and cert["value"] <= cell["total_budget"]
        }
        rows = [
            r
            for r in by_task
            if r["model_id"] == cell["model_id"]
            and r["total_budget"] == cell["total_budget"]
            and r["k"] == cell["k"]
            and r["task_id"] in feasible
        ]
        cell["structural_success_ceiling"] = len(feasible) / len(config.tasks)
        cell["exact_min_coverage"] = 1.0
        cell["feasible_tasks"] = len(feasible)
        cell["feasible_subset_pass_at_k"] = (
            fmean(r["pass_at_k"] for r in rows)
            if rows and all(r["n"] == config.samples_per_task for r in rows)
            else None
        )
    comparisons = []
    reference = config.models[0].model_id
    for candidate in config.models[1:]:
        for budget in config.budgets:
            for k in config.ks:
                pairs = {}
                for model in (reference, candidate.model_id):
                    pairs[model] = {
                        r["task_id"]: r["pass_at_k"]
                        for r in by_task
                        if r["model_id"] == model
                        and r["total_budget"] == budget
                        and r["k"] == k
                        and r["n"] == config.samples_per_task
                    }
                if len(pairs[reference]) == len(config.tasks) and len(
                    pairs[candidate.model_id]
                ) == len(config.tasks):
                    delta = paired_bootstrap_delta(
                        pairs[reference],
                        pairs[candidate.model_id],
                        seed=config.seed,
                        replicates=config.bootstrap_replicates,
                    )
                else:
                    delta = None
                comparisons.append(
                    {
                        "reference": reference,
                        "candidate": candidate.model_id,
                        "total_budget": budget,
                        "k": k,
                        "delta": delta,
                    }
                )
    summary.update(
        {
            "schema_version": "ard-budget-summary-v1",
            "evidence_kind": "demo_synthetic",
            "protocol_id": "toolkit_v1",
            "budget_mode": "budget_aware_rerollout",
            "config_hash": config.config_hash,
            "certificates": certificates,
            "comparisons": comparisons,
        }
    )
    return summary


def run_budget(config: BudgetConfig, output: Path, *, project_root: Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    tick = time.perf_counter()
    run_id = f"budget-{config.config_hash[:16]}"
    with new_output_directory(output) as staging:
        records = list(generate_trajectories(config, run_id))
        summary = build_summary(config, records)
        write_json(staging / "config.json", config.raw)
        write_json(staging / "summary.json", summary)
        with (staging / "trajectories.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record.to_dict(), sort_keys=True, allow_nan=False) + "\n")
        with (staging / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary["cells"][0]))
            writer.writeheader()
            writer.writerows(summary["cells"])
        artifact_names = ("config.json", "summary.json", "trajectories.jsonl", "metrics.csv")
        manifest = {
            "schema_version": "ard-manifest-v1",
            "trajectory_schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "status": "complete",
            "evidence_kind": "demo_synthetic",
            "protocol_id": "toolkit_v1",
            "config_hash": config.config_hash,
            "protocol_hash": canonical_hash(
                {
                    "protocol": "toolkit_v1",
                    "runner": __version__,
                    "budget_mode": "budget_aware_rerollout",
                }
            ),
            "dataset_hash": canonical_hash([asdict(task) for task in config.tasks]),
            "state_pool_hash": None,
            "models": [
                {
                    "model_id": m.model_id,
                    "checkpoint_id": m.checkpoint_id,
                    "kind": "scripted_policy",
                }
                for m in config.models
            ],
            "sampling": {
                "seed": config.seed,
                "independent_rng_per_slot": True,
                "budget_supplied_each_action": True,
                "temperature": None,
                "top_p": None,
            },
            "environment": {
                "name": "frozenlake_deterministic",
                "version": "ard-grid-v1",
                "official_benchmark": False,
            },
            "dependencies": {
                "python": platform.python_version(),
                "package": __version__,
                "runtime": "stdlib_only",
            },
            "source": git_identity(project_root),
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "cost": {
                "wall_seconds": time.perf_counter() - tick,
                "actions": sum(len(r.steps) for r in records),
                "trajectories": len(records),
                "tokens": None,
                "gpu_required": False,
                "network_requests": 0,
            },
            "artifacts": {name: file_hash(staging / name) for name in artifact_names},
        }
        (staging / "report.html").write_text(render_report(summary, manifest), encoding="utf-8")
        manifest["artifacts"]["report.html"] = file_hash(staging / "report.html")
        write_json(staging / "manifest.json", manifest)
    return {
        "run_id": run_id,
        "output": str(output),
        "report": str(output / "report.html"),
        "trajectories": len(records),
        "actions": manifest["cost"]["actions"],
        "evidence_kind": "demo_synthetic",
    }
