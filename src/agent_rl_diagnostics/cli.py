from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from importlib.resources import files
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .agent_config import load_agent_config
from .artifacts.io import (
    canonical_hash,
    file_hash,
    new_output_directory,
    read_json,
    verify_artifacts,
    write_json,
)
from .config import load_config, parse_config
from .contracts import SchemaError, Trajectory
from .contracts.schema import integer
from .environments.execution import ExecutorError
from .environments.frozenlake import GridEnvironment, finite_horizon_q
from .evaluators.agent import audit_agent_budget, run_agent_budget
from .evaluators.audit import audit_fixed_or_recovery
from .evaluators.budget import build_summary, run_budget
from .evaluators.fixed import check_pool_config, run_fixed
from .evaluators.recovery import run_recovery
from .integrations.checkpoint import (
    audit_checkpoint_validation,
    checkpoint_identity,
    run_checkpoint_validation,
)
from .integrations.tracking import DiagnosticTracker
from .models import BackendError
from .pools.errors import build_error_pool
from .pools.frozen import build_pool, load_pool
from .pools.witness import build_witness_pool
from .reporting.diagnostic import render_diagnostic_report
from .reporting.dynamics import report_dynamics
from .reporting.html import render_report
from .training.plan import audit_plan, create_plan


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise SchemaError(message)


def _parser() -> argparse.ArgumentParser:
    parser = Parser(
        description="Budget, fixed-state quality and recovery diagnostics for multi-turn agents"
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    traces = sub.add_parser("traces", help="Import and compare caller-attested external JSONL")
    trace_commands = traces.add_subparsers(dest="trace_command", required=True)
    trace_import = trace_commands.add_parser("import")
    trace_import.add_argument("--input", type=Path, required=True)
    trace_import.add_argument("--design", type=Path, required=True)
    trace_import.add_argument("--mapping", type=Path)
    trace_import.add_argument("--label", required=True)
    trace_import.add_argument("--output", type=Path, required=True)
    trace_inspect = trace_commands.add_parser(
        "inspect", help="Import a local Inspect .eval/.json log"
    )
    trace_inspect.add_argument("--input", type=Path, required=True)
    trace_inspect.add_argument("--design", type=Path, required=True)
    trace_inspect.add_argument("--scorer", required=True)
    trace_inspect.add_argument("--label", required=True)
    trace_inspect.add_argument("--output", type=Path, required=True)
    trace_audit = trace_commands.add_parser("audit")
    trace_audit.add_argument("--input", type=Path, required=True)
    trace_compare = trace_commands.add_parser("compare")
    trace_compare.add_argument("--base", type=Path, required=True)
    trace_compare.add_argument("--candidate", type=Path, required=True)
    trace_compare.add_argument("--output", type=Path, required=True)
    trace_compare.add_argument("--seed", type=int, default=0)
    trace_compare.add_argument(
        "--min-delta",
        type=float,
        help="Minimum candidate-minus-base success-rate change; exit 1 on failure",
    )
    walkthrough = sub.add_parser("toolkit-demo", help="Offline agent-v2 toolkit walkthrough")
    walkthrough.add_argument("--output", type=Path, required=True)
    likelihood = sub.add_parser("likelihood", help="Teacher-forced NLL/LLR mechanism analysis")
    likelihood_commands = likelihood.add_subparsers(dest="likelihood_command", required=True)
    score = likelihood_commands.add_parser(
        "score", help="Score frozen responses under a local HF checkpoint"
    )
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--model", type=Path, required=True)
    score.add_argument("--model-id", required=True)
    score.add_argument("--device", default="cpu")
    score.add_argument("--max-length", type=int, default=4096)
    score.add_argument("--failure-policy", choices=["error", "record"], required=True)
    score.add_argument("--output", type=Path, required=True)
    analyze = likelihood_commands.add_parser(
        "analyze", help="Pair policy scores and length controls"
    )
    analyze.add_argument("--base", type=Path, required=True)
    analyze.add_argument("--rl", type=Path, required=True)
    analyze.add_argument("--token-caliper", type=float, required=True)
    analyze.add_argument("--step-caliper", type=int, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    compare = sub.add_parser("compare", help="Audit runs and plot matched Base/RL summaries")
    compare.add_argument("--input", type=Path, nargs="+", required=True)
    compare.add_argument("--labels", nargs="+", required=True)
    compare.add_argument("--pool", type=Path)
    compare.add_argument("--output", type=Path, required=True)
    doctor = sub.add_parser("doctor", help="Inspect the CPU core and optional dependencies")
    doctor.add_argument("--json", action="store_true", help="Emit machine-readable status")
    demo = sub.add_parser("demo", help="Run a synthetic scripted-policy demo without network/GPU")
    demo.add_argument("--offline", required=True, action="store_true")
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--dry-run", action="store_true")
    mix = sub.add_parser("budgetmix", help="Plan/audit occurrence budgets without training")
    mix_commands = mix.add_subparsers(dest="mix_command", required=True)
    mix_plan = mix_commands.add_parser("plan")
    mix_plan.add_argument(
        "--task-ids",
        type=Path,
        required=True,
        help="JSON list; duplicates represent distinct occurrences",
    )
    mix_plan.add_argument("--tmax", type=int, required=True)
    mix_plan.add_argument("--seed", type=int, default=5422)
    mix_plan.add_argument("--update", type=int, default=1)
    mix_plan.add_argument("--group-size", type=int, default=8)
    mix_plan.add_argument("--output", type=Path, required=True)
    mix_audit = mix_commands.add_parser("audit")
    mix_audit.add_argument("--input", type=Path, required=True)
    dynamics = sub.add_parser(
        "dynamics", help="Audit raw checkpoint series and draw CPU-only curves"
    )
    dynamics.add_argument("--run", type=Path, required=True)
    dynamics.add_argument("--expected-steps", type=int, required=True)
    dynamics.add_argument("--output", type=Path, required=True)
    dynamics.add_argument("--verify-weights", action="store_true")
    evaluate = sub.add_parser("eval", help="Run an implemented evaluator")
    evaluators = evaluate.add_subparsers(dest="evaluator", required=True)
    budget = evaluators.add_parser(
        "budget", help="Independently sample model or demo rollouts per budget"
    )
    budget.add_argument("--config", type=Path, required=True)
    budget.add_argument("--output", type=Path)
    budget.add_argument("--dry-run", action="store_true")
    checkpoint = evaluators.add_parser(
        "checkpoint",
        help="Bind multi-turn and fixed-state validation to one immutable local checkpoint",
    )
    checkpoint.add_argument("--config", type=Path, required=True)
    checkpoint.add_argument("--pool", type=Path, required=True)
    checkpoint.add_argument("--checkpoint", type=Path, required=True)
    checkpoint.add_argument("--checkpoint-step", type=int, required=True)
    checkpoint.add_argument("--samples", type=int, default=8)
    checkpoint.add_argument("--output", type=Path, required=True)
    checkpoint.add_argument("--tracking-dir", type=Path)
    checkpoint.add_argument("--wandb-offline", action="store_true")
    checkpoint.add_argument("--dry-run", action="store_true")
    for name, default_samples in (("step", 8), ("recovery", 1)):
        evaluator = evaluators.add_parser(name, help="Evaluate immutable frozen inputs")
        evaluator.add_argument("--config", type=Path, required=True)
        evaluator.add_argument("--pool", type=Path, required=True)
        evaluator.add_argument("--output", type=Path, required=True)
        evaluator.add_argument("--samples", type=int, default=default_samples)
        evaluator.add_argument("--dry-run", action="store_true")
    pool = sub.add_parser("pool", help="Build, audit and perturb frozen state pools")
    pool_commands = pool.add_subparsers(dest="pool_command", required=True)
    build = pool_commands.add_parser("build")
    build.add_argument("--input", type=Path, nargs="+", required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--kind", choices=["base_monitor", "matched"], default="base_monitor")
    build.add_argument("--base-model-id", required=True)
    build.add_argument("--rl-model-id")
    build.add_argument("--size", type=int, default=16)
    build.add_argument("--seed", type=int, default=5422)
    build.add_argument("--max-per-task", type=int, default=8)
    build.add_argument("--min-tasks", type=int, default=1)
    build.add_argument("--training-task-ids", type=Path)
    pool_audit = pool_commands.add_parser("audit")
    pool_audit.add_argument("--input", type=Path, required=True)
    errors = pool_commands.add_parser("errors")
    errors.add_argument("--input", type=Path, required=True)
    errors.add_argument("--output", type=Path, required=True)
    errors.add_argument("--seed", type=int, default=5422)
    errors.add_argument("--require-recoverable", action="store_true")
    errors.add_argument("--references", type=Path)
    witness = pool_commands.add_parser(
        "witness",
        help="Independent geometric prefixes for recovery probes; not Base-collected states",
    )
    witness.add_argument("--config", type=Path, required=True)
    witness.add_argument("--output", type=Path, required=True)
    witness.add_argument("--budget", type=int, required=True)
    witness.add_argument("--size", type=int, default=4)
    audit = sub.add_parser("audit", help="Verify run hashes, schema and metric recomputation")
    audit.add_argument("--input", type=Path, required=True)
    audit.add_argument("--pool", type=Path)
    audit.add_argument("--verify-weights", action="store_true")
    audit.add_argument(
        "--replay-code", action="store_true", help="Rerun coding programs in namespace isolation"
    )
    report = sub.add_parser("report", help="Regenerate a static report from a verified run")
    report.add_argument("--input", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--pool", type=Path)
    return parser


def _project_root() -> Path:
    # For a wheel there is no source checkout; git_identity will return null.
    return Path(__file__).resolve().parents[2]


def _audit_run(
    directory: Path,
    pool: Path | None = None,
    *,
    replay_code: bool = False,
    verify_weights: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = verify_artifacts(directory)
    if manifest.get("kind") == "checkpoint_validation":
        if pool is None:
            raise SchemaError("checkpoint audit requires --pool")
        point = audit_checkpoint_validation(
            directory, pool, replay_code=replay_code, verify_weights=verify_weights
        )
        child = read_json(directory / "budget" / "manifest.json")
        manifest.update(
            run_id=point["event_id"],
            environment=child["environment"],
            evidence_kind=child["evidence_kind"],
        )
        return manifest, point
    if manifest.get("kind") in {"agent_budget", "step_quality", "recovery"}:
        for name in ("config.json", "summary.json"):
            if name not in manifest["artifacts"]:
                raise SchemaError(f"manifest does not hash required artifact: {name}")
        record_name = {
            "agent_budget": "trajectories.jsonl",
            "step_quality": "actions.jsonl",
            "recovery": "continuations.jsonl",
        }[manifest["kind"]]
        if record_name not in manifest["artifacts"]:
            raise SchemaError("manifest does not hash raw evaluation records")
        if manifest["kind"] == "agent_budget":
            summary = audit_agent_budget(directory, manifest, replay_code=replay_code)
        else:
            if pool is None:
                raise SchemaError(
                    "fixed/recovery audit requires --pool pointing to the original immutable pool"
                )
            summary = audit_fixed_or_recovery(directory, manifest, pool, replay_code=replay_code)
        return manifest, summary
    if manifest.get("status") != "complete":
        raise SchemaError("run manifest is not complete")
    if (
        manifest.get("protocol_id") != "toolkit_v1"
        or manifest.get("evidence_kind") != "demo_synthetic"
    ):
        raise SchemaError("manifest protocol/evidence does not match runner")
    for name in ("config.json", "summary.json", "trajectories.jsonl"):
        if name not in manifest["artifacts"]:
            raise SchemaError(f"manifest does not hash required artifact: {name}")
    config = load_config(directory / "config.json")
    if config.config_hash != manifest.get("config_hash"):
        raise SchemaError("manifest/config identity mismatch")
    with (directory / "trajectories.jsonl").open(encoding="utf-8") as stream:
        records = [Trajectory.from_dict(json.loads(line)) for line in stream]
    expected_run_id = f"budget-{config.config_hash[:16]}"
    expected_checkpoints = {m.model_id: m.checkpoint_id for m in config.models}
    if manifest.get("run_id") != expected_run_id:
        raise SchemaError("unexpected run_id")
    tasks = {task.task_id: task for task in config.tasks}
    for record in records:
        if record.run_id != expected_run_id or record.checkpoint_id != expected_checkpoints.get(
            record.model_id
        ):
            raise SchemaError("trajectory does not match run/model identity")
        if record.protocol_id != "toolkit_v1" or record.evidence_kind != "demo_synthetic":
            raise SchemaError("trajectory protocol/evidence does not match config")
        if record.budget_mode != config.raw["budget_mode"]:
            raise SchemaError("trajectory budget_mode does not match config")
        task = tasks.get(record.task_id)
        if task is None:
            raise SchemaError("trajectory contains an unexpected task")
        env = GridEnvironment(task)
        q_table = finite_horizon_q(task, record.total_budget, gamma=0.95)
        for step in record.steps:
            if step.state != env.state:
                raise SchemaError("trajectory does not match environment initial/current state")
            expected_score = (
                q_table[step.remaining_budget][step.state][step.action] if step.action < 4 else None
            )
            if env.step(step.action) != (
                step.next_state,
                step.reward,
                step.terminated,
                step.success,
            ):
                raise SchemaError("trajectory transition does not match environment replay")
            if (
                step.local_score != expected_score
                or step.scorer != "frozenlake_q_star_gamma_0.95_v1"
            ):
                raise SchemaError("trajectory local score does not match Q* replay")
    summary = read_json(directory / "summary.json")
    if build_summary(config, records) != summary:
        raise SchemaError("stored metrics do not match recomputed trajectories")
    return manifest, summary


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "traces":
        from .external_traces import audit_traces, compare_traces, import_traces

        if args.trace_command == "inspect":
            from .integrations.inspect_logs import import_inspect

            return import_inspect(
                args.input,
                scorer=args.scorer,
                design=read_json(args.design),
                label=args.label,
                output=args.output,
            )
        if args.trace_command == "import":
            return import_traces(
                args.input,
                read_json(args.design),
                read_json(args.mapping) if args.mapping else {},
                args.label,
                args.output,
            )
        if args.trace_command == "audit":
            _, _, summary = audit_traces(args.input)
            return {
                "status": "verified",
                "checks": ["sha256", "schema", "metric_recomputation"],
                "environment_replayed": False,
                "summary": summary,
            }
        return compare_traces(
            args.base, args.candidate, args.output, min_delta=args.min_delta, seed=args.seed
        )
    if args.command == "toolkit-demo":
        from .toolkit_demo import run_toolkit_demo

        return run_toolkit_demo(args.output)
    if args.command == "compare":
        from .reporting.comparison import compare_summaries, render_analysis

        audited = [_audit_run(path.resolve(), args.pool) for path in args.input]
        protocols = [
            canonical_hash(
                {
                    key: value
                    for key, value in read_json(path / "config.json").items()
                    if key not in {"model", "models", "output"}
                }
            )
            for path in args.input
        ]
        if len(set(protocols)) != 1:
            raise SchemaError(
                "comparison requires identical evaluation configuration except model/output"
            )
        report = compare_summaries([summary for _, summary in audited], labels=args.labels)
        report["sources"] = [
            {"label": label, "manifest_hash": file_hash(path / "manifest.json")}
            for label, path in zip(args.labels, args.input, strict=True)
        ]
        with new_output_directory(args.output.resolve()) as output:
            write_json(output / "summary.json", report)
            (output / "report.html").write_text(render_analysis(report), encoding="utf-8")
        return {"output": str(args.output.resolve()), "source_verified": True}
    if args.command == "likelihood":
        from .analysis.likelihood import length_control, pair_scores
        from .reporting.comparison import render_analysis

        if args.likelihood_command == "score":
            from .models.teacher_forcing import TeacherForcingScorer

            integer(args.max_length, "max_length", 2)
            events = read_json(args.input)
            if not isinstance(events, list) or not events:
                raise SchemaError("likelihood input must be a nonempty JSON event array")
            if any(
                not isinstance(event, dict)
                or not isinstance(event.get("event_id"), str)
                or not event["event_id"]
                for event in events
            ):
                raise SchemaError("every likelihood event requires a nonempty event_id")
            if len({event["event_id"] for event in events}) != len(events):
                raise SchemaError("duplicate likelihood event IDs")
            with new_output_directory(args.output.resolve()) as output:
                scorer = TeacherForcingScorer(
                    args.model,
                    model_id=args.model_id,
                    device=args.device,
                    max_length=args.max_length,
                )
                rows = []
                for event in events:
                    try:
                        rows.append(scorer.score(event))
                    except (ValueError, RuntimeError) as exc:
                        if args.failure_policy == "error":
                            raise
                        rows.append(
                            {
                                "event_id": event["event_id"],
                                "status": "scoring_failure",
                                **{
                                    name: event.get(name)
                                    for name in ("task_id", "trajectory_id", "source", "success")
                                },
                                "error": str(exc),
                                "model_id": args.model_id,
                            }
                        )
                write_json(output / "scores.json", rows)
                write_json(
                    output / "provenance.json",
                    {
                        "input_sha256": file_hash(args.input),
                        "model": str(args.model.resolve()),
                        "model_id": args.model_id,
                        "distribution": "raw_temperature_1_no_top_p",
                        "failure_policy": args.failure_policy,
                    },
                )
            return {
                "output": str(args.output.resolve()),
                "events": len(rows),
                "scoring_failures": sum(row["status"] != "ok" for row in rows),
            }
        report = pair_scores(read_json(args.base), read_json(args.rl))
        report["length_control"] = length_control(
            report["paired"],
            token_caliper=args.token_caliper,
            step_caliper=args.step_caliper,
            excluded_events=report["excluded"],
        )
        report["source_hashes"] = {"base": file_hash(args.base), "rl": file_hash(args.rl)}
        report["series"] = [
            {
                "label": row["source"],
                "metric": metric,
                "x": row["response_tokens"],
                "x_unit": "response_tokens",
                "value": row[metric],
            }
            for row in report["paired"]
            for metric in ("base_nll_per_token", "llr_per_token", "base_action_nll_per_token")
        ]
        with new_output_directory(args.output.resolve()) as output:
            write_json(output / "summary.json", report)
            (output / "report.html").write_text(render_analysis(report), encoding="utf-8")
        return {
            "output": str(args.output.resolve()),
            "paired": len(report["paired"]),
            "excluded": len(report["excluded"]),
        }
    if args.command == "dynamics":
        report = report_dynamics(
            args.run.resolve(),
            args.output.resolve(),
            expected_steps=args.expected_steps,
            verify_weights=args.verify_weights,
        )
        return {
            "status": report["status"],
            "output": str(args.output.resolve()),
            "raw_validation_audited": True,
            "weights_rehashed": args.verify_weights,
            "observed_adjacent_tradeoffs": len(report["observed_adjacent_tradeoffs"]),
        }
    if args.command == "budgetmix":
        if args.mix_command == "audit":
            return audit_plan(args.input.resolve())
        return create_plan(
            read_json(args.task_ids),
            tmax=args.tmax,
            seed=args.seed,
            update=args.update,
            group_size=args.group_size,
            output=args.output.resolve(),
        )
    if args.command == "doctor":
        return {
            "package": "agent-rl-diagnostics",
            "version": __version__,
            "python": platform.python_version(),
            "core": "ready",
            "runtime": "stdlib_only",
            "implemented": [
                "toolkit-demo",
                "demo",
                "eval budget",
                "pool build/audit/errors",
                "eval step",
                "eval recovery",
                "eval checkpoint",
                "pool witness",
                "budgetmix plan/audit",
                "dynamics",
                "compare",
                "likelihood score/analyze",
                "audit",
                "report",
            ],
            "optional_dependencies": {
                name: "available" if importlib.util.find_spec(name) else "not_installed"
                for name in ("torch", "ray", "vllm", "verl", "wandb")
            },
            "integration_status": {
                "verl": "single_rank_ray_vllm_smoke_verified_multirank_and_full_parity_pending",
                "tracking": "local_jsonl_optional_wandb_offline",
                "llm_rollout": "http_or_local_transformers",
                "coding_execution": "Linux_namespace_executor",
            },
        }
    if args.command == "pool":
        if args.pool_command == "witness":
            return build_witness_pool(
                load_agent_config(args.config),
                args.output.resolve(),
                budget=args.budget,
                size=args.size,
            )
        if args.pool_command == "audit":
            manifest, states, _ = load_pool(args.input.resolve())
            return {
                "status": "verified",
                "pool_hash": manifest["pool_hash"],
                "states": len(states),
                "kind": manifest["kind"],
            }
        if args.pool_command == "errors":
            return build_error_pool(
                args.input.resolve(),
                args.output.resolve(),
                seed=args.seed,
                require_recoverable=args.require_recoverable,
                references=args.references,
            )
        training_ids = read_json(args.training_task_ids) if args.training_task_ids else None
        if training_ids is not None and (
            not isinstance(training_ids, list) or not all(isinstance(t, str) for t in training_ids)
        ):
            raise SchemaError("training-task-ids must be a JSON array of strings")
        return build_pool(
            [path.resolve() for path in args.input],
            args.output.resolve(),
            kind=args.kind,
            size=args.size,
            seed=args.seed,
            base_model_id=args.base_model_id,
            rl_model_id=args.rl_model_id,
            max_per_task=args.max_per_task,
            min_tasks=args.min_tasks,
            training_task_ids=set(training_ids) if training_ids is not None else None,
        )
    if args.command == "eval" and args.evaluator == "checkpoint":
        config = load_agent_config(args.config)
        integer(args.checkpoint_step, "checkpoint_step")
        integer(args.samples, "samples", 1)
        if args.wandb_offline and args.tracking_dir is None:
            raise SchemaError("--wandb-offline requires --tracking-dir")
        if args.dry_run:
            if (
                config.raw["split"] not in {"validation", "smoke"}
                or config.model["backend"] != "transformers"
            ):
                raise SchemaError(
                    "checkpoint validation requires local Transformers and validation/smoke split"
                )
            identity = checkpoint_identity(args.checkpoint)
            manifest, states, tasks = load_pool(args.pool)
            if manifest["selection"].get("kind") != "base_monitor":
                raise SchemaError("training dynamics require a Base-only monitor pool")
            check_pool_config(config, manifest, states, tasks)
            return {
                "dry_run": True,
                "checkpoint": identity,
                "checkpoint_step": args.checkpoint_step,
                "state_pool_hash": manifest["pool_hash"],
                "model_load": False,
            }
        point = run_checkpoint_validation(
            config,
            args.pool.resolve(),
            args.checkpoint.resolve(),
            args.output.resolve(),
            checkpoint_step=args.checkpoint_step,
            samples=args.samples,
        )
        if args.tracking_dir:
            tracker = DiagnosticTracker(
                args.tracking_dir.resolve(), wandb_offline=args.wandb_offline
            )
            try:
                tracker.log(point)
            finally:
                tracker.close()
        return point
    if args.command == "eval" and args.evaluator in {"step", "recovery"}:
        config = load_agent_config(args.config)
        integer(args.samples, "samples", 1)
        if args.dry_run:
            manifest, states, tasks = load_pool(args.pool.resolve())
            check_pool_config(config, manifest, states, tasks)
            if args.evaluator == "recovery" and manifest["kind"] != "error_pool":
                raise SchemaError("recovery requires an error pool")
            return {
                "dry_run": True,
                "kind": args.evaluator,
                "pool_hash": manifest["pool_hash"],
                "states": len(states),
                "samples": args.samples,
                "generation_upper_bound": len(states)
                * args.samples
                * (1 if args.evaluator == "step" else max(s["remaining_budget"] for s in states)),
            }
        evaluator = run_fixed if args.evaluator == "step" else run_recovery
        return evaluator(config, args.pool.resolve(), args.output.resolve(), samples=args.samples)
    if args.command in {"demo", "eval"}:
        if args.command == "demo":
            value = json.loads(
                files("agent_rl_diagnostics").joinpath("resources/demo.json").read_text()
            )
            config = parse_config(value, base_directory=Path.cwd())
        else:
            value = read_json(args.config)
            if isinstance(value, dict) and value.get("schema_version") == "ard-agent-config-v1":
                agent_config = load_agent_config(args.config)
                output = args.output.resolve() if args.output else agent_config.output
                if output is None:
                    raise SchemaError("provide --output or config output")
                if args.dry_run:
                    return {
                        "dry_run": True,
                        "output": str(output),
                        "config_hash": agent_config.config_hash,
                        "environment": agent_config.environment,
                        "backend": agent_config.model["backend"],
                        "generation_upper_bound": len(agent_config.tasks)
                        * agent_config.raw["samples_per_task"]
                        * sum(agent_config.raw["budgets"]),
                        "decoding": value.get("decoding", {}),
                        "model_load": False,
                    }
                return run_agent_budget(agent_config, output)
            config = load_config(args.config)
        output = args.output.resolve() if args.output else config.output
        if output is None:
            raise SchemaError("provide --output or a config-relative output field")
        if args.dry_run:
            return {
                "dry_run": True,
                "output": str(output),
                "config_hash": config.config_hash,
                "task_count": len(config.tasks),
                "samples_per_task": config.samples_per_task,
                "action_upper_bound": config.estimated_action_upper_bound,
                "gpu_required": False,
                "network_requests": 0,
                "evidence_kind": "demo_synthetic",
            }
        return run_budget(config, output, project_root=_project_root())
    directory = args.input.resolve()
    manifest, summary = _audit_run(
        directory,
        args.pool,
        replay_code=getattr(args, "replay_code", False),
        verify_weights=getattr(args, "verify_weights", False),
    )
    if args.command == "audit":
        return {
            "status": "verified",
            "run_id": manifest["run_id"],
            "artifacts": len(manifest["artifacts"]),
            "checks": [
                "sha256",
                "trajectory_schema",
                "identity",
                "environment_replay"
                if manifest.get("environment") == "frozenlake_deterministic"
                or manifest.get("kind") is None
                or getattr(args, "replay_code", False)
                else "code_replay_not_requested",
                "metric_recomputation",
            ],
        }
    output = args.output.resolve()
    with new_output_directory(output) as staging:
        write_json(staging / "summary.json", summary)
        if manifest.get("kind"):
            html = render_diagnostic_report(summary, manifest, manifest_href="source-manifest.json")
            write_json(staging / "source-manifest.json", manifest)
        else:
            html = render_report(summary, manifest)
        (staging / "report.html").write_text(html, encoding="utf-8")
        write_json(
            staging / "report-source.json",
            {
                "source_run_id": manifest["run_id"],
                "source_manifest_hash": file_hash(directory / "manifest.json"),
                "source_summary_hash": file_hash(
                    directory
                    / (
                        "metrics.json"
                        if manifest.get("kind") == "checkpoint_validation"
                        else "summary.json"
                    )
                ),
            },
        )
    return {"output": str(output), "report": str(output / "report.html"), "source_verified": True}


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        result = _execute(args)
    except (ValueError, OSError, BackendError, ExecutorError, ImportError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if args.command == "traces" and args.trace_command == "compare":
        if result["gate"]["status"] in {"fail", "incomplete"}:
            return 1
    return 0
