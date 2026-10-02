"""Offline agent-v2 evaluation walkthrough with explicitly scripted policies."""

import argparse
from pathlib import Path

from .agent_config import parse_agent_config
from .artifacts.io import canonical_hash, new_output_directory, write_json
from .evaluators.agent import run_agent_budget
from .evaluators.fixed import run_fixed
from .evaluators.recovery import run_recovery
from .models import Generation
from .pools.errors import build_error_pool
from .pools.frozen import build_pool
from .training.plan import audit_plan, create_plan


class ScriptedPolicy:
    """Alternate a wasted action with RIGHT, or always choose RIGHT; no learned model."""

    def __init__(self, *, waste_actions):
        self.waste_actions = waste_actions
        self.metadata = {"kind": "demo_scripted_policy", "seed_support": "deterministic"}

    def generate(self, messages, *, seed):
        depth = sum(message["role"] == "assistant" for message in messages)
        action = "UP" if self.waste_actions and depth % 2 == 0 else "RIGHT"
        return Generation(
            action, {"input_messages_hash": canonical_hash(messages), "seed_requested": seed}
        )


def run_toolkit_demo(output: Path):
    # Reuse the same audit/config checks as the public CLI, without parsing user commands.
    from .cli import _audit_run, _execute

    output = output.resolve()
    with new_output_directory(output) as root:
        configs = {}
        policies = {}
        for label, waste in (("Base", True), ("RL", False)):
            configs[label] = parse_agent_config(
                {
                    "schema_version": "ard-agent-config-v1",
                    "protocol_id": "toolkit_v1",
                    "evidence_kind": "demo_synthetic",
                    "environment": "frozenlake_deterministic",
                    "tasks": [
                        {"task_id": "short", "desc": ["SFFG"]},
                        {"task_id": "long", "desc": ["SFFFG"]},
                    ],
                    # The injected ChatModel bypasses this endpoint. It is never contacted.
                    "model": {
                        "backend": "http",
                        "model_id": label,
                        "revision": "scripted-v1",
                        "base_url": "http://127.0.0.1:1/v1",
                    },
                    "decoding": {"temperature": 0, "top_p": 1, "max_new_tokens": 8},
                    "budgets": [3, 5, 9],
                    "ks": [1, 2],
                    "samples_per_task": 2,
                    "bootstrap_replicates": 40,
                    "seed": 5422,
                },
                base=root,
            )
            policies[label] = ScriptedPolicy(waste_actions=waste)
            run_agent_budget(configs[label], root / f"budget-{label}", policies[label])
        pool = root / "pool"
        build_pool(
            [root / "budget-Base", root / "budget-RL"],
            pool,
            kind="matched",
            size=8,
            seed=5422,
            base_model_id="Base",
            rl_model_id="RL",
            max_per_task=4,
            min_tasks=2,
        )
        errors = root / "errors"
        build_error_pool(pool, errors, seed=5422)
        monitor = root / "base-monitor"
        build_pool(
            [root / "budget-Base"],
            monitor,
            kind="base_monitor",
            size=6,
            seed=5422,
            base_model_id="Base",
            max_per_task=3,
            min_tasks=2,
        )
        jobs = [
            ("budget", None),
            ("step", pool),
            ("error-step", errors),
            ("recovery", errors),
            ("monitor-step", monitor),
        ]
        audits = {}
        for label in configs:
            for kind, frozen in jobs:
                target = root / f"{kind}-{label}"
                if kind != "budget":
                    evaluate = run_recovery if kind == "recovery" else run_fixed
                    evaluate(configs[label], frozen, target, samples=2, model=policies[label])
                _audit_run(target, frozen)
                audits[target.name] = "verified"
        for kind, frozen in jobs:
            _execute(
                argparse.Namespace(
                    command="compare",
                    input=[root / f"{kind}-{label}" for label in configs],
                    labels=list(configs),
                    pool=frozen,
                    output=root / f"compare-{kind}",
                )
            )
        create_plan(
            ["short", "long", "short"],
            tmax=9,
            seed=5422,
            update=1,
            group_size=2,
            output=root / "budgetmix",
        )
        audit_plan(root / "budgetmix")
        summary = {
            "schema_version": "ard-toolkit-demo-v1",
            "evidence_kind": "demo_synthetic",
            "paper_reproduction": False,
            "trained_models": False,
            "checkpoint_training_validation": False,
            "audited_runs": audits,
            "reports": [f"compare-{kind}/report.html" for kind, _ in jobs],
            "budgetmix": "verified_plan_no_training",
        }
        write_json(root / "walkthrough.json", summary)
        links = "".join(
            f'<li><a href="{path}">{path.split("/")[0]}</a></li>' for path in summary["reports"]
        )
        (root / "index.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>CPU toolkit walkthrough</title>'
            "<h1>Scripted CPU walkthrough</h1><p>Synthetic policies, no trained models "
            "or paper numerical reproduction. The Base monitor is frozen evaluation input; "
            "this demonstration does not validate training checkpoints.</p><ul>" + links + "</ul>",
            encoding="utf-8",
        )
    return {"output": str(output), "index": str(output / "index.html"), **summary}
