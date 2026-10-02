#!/usr/bin/env python3
"""Fail-closed, read-only acceptance of this project's one-rank Ray smoke.

This is not distributed/on-off parity, optimizer-resume or paper acceptance.
Requires finished upstream file history and independently exported offline W&B history.
"""

import argparse
import json
import math
from pathlib import Path

from agent_rl_diagnostics.artifacts.io import canonical_hash, read_json, write_json
from agent_rl_diagnostics.contracts.schema import SchemaError
from agent_rl_diagnostics.integrations.budgetmix_audit import audit_budgetmix_traces
from agent_rl_diagnostics.integrations.checkpoint import audit_checkpoint_validation
from agent_rl_diagnostics.integrations.grid_fixture import TRAIN_GRID_TASKS
from agent_rl_diagnostics.integrations.grid_trace import audit_grid_trace
from agent_rl_diagnostics.integrations.tensor_identity import tensor_identity


def require(condition, message):
    if not condition:
        raise SchemaError(message)


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def audit(run, base, history, wandb_history):
    returned = read_json(run / "trainer-returned.json")
    steps = returned["steps_requested"]
    require(type(steps) is int and steps >= 2, "need at least two completed optimizer steps")
    native = list(read_jsonl(history))
    wandb = list(read_jsonl(wandb_history))
    require([r["step"] for r in native] == list(range(steps + 1)), "native steps incomplete")
    require(len(wandb) == steps + 1, "offline W&B steps incomplete")
    require(
        [r.get("checkpoint_step") for r in wandb] == list(range(steps + 1)),
        "offline W&B checkpoint steps mismatch",
    )
    raw_traces = [read_json(p) for p in sorted((run / "rollouts/grid-traces").glob("*.json"))]
    traces = [audit_grid_trace(trace) for trace in raw_traces]
    require(
        traces and {t["weight_version"] for t in traces} == set(range(steps + 1)),
        "rollout weight versions incomplete",
    )
    require(
        any(t["environment_tokens"] > 0 for t in traces), "no actual multi-turn feedback exercised"
    )
    # Independently decode recorded generated IDs; do not trust raw response text alone.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    for trace in raw_traces:
        initial = tokenizer.apply_chat_template(
            trace["initial_messages"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        require(initial == trace["initial_prompt_ids"], "initial tokenized prompt mismatch")
        for action in trace["steps"]:
            decoded = tokenizer.decode(action["response_ids"], skip_special_tokens=True)
            require(decoded == action["response"], "generated token IDs disagree with action text")
    previous = tensor_identity(base)
    transitions = []
    for step in range(steps + 1):
        metrics = audit_checkpoint_validation(
            run / f"diagnostics/validation-{step}", run / "pool", verify_weights=True
        )
        require(metrics["metrics"]["diag/health/incomplete"] == 0, "incomplete diagnostic point")
        require(metrics["checkpoint_step"] == step, "diagnostic checkpoint step mismatch")
        require(
            native[step]["data"].get("checkpoint_step") == step, "driver checkpoint step mismatch"
        )
        for key, value in metrics["metrics"].items():
            require(native[step]["data"].get(key) == value, f"driver diagnostic mismatch: {key}")
            require(wandb[step].get(key) == value, f"offline W&B diagnostic mismatch: {key}")
        if step == 0:
            continue
        gradient = native[step]["data"].get("actor/grad_norm")
        require(
            isinstance(gradient, (int, float)) and math.isfinite(gradient) and gradient > 0,
            "missing/nonfinite/zero actor gradient",
        )
        require(wandb[step].get("actor/grad_norm") == gradient, "offline W&B gradient mismatch")
        current = tensor_identity(run / f"checkpoints/global_step_{step}/actor/huggingface")
        common = set(current["tensors"]) & set(previous["tensors"])
        require(
            all(
                current["tensors"][name][key] == previous["tensors"][name][key]
                for name in common
                for key in ("dtype", "shape")
            ),
            "checkpoint tensor dtype/shape changed; cannot infer optimizer update",
        )
        changed = [
            name
            for name in common
            if current["tensors"][name]["sha256"] != previous["tensors"][name]["sha256"]
        ]
        require(common and changed, "checkpoint tensor contents did not change")
        transitions.append(
            {
                "step": step,
                "gradient_norm": gradient,
                "changed_common_tensors": len(changed),
                "common_tensors": len(common),
                "added_tensor_names": sorted(set(current["tensors"]) - set(previous["tensors"])),
                "removed_tensor_names": sorted(set(previous["tensors"]) - set(current["tensors"])),
                "fingerprint": current["fingerprint"],
            }
        )
        previous = current
    mix = []
    if returned.get("budgetmix_enabled", False):
        for update in range(1, steps + 1):
            selected = [t for t in raw_traces if t.get("budgetmix", {}).get("update") == update]
            point = audit_budgetmix_traces(
                selected, group_size=returned["samples_per_training_task"]
            )
            require(
                point["groups"] == len(TRAIN_GRID_TASKS), "BudgetMix training groups incomplete"
            )
            require(
                point["weight_version"] == update - 1, "BudgetMix trainer/rollout version mismatch"
            )
            plan = read_json(run / f"budgetmix/update-{update:06d}/assignments.json")
            observed = {t["budgetmix"]["task_occurrence_id"]: t["budgetmix"] for t in selected}
            require(
                canonical_hash(sorted(plan, key=lambda r: r["row"]))
                == canonical_hash(sorted(observed.values(), key=lambda r: r["row"])),
                "BudgetMix completed traces differ from pre-repeat assignments",
            )
            expected = {
                "train/budgetmix/budget_mean": point["budget_mean"],
                "train/budgetmix/groups": point["groups"],
                **{
                    f"train/budgetmix/groups_t{t}": count
                    for t, count in point["budget_histogram"].items()
                },
            }
            for key, value in expected.items():
                require(
                    native[update]["data"].get(key) == value, "native BudgetMix metrics mismatch"
                )
                require(wandb[update].get(key) == value, "offline W&B BudgetMix metrics mismatch")
            mix.append({"update": update, **point})
    process = None
    if returned.get("process_signal_settings") is not None:
        from agent_rl_diagnostics.integrations.process_audit import audit_process_exports

        settings = returned["process_signal_settings"]
        require(
            read_json(run / "verl-config.json").get("ard_process_signal") == settings,
            "resolved process settings differ from returned trainer manifest",
        )
        exports = [
            (phase, step, row)
            for phase, folder, indices in (
                ("train", "rollouts", range(1, steps + 1)),
                ("validation", "native-validation", range(steps + 1)),
            )
            for step in indices
            for row in read_jsonl(run / folder / f"{step}.jsonl")
        ]
        process = audit_process_exports(
            raw_traces,
            exports,
            settings=settings,
            steps=steps,
            group_size=returned["samples_per_training_task"],
            train_task_ids={task["task_id"] for task in TRAIN_GRID_TASKS},
            validation_task_ids={
                task["task_id"] for task in read_json(run / "monitor-config.json")["tasks"]
            },
            decode=lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
        )
    return {
        "scope": "single_rank_RayPPOTrainer_vllm_FSDP_REINFORCE++_smoke",
        "accepted": True,
        "steps": steps,
        "trace_count": len(traces),
        "environment_tokens": sum(t["environment_tokens"] for t in traces),
        "tensor_transitions": transitions,
        "native_and_offline_wandb_diagnostic_points_match": True,
        "generated_response_token_decode_verified": True,
        "raw_dual_validation_recomputed_and_weights_rehashed": True,
        "distributed_parity_verified": False,
        "optimizer_state_parity_verified": False,
        "full_trainer_diagnostics_on_off_verified": False,
        "standalone_output_parity_verified": False,
        "optimizer_resume_verified": False,
        "paper_reproduction": False,
        "budgetmix_completed_group_points": mix,
        "budgetmix_training_fixture_verified": bool(mix),
        "process_signal_native_exports": process,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--wandb-history", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "acceptance output exists")
    write_json(args.output, audit(args.run, args.base, args.history, args.wandb_history))


if __name__ == "__main__":
    main()
