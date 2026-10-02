#!/usr/bin/env python3
"""Small genuine-model acceptance run; not a paper experiment or training job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_rl_diagnostics.agent_config import parse_agent_config
from agent_rl_diagnostics.artifacts.io import read_json, verify_artifacts, write_json
from agent_rl_diagnostics.contracts.schema import SchemaError
from agent_rl_diagnostics.evaluators.agent import audit_agent_budget, run_agent_budget
from agent_rl_diagnostics.evaluators.audit import audit_fixed_or_recovery
from agent_rl_diagnostics.evaluators.fixed import run_fixed
from agent_rl_diagnostics.evaluators.recovery import run_recovery
from agent_rl_diagnostics.models.chat import TransformersChatModel
from agent_rl_diagnostics.pools.errors import build_error_pool
from agent_rl_diagnostics.pools.frozen import build_pool, load_pool
from agent_rl_diagnostics.pools.witness import build_witness_pool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--revision",
        required=True,
        help="Operator-declared checkpoint identity; not an automatically verified weight hash",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument(
        "--prompt-profile", choices=["legacy_v1", "action_only_v2"], default="legacy_v1"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new directory")
    examples = Path(__file__).resolve().parents[1] / "configs" / "examples"
    spec = {
        "backend": "transformers",
        "model_id": "local-smoke-base",
        "revision": args.revision,
        "path": str(args.model_path.resolve()),
        "device": args.device,
        "dtype": args.dtype,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    configs = []
    for name in ("frozenlake_agent_http.json", "deepcoder_agent_http.json"):
        raw = read_json(examples / name)
        raw["model"] = spec.copy()
        raw[
            "grid_prompt_profile"
            if raw["environment"] == "frozenlake_deterministic"
            else "coding_prompt_profile"
        ] = args.prompt_profile
        configs.append(parse_agent_config(raw, base=examples))
    print(json.dumps({"stage": "loading_local_model", "device": args.device}), flush=True)
    model = TransformersChatModel(spec, configs[0].sampling)
    args.output.mkdir(parents=True)
    # A same-checkpoint repeat must preserve the trainer's CPU/device RNG and mode.
    torch = model.torch
    cpu_before = torch.get_rng_state().clone()
    gpu_before = (
        torch.cuda.get_rng_state(model.device).clone() if model.device.type == "cuda" else None
    )
    mode_before = model.model.training
    messages = [{"role": "user", "content": "Reply with exactly the word RIGHT."}]
    first = model.generate(messages, seed=1234)
    second = model.generate(messages, seed=1234)
    checks = {
        "same_seed_output_equal": first.text == second.text,
        "cpu_rng_preserved": torch.equal(cpu_before, torch.get_rng_state()),
        "device_rng_preserved": gpu_before is None
        or torch.equal(gpu_before, torch.cuda.get_rng_state(model.device)),
        "training_mode_preserved": mode_before == model.model.training,
    }
    write_json(
        args.output / "generation-checks.json",
        {
            "checks": checks,
            "model": model.metadata,
            "first": {"text": first.text, "metadata": first.metadata},
            "second": {"text": second.text, "metadata": second.metadata},
        },
    )
    if not all(checks.values()):
        raise RuntimeError("generation reproducibility/state-preservation acceptance failed")
    results = {"generation_checks": checks, "paper_reproduction": False, "split": "smoke"}
    for config, label, size in zip(configs, ("grid", "coding"), (8, 2), strict=True):
        root = args.output / label
        root.mkdir()
        source, pool, step, errors, recovery = [
            root / name for name in ("budget", "pool", "step", "errors", "recovery")
        ]
        print(json.dumps({"stage": label + "_budget"}), flush=True)
        run_agent_budget(config, source, model)
        audit_agent_budget(source, verify_artifacts(source), replay_code=True)
        build_pool(
            [source],
            pool,
            kind="base_monitor",
            size=size,
            seed=5422,
            base_model_id=config.model_id,
            max_per_task=size,
            min_tasks=2,
        )
        load_pool(pool)
        print(json.dumps({"stage": label + "_step"}), flush=True)
        run_fixed(config, pool, step, samples=2, model=model)
        audit_fixed_or_recovery(step, verify_artifacts(step), pool, replay_code=True)
        recovery_source = "model_collected_base_pool"
        try:
            build_error_pool(
                pool,
                errors,
                seed=5422,
                require_recoverable=True,
                references=examples / "coding_smoke_references.json" if label == "coding" else None,
            )
        except SchemaError as exc:
            if label != "grid" or not str(exc).startswith("no eligible clean/error pairs"):
                raise
            print(
                json.dumps(
                    {
                        "stage": "grid_base_pool_has_no_eligible_errors",
                        "action": "use_separate_explicit_oracle_reference_pool_not_base_monitor",
                    }
                ),
                flush=True,
            )
            witness = root / "independent-witness-pool"
            build_witness_pool(config, witness, budget=max(config.raw["budgets"]), size=4)
            build_error_pool(witness, errors, seed=5422, require_recoverable=True)
            recovery_source = "independent_exact_geometry_witness_not_base"
        load_pool(errors)
        print(json.dumps({"stage": label + "_recovery"}), flush=True)
        run_recovery(config, errors, recovery, samples=1, model=model)
        audit_fixed_or_recovery(recovery, verify_artifacts(recovery), errors, replay_code=True)
        results[label] = {
            name: read_json(path / "summary.json")
            for name, path in (("budget", source), ("step", step), ("recovery", recovery))
        }
        results[label]["recovery_source"] = recovery_source
        print(
            json.dumps(
                {
                    "stage": label + "_verified",
                    "r_single": results[label]["step"]["r_single"],
                    "clean_success": results[label]["recovery"]["clean_success"],
                    "error_success": results[label]["recovery"]["error_success"],
                }
            ),
            flush=True,
        )
    write_json(args.output / "acceptance.json", results)
    print(json.dumps({"stage": "accepted", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
