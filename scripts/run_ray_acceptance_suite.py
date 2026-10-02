#!/usr/bin/env python3
"""One authorized GPU, isolated interpreter: real fixed off/on and BudgetMix runs.

Run on the compute server itself. No SSH/password handling, uploads or driver resets.
Each child writes real stdout/stderr; failed stages retain their original evidence.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from agent_rl_diagnostics.artifacts.io import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--verl-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--actor-seed", type=int, default=None)
    args = parser.parse_args()
    if args.actor_seed is not None and not 0 <= args.actor_seed <= 2**32 - 1:
        parser.error("actor-seed must be in [0, 2**32-1]")
    if args.output.exists() or args.output.is_symlink():
        parser.error("choose a new output directory")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdecimal():
        parser.error("select one physical CUDA_VISIBLE_DEVICES index explicitly")
    repo = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True)
    logs = output / "logs"
    logs.mkdir()
    environment = dict(os.environ)
    environment.update(
        WANDB_MODE="offline",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        TOKENIZERS_PARALLELISM="false",
        TORCHINDUCTOR_CACHE_DIR=str(output / "inductor-cache"),
        TRITON_CACHE_DIR=str(output / "triton-cache"),
        VLLM_CACHE_ROOT=str(output / "vllm-cache"),
    )
    environment["PYTHONPATH"] = str(args.verl_source.resolve()) + os.pathsep + str(repo / "src")
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        environment.pop(key, None)
    stages = []

    def run(label, command, *, comparison=False):
        write_json(
            output / "progress.json",
            {"stage": label, "status": "running", "completed_stages": stages},
        )
        print(json.dumps({"stage": label, "status": "running"}), flush=True)
        with (
            (logs / f"{label}.stdout").open("x") as stdout,
            (logs / f"{label}.stderr").open("x") as stderr,
        ):
            completed = subprocess.run(
                command, cwd=repo, env=environment, stdout=stdout, stderr=stderr
            )
        stages.append({"stage": label, "returncode": completed.returncode})
        write_json(
            output / "progress.json",
            {"stage": label, "status": "returned", "completed_stages": stages},
        )
        if completed.returncode and not comparison:
            raise RuntimeError(
                f"{label} failed with {completed.returncode}; inspect its original logs"
            )

    def script(name, *arguments):
        return [sys.executable, str(repo / "scripts" / name), *map(str, arguments)]

    run(
        "runtime",
        script(
            "check_runtime.py",
            "--python",
            sys.executable,
            "--cuda-device",
            visible,
            "--timeout",
            30,
            "--output",
            output / "runtime.json",
        ),
    )
    run(
        "base-monitor",
        script(
            "prepare_ray_monitor.py",
            "--model-path",
            args.model_path,
            "--template",
            repo / "configs/examples/frozenlake_ray_monitor.json",
            "--output",
            output / "base-monitor",
        ),
    )
    base = output / "base-monitor"
    common = [
        "--model-path",
        args.model_path,
        "--verl-source",
        args.verl_source,
        "--monitor-config",
        base / "monitor-config.json",
        "--pool",
        base / "pool",
        "--steps",
        2,
        "--train-budget",
        2,
        "--group-size",
        16,
        "--capture-training-state",
    ]
    for label, extras in (
        ("fixed-off", ["--diagnostics-off"]),
        ("fixed-on", []),
        ("budgetmix-on", ["--budgetmix"]),
    ):
        if args.actor_seed is not None:
            extras = [*extras, "--actor-seed", args.actor_seed]
        root = output / label
        run(label, script("run_verl_ray_smoke.py", *common, "--output", root, *extras))
        offline = list((root / "wandb").glob("offline-run-*/run-*.wandb"))
        if len(offline) != 1:
            raise RuntimeError("exactly one actual offline W&B file per training run required")
        run(
            label + "-history",
            script(
                "export_wandb_history.py",
                "--input",
                offline[0],
                "--output",
                root / "offline-history.jsonl",
            ),
        )
        if label != "fixed-off":
            run(
                label + "-audit",
                script(
                    "audit_verl_ray_smoke.py",
                    "--run",
                    root,
                    "--base",
                    args.model_path,
                    "--history",
                    root / "trainer-history.jsonl",
                    "--wandb-history",
                    root / "offline-history.jsonl",
                    "--output",
                    root / "acceptance.json",
                ),
            )
    run(
        "paired-observations",
        script(
            "compare_ray_training_pair.py",
            "--left",
            output / "fixed-off",
            "--right",
            output / "fixed-on",
            "--output",
            output / "pair-comparison.json",
        ),
        comparison=True,
    )
    root = output / "fixed-on"
    run(
        "standalone-step1",
        [
            sys.executable,
            "-m",
            "agent_rl_diagnostics",
            "eval",
            "checkpoint",
            "--config",
            str(root / "monitor-config.json"),
            "--pool",
            str(root / "pool"),
            "--checkpoint",
            str(root / "checkpoints/global_step_1/actor/huggingface"),
            "--checkpoint-step",
            "1",
            "--samples",
            "2",
            "--output",
            str(output / "standalone-step1"),
        ],
    )
    run(
        "standalone-comparison",
        script(
            "compare_checkpoint_validation.py",
            "--left",
            root / "diagnostics/validation-1",
            "--right",
            output / "standalone-step1",
            "--pool",
            root / "pool",
            "--verify-weights",
            "--output",
            output / "standalone-comparison.json",
        ),
        comparison=True,
    )
    pair = read_json(output / "pair-comparison.json")
    standalone = read_json(output / "standalone-comparison.json")
    report = {
        "scope": "real_single_rank_fixed_and_BudgetMix_fixture_suite",
        "stages": stages,
        "fixed_training_accepted": read_json(output / "fixed-on/acceptance.json")["accepted"],
        "budgetmix_training_accepted": read_json(output / "budgetmix-on/acceptance.json")[
            "budgetmix_training_fixture_verified"
        ],
        "paired_observations_match": pair["status"] == "matched_observed_actor_and_rollouts",
        "standalone_outputs_match": standalone["status"] == "matched",
        "full_trainer_parity_verified": False,
        "distributed_parity_verified": False,
        "paper_reproduction": False,
    }
    write_json(output / "verification.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
