#!/usr/bin/env python3
"""Actual pinned RayPPOTrainer + vLLM + multi-turn grid + HF checkpoint sidecar.

This is an integration smoke, not the paper's algorithm/benchmark recipe.
"""

import argparse
import importlib.metadata
import json
import os
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path

from agent_rl_diagnostics.artifacts.io import read_json, write_json
from agent_rl_diagnostics.integrations.verl import verify_verl_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--verl-source", type=Path, required=True)
    parser.add_argument("--monitor-config", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--train-budget", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--capture-training-state", action="store_true")
    parser.add_argument("--diagnostics-off", action="store_true")
    parser.add_argument("--actor-seed", type=int, default=None)
    parser.add_argument(
        "--budgetmix",
        action="store_true",
        help="Sample unique ceil-deciles up to --train-budget; one-rank fixture verified",
    )
    parser.add_argument(
        "--process-coefficient",
        type=float,
        help="Opt-in explicit mean-step oracle reward coefficient; not a paper preset",
    )
    parser.add_argument("--process-failure-policy", choices=["error", "terminal_only"])
    args = parser.parse_args()
    process_settings = None
    if (args.process_coefficient is None) != (args.process_failure_policy is None):
        parser.error("process-coefficient and process-failure-policy must be supplied together")
    if args.process_coefficient is not None:
        from agent_rl_diagnostics.training.process_signals import process_signal_settings

        try:
            process_settings = process_signal_settings(
                args.process_coefficient, args.process_failure_policy
            )
        except ValueError as exc:
            parser.error(str(exc))
    if args.actor_seed is not None and not 0 <= args.actor_seed <= 2**32 - 1:
        parser.error("actor-seed must be in [0, 2**32-1]")
    if (
        args.output.exists()
        or args.steps < 1
        or not 1 <= args.train_budget <= 8
        or not 2 <= args.group_size <= 16
    ):
        parser.error(
            "new output, steps>=1, train-budget in [1,8] and group-size in [2,16] required"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        parser.error("select exactly one physical GPU explicitly using CUDA_VISIBLE_DEVICES")
    lock = verify_verl_source(args.verl_source.resolve())
    sys.path.insert(0, str(args.verl_source.resolve()))
    expected = {
        "torch": "2.9.0",
        "vllm": "0.11.2",
        "transformers": "4.57.1",
        "ray": "2.54.0",
        "pyarrow": "21.0.0",
        "pandas": "2.2.3",
        "peft": "0.18.1",
    }
    versions = {name: importlib.metadata.version(name) for name in expected}
    if any(versions[name].split("+")[0] != value for name, value in expected.items()):
        parser.error(
            "this candidate recipe requires its explicitly pinned environment: "
            + json.dumps(expected)
        )
    from unittest.mock import patch

    import ray
    from datasets import Dataset
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict
    from verl.experimental.reward_loop import migrate_legacy_reward_impl
    from verl.trainer import main_ppo
    from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica
    from vllm.config import ModelConfig

    from agent_rl_diagnostics.agent_config import parse_agent_config
    from agent_rl_diagnostics.environments.text_grid import TextGrid
    from agent_rl_diagnostics.integrations.checkpoint import run_checkpoint_validation
    from agent_rl_diagnostics.integrations.grid_fixture import TRAIN_GRID_TASKS, check_grid_holdout
    from agent_rl_diagnostics.integrations.ray_checkpoint import diagnostic_trainer_class
    from agent_rl_diagnostics.integrations.ray_paths import create_ray_runtime_directory
    from agent_rl_diagnostics.integrations.ray_worker import preload_arrow
    from agent_rl_diagnostics.integrations.tracking import DiagnosticTracker
    from agent_rl_diagnostics.integrations.verl_budgetmix import budgetmix_trainer_class
    from agent_rl_diagnostics.integrations.verl_grid import GridAgentLoop
    from agent_rl_diagnostics.integrations.versioned_worker import versioned_worker_class

    # Exercise the real architecture-inspection child before expensive Base diagnostics.
    ModelConfig(model=str(args.model_path.resolve()), max_model_len=1024)

    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "imports_verified_only_not_training",
                    "source": lock,
                    "packages": versions,
                    "entry_points": [
                        GridAgentLoop.__name__,
                        AsyncActorRolloutRefWorker.__name__,
                        vLLMReplica.__name__,
                    ],
                }
            )
        )
        return

    args.output.mkdir(parents=True)
    output = args.output.resolve()
    shutil.copytree(args.pool, output / "pool")
    monitor_raw = read_json(args.monitor_config)
    monitor_raw["model"]["path"] = str(args.model_path.resolve())
    monitor = parse_agent_config(monitor_raw, base=args.monitor_config.parent)
    write_json(output / "monitor-config.json", monitor.raw)
    if not args.diagnostics_off:
        initial = run_checkpoint_validation(
            monitor,
            output / "pool",
            args.model_path,
            output / "diagnostics/validation-0",
            checkpoint_step=0,
            samples=2,
        )
        DiagnosticTracker(output / "tracking").log(initial)
    tasks = list(TRAIN_GRID_TASKS)
    check_grid_holdout(monitor.tasks)

    def rows(selected, budget, phase):
        return [
            {
                "data_source": "ard_grid_smoke",
                "agent_name": "ard_grid",
                "prompt": TextGrid(task, prompt_profile="action_only_v2").initial_messages(budget),
                "ability": "grid",
                "reward_model": {"style": "rule", "ground_truth": task["task_id"]},
                "extra_info": {"task": task, "budget": budget, "index": index, "ard_phase": phase},
            }
            for index, task in enumerate(selected)
        ]

    Dataset.from_list(rows(tasks, args.train_budget, "train")).to_parquet(output / "train.parquet")
    Dataset.from_list(rows(monitor.tasks, 8, "validation")).to_parquet(output / "val.parquet")
    write_json(
        output / "agent-loop.json",
        [
            {
                "name": "ard_grid",
                "_target_": "agent_rl_diagnostics.integrations.verl_grid.GridAgentLoop",
            }
        ],
    )
    with initialize_config_dir(
        config_dir=str(args.verl_source.resolve() / "verl/trainer/config"), version_base=None
    ):
        config = compose(config_name="ppo_trainer")
    overrides = {
        "data.train_files": str(output / "train.parquet"),
        "data.val_files": str(output / "val.parquet"),
        "data.train_batch_size": 4,
        "data.max_prompt_length": 256,
        "data.max_response_length": 768,
        "data.apply_chat_template_kwargs": {"enable_thinking": False},
        "data.dataloader_num_workers": 0,
        "actor_rollout_ref.model.path": str(args.model_path.resolve()),
        "actor_rollout_ref.model.use_remove_padding": False,
        "actor_rollout_ref.model.override_config": {"attn_implementation": "sdpa"},
        "actor_rollout_ref.actor.ppo_mini_batch_size": 4,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.use_torch_compile": False,
        "actor_rollout_ref.actor.fsdp_config.use_torch_compile": False,
        "actor_rollout_ref.actor.fsdp_config.param_offload": True,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
        "actor_rollout_ref.actor.fsdp_config.model_dtype": "bf16",
        "actor_rollout_ref.actor.optim.lr": 1e-6,
        "actor_rollout_ref.actor.checkpoint.save_contents": ["hf_model"],
        "actor_rollout_ref.actor.checkpoint.load_contents": ["model"],
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.mode": "async",
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.35,
        "actor_rollout_ref.rollout.enforce_eager": True,
        "actor_rollout_ref.rollout.load_format": "safetensors",
        "actor_rollout_ref.rollout.max_model_len": 1024,
        "actor_rollout_ref.rollout.max_num_batched_tokens": 1024,
        "actor_rollout_ref.rollout.max_num_seqs": 8,
        "actor_rollout_ref.rollout.n": args.group_size,
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.top_p": 1.0,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.agent.num_workers": 1,
        "actor_rollout_ref.rollout.agent.agent_loop_config_path": str(output / "agent-loop.json"),
        "actor_rollout_ref.rollout.val_kwargs.n": 1,
        "actor_rollout_ref.rollout.val_kwargs.temperature": 0.6,
        "actor_rollout_ref.rollout.val_kwargs.top_p": 0.9,
        "algorithm.adv_estimator": "reinforce_plus_plus",
        "algorithm.use_kl_in_reward": False,
        "reward.num_workers": 1,
        "trainer.project_name": "agent-rl-diagnostics",
        "trainer.experiment_name": output.name,
        "trainer.logger": ["console", "wandb", "file"],
        "trainer.n_gpus_per_node": 1,
        "trainer.nnodes": 1,
        "trainer.total_training_steps": args.steps,
        "trainer.total_epochs": args.steps,
        "trainer.save_freq": 1,
        "trainer.test_freq": 1,
        "trainer.resume_mode": "disable",
        "trainer.default_local_dir": str(output / "checkpoints"),
        "trainer.rollout_data_dir": str(output / "rollouts"),
        "trainer.validation_data_dir": str(output / "native-validation"),
        "trainer.use_legacy_worker_impl": "enable",
    }
    with open_dict(config):
        if process_settings is not None:
            config.ard_process_signal = process_settings
        if args.actor_seed is not None:
            config.ard_actor_initialization = {
                "protocol": "one_time_before_native_init_model",
                "seed": args.actor_seed,
            }
        for key, value in overrides.items():
            OmegaConf.update(config, key, value, force_add=True)
        config.ard_diagnostics = {
            "config": str(output / "monitor-config.json"),
            "pool": str(output / "pool"),
            "output": str(output / "diagnostics"),
            "initial_output": str(output / "diagnostics/validation-0"),
            "tracking": str(output / "tracking"),
            "samples": 2,
            "every_n_steps": 1,
            "timeout_seconds": 300,
            "cuda_visible_devices": visible,
        }
        if args.budgetmix:
            config.ard_budgetmix = {
                "tmax": args.train_budget,
                "seed": 5422,
                "output": str(output / "budgetmix"),
            }
    config = migrate_legacy_reward_impl(config)
    write_json(output / "verl-config.json", OmegaConf.to_container(config, resolve=True))
    write_json(output / "source-lock.json", lock)
    write_json(output / "environment.json", versions)
    env_vars = {
        "PYTHONPATH": str(args.verl_source.resolve())
        + os.pathsep
        + os.environ.get("PYTHONPATH", ""),
        "WANDB_MODE": "offline",
        "WANDB_DIR": str(output),
        "VERL_FILE_LOGGER_PATH": str(output / "trainer-history.jsonl"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_USE_V1": "1",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        "ARD_TRAIN_STATE_DIR": str(output / "state-proofs") if args.capture_training_state else "",
        "ARD_ACTOR_INIT_SEED": str(args.actor_seed) if args.actor_seed is not None else "",
    }

    # Real upstream construction site, overridden only in the private controller process.
    class DiagnosticTaskRunner(main_ppo.TaskRunner):
        def add_actor_rollout_worker(self, config):
            from verl.trainer.ppo.ray_trainer import Role

            native, group = super().add_actor_rollout_worker(config)
            if native is not AsyncActorRolloutRefWorker:
                raise RuntimeError("candidate requires the locked legacy FSDP async worker")
            actor = versioned_worker_class(args.verl_source.resolve())
            self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor)
            return actor, group

        def run(self, config):
            context = nullcontext()
            trainer = main_ppo.RayPPOTrainer
            if not args.diagnostics_off:
                trainer = diagnostic_trainer_class(args.verl_source.resolve())
            if args.budgetmix:
                trainer = budgetmix_trainer_class(args.verl_source.resolve(), base_class=trainer)
            if not args.diagnostics_off or args.budgetmix:
                context = patch.object(main_ppo, "RayPPOTrainer", trainer)
            # Ray shutdown can terminate the SDK service before a deferred Tracking
            # destructor flushes it. Finish inside its owning controller process,
            # while Ray is still alive; do not reconstruct history after the fact.
            try:
                with context:
                    return super().run(config)
            finally:
                import wandb

                if wandb.run is not None:
                    wandb.finish()

    ray_directory = create_ray_runtime_directory(output)
    try:
        ray.init(
            num_cpus=8,
            num_gpus=1,
            include_dashboard=False,
            _node_ip_address="127.0.0.1",
            _temp_dir=str(ray_directory),
            object_store_memory=512 * 1024 * 1024,
            runtime_env={"env_vars": env_vars, "worker_process_setup_hook": preload_arrow},
        )
        main_ppo.run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(DiagnosticTaskRunner))
    finally:
        ray.shutdown()
    write_json(
        output / "trainer-returned.json",
        {
            "scope": "actual_single_rank_RayPPOTrainer_vllm_FSDP",
            "algorithm": "reinforce_plus_plus",
            "steps_requested": args.steps,
            "train_action_budget": args.train_budget,
            "samples_per_training_task": args.group_size,
            "checkpoint_diagnostics_enabled": not args.diagnostics_off,
            "actor_state_capture_enabled": args.capture_training_state,
            "budgetmix_enabled": args.budgetmix,
            "process_signal_settings": process_settings,
            "actor_initialization_seed": args.actor_seed,
            "paper_reproduction": False,
            "acceptance": "requires_separate_raw_checkpoint_and_gradient_audit",
        },
    )


if __name__ == "__main__":
    main()
