#!/usr/bin/env python3
"""Real single-GPU verl actor smoke with explicit GRPO/REINFORCE. NOT RayPPOTrainer."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

from agent_rl_diagnostics.agent_config import parse_agent_config
from agent_rl_diagnostics.artifacts.io import read_json, write_json
from agent_rl_diagnostics.evaluators.agent import rollout, run_agent_budget, write_records
from agent_rl_diagnostics.integrations.checkpoint import (
    checkpoint_identity,
    run_checkpoint_validation,
)
from agent_rl_diagnostics.integrations.tracking import DiagnosticTracker
from agent_rl_diagnostics.integrations.verl import verify_verl_source
from agent_rl_diagnostics.models.chat import Sampling, TransformersChatModel
from agent_rl_diagnostics.pools.frozen import build_pool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--verl-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--algorithm", choices=["grpo", "reinforce"], default="grpo")
    args = parser.parse_args()
    if args.output.exists() or args.steps < 1 or args.group_size < 2:
        parser.error("new output, steps>=1 and group-size>=2 required")
    lock = verify_verl_source(args.verl_source.resolve())
    sys.path.insert(0, str(args.verl_source.resolve()))
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from verl import DataProto
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.workers.config import FSDPActorConfig

    args.output.mkdir(parents=True)
    write_json(
        args.output / "training-config.json",
        {
            "algorithm": args.algorithm,
            "group_size": args.group_size,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "upstream": lock,
            "seed": 5422,
            "reward": "terminal_success_only",
            "lora_rank": 4,
            "kl": False,
            "scope": "smoke_not_paper_recipe",
        },
    )
    torch.manual_seed(5422)
    np.random.seed(5422)
    torch.distributed.init_process_group(
        "gloo",
        init_method="file://" + str((args.output / "dist-init").resolve()),
        rank=0,
        world_size=1,
    )
    examples = Path(__file__).resolve().parents[1] / "configs" / "examples"
    raw = read_json(examples / "frozenlake_agent_http.json")
    original = checkpoint_identity(args.model_path)
    raw.update(split="validation", grid_prompt_profile="action_only_v2", budgets=[6, 8])
    raw["model"] = {
        "backend": "transformers",
        "model_id": "qwen17-base-monitor",
        "revision": "hf-sha256:" + original["fingerprint"],
        "path": str(args.model_path.resolve()),
        "device": "cuda:0",
        "dtype": "bfloat16",
        "chat_template_kwargs": {"enable_thinking": False},
    }
    raw["decoding"]["max_new_tokens"] = 32
    config = parse_agent_config(raw, base=examples)
    print(json.dumps({"stage": "load_base", "verl": lock}), flush=True)
    backend = TransformersChatModel(config.model, config.sampling)
    train_tasks = [
        {"task_id": "train-right-one", "desc": ["SG"]},
        {"task_id": "train-left-one", "desc": ["GS"]},
        {"task_id": "train-down-one", "desc": ["S", "G"]},
        {"task_id": "train-up-one", "desc": ["G", "S"]},
    ]
    train_ids = {task["task_id"] for task in train_tasks}
    assert not ({tuple(t["desc"]) for t in train_tasks} & {tuple(t["desc"]) for t in config.tasks})
    print(json.dumps({"stage": "collect_heldout_base_pool"}), flush=True)
    run_agent_budget(config, args.output / "base-collection", backend)
    pool = args.output / "pool"
    build_pool(
        [args.output / "base-collection"],
        pool,
        kind="base_monitor",
        size=8,
        seed=5422,
        base_model_id=config.model_id,
        max_per_task=4,
        min_tasks=2,
        training_task_ids=train_ids,
    )
    backend.model = get_peft_model(
        backend.model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    backend.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    backend.model.enable_input_require_grads()
    parameters = [p for p in backend.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0)
    pad = backend.tokenizer.pad_token_id or backend.tokenizer.eos_token_id
    tracker = DiagnosticTracker(args.output / "tracking", wandb_offline=args.wandb_offline)

    def fingerprint(values):
        h = hashlib.sha256()
        for value in values:
            h.update(
                value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
            )
        return h.hexdigest()

    def optimizer_identity(value):
        if isinstance(value, torch.Tensor):
            return {
                "tensor_hash": fingerprint([value]),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        if isinstance(value, dict):
            return {str(k): optimizer_identity(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [optimizer_identity(v) for v in value]
        return value

    def export(step):
        directory = args.output / "checkpoints" / f"step-{step}"
        directory.mkdir(parents=True)
        merged = copy.deepcopy(backend.model).cpu().merge_and_unload()
        merged.save_pretrained(directory, safe_serialization=True, max_shard_size="2GB")
        backend.tokenizer.save_pretrained(directory)
        del merged
        return directory

    def evaluate(step, checkpoint, name):
        print(json.dumps({"stage": "dual_validation", "step": step}), flush=True)
        point = run_checkpoint_validation(
            config, pool, checkpoint, args.output / name, checkpoint_step=step, samples=2
        )
        return point

    initial = export(0)
    base_point = evaluate(0, initial, "validation-0")
    tracker.log(base_point)
    updates, parity = [], None
    for step in range(1, args.steps + 1):
        print(json.dumps({"stage": "real_training_rollouts", "step": step}), flush=True)
        train_raw = copy.deepcopy(config.raw)
        live_revision = (
            "frozen-base:" + original["fingerprint"] + ":lora:" + fingerprint(parameters)
        )
        train_raw["model"].update(model_id="qwen17-native-actor", revision=live_revision)
        backend.metadata.update(
            model_id="qwen17-native-actor",
            revision=live_revision,
            revision_status="live_base_and_lora_sha256",
        )
        train_raw.update(
            tasks=train_tasks,
            split="train",
            seed=5422 + 100 * step,
            budgets=[2, 3],
            samples_per_task=args.group_size,
            ks=[1],
        )
        train_raw["decoding"] = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": 32}
        train = parse_agent_config(train_raw, base=examples)
        previous_sampling = backend.sampling
        backend.sampling = Sampling(**train_raw["decoding"])
        episodes = [
            rollout(
                train,
                task,
                backend,
                run_id=f"native-{args.algorithm}-step-{step}",
                budget=3 if i % 2 == 0 else 2,
                sample_id=sample,
            )
            for i, task in enumerate(train_tasks)
            for sample in range(args.group_size)
        ]
        backend.sampling = previous_sampling
        for episode in episodes:
            for action in episode.steps:
                action.generation["live_actor_revision"] = live_revision
        backend.model.save_pretrained(args.output / "checkpoints" / f"adapter-before-step-{step}")
        write_json(args.output / f"train-config-{step}.json", train.raw)
        write_records(args.output / f"train-episodes-{step}.jsonl", [r.to_dict() for r in episodes])
        rewards = torch.tensor([float(e.success) for e in episodes], device="cuda")
        groups = np.array([e.task_id + f":occurrence-{step}" for e in episodes])
        if args.algorithm == "grpo":
            episode_advantages, _ = compute_grpo_outcome_advantage(
                rewards[:, None], torch.ones((len(episodes), 1), device="cuda"), groups
            )
        else:
            # Explicit uncentered terminal-return REINFORCE, not a silent GRPO fallback.
            episode_advantages = rewards[:, None]
        if not torch.any(episode_advantages != 0):
            raise RuntimeError(
                f"real {args.algorithm} rollouts produced zero advantages; "
                "no genuine learning signal; refusing fake update"
            )
        actions = [
            (e, s, episode_advantages[i, 0]) for i, e in enumerate(episodes) for s in e.steps
        ]
        pmax = max(len(s.generation["prompt_token_ids"]) for _, s, _ in actions)
        rmax = max(len(s.generation["response_token_ids"]) for _, s, _ in actions)
        prompts = torch.full((len(actions), pmax), pad, dtype=torch.long, device="cuda")
        responses = torch.full((len(actions), rmax), pad, dtype=torch.long, device="cuda")
        attention = torch.zeros((len(actions), pmax + rmax), dtype=torch.long, device="cuda")
        response_mask = torch.zeros((len(actions), rmax), dtype=torch.long, device="cuda")
        advantages = torch.zeros((len(actions), rmax), dtype=torch.float32, device="cuda")
        for i, (_, action, advantage) in enumerate(actions):
            prompt_ids, response_ids = (
                action.generation["prompt_token_ids"],
                action.generation["response_token_ids"],
            )
            prompts[i, -len(prompt_ids) :] = torch.tensor(prompt_ids, device="cuda")
            responses[i, : len(response_ids)] = torch.tensor(response_ids, device="cuda")
            attention[i, pmax - len(prompt_ids) : pmax + len(response_ids)] = 1
            response_mask[i, : len(response_ids)] = 1
            advantages[i, : len(response_ids)] = advantage
        input_ids = torch.cat([prompts, responses], dim=1)
        positions = (attention.cumsum(-1) - 1).clamp_min(0)
        actor_config = FSDPActorConfig(
            ppo_mini_batch_size=len(actions),
            ppo_micro_batch_size_per_gpu=1,
            ppo_infer_micro_batch_size_per_gpu=1,
            rollout_n=args.group_size,
            use_torch_compile=False,
            use_remove_padding=False,
            calculate_entropy=False,
            entropy_coeff=0,
            use_kl_loss=False,
            ppo_epochs=1,
            shuffle=False,
        )
        actor = DataParallelPPOActor(actor_config, backend.model, optimizer)
        data = DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "responses": responses,
                "attention_mask": attention,
                "position_ids": positions,
                "response_mask": response_mask,
                "advantages": advantages,
            },
            meta_info={
                "temperature": 1.0,
                "micro_batch_size": 1,
                "use_dynamic_bsz": False,
                "pad_token_id": pad,
            },
        )
        data.batch["old_log_probs"] = actor.compute_log_prob(data)["log_probs"].detach()
        before = fingerprint(parameters)
        captured = []
        handle = optimizer.register_step_pre_hook(
            lambda *unused: captured.append(
                [
                    p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                    for p in parameters
                ]
            )
        )
        if step == 1:
            weights = [p.detach().clone() for p in parameters]
            optimizer_before = copy.deepcopy(optimizer.state_dict())
            rng_cpu, rng_gpu = torch.get_rng_state(), torch.cuda.get_rng_state()
            rng_python, rng_numpy = random.getstate(), np.random.get_state()
            immutable_batch = fingerprint([data.batch[k] for k in sorted(data.batch.keys())])
            off_metrics = actor.update_policy(data)
            off_weights = [p.detach().clone() for p in parameters]
            off_gradients = captured[-1]
            off_rng_cpu, off_rng_gpu = torch.get_rng_state(), torch.cuda.get_rng_state()
            off_rng_python, off_rng_numpy = random.getstate(), np.random.get_state()
            off_optimizer = optimizer_identity(optimizer.state_dict())
            for p, value in zip(parameters, weights, strict=True):
                p.data.copy_(value)
            optimizer.load_state_dict(optimizer_before)
            torch.set_rng_state(rng_cpu)
            torch.cuda.set_rng_state(rng_gpu)
            random.setstate(rng_python)
            np.random.set_state(rng_numpy)
            evaluate(0, initial, "parity-validation-0")
            metrics = actor.update_policy(data)
            parity = {
                "updated_weights_equal": all(
                    torch.equal(a, b) for a, b in zip(off_weights, parameters, strict=True)
                ),
                "clipped_gradients_equal": all(
                    torch.equal(a, b) for a, b in zip(off_gradients, captured[-1], strict=True)
                ),
                "actor_metrics_equal": off_metrics == metrics,
                "cpu_rng_equal": torch.equal(off_rng_cpu, torch.get_rng_state()),
                "gpu_rng_equal": torch.equal(off_rng_gpu, torch.cuda.get_rng_state()),
                "training_batch_unchanged": immutable_batch
                == fingerprint([data.batch[k] for k in sorted(data.batch.keys())]),
                "python_rng_equal": off_rng_python == random.getstate(),
                "numpy_rng_equal": off_rng_numpy[0] == np.random.get_state()[0]
                and np.array_equal(off_rng_numpy[1], np.random.get_state()[1])
                and off_rng_numpy[2:] == np.random.get_state()[2:],
                "optimizer_state_equal": off_optimizer
                == optimizer_identity(optimizer.state_dict()),
            }
            write_json(args.output / "diagnostics-on-off-parity.json", parity)
            if not all(parity.values()):
                raise RuntimeError("native actor diagnostics-on/off parity failed")
        else:
            metrics = actor.update_policy(data)
        handle.remove()
        after = fingerprint(parameters)
        if before == after:
            raise RuntimeError("optimizer did not change any trainable weights")
        update = {
            "step": step,
            "trainable_before": before,
            "trainable_after": after,
            "mean_success_reward": rewards.mean().item(),
            "nonzero_advantage_episodes": int((episode_advantages[:, 0] != 0).sum()),
            "actions": len(actions),
            "native_actor_metrics": metrics,
        }
        updates.append(update)
        checkpoint = export(step)
        point = evaluate(step, checkpoint, f"validation-{step}")
        point["metrics"].update(
            {
                "train/success_reward_mean": update["mean_success_reward"],
                "train/nonzero_advantage_episodes": update["nonzero_advantage_episodes"],
                "train/actor_pg_loss": metrics["actor/pg_loss"],
                "train/actor_grad_norm": float(np.mean(metrics["actor/grad_norm"])),
            }
        )
        tracker.log(point)
        print(json.dumps({"stage": "update_verified", **update}), flush=True)
    tracker.close()
    torch.distributed.destroy_process_group()
    write_json(
        args.output / "acceptance.json",
        {
            "schema_version": "ard-native-actor-acceptance-v1",
            "scope": "single_gpu_native_verl_actor_not_ray_trainer",
            "algorithm": args.algorithm,
            "upstream": lock,
            "updates": updates,
            "diagnostic_parity": parity,
            "full_ray_trainer_tested": False,
            "paper_reproduction": False,
        },
    )
    print(
        json.dumps({"stage": "accepted", "scope": "native_verl_actor", "output": str(args.output)}),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        torch_module = sys.modules.get("torch")
        if torch_module is not None and torch_module.distributed.is_initialized():
            torch_module.distributed.destroy_process_group()
        wandb_module = sys.modules.get("wandb")
        if wandb_module is not None and wandb_module.run is not None:
            wandb_module.finish()
