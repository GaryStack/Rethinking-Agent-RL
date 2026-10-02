"""Locked VeRL reward-to-optimizer CPU check with scripted actions and tiny random HF.

No Ray cluster, vLLM generation, research weights, GPU or paper training acceptance.
Only native actor device selection is patched to CPU; upstream algorithms are unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch


def verify(source, output):
    from agent_rl_diagnostics.artifacts.io import canonical_hash, new_output_directory, write_json
    from agent_rl_diagnostics.integrations.verl import verify_verl_source

    lock = verify_verl_source(source)
    sys.path.insert(0, str(source.resolve()))
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
    from verl.trainer.ppo.ray_trainer import compute_advantage
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.workers.config import FSDPActorConfig

    from agent_rl_diagnostics.environments.text_grid import TextGrid
    from agent_rl_diagnostics.integrations.grid_trace import audit_grid_trace
    from agent_rl_diagnostics.integrations.verl_grid import GridAgentLoop

    torch.set_num_threads(2)
    torch.manual_seed(5422)
    backend = Tokenizer(
        WordLevel(
            {"PAD": 0, "PROMPT": 1, "FEEDBACK": 2, "RIGHT": 3, "LEFT": 4, "UNK": 5}, unk_token="UNK"
        )
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="PAD", unk_token="UNK")
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=6,
            n_positions=16,
            n_embd=16,
            n_layer=1,
            n_head=2,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
            pad_token_id=0,
        )
    )
    initial = copy.deepcopy(model.state_dict())
    task = {"task_id": "authored-two-action-failure", "desc": ["SFG"]}

    class ScriptedServer:
        def __init__(self, actions):
            self.actions = iter(actions)

        async def generate(self, **kwargs):
            return NS(
                stop_reason="completed",
                extra_fields={"global_steps": 0},
                token_ids=[next(self.actions)],
                log_probs=None,
            )

    async def collect(coefficient, phase, directory):
        outputs = []
        for actions in ([3, 4], [4, 4]):
            loop = GridAgentLoop.__new__(GridAgentLoop)
            loop.config = OmegaConf.create(
                {
                    "trainer": {"rollout_data_dir": str(directory)},
                    "ard_process_signal": {
                        "enabled": True,
                        "coefficient": coefficient,
                        "failure_policy": "error",
                    },
                }
            )
            loop.rollout_config = NS(response_length=8)
            loop.server_manager = ScriptedServer(actions)
            loop.tokenizer = tokenizer

            async def template(messages, remove_system_prompt=False):
                return [2] if remove_system_prompt else [1]

            loop.apply_chat_template = template
            outputs.append(
                await loop.run(
                    {"temperature": 1.0},
                    extra_info={"task": task, "budget": 2, "ard_phase": phase},
                    raw_prompt=TextGrid(task, prompt_profile="action_only_v2").initial_messages(2),
                )
            )
        # Invoke actual upstream padding/reward placement, not a reimplementation.
        worker = AgentLoopWorker.__new__(AgentLoopWorker)
        worker.rollout_config = NS(prompt_length=1, response_length=8)
        worker.tokenizer, worker.processor = tokenizer, None
        worker.reward_loop_worker_handles = None
        worker.distillation_enabled = False
        padded = [
            await worker._agent_loop_postprocess(item, phase == "validation", raw_prompt=[])
            for item in outputs
        ]
        array = np.empty(2, dtype=object)
        array[:] = ["same-task-occurrence"] * 2
        data = worker._postprocess(
            padded, input_non_tensor_batch={"uid": array}, validate=phase == "validation"
        )
        traces = [json.loads(p.read_text()) for p in (directory / "grid-traces").glob("*.json")]
        for trace in traces:
            audit_grid_trace(trace)
            assert not trace["success"] and trace["terminal_reward"] == 0
            assert ("process_signal" in trace) == (phase == "train")
        return data, outputs

    with new_output_directory(output) as staging:
        batches = {}
        for name, coefficient, phase in (
            ("off", 0, "train"),
            ("on", 0.5, "train"),
            ("validation", 0.5, "validation"),
        ):
            data, outputs = asyncio.run(collect(coefficient, phase, staging / name))
            scores = [item.reward_score for item in outputs]
            assert torch.allclose(data.batch["rm_scores"].sum(-1), torch.tensor(scores))
            data.batch["token_level_rewards"] = data.batch["rm_scores"].clone()
            data = compute_advantage(
                data, adv_estimator="reinforce_plus_plus", config=OmegaConf.create({"gamma": 1.0})
            )
            assert not torch.any(data.batch["advantages"][data.batch["response_mask"] == 0])
            data.meta_info.update(
                temperature=1.0, micro_batch_size=1, use_dynamic_bsz=False, pad_token_id=0
            )
            batches[name] = (data, scores)
        off, on, validation = (batches[x][0] for x in ("off", "on", "validation"))
        assert torch.equal(off.batch["responses"], on.batch["responses"])
        assert torch.equal(off.batch["response_mask"], on.batch["response_mask"])
        assert not torch.any(off.batch["advantages"])
        assert torch.any(on.batch["advantages"])
        assert not torch.any(validation.batch["advantages"])
        torch.distributed.init_process_group(
            "gloo", init_method="file://" + str(staging / "dist-init"), rank=0, world_size=1
        )
        outcomes = {}
        try:
            for name in ("off", "on"):
                candidate = copy.deepcopy(model)
                candidate.load_state_dict(initial)
                optimizer = torch.optim.AdamW(candidate.parameters(), lr=0.001, weight_decay=0)
                config = FSDPActorConfig(
                    ppo_mini_batch_size=2,
                    ppo_micro_batch_size_per_gpu=1,
                    ppo_infer_micro_batch_size_per_gpu=1,
                    rollout_n=2,
                    use_torch_compile=False,
                    use_remove_padding=False,
                    calculate_entropy=False,
                    entropy_coeff=0,
                    use_kl_loss=False,
                    ppo_epochs=1,
                    shuffle=False,
                )
                actor = DataParallelPPOActor(config, candidate, optimizer)
                data = batches[name][0]
                # Native get_device_id returns accelerator index; CPU is an explicit test-only shim.
                with patch(
                    "verl.workers.actor.dp_actor.get_device_id", return_value=torch.device("cpu")
                ):
                    data.batch["old_log_probs"] = actor.compute_log_prob(data)["log_probs"].detach()
                    metrics = actor.update_policy(data)
                changed = sum(
                    not torch.equal(initial[key], value)
                    for key, value in candidate.state_dict().items()
                )
                outcomes[name] = {
                    "changed_tensors": changed,
                    "gradient_norm": float(np.mean(metrics["actor/grad_norm"])),
                    "rewards": batches[name][1],
                }
            assert outcomes["off"]["changed_tensors"] == 0
            assert outcomes["on"]["changed_tensors"] > 0
            assert outcomes["on"]["gradient_norm"] > 0
        finally:
            torch.distributed.destroy_process_group()
        assert not torch.cuda.is_initialized()
        report = {
            "status": "verified",
            "scope": "locked_native_scripted_rollout_tiny_hf_cpu_update",
            "upstream": lock,
            "outcomes": outcomes,
            "validation_rewards": batches["validation"][1],
            "response_mask_hash": canonical_hash(on.batch["response_mask"].tolist()),
            "native_actor_optimizer_updates": True,
            "scripted_rollouts": True,
            "tiny_random_model": True,
            "device_resolution_shim": "dp_actor.get_device_id=cpu",
            "RayPPOTrainer_fit_verified": False,
            "vllm_generation_verified": False,
            "research_model_training_verified": False,
            "paper_reproduction_verified": False,
        }
        write_json(staging / "verification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verl-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("set CUDA_VISIBLE_DEVICES= explicitly for this CPU-only check")
    print(json.dumps(verify(args.verl_source.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
