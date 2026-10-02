#!/usr/bin/env python3
"""Exercise the real locked trainer pre-repeat hook and DataProto on CPU.

No trainer construction, Ray cluster, model weights, optimizer update or CUDA.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verl-source", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("set CUDA_VISIBLE_DEVICES= explicitly; this verification is CPU-only")
    from agent_rl_diagnostics.integrations.verl import verify_verl_source

    lock = verify_verl_source(args.verl_source)
    sys.path.insert(0, str(args.verl_source.resolve()))
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from verl.protocol import DataProto

    from agent_rl_diagnostics.artifacts.io import canonical_hash, write_json
    from agent_rl_diagnostics.environments.text_grid import TextGrid
    from agent_rl_diagnostics.integrations.verl_budgetmix import budgetmix_trainer_class
    from agent_rl_diagnostics.training.budgetmix import audit_groups

    trainer_type = budgetmix_trainer_class(args.verl_source)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    trainer = trainer_type.__new__(trainer_type)
    trainer.config = NS(
        ard_budgetmix=NS(tmax=2, seed=5422, output=str(args.output / "plans")),
        trainer=NS(resume_mode="disable"),
        actor_rollout_ref=NS(rollout=NS(n=4)),
    )
    tasks = [{"task_id": "authored-three-move", "desc": ["SFFG"]}] * 2 + [
        {"task_id": "authored-other", "desc": ["S", "F", "F", "G"]}
    ]
    results = []
    for update in (1, 2):
        trainer.global_steps = update
        records = [{"task": t, "budget": 2} for t in tasks]
        prompts = [TextGrid(t, prompt_profile="action_only_v2").initial_messages(2) for t in tasks]
        fields = {
            "extra_info": records,
            "raw_prompt": prompts,
            "uid": [f"actual-native-group-{i}" for i in range(3)],
        }
        arrays = {}
        for key, values in fields.items():
            array = np.empty(len(values), dtype=object)
            array[:] = values
            arrays[key] = array
        batch = DataProto.from_dict(
            tensors={"dummy_tensor": torch.zeros(3, 1, dtype=torch.uint8)}, non_tensors=arrays
        )
        generated = trainer._get_gen_batch(batch)
        repeated = generated.repeat(repeat_times=4, interleave=True)
        assignments = [r["budgetmix"] for r in repeated.non_tensor_batch["extra_info"]]
        summary = audit_groups(assignments, group_size=4)
        for index, entry in enumerate(repeated.non_tensor_batch["extra_info"]):
            messages = TextGrid(entry["task"], prompt_profile="action_only_v2").initial_messages(
                entry["budget"]
            )
            assert repeated.non_tensor_batch["raw_prompt"][index] == messages
            ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
            )
            assert 0 < len(ids) <= 256
            assert repeated.non_tensor_batch["uid"][index] == f"actual-native-group-{index // 4}"
        assert batch.batch["dummy_tensor"].device.type == "cpu"
        assert not torch.cuda.is_initialized()
        results.append(
            {
                "update": update,
                "metrics": summary,
                "repeated_assignments_sha256": canonical_hash(assignments),
                "structurally_infeasible_groups_retained": 3,
            }
        )
    report = {
        "schema_version": "ard-budgetmix-cpu-verification-v1",
        "scope": "actual_native_pre_repeat_hook_and_local_tokenizer",
        "source": lock,
        "updates": results,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "gpu_training_verified": False,
        "ppo_critic_verified": False,
        "gigpo_training_verified": False,
        "model_weights_loaded": False,
        "cuda_initialized": torch.cuda.is_initialized(),
        "paper_reproduction": False,
    }
    write_json(args.output / "verification.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
