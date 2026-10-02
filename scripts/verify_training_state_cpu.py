#!/usr/bin/env python3
"""Real tiny CPU AdamW state capture/checks, explicitly not verl/GPU acceptance."""

import argparse
import copy
import json
import random
from pathlib import Path

from agent_rl_diagnostics.integrations.training_state import (
    capture_training_state,
    compare_training_states,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--omegaconf-betas", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output directory")
    import numpy as np
    import torch

    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(5422)
    np.random.seed(5422)
    random.seed(5422)
    model = torch.nn.Linear(4, 2)
    initial = copy.deepcopy(model.state_dict())
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    batch = {
        "inputs": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        "labels": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    }
    for name in ("off", "on"):
        model.load_state_dict(initial)
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        betas = (0.9, 0.999)
        if args.omegaconf_betas:
            from omegaconf import OmegaConf

            betas = OmegaConf.create(list(betas))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=betas)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        before = capture_training_state(model, optimizer, scheduler, batch)
        assert capture_training_state(model, optimizer, scheduler, batch) == before
        if name == "on":
            # A real deterministic diagnostic calculation that consumes no RNG.
            with torch.no_grad():
                diagnostic = model(batch["inputs"]).square().mean().item()
            assert diagnostic >= 0
            assert capture_training_state(model, optimizer, scheduler, batch) == before
        optimizer.zero_grad()
        loss = (model(batch["inputs"]) - batch["labels"]).square().mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        optimizer.step()
        scheduler.step()
        after = capture_training_state(model, optimizer, scheduler, batch)
        directory = args.output / name
        directory.mkdir()
        proof = {
            "schema_version": "ard-training-state-v1",
            "update": 1,
            "scope": "cpu_optimizer_fixture",
            "before": before,
            "after": after,
            "framework_versions": {"torch": torch.__version__, "numpy": np.__version__},
            "actor_grad_norms": [norm],
        }
        with (directory / "update-000001.json").open("x") as stream:
            json.dump(proof, stream, indent=2)
    report = compare_training_states(args.output / "off", args.output / "on")
    assert report["status"] == "matched_observed_actor_states"
    assert not report["cuda_rng_observed"] and not report["full_trainer_parity_verified"]
    with (args.output / "verification.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
