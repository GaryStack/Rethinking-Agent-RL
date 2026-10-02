#!/usr/bin/env python3
"""Actual CPU initialization regression, explicitly not GPU/VeRL acceptance."""

import argparse
import json
import random
from pathlib import Path

from agent_rl_diagnostics.artifacts.io import canonical_hash, write_json
from agent_rl_diagnostics.integrations.training_state import capture_training_state
from agent_rl_diagnostics.integrations.worker_seed import initialize_actor_rng


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output directory")
    import numpy as np
    import torch

    torch.set_num_threads(1)

    def sample(seed):
        protocol = initialize_actor_rng(seed, cuda_device=None)
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        batch = {"inputs": torch.rand(2, 4)}
        streams = [random.random(), np.random.random(), torch.rand(1).item()]
        state = capture_training_state(model, optimizer, None, batch)
        assert capture_training_state(model, optimizer, None, batch) == state
        return {"protocol": protocol, "state": state, "streams": streams}

    first, second, other = sample(5422), sample(5422), sample(5423)
    assert first == second and first != other
    report = {
        "scope": "actual_cpu_initialization_fixture",
        "same_seed_matches": True,
        "different_seed_differs": True,
        "capture_does_not_reset_rng": True,
        "same_seed_hash": canonical_hash(first),
        "other_seed_hash": canonical_hash(other),
        "GPU_verified": False,
        "full_trainer_parity_verified": False,
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "verification.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
