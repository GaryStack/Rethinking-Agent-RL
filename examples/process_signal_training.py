"""CPU REINFORCE demonstration: an external judge changes actual optimizer updates.

This is an executable integration example, not a PPO/GRPO/GiGPO implementation.
"""

import argparse
import json
import math

from agent_rl_diagnostics.training.process_signals import CachedJudge, reward_for_episode


def train(coefficient):
    logit = 0.0
    judge = CachedJudge(lambda step: float(step["action"] == "good"), identity="toy-good-v1")
    history = []
    for action in ("good", "bad", "good", "good"):
        p = 1 / (1 + math.exp(-logit))
        mixed = reward_for_episode(
            0.0,
            [{"action": action}],
            score=judge.score,
            coefficient=coefficient,
            failure_policy="error",
        )
        # d log pi(action) / d logit = I(good)-p. Gradient ascent on reward.
        gradient = mixed["reward"] * (float(action == "good") - p)
        logit += 0.1 * gradient
        history.append({**mixed, "action": action, "gradient": gradient, "logit": logit})
    return {
        "algorithm": "toy_reinforce",
        "evidence_kind": "demo_synthetic",
        "coefficient": coefficient,
        "final_logit": logit,
        "history": history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coefficient", type=float, required=True)
    print(json.dumps(train(parser.parse_args().coefficient), indent=2, allow_nan=False))
