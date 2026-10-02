"""Explicit terminal + process reward composition, with content-addressed judge cache."""

from __future__ import annotations

import math

from ..artifacts.io import canonical_hash
from ..contracts.schema import SchemaError, finite


def mix_rewards(terminal_reward, scores, *, coefficient, failure_policy):
    """R = terminal + coefficient * mean(step scores). No paper coefficient is inferred."""
    if failure_policy not in {"error", "terminal_only"}:
        raise SchemaError("failure_policy must be error or terminal_only")
    finite(coefficient, "coefficient")
    finite(terminal_reward, "terminal_reward")
    if coefficient < 0:
        raise SchemaError("finite terminal reward and nonnegative coefficient required")
    for value in scores:
        if value is not None and not 0 <= finite(value, "process score") <= 1:
            raise SchemaError("process scores must be finite in [0,1]")
    if not scores or any(x is None for x in scores):
        if failure_policy == "error":
            raise SchemaError("process scoring unavailable")
        return {"reward": terminal_reward, "process_mean": None, "status": "terminal_only"}
    mean = math.fsum(scores) / len(scores)
    return {"reward": terminal_reward + coefficient * mean, "process_mean": mean, "status": "ok"}


class CachedJudge:
    """Caller owns cache persistence and judge identity (revision/prompt/scoring rule)."""

    def __init__(self, judge, *, identity, cache=None):
        if not identity:
            raise SchemaError("explicit judge identity required")
        self.judge, self.identity = judge, identity
        self.cache = cache if cache is not None else {}

    def score(self, step):
        key = canonical_hash([self.identity, step])
        if key not in self.cache:
            value = self.judge(step)
            if value is not None:
                mix_rewards(0, [value], coefficient=1, failure_policy="error")
                self.cache[key] = value
            return value
        value = self.cache[key]
        mix_rewards(0, [value], coefficient=1, failure_policy="error")
        return value


def oracle_score(step, *, scorer):
    if step.get("scorer") != scorer:
        raise SchemaError("oracle scoring units differ")
    value = step.get("local_score")
    mix_rewards(0, [value], coefficient=1, failure_policy="error")
    return value


def reward_for_episode(terminal_reward, steps, *, score, coefficient, failure_policy):
    # Judge exceptions propagate; only explicitly unavailable None triggers fallback.
    result = mix_rewards(
        terminal_reward,
        [score(step) for step in steps],
        coefficient=coefficient,
        failure_policy=failure_policy,
    )
    return {
        **result,
        "coefficient": coefficient,
        "steps": len(steps),
        "terminal_reward": terminal_reward,
        "failure_policy": failure_policy,
        "aggregation": "mean_steps",
    }


def grid_process_reward(trace, *, coefficient, failure_policy):
    """Pinned grid agent-loop reward_score path, audited oracle Q* inputs."""
    from ..integrations.grid_trace import audit_grid_trace

    audit_grid_trace(trace)
    steps = [step["outcome"] for step in trace["steps"]]
    return reward_for_episode(
        trace["terminal_reward"],
        steps,
        score=lambda step: oracle_score(step, scorer="frozenlake_q_star_gamma_0.95_v1"),
        coefficient=coefficient,
        failure_policy=failure_policy,
    )


def process_signal_settings(coefficient, failure_policy):
    """Validated explicit settings shared by launchers and the native loop."""
    mix_rewards(0, [0], coefficient=coefficient, failure_policy=failure_policy)
    return {"enabled": True, "coefficient": coefficient, "failure_policy": failure_policy}


def process_training_phase(info):
    """Never infer validation from temperature: require explicit dataset provenance."""
    phase = info.get("ard_phase")
    if phase not in {"train", "validation"}:
        raise SchemaError("process signal requires extra_info.ard_phase=train or validation")
    return phase
