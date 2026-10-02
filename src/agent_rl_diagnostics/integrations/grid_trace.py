"""Dependency-free replay of real optional verl grid-loop masks and outcomes."""

from ..contracts.schema import SchemaError, integer
from ..environments.text_grid import TextGrid
from ..training.budgetmix import validate_assignment


def audit_grid_trace(trace: dict) -> dict:
    if trace.get("schema_version") != "ard-verl-grid-rollout-v1":
        raise SchemaError("unsupported training trace")
    budget = integer(trace["budget"], "budget", 1)
    if "budgetmix" in trace:
        validate_assignment(trace["budgetmix"])
        if (
            trace["budgetmix"]["budget"] != budget
            or trace["budgetmix"]["task_id"] != trace["task"]["task_id"]
        ):
            raise SchemaError("BudgetMix trace task/budget mismatch")
    integer(trace["weight_version"], "weight_version")
    env = TextGrid(trace["task"], prompt_profile="action_only_v2")
    if trace["initial_messages"] != env.initial_messages(budget):
        raise SchemaError("initial training prompt mismatch")
    if not trace["steps"] or len(trace["steps"]) > budget:
        raise SchemaError("invalid training action count")
    prompt, responses, masks = list(trace["initial_prompt_ids"]), [], []
    success, terminal = False, False
    for turn, step in enumerate(trace["steps"]):
        if terminal or step["turn"] != turn or step["remaining_budget"] != budget - turn:
            raise SchemaError("training budget/terminal mismatch")
        if step["weight_version"] != trace["weight_version"]:
            raise SchemaError("mixed training weight versions")
        if step["prompt_ids"] != prompt or step["snapshot_before"] != env.snapshot():
            raise SchemaError("training prefix/snapshot mismatch")
        tokens = step["response_ids"]
        if not tokens:
            raise SchemaError("empty training action")
        outcome = env.apply(step["response"], budget - turn)
        if outcome != step["outcome"] or step["snapshot_after"] != env.snapshot():
            raise SchemaError("training environment replay mismatch")
        terminal, success = outcome["terminated"], outcome["success"]
        prompt.extend(tokens)
        responses.extend(tokens)
        masks.extend([1] * len(tokens))
        if "feedback_ids" in step:
            if terminal or turn + 1 >= budget:
                raise SchemaError("feedback after termination")
            expected = [
                {
                    "role": "user",
                    "content": env.budget_observation(
                        budget, budget - turn - 1, outcome["observation"]
                    ),
                }
            ]
            if step["feedback_messages"] != expected:
                raise SchemaError("budget feedback mismatch")
            prompt.extend(step["feedback_ids"])
            responses.extend(step["feedback_ids"])
            masks.extend([0] * len(step["feedback_ids"]))
        elif not terminal and turn + 1 < len(trace["steps"]):
            raise SchemaError("missing inter-action feedback")
    if not terminal and len(trace["steps"]) != budget:
        raise SchemaError("silent action-budget truncation")
    reason = "success" if success else "environment_terminal" if terminal else "budget_exhausted"
    if (
        trace["success"] != success
        or trace["terminal_reward"] != float(success)
        or trace["termination_reason"] != reason
    ):
        raise SchemaError("terminal reward/termination mismatch")
    if trace["response_ids"] != responses or trace["response_mask"] != masks:
        raise SchemaError("generated/environment token mask mismatch")
    if "process_signal" in trace:
        if trace.get("ard_phase") not in {None, "train"}:
            raise SchemaError("process reward cannot alter native validation")
        from ..training.process_signals import oracle_score, reward_for_episode

        signal = trace["process_signal"]
        recomputed = reward_for_episode(
            trace["terminal_reward"],
            [step["outcome"] for step in trace["steps"]],
            score=lambda step: oracle_score(step, scorer="frozenlake_q_star_gamma_0.95_v1"),
            coefficient=signal["coefficient"],
            failure_policy=signal["failure_policy"],
        )
        if signal != recomputed:
            raise SchemaError("process reward does not match audited step scores")
    return {
        "actions": len(trace["steps"]),
        "generated_tokens": sum(masks),
        "environment_tokens": len(masks) - sum(masks),
        "terminal_reward": float(success),
        "weight_version": trace["weight_version"],
    }
