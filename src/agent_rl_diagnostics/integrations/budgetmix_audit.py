"""Replay completed BudgetMix grid groups; authored fixtures are not GPU evidence."""

from ..contracts.schema import SchemaError
from ..training.budgetmix import audit_groups
from .grid_trace import audit_grid_trace


def audit_budgetmix_traces(traces, *, group_size):
    if not isinstance(traces, list) or not traces:
        raise SchemaError("nonempty completed BudgetMix traces required")
    ids, assignments, groups, versions = set(), [], {}, set()
    exhausted = 0
    for trace in traces:
        audit_grid_trace(trace)
        request = trace.get("request_id")
        if not isinstance(request, str) or not request or request in ids:
            raise SchemaError("missing/duplicate completed rollout request ID")
        ids.add(request)
        assignment = trace.get("budgetmix")
        if assignment is None:
            raise SchemaError("BudgetMix group trace lacks an observed assignment")
        assignments.append(assignment)
        groups.setdefault(assignment["task_occurrence_id"], []).append(trace["terminal_reward"])
        versions.add(trace["weight_version"])
        exhausted += trace["termination_reason"] == "budget_exhausted"
    summary = audit_groups(assignments, group_size=group_size)
    if len(versions) != 1:
        raise SchemaError("mixed BudgetMix rollout weight versions")
    return {
        **summary,
        "weight_version": versions.pop(),
        "budget_terminated_fraction": exhausted / len(traces),
        "zero_reward_group_fraction": sum(all(r == 0 for r in values) for values in groups.values())
        / len(groups),
        "generated_tokens": sum(sum(t["response_mask"]) for t in traces),
        "actions": sum(len(t["steps"]) for t in traces),
    }
