"""Public external-scoring adapter. Reuses the fixed-state metric and cluster CI."""

import copy

from ..contracts.schema import SchemaError, finite, integer
from .fixed import summarize_fixed


def evaluate_external_step(states, *, generate, score, scorer, model_id, samples=8):
    """generate(state, sample) -> response; score(state, response) -> [0,1].

    State schema: state_id, task_id, prefix_depth, remaining_budget, plus external
    payload (messages/snapshot recommended). Callbacks receive independent copies.
    This Python API does not register a new built-in rollout environment.
    """
    integer(samples, "samples", 1)
    if not states or len({s["state_id"] for s in states}) != len(states):
        raise SchemaError("external states must be nonempty and have unique IDs")
    records = []
    for state in states:
        integer(state["prefix_depth"], "prefix_depth")
        integer(state["remaining_budget"], "remaining_budget", 1)
        for sample in range(samples):
            response = generate(copy.deepcopy(state), sample)
            value = finite(score(copy.deepcopy(state), response), "external local_score")
            if not 0 <= value <= 1:
                raise SchemaError("external scores must be in [0,1]")
            records.append(
                {
                    "state_id": state["state_id"],
                    "sample_id": sample,
                    "model_id": model_id,
                    "status": "completed",
                    "response": response,
                    "step": {"local_score": value, "scorer": scorer},
                }
            )
    return {
        "records": records,
        "summary": summarize_fixed(
            records, states, samples=samples, scorer=scorer, model_id=model_id
        ),
    }
