import pytest

from agent_rl_diagnostics.contracts.schema import SchemaError
from agent_rl_diagnostics.evaluators.external import evaluate_external_step


def test_external_scorer_restores_inputs_and_weights_states():
    states = [
        {
            "state_id": str(i),
            "task_id": "task",
            "prefix_depth": i,
            "remaining_budget": 2,
            "snapshot": {"value": i},
        }
        for i in range(2)
    ]

    def generate(state, sample):
        state["snapshot"]["value"] = 99
        return sample

    result = evaluate_external_step(
        states,
        generate=generate,
        score=lambda state, reply: state["snapshot"]["value"],
        scorer="toy_v1",
        model_id="toy",
        samples=3,
    )
    assert result["summary"]["r_single"] == 0.5
    assert states[0]["snapshot"]["value"] == 0
    with pytest.raises(SchemaError):
        evaluate_external_step(
            states, generate=generate, score=lambda state, reply: 2, scorer="bad", model_id="toy"
        )
