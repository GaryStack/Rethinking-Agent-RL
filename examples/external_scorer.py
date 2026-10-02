"""New external scorer without modifying the package; CPU-only authored fixture."""

import json

from agent_rl_diagnostics.evaluators.external import evaluate_external_step

states = [
    {
        "state_id": "toy-a",
        "task_id": "toy",
        "prefix_depth": 0,
        "remaining_budget": 2,
        "messages": [{"role": "user", "content": "Say YES"}],
        "expected": "YES",
    }
]

if __name__ == "__main__":
    result = evaluate_external_step(
        states,
        generate=lambda state, sample: "YES",
        score=lambda state, response: float(response == state["expected"]),
        scorer="toy_exact_match_v1",
        model_id="scripted",
        samples=2,
    )
    print(json.dumps(result, indent=2))
