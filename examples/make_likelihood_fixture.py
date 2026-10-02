"""Regenerate authored token scores, not real model measurements."""

import json
from pathlib import Path

from agent_rl_diagnostics.analysis.likelihood import score_record


def generate(output):
    output.mkdir(parents=True, exist_ok=True)
    events, base, rl = [], [], []
    for source in ("base", "rl"):
        event = {
            "event_id": f"{source}-event",
            "task_id": "toy-task",
            "trajectory_id": f"{source}-trajectory",
            "source": source,
            "success": True,
            "prompt": "Toy prompt: ",
            "response": "a b",
            "action_span": [2, 3],
        }
        events.append(event)
        for model, logs, sink in (("toy-base", [-1.0, -2.0], base), ("toy-rl", [-0.5, -1.5], rl)):
            sink.append(
                score_record(
                    event,
                    model_id=model,
                    prompt_ids=[10],
                    response_ids=[1, 2],
                    logprobs=logs,
                    offsets=[(0, 1), (2, 3)],
                )
            )
    for name, value in (("events", events), ("base-scores", base), ("rl-scores", rl)):
        (output / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")


if __name__ == "__main__":
    generate(Path(__file__).parent / "likelihood")
