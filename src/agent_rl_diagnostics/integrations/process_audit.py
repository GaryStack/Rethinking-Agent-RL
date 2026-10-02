"""Join replayed process traces to native trainer reward exports without Torch."""

import math
from collections import Counter, defaultdict

from ..contracts.schema import SchemaError, finite, integer
from ..training.process_signals import process_signal_settings
from .grid_trace import audit_grid_trace


def require(condition, message):
    if not condition:
        raise SchemaError(message)


def audit_process_exports(
    traces, exports, *, settings, steps, group_size, train_task_ids, validation_task_ids, decode
):
    integer(steps, "steps", 1)
    integer(group_size, "group_size", 1)
    validated = process_signal_settings(settings["coefficient"], settings["failure_policy"])
    require(settings == validated, "process settings are not explicitly enabled/validated")
    require(train_task_ids and validation_task_ids, "process audit needs both task sets")
    expected, observed = defaultdict(list), defaultdict(list)
    counts = Counter()
    components = []
    for trace in traces:
        audit_grid_trace(trace)
        phase = trace.get("ard_phase")
        require(phase in {"train", "validation"}, "process trace needs explicit phase")
        step = trace["weight_version"] + (phase == "train")
        task_id = trace["task"]["task_id"]
        if phase == "train":
            require(1 <= step <= steps and task_id in train_task_ids, "unexpected train slot")
            signal = trace.get("process_signal")
            require(isinstance(signal, dict), "training trace lacks process signal")
            require(
                all(signal[key] == settings[key] for key in ("coefficient", "failure_policy")),
                "trace process settings differ from trainer",
            )
            reward = signal["reward"]
            component = reward - trace["terminal_reward"]
            components.append(component)
        else:
            require(
                0 <= step <= steps and task_id in validation_task_ids, "unexpected validation slot"
            )
            require("process_signal" not in trace, "validation process signal is forbidden")
            reward, component = trace["terminal_reward"], 0.0
        key = (
            phase,
            step,
            task_id,
            decode(trace["initial_prompt_ids"]),
            decode(trace["response_ids"]),
        )
        expected[key].append(
            (reward, trace["terminal_reward"], component, len(trace["steps"]), trace["budget"])
        )
        counts[(phase, step, task_id)] += 1
    required = {
        (phase, step, task): samples
        for phase, indices, tasks, samples in (
            ("train", range(1, steps + 1), train_task_ids, group_size),
            ("validation", range(steps + 1), validation_task_ids, 1),
        )
        for step in indices
        for task in tasks
    }
    require(dict(counts) == required, "process trace sample slots incomplete or duplicated")
    for phase, step, row in exports:
        require(row["step"] == step, "native export step mismatch")
        values = tuple(
            finite(row[key], key)
            for key in (
                "score",
                "grid_success",
                "process_component",
                "grid_actions",
                "action_budget",
            )
        )
        require(
            math.isclose(
                values[0],
                finite(row["composed_reward"], "composed_reward"),
                rel_tol=0,
                abs_tol=1e-6,
            ),
            "native score differs from exported composed reward",
        )
        observed[(phase, step, row["gts"], row["input"], row["output"])].append(values)
    require(
        set(expected) == set(observed), "native exports and traced generation identities differ"
    )
    for key, values in expected.items():
        actual = observed[key]
        require(len(actual) == len(values), "native export multiplicity mismatch")
        for left, right in zip(sorted(values), sorted(actual), strict=True):
            require(
                all(
                    math.isclose(x, y, rel_tol=0, abs_tol=1e-6)
                    for x, y in zip(left, right, strict=True)
                ),
                "native score/components disagree with replayed process reward",
            )
    return {
        "scope": "replayed_process_rewards_joined_to_native_token_level_score_exports",
        "accepted": True,
        "training_rows": len(components),
        "nonzero_process_training_rows": sum(value > 0 for value in components),
        "validation_rows": len(validation_task_ids) * (steps + 1),
        "native_training_score_and_composed_reward_match": True,
        "validation_terminal_only": True,
        "reward_absolute_tolerance": 1e-6,
        "paper_reproduction": False,
    }
