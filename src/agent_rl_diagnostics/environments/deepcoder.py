"""Derived stdin/stdout coding protocol; not the official LiveCodeBench checker."""

from __future__ import annotations

import re
from typing import Any

from ..artifacts.io import canonical_hash
from ..contracts.schema import SchemaError, integer
from .execution import NamespaceExecutor

SCORER = "derived_stdin_test_fraction_v1"


def parse_code_action(response: str) -> dict[str, Any]:
    final = response.rsplit("</think>", 1)[-1].strip()
    match = re.fullmatch(
        r"(?:(RUN|SUBMIT)\s*)?```(?:python|py)?\s*\n(.*?)\n```\s*",
        final,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return {"kind": "invalid"}
    # Explicitly documented implicit submission for bare fenced programs.
    return {"kind": (match.group(1) or "SUBMIT").lower(), "code": match.group(2)}


class DeepCoder:
    name = "deepcoder_stdin"
    scorer = SCORER

    def __init__(
        self,
        task: dict[str, Any],
        executor: NamespaceExecutor,
        *,
        prompt_profile: str = "legacy_v1",
    ) -> None:
        if prompt_profile not in {"legacy_v1", "action_only_v2"}:
            raise SchemaError("unsupported coding prompt profile")
        self.prompt_profile = prompt_profile
        self.task, self.executor = dict(task), executor
        self.code: str | None = None
        self.action_count = 0
        self.terminal = False

    def snapshot(self) -> dict[str, Any]:
        snapshot = {
            "version": "ard-code-snapshot-v1",
            "task_hash": canonical_hash(self.task),
            "current_code": self.code,
            "action_count": self.action_count,
            "terminal": self.terminal,
        }
        if self.prompt_profile != "legacy_v1":
            snapshot["prompt_profile"] = self.prompt_profile
        return snapshot

    def restore(self, snapshot: dict[str, Any]) -> None:
        if snapshot.get("prompt_profile", "legacy_v1") != self.prompt_profile:
            raise SchemaError("coding snapshot/prompt profile mismatch")
        if snapshot.get("version") != "ard-code-snapshot-v1" or snapshot.get(
            "task_hash"
        ) != canonical_hash(self.task):
            raise SchemaError("coding snapshot/task mismatch")
        if snapshot.get("terminal") is not False:
            raise SchemaError("cannot restore terminal coding state for generation")
        code = snapshot.get("current_code")
        if code is not None and not isinstance(code, str):
            raise SchemaError("snapshot current_code must be text or null")
        self.code = code
        self.action_count = integer(snapshot.get("action_count"), "action_count")
        self.terminal = False

    def initial_messages(self, total_budget: int) -> list[dict[str, str]]:
        examples = "\n\n".join(
            f"Input:\n{case['input']}\nOutput:\n{case['output']}"
            for case in self.task["public_tests"]
        )
        return [
            {
                "role": "system",
                "content": "Solve the programming task using Python stdin/stdout. Each action consumes one turn. Use RUN followed by a fenced Python program for scratch execution with empty stdin, or SUBMIT followed by a fenced Python program for final evaluation. A bare fenced program is also a final submission. SUBMIT ends the episode whether it passes or fails. Hidden tests are not available in scratch execution."
                + (
                    " Return ONLY the action name on its own line followed by one fenced Python program. No explanation, prose, comments outside the program, or text before or after the action. RUN cannot receive task input; input()-based programs may fail in scratch while still being correct submissions. With one action remaining, use SUBMIT, never RUN."
                    if self.prompt_profile == "action_only_v2"
                    else ""
                ),
            },
            {
                "role": "user",
                "content": self.budget_observation(
                    total_budget,
                    total_budget,
                    self.task["question"] + "\nPublic examples:\n" + examples,
                ),
            },
        ]

    def budget_observation(self, total_budget: int, remaining: int, observation: str) -> str:
        reminder = (
            "Return only SUBMIT and one fenced Python program; this is your last action.\n"
            if self.prompt_profile == "action_only_v2" and remaining == 1
            else ""
        )
        return f"Total action budget: {total_budget}. Remaining action budget including this action: {remaining}.\n{reminder}{observation}"

    def grade(self, code: str) -> dict[str, Any]:
        cases = self.task["public_tests"] + self.task["hidden_tests"]
        verdicts = []
        for case in cases:
            outcome = self.executor.run(code, case["input"])
            verdicts.append(
                {
                    "passed": outcome["status"] == "ok"
                    and outcome["stdout"].strip().split() == case["output"].strip().split(),
                    "execution_status": outcome["status"],
                }
            )
        count = sum(case["passed"] for case in verdicts)
        return {
            "passed_tests": count,
            "total_tests": len(cases),
            "test_fraction": count / len(cases),
            "passed": count == len(cases),
            "tests": verdicts,
            "checker": SCORER,
        }

    def apply(self, response: str, remaining: int) -> dict[str, Any]:
        integer(remaining, "remaining", 1)
        if self.terminal:
            raise SchemaError("cannot step terminal coding state")
        action = parse_code_action(response)
        self.action_count += 1
        success, score = False, 0.0
        observation = "Invalid action. Use RUN or SUBMIT followed by a fenced Python program."
        if action["kind"] != "invalid":
            self.code = action["code"]
            verdict = self.grade(self.code)
            score = verdict["test_fraction"]
            if action["kind"] == "run":
                result = self.executor.run(self.code)
                observation = f"Scratch execution status: {result['status']}\nSTDOUT:\n{result['stdout']}\nSTDERR:\n{result['stderr']}\nContinue with RUN or SUBMIT."
            else:
                self.terminal = True
                success = verdict["passed"]
                observation = "Submission evaluator: " + ("PASS" if success else "FAIL")
        return {
            "action": action,
            "observation": observation,
            "reward": float(success),
            "terminated": self.terminal,
            "success": success,
            "local_score": score,
            "scorer": SCORER,
        }
