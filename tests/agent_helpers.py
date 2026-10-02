from pathlib import Path

from agent_rl_diagnostics.agent_config import parse_agent_config
from agent_rl_diagnostics.artifacts.io import canonical_hash
from agent_rl_diagnostics.models import Generation

CODE = "print(sum(map(int, input().split())))"


def agent_config(tmp_path: Path, *, coding=False, model_id="base", evidence="demo_synthetic"):
    tasks = [{"task_id": "corridor", "desc": ["SFFG"]}]
    if coding:
        tasks = [
            {
                "task_id": "sum",
                "question": "Read two integers and print their sum.",
                "public_tests": [{"input": "1 2\n", "output": "3\n"}],
                "hidden_tests": [{"input": "17 29\n", "output": "46\n"}],
            }
        ]
    return parse_agent_config(
        {
            "schema_version": "ard-agent-config-v1",
            "protocol_id": "toolkit_v1",
            "evidence_kind": evidence,
            "environment": "deepcoder_stdin" if coding else "frozenlake_deterministic",
            "tasks": tasks,
            "model": {
                "backend": "http",
                "model_id": model_id,
                "revision": model_id + "-r1",
                "base_url": "http://127.0.0.1:1/v1",
            },
            "decoding": {"temperature": 0.6, "top_p": 0.9, "max_new_tokens": 64},
            "budgets": [5],
            "ks": [1, 2],
            "samples_per_task": 2,
            "bootstrap_replicates": 20,
            "seed": 7,
        },
        base=tmp_path,
    )


class FixtureModel:
    metadata = {"kind": "test_fixture", "seed_support": "fixture_deterministic"}

    def __init__(self, response="RIGHT"):
        self.response = response
        self.calls = []

    def generate(self, messages, *, seed):
        self.calls.append((messages.copy(), seed))
        return Generation(
            self.response, {"input_messages_hash": canonical_hash(messages), "seed_requested": seed}
        )


class FixtureExecutor:
    """Authored-case lookup only; never executes candidate code outside the sandbox."""

    def preflight(self):
        pass

    def run(self, code, stdin=""):
        if code == CODE and stdin:
            return {
                "status": "ok",
                "returncode": 0,
                "stdout": str(sum(map(int, stdin.split()))) + "\n",
                "stderr": "",
            }
        return {"status": "program_error", "returncode": 1, "stdout": "", "stderr": "SyntaxError"}
