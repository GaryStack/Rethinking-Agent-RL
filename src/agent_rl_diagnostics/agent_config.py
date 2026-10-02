from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .artifacts.io import canonical_hash, read_json
from .contracts.schema import SchemaError, finite, integer, text
from .environments.execution import NamespaceExecutor
from .environments.frozenlake import GridTask
from .models import Sampling


@dataclass(frozen=True)
class AgentConfig:
    raw: dict[str, Any]
    tasks: tuple[dict[str, Any], ...]
    model: dict[str, Any]
    sampling: Sampling
    output: Path | None

    @property
    def config_hash(self) -> str:
        return canonical_hash({k: v for k, v in self.raw.items() if k != "output"})

    @property
    def model_id(self) -> str:
        return self.model["model_id"]

    @property
    def checkpoint_id(self) -> str:
        return self.model["revision"]

    @property
    def environment(self) -> str:
        return self.raw["environment"]

    def executor(self) -> NamespaceExecutor:
        return NamespaceExecutor(**self.raw.get("executor", {}))


def parse_agent_config(value: Any, *, base: Path, recorded: bool = False) -> AgentConfig:
    if not isinstance(value, dict):
        raise SchemaError("agent config must be an object")
    raw = copy.deepcopy(value)
    allowed = {
        "schema_version",
        "protocol_id",
        "evidence_kind",
        "environment",
        "tasks",
        "tasks_file",
        "model",
        "decoding",
        "budgets",
        "ks",
        "samples_per_task",
        "seed",
        "bootstrap_replicates",
        "output",
        "executor",
        "failure_policy",
        "split",
        "coding_prompt_profile",
        "grid_prompt_profile",
    }
    if set(raw) - allowed or raw.get("schema_version") != "ard-agent-config-v1":
        raise SchemaError("unsupported agent config schema/keys")
    if raw.get("protocol_id") != "toolkit_v1" or raw.get("evidence_kind") not in {
        "raw_recomputable",
        "demo_synthetic",
    }:
        raise SchemaError("agent config requires toolkit_v1 and explicit runnable evidence kind")
    if raw.get("environment") not in {"frozenlake_deterministic", "deepcoder_stdin"}:
        raise SchemaError("unsupported environment")
    if "coding_prompt_profile" in raw and (
        raw["environment"] != "deepcoder_stdin"
        or raw["coding_prompt_profile"] not in {"legacy_v1", "action_only_v2"}
    ):
        raise SchemaError("coding_prompt_profile is legacy_v1 or action_only_v2, coding only")
    if "grid_prompt_profile" in raw and (
        raw["environment"] != "frozenlake_deterministic"
        or raw["grid_prompt_profile"] not in {"legacy_v1", "action_only_v2"}
    ):
        raise SchemaError("grid_prompt_profile is legacy_v1 or action_only_v2, grid only")
    if ("tasks" in raw) == ("tasks_file" in raw):
        raise SchemaError("provide exactly one of tasks or tasks_file")
    if "tasks_file" in raw:
        if recorded:
            raise SchemaError("recorded configurations must contain embedded tasks")
        text(raw["tasks_file"], "tasks_file")
        raw["tasks"] = read_json((base / raw.pop("tasks_file")).resolve())
    if not isinstance(raw["tasks"], list) or not raw["tasks"]:
        raise SchemaError("tasks must be a nonempty array")
    seen_ids, seen_content = set(), set()
    for task in raw["tasks"]:
        if not isinstance(task, dict):
            raise SchemaError("task must be an object")
        text(task.get("task_id"), "task_id")
        if raw["environment"] == "frozenlake_deterministic":
            if (
                set(task) != {"task_id", "desc"}
                or not isinstance(task["desc"], list)
                or not all(isinstance(row, str) for row in task["desc"])
            ):
                raise SchemaError("grid tasks require task_id and desc string array")
            GridTask(task["task_id"], tuple(task["desc"]))
        else:
            if set(task) != {"task_id", "question", "public_tests", "hidden_tests"}:
                raise SchemaError(
                    "coding tasks require task_id, question, public_tests, hidden_tests"
                )
            text(task["question"], "question")
            for key in ("public_tests", "hidden_tests"):
                if not isinstance(task[key], list):
                    raise SchemaError("test sets must be arrays")
                for case in task[key]:
                    if (
                        not isinstance(case, dict)
                        or set(case) != {"input", "output"}
                        or not all(isinstance(v, str) for v in case.values())
                    ):
                        raise SchemaError("tests require input/output strings")
            if not task["public_tests"] + task["hidden_tests"]:
                raise SchemaError("coding task has no tests")
        digest = canonical_hash({k: v for k, v in task.items() if k != "task_id"})
        if task["task_id"] in seen_ids or digest in seen_content:
            raise SchemaError("duplicate task ID or task content")
        seen_ids.add(task["task_id"])
        seen_content.add(digest)
    model = raw.get("model")
    if not isinstance(model, dict) or model.get("backend") not in {"http", "transformers"}:
        raise SchemaError("model requires http/transformers backend")
    model_allowed = {
        "backend",
        "model_id",
        "revision",
        "path",
        "device",
        "dtype",
        "base_url",
        "api_key_env",
        "send_seed",
        "send_top_k",
        "seed_support",
        "timeout_seconds",
        "chat_template_kwargs",
        "require_response_revision",
        "expected_response_model",
    }
    if set(model) - model_allowed:
        raise SchemaError("unknown model keys; credentials must be passed through api_key_env")
    text(model.get("model_id"), "model_id")
    text(model.get("revision"), "revision")
    required = "path" if model["backend"] == "transformers" else "base_url"
    text(model.get(required), required)
    if model["backend"] == "transformers":
        if recorded:
            if not Path(model["path"]).is_absolute():
                raise SchemaError("recorded model path must be absolute")
        else:
            model["path"] = str((base / model["path"]).resolve())
        if not re.fullmatch(r"cpu|cuda(?::\d+)?", model.get("device", "cpu")):
            raise SchemaError("device must be cpu or cuda[:index]")
        if model.get("dtype", "float32") not in {"float32", "float16", "bfloat16"}:
            raise SchemaError("unsupported model dtype")
    else:
        parsed = urlsplit(model["base_url"])
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.query
            or parsed.fragment
        ):
            raise SchemaError("base_url requires http(s), without credentials, query or fragment")
        if "timeout_seconds" in model and finite(model["timeout_seconds"], "timeout_seconds") <= 0:
            raise SchemaError("model timeout_seconds must be positive")
    if model.get("api_key_env") and not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*", model["api_key_env"]
    ):
        raise SchemaError("api_key_env must be an environment variable name")
    for key in ("send_seed", "send_top_k", "require_response_revision"):
        if key in model and type(model[key]) is not bool:
            raise SchemaError(f"{key} must be boolean")
    if model.get("seed_support", "unknown") not in {"unknown", "supported", "unsupported"}:
        raise SchemaError("invalid declared seed support")
    if not isinstance(model.get("chat_template_kwargs", {}), dict):
        raise SchemaError("chat_template_kwargs must be an object")
    try:
        sampling = Sampling(**raw.get("decoding", {}))
    except TypeError as exc:
        raise SchemaError("invalid decoding settings") from exc
    for key in ("budgets", "ks"):
        if not isinstance(raw.get(key), list) or not raw[key]:
            raise SchemaError(f"{key} must be a nonempty array")
        for item in raw[key]:
            integer(item, key, 1)
        if len(set(raw[key])) != len(raw[key]):
            raise SchemaError(f"{key} values must be unique")
    samples = integer(raw.get("samples_per_task"), "samples_per_task", 1)
    if max(raw["ks"]) > samples:
        raise SchemaError("samples_per_task must cover max(K)")
    raw["seed"] = integer(raw.get("seed", 5422), "seed")
    raw["bootstrap_replicates"] = integer(
        raw.get("bootstrap_replicates", 1000), "bootstrap_replicates", 1
    )
    raw["failure_policy"] = raw.get("failure_policy", "fail")
    if raw["failure_policy"] not in {"fail", "record"}:
        raise SchemaError("failure_policy must be fail or record")
    raw["split"] = raw.get("split", "unspecified")
    if raw["split"] not in {"unspecified", "train", "validation", "test", "smoke"}:
        raise SchemaError("unsupported split label")
    output = raw.get("output")
    if output is not None:
        text(output, "output")
        output = (base / output).resolve()
    if "executor" in raw and (
        not isinstance(raw["executor"], dict)
        or set(raw["executor"]) - {"timeout_seconds", "memory_mb", "output_bytes"}
    ):
        raise SchemaError("invalid executor settings")
    return AgentConfig(raw, tuple(raw["tasks"]), model, sampling, output)


def load_agent_config(path: Path) -> AgentConfig:
    return parse_agent_config(read_json(path), base=path.resolve().parent)


def load_recorded_agent_config(path: Path) -> AgentConfig:
    """Audit the original path identity without host-dependent symlink rebasing.

    Never use this loader to generate new model outputs. Weight rehashing remains
    a separate explicit operation using the current filesystem.
    """
    return parse_agent_config(read_json(path), base=path.resolve().parent, recorded=True)
