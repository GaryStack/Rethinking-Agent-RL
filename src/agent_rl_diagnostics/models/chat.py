"""Chat generation adapters. Optional GPU dependencies are imported lazily."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from ..artifacts.io import canonical_hash, file_hash
from ..contracts.schema import SchemaError, finite, integer, text


class BackendError(RuntimeError):
    """Generation failed; this is not a failed task completion."""


@dataclass(frozen=True)
class Sampling:
    temperature: float = 0.6
    top_p: float = 0.9
    max_new_tokens: int = 128
    top_k: int = 0

    def __post_init__(self) -> None:
        finite(self.temperature, "temperature")
        finite(self.top_p, "top_p")
        if self.temperature < 0 or not 0 < self.top_p <= 1:
            raise SchemaError("temperature must be >=0 and top_p in (0,1]")
        integer(self.max_new_tokens, "max_new_tokens", 1)
        integer(self.top_k, "top_k")


@dataclass(frozen=True)
class Generation:
    text: str
    metadata: dict[str, Any]


class ChatModel(Protocol):
    metadata: dict[str, Any]

    def generate(self, messages: Sequence[dict[str, str]], *, seed: int) -> Generation: ...


def validate_messages(messages: Sequence[dict[str, str]]) -> None:
    if not messages:
        raise SchemaError("chat requires complete nonempty messages")
    for message in messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise SchemaError("messages require exactly role and content")
        if message["role"] not in {"system", "user", "assistant"}:
            raise SchemaError("unsupported message role")
        if not isinstance(message["content"], str):
            raise SchemaError("message content must be text")
    if messages[-1]["role"] != "user":
        raise SchemaError("generation input must end with a user observation")


class HTTPChatModel:
    def __init__(self, spec: dict[str, Any], sampling: Sampling) -> None:
        self.spec, self.sampling = dict(spec), sampling
        parsed = urllib.parse.urlsplit(spec["base_url"])
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.query
            or parsed.fragment
        ):
            raise SchemaError("base_url must be http(s), without embedded credentials or query")
        self.url = spec["base_url"].rstrip("/") + "/chat/completions"
        self.metadata = {
            "kind": "http_chat",
            "model_id": spec["model_id"],
            "revision": spec["revision"],
            "revision_status": "response_attested"
            if spec.get("require_response_revision")
            else "operator_declared",
            "seed_support": spec.get("seed_support", "unknown"),
            "teacher_forced_logprobs": False,
            "decoding": asdict(sampling),
            "top_k_control": "sent" if spec.get("send_top_k", False) else "not_sent",
            "chat_template_kwargs": spec.get("chat_template_kwargs", {}),
        }

    def generate(self, messages: Sequence[dict[str, str]], *, seed: int) -> Generation:
        import urllib.request

        validate_messages(messages)
        integer(seed, "seed")
        body: dict[str, Any] = {
            "model": self.spec["model_id"],
            "messages": list(messages),
            "n": 1,
            "temperature": self.sampling.temperature,
            "top_p": self.sampling.top_p,
            "max_tokens": self.sampling.max_new_tokens,
        }
        if self.spec.get("send_seed", True):
            body["seed"] = seed
        # top_k is an extension; opt in for vLLM/SGLang, not arbitrary providers.
        if self.spec.get("send_top_k", False):
            body["top_k"] = self.sampling.top_k
        if self.spec.get("chat_template_kwargs"):
            body["chat_template_kwargs"] = self.spec["chat_template_kwargs"]
        headers = {"Content-Type": "application/json"}
        if self.spec.get("api_key_env"):
            key = os.environ.get(self.spec["api_key_env"])
            if not key:
                raise BackendError("configured API key environment variable is unset")
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(
            self.url, json.dumps(body).encode(), headers, method="POST"
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.spec.get("timeout_seconds", 120)
            ) as response:
                payload = json.load(response)
                if not isinstance(payload, dict):
                    raise ValueError("API response must be an object")
                revision = response.headers.get("X-Model-Revision", payload.get("model_revision"))
        except urllib.error.HTTPError as exc:
            # Do not echo server bodies, credentials, URLs or prompt contents.
            raise BackendError(f"model API HTTP status {exc.code}") from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise BackendError(f"model API transport/JSON failure: {type(exc).__name__}") from exc
        try:
            choices = payload["choices"]
            if len(choices) != 1:
                raise ValueError("expected one completion")
            choice = choices[0]
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("completion content is not text")
        except (KeyError, TypeError, IndexError, ValueError) as exc:
            raise BackendError("model API returned an invalid completion shape") from exc
        if self.spec.get("require_response_revision") and revision != self.spec["revision"]:
            raise BackendError("response model revision does not match the requested checkpoint")
        expected = self.spec.get("expected_response_model")
        if expected and payload.get("model") != expected:
            raise BackendError("response model does not match expected_response_model")
        return Generation(
            content,
            {
                "response_model": payload.get("model"),
                "response_revision": revision,
                "finish_reason": choice.get("finish_reason"),
                "usage": payload.get("usage", {}),
                "seed_requested": seed if self.spec.get("send_seed", True) else None,
                "seed_support": self.metadata["seed_support"],
                "input_messages_hash": canonical_hash(messages),
            },
        )


class TransformersChatModel:
    def __init__(self, spec: dict[str, Any], sampling: Sampling) -> None:
        # Local checkpoints only: no implicit model download or remote code execution.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.sampling, self.spec = torch, sampling, dict(spec)
        path = Path(spec["path"])
        if not path.is_dir():
            raise SchemaError("transformers path must be an existing local checkpoint directory")
        device = spec.get("device", "cpu")
        if not re.fullmatch(r"cpu|cuda(?::\d+)?", device):
            raise SchemaError("device must be cpu or cuda[:index]")
        self.device = torch.device(device)
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(spec.get("dtype", "float32"))
        if dtype is None:
            raise SchemaError("dtype must be float32, float16 or bfloat16")
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        if not self.tokenizer.chat_template:
            raise SchemaError("checkpoint tokenizer has no chat template")
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        ).to(self.device)
        files = ["config.json", "tokenizer_config.json", "model.safetensors.index.json"]
        self.metadata = {
            "kind": "transformers_local",
            "model_id": spec["model_id"],
            "revision": spec["revision"],
            "revision_status": "operator_declared_local_files",
            "seed_support": "local_fork_rng",
            "teacher_forced_logprobs": False,
            "decoding": asdict(sampling),
            "chat_template_hash": canonical_hash(self.tokenizer.chat_template),
            "local_metadata_hashes": {
                name: file_hash(path / name) for name in files if (path / name).is_file()
            },
            "full_weight_hash_verified": False,
            "torch_version": torch.__version__,
        }

    def generate(self, messages: Sequence[dict[str, str]], *, seed: int) -> Generation:
        validate_messages(messages)
        integer(seed, "seed")
        torch = self.torch
        inputs = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            **self.spec.get("chat_template_kwargs", {}),
        ).to(self.device)
        options: dict[str, Any] = {
            "max_new_tokens": self.sampling.max_new_tokens,
            "do_sample": self.sampling.temperature > 0,
            "num_beams": 1,
            "repetition_penalty": 1.0,
            "pad_token_id": self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id,
        }
        if self.sampling.temperature > 0:
            options.update(
                temperature=self.sampling.temperature,
                top_p=self.sampling.top_p,
                top_k=self.sampling.top_k,
                min_p=None,
                typical_p=1.0,
                epsilon_cutoff=0.0,
                eta_cutoff=0.0,
            )
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        previous_mode = self.model.training
        try:
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                torch.random.default_generator.manual_seed(seed)
                if devices:
                    with torch.cuda.device(self.device):
                        torch.cuda.manual_seed(seed)
                self.model.eval()
                tokens = self.model.generate(**inputs, **options)
        finally:
            self.model.train(previous_mode)
        prompt_tokens = inputs["input_ids"].shape[-1]
        generated = tokens[0, prompt_tokens:]
        return Generation(
            self.tokenizer.decode(generated, skip_special_tokens=True),
            {
                "seed_requested": seed,
                "seed_support": "local_fork_rng",
                "input_messages_hash": canonical_hash(messages),
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(generated)},
                "prompt_token_ids": inputs["input_ids"][0].tolist(),
                "response_token_ids": generated.tolist(),
                "finish_reason": "length"
                if len(generated) == self.sampling.max_new_tokens
                else "stop",
            },
        )


def create_model(spec: dict[str, Any], sampling: Sampling) -> ChatModel:
    text(spec.get("model_id"), "model_id")
    text(spec.get("revision"), "revision")
    if spec.get("backend") == "http":
        return HTTPChatModel(spec, sampling)
    if spec.get("backend") == "transformers":
        return TransformersChatModel(spec, sampling)
    raise SchemaError("backend must be http or transformers")
