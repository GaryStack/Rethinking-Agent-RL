"""Optional local HF scorer. No sampling, truncation, downloads or remote code."""

from pathlib import Path

from ..analysis.likelihood import score_record
from ..contracts.schema import SchemaError


class TeacherForcingScorer:
    def __init__(self, path, *, model_id, device="cpu", max_length=4096):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not Path(path).is_dir():
            raise SchemaError("teacher forcing requires a local checkpoint")
        self.torch = torch
        self.model_id = model_id
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, use_fast=True, local_files_only=True, trust_remote_code=False
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                path, local_files_only=True, trust_remote_code=False
            )
            .to(device)
            .eval()
        )

    def score(self, event):
        if not isinstance(event, dict) or any(
            not isinstance(event.get(name), str) or not event[name]
            for name in ("prompt", "response")
        ):
            raise SchemaError("nonempty prompt and response strings required")
        prompt, response = event["prompt"], event["response"]
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        encoded = self.tokenizer(
            prompt + response, add_special_tokens=False, return_offsets_mapping=True
        )
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        p = len(prompt_ids)
        if not p or ids[:p] != prompt_ids or any(left < len(prompt) for left, _ in offsets[p:]):
            raise SchemaError("prompt/response boundary crosses a token; supply a stable boundary")
        limit = min(
            self.max_length, getattr(self.model.config, "max_position_embeddings", self.max_length)
        )
        if len(ids) > limit:
            raise SchemaError("context_plus_response_too_long; no implicit truncation")
        if len(ids) == p:
            raise SchemaError("response has no tokens")
        tensor = self.torch.tensor([ids], device=self.model.device)
        with self.torch.inference_mode():
            logits = self.model(input_ids=tensor, use_cache=False).logits[:, p - 1 : -1, :]
            logprobs = (
                logits.float()
                .log_softmax(-1)
                .gather(-1, tensor[:, p:].unsqueeze(-1))
                .flatten()
                .cpu()
                .tolist()
            )
        return score_record(
            event,
            model_id=self.model_id,
            prompt_ids=prompt_ids,
            response_ids=ids[p:],
            logprobs=logprobs,
            offsets=[(a - len(prompt), b - len(prompt)) for a, b in offsets[p:]],
        )
