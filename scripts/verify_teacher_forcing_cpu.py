"""Optional CPU HF integration check using a newly initialized tiny local checkpoint."""

import json
import tempfile
from pathlib import Path


def verify():
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from agent_rl_diagnostics.contracts.schema import SchemaError
    from agent_rl_diagnostics.models.teacher_forcing import TeacherForcingScorer

    with tempfile.TemporaryDirectory(prefix="ard-hf-cpu-") as directory:
        path = Path(directory)
        backend = Tokenizer(WordLevel({"[UNK]": 0, "input": 1, "a": 2, "b": 3}, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
        tokenizer.save_pretrained(path)
        torch.manual_seed(42)
        model = GPT2LMHeadModel(
            GPT2Config(vocab_size=4, n_positions=16, n_embd=8, n_layer=1, n_head=1)
        ).eval()
        model.save_pretrained(path)
        scorer = TeacherForcingScorer(path, model_id="authored-tiny-random-v1", max_length=16)
        event = {
            "event_id": "e",
            "task_id": "t",
            "trajectory_id": "tr",
            "source": "base",
            "success": True,
            "prompt": "input ",
            "response": "a b",
            "action_span": [2, 3],
        }
        result = scorer.score(event)
        with torch.inference_mode():
            logits = model(torch.tensor([[1, 2, 3]])).logits[0]
        expected = [float(logits[0].log_softmax(-1)[2]), float(logits[1].log_softmax(-1)[3])]
        assert (
            max(abs(a - b) for a, b in zip(result["token_logprobs"], expected, strict=True)) < 1e-6
        )
        assert result["action_mask"] == [False, True]
        assert result["response_ids"] == [2, 3]
        try:
            scorer.score({**event, "prompt": "a", "response": "b", "action_span": None})
        except SchemaError:
            pass
        else:
            raise AssertionError("cross-boundary tokenization should fail")
        scorer.max_length = 2
        try:
            scorer.score(event)
        except SchemaError:
            pass
        else:
            raise AssertionError("oversized contexts must not truncate")
        return {
            "status": "verified",
            "scope": "tiny_random_local_hf_cpu",
            "real_research_checkpoint_verified": False,
            "response_tokens": 2,
            "checks": ["causal_shift", "raw_logprobs", "action_mask", "boundary", "length"],
        }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
