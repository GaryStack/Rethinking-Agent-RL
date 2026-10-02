import io
import json
import urllib.request

import pytest

from agent_rl_diagnostics.artifacts.io import canonical_hash
from agent_rl_diagnostics.models.chat import BackendError, HTTPChatModel, Sampling


class Response(io.BytesIO):
    headers = {"X-Model-Revision": "r1"}


def spec(**kwargs):
    return {"model_id": "base", "revision": "r1", "base_url": "http://localhost:8000/v1", **kwargs}


def test_http_complete_messages_controls_and_credentials(monkeypatch):
    seen = {}

    def urlopen(request, timeout):
        seen.update(body=json.loads(request.data), headers=dict(request.header_items()))
        return Response(
            json.dumps(
                {
                    "model": "base",
                    "choices": [{"message": {"content": "RIGHT"}, "finish_reason": "stop"}],
                }
            ).encode()
        )

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("ARD_TEST_KEY", "private-credential")
    model = HTTPChatModel(
        spec(api_key_env="ARD_TEST_KEY", send_top_k=True, require_response_revision=True),
        Sampling(),
    )
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "observation"}]
    result = model.generate(messages, seed=123)
    assert seen["body"]["messages"] == messages
    assert seen["body"]["seed"] == 123 and seen["body"]["top_k"] == 0
    assert seen["headers"]["Authorization"] == "Bearer private-credential"
    assert result.metadata["input_messages_hash"] == canonical_hash(messages)
    assert "private-credential" not in json.dumps(model.metadata)


@pytest.mark.parametrize(
    "payload", [{}, {"choices": []}, {"choices": [{"message": {"content": None}}]}]
)
def test_http_bad_response_is_infrastructure_failure(monkeypatch, payload):
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: Response(json.dumps(payload).encode())
    )
    with pytest.raises(BackendError):
        HTTPChatModel(spec(), Sampling()).generate([{"role": "user", "content": "x"}], seed=0)


def test_http_revision_mismatch_fails(monkeypatch):
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **k: Response(b'{"choices":[{"message":{"content":"RIGHT"}}]}'),
    )
    with pytest.raises(BackendError, match="revision"):
        HTTPChatModel(spec(revision="r2", require_response_revision=True), Sampling()).generate(
            [{"role": "user", "content": "x"}], seed=0
        )


def test_http_unsent_top_k_explicit():
    assert HTTPChatModel(spec(), Sampling()).metadata["top_k_control"] == "not_sent"


def test_agent_record_in_memory_roundtrip(tmp_path):
    from agent_rl_diagnostics.contracts.agent import AgentTrajectory
    from agent_rl_diagnostics.evaluators.agent import rollout

    from .agent_helpers import FixtureModel, agent_config

    record = rollout(
        agent_config(tmp_path),
        {"task_id": "corridor", "desc": ["SFFG"]},
        FixtureModel(),
        run_id="test",
        budget=5,
        sample_id=0,
    )
    assert AgentTrajectory.from_dict(record.to_dict()) == record
