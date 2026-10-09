import datetime
from unittest.mock import MagicMock

import pytest
import requests

from study_coach import ai


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GROQ_MODEL", raising=False)


@pytest.fixture
def no_secrets(monkeypatch):
    monkeypatch.setattr(ai, "_secret", lambda name: "")


def _ok_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


def test_missing_key_raises(monkeypatch, no_secrets):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ai.GroqError, match="GROQ_API_KEY not found"):
        ai.chat("hi")


def test_timeout_raises(monkeypatch, api_key):
    def fake_post(*args, **kwargs):
        raise requests.exceptions.Timeout()

    monkeypatch.setattr(ai.requests, "post", fake_post)
    with pytest.raises(ai.GroqError, match="timed out"):
        ai.chat("hi")


def test_http_error_raises(monkeypatch, api_key):
    resp = MagicMock()
    resp.status_code = 429
    resp.text = "rate limited"
    resp.raise_for_status.side_effect = requests.exceptions.HTTPError(response=resp)
    monkeypatch.setattr(ai.requests, "post", lambda *a, **k: resp)
    with pytest.raises(ai.GroqError, match="429"):
        ai.chat("hi")


def test_unexpected_error_raises(monkeypatch, api_key):
    resp = _ok_response("x")
    resp.json.return_value = {}  # malformed body -> KeyError
    monkeypatch.setattr(ai.requests, "post", lambda *a, **k: resp)
    with pytest.raises(ai.GroqError):
        ai.chat("hi")


def test_chat_success_uses_env_model_and_timeout(monkeypatch, api_key):
    monkeypatch.setenv("GROQ_MODEL", "custom/model")
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured.update(url=url, headers=headers, json=json, timeout=timeout)
        return _ok_response("  plan text  ")

    monkeypatch.setattr(ai.requests, "post", fake_post)
    assert ai.chat("prompt", system="sys") == "plan text"
    assert captured["url"] == ai.GROQ_URL
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["model"] == "custom/model"
    assert captured["json"]["messages"][0] == {"role": "system", "content": "sys"}
    assert captured["json"]["messages"][1] == {"role": "user", "content": "prompt"}
    assert captured["timeout"] == ai.REQUEST_TIMEOUT


def test_default_model(monkeypatch, no_secrets):
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    assert ai.resolve_model() == "openai/gpt-oss-120b"


def test_generate_study_plan_prompt_contains_inputs(monkeypatch, api_key):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["prompt"] = json["messages"][1]["content"]
        return _ok_response("ok")

    monkeypatch.setattr(ai.requests, "post", fake_post)
    ai.generate_study_plan(
        ["Math", "DSA"],
        3.0,
        {"Math": 5, "DSA": 2},
        datetime.time(9, 0),
        datetime.time(12, 0),
    )
    assert "Math (priority 5/5)" in captured["prompt"]
    assert "DSA (priority 2/5)" in captured["prompt"]
    assert "09:00 AM to 12:00 PM (3.0 hours total)" in captured["prompt"]


def test_feedback_prompt_includes_average(monkeypatch, api_key):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["prompt"] = json["messages"][1]["content"]
        return _ok_response("ok")

    monkeypatch.setattr(ai.requests, "post", fake_post)
    ai.get_ai_feedback("some plan", [6, 8])
    assert "average recent focus score is 7.0/10" in captured["prompt"]
