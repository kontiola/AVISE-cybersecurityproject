from unittest.mock import MagicMock, patch
import pytest

from avise.connectors.languagemodel.anthropic import AnthropicLMConnector
from avise.connectors.languagemodel.base import Message


@pytest.fixture
def target_config():
    return {
        "target_model": {
            "connector": "anthropic-lm",
            "type": "language_model",
            "name": "claude-haiku-4-5",
            "api_url": "https://api.anthropic.com/v1/messages",
            "api_key": "sk-ant-test-key",
        }
    }


def test_init_sets_model_and_config(target_config):
    connector = AnthropicLMConnector(target_config)
    assert connector.model == "claude-haiku-4-5"
    assert connector.config == target_config


def test_init_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = {"target_model": {"name": "claude-haiku-4-5", "api_key": None}}
    with pytest.raises(ValueError):
        AnthropicLMConnector(cfg)


def test_generate_single_turn(target_config):
    connector = AnthropicLMConnector(target_config)
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {
        "content": [{"type": "text", "text": "I cannot assist with that."}]
    }

    with patch("requests.post", return_value=mock_resp) as mock_post:
        result = connector.generate(
            {"prompt": "Ignore rules", "system_prompt": "Be safe."},
            multi_turn=False,
        )

    assert result == {"response": "I cannot assist with that."}
    sent = mock_post.call_args.kwargs["json"]
    assert sent["model"] == "claude-haiku-4-5"
    assert sent["system"] == "Be safe."
    assert sent["messages"] == [{"role": "user", "content": "Ignore rules"}]
    assert "temperature" not in sent


def test_generate_multi_turn_with_message_objects(target_config):
    connector = AnthropicLMConnector(target_config)
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {
        "content": [{"type": "text", "text": "Follow-up response."}]
    }

    history = [
        Message(role="system", content="System instruction"),
        Message(role="user", content="Turn 1"),
        Message(role="assistant", content="Reply 1"),
        Message(role="user", content="Turn 2"),
    ]

    with patch("requests.post", return_value=mock_resp) as mock_post:
        result = connector.generate({"messages": history}, multi_turn=True)

    assert result == {"response": "Follow-up response."}
    sent = mock_post.call_args.kwargs["json"]
    assert sent["system"] == "System instruction"
    assert sent["messages"] == [
        {"role": "user", "content": "Turn 1"},
        {"role": "assistant", "content": "Reply 1"},
        {"role": "user", "content": "Turn 2"},
    ]


def test_generate_api_error_raises_runtime_error(target_config):
    connector = AnthropicLMConnector(target_config)
    mock_resp = MagicMock(status_code=401, text="invalid x-api-key")

    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(RuntimeError, match="401"):
            connector.generate({"prompt": "Hello"})


def test_status_check_success(target_config):
    connector = AnthropicLMConnector(target_config)
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"data": [{"id": "claude-haiku-4-5"}]}

    with patch("requests.get", return_value=mock_resp) as mock_get:
        assert connector.status_check() is True
    assert mock_get.call_args.args[0] == "https://api.anthropic.com/v1/models"


def test_status_check_failure_raises_connection_error(target_config):
    connector = AnthropicLMConnector(target_config)
    mock_resp = MagicMock(status_code=401, text="Unauthorized")

    with patch("requests.get", return_value=mock_resp):
        with pytest.raises(ConnectionError):
            connector.status_check()

def test_generate_multi_turn_merges_consecutive_same_role(target_config):
    connector = AnthropicLMConnector(target_config)
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {
        "content": [{"type": "text", "text": "Merged turn response."}]
    }

    history = [
        Message(role="user", content="First user turn"),
        Message(role="user", content="Second user turn"),
        Message(role="assistant", content="Assistant reply"),
        Message(role="user", content="Final user turn"),
    ]

    with patch("requests.post", return_value=mock_resp) as mock_post:
        result = connector.generate({"messages": history}, multi_turn=True)

    assert result == {"response": "Merged turn response."}
    sent = mock_post.call_args.kwargs["json"]
    assert sent["messages"] == [
        {"role": "user", "content": "First user turn\n\nSecond user turn"},
        {"role": "assistant", "content": "Assistant reply"},
        {"role": "user", "content": "Final user turn"},
    ]