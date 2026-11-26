import pytest
from unittest.mock import Mock, patch

from avise.connectors.languagemodel.deepseek import (
    DeepSeekLMConnector,
)
from avise.connectors.languagemodel.base import Message


TEST_CONFIG = {
    "target_model": {
        "name": "deepseek-flash",
        "api_key": "test-key",
        "api_url": "https://api.deepseek.com",
        "headers": None,
        "max_tokens": 100,
        "completion_kwargs": {},
    }
}

def test_init():

    connector = DeepSeekLMConnector(TEST_CONFIG)

    assert connector.model == "deepseek-flash"
    assert connector.api_key == "test-key"
    assert connector.base_url == "https://api.deepseek.com"

@patch("requests.post")
def test_single_turn_generation(mock_post):

    mock_response = Mock()

    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "test response"
                }
            }
        ]
    }

    mock_response.raise_for_status.return_value = None

    mock_post.return_value = mock_response

    connector = DeepSeekLMConnector(TEST_CONFIG)

    response = connector.generate(
        {
            "prompt": "hello"
        }
    )

    assert response["response"] == "test response"

@patch("requests.post")
def test_system_prompt(mock_post):

    mock_response = Mock()

    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "response"
                }
            }
        ]
    }

    mock_response.raise_for_status.return_value = None

    mock_post.return_value = mock_response

    connector = DeepSeekLMConnector(TEST_CONFIG)

    connector.generate(
        {
            "prompt": "hello",
            "system_prompt": "you are helpful"
        }
    )

    payload = mock_post.call_args.kwargs["json"]

    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"

@patch("requests.post")
def test_multi_turn_generation(mock_post):

    mock_response = Mock()

    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "assistant reply"
                }
            }
        ]
    }

    mock_response.raise_for_status.return_value = None

    mock_post.return_value = mock_response

    connector = DeepSeekLMConnector(TEST_CONFIG)

    messages = [
        Message("user", "hello"),
        Message("assistant", "hi"),
        Message("user", "how are you"),
    ]

    response = connector.generate(
        {
            "messages": messages
        },
        multi_turn=True,
    )

    assert response["response"] == "assistant reply"

@patch("requests.post")
def test_invalid_response(mock_post):

    mock_response = Mock()

    mock_response.json.return_value = {}

    mock_response.raise_for_status.return_value = None

    mock_post.return_value = mock_response

    connector = DeepSeekLMConnector(TEST_CONFIG)

    with pytest.raises(RuntimeError):
        connector.generate(
            {
                "prompt": "hello"
            }
        )

@patch("requests.get")
def test_list_models(mock_get):

    mock_response = Mock()

    mock_response.json.return_value = {
        "data": [
            {"id": "deepseek-flash"},
            {"id": "deepseek-reasoner"},
        ]
    }

    mock_response.raise_for_status.return_value = None

    mock_get.return_value = mock_response

    connector = DeepSeekLMConnector(TEST_CONFIG)

    models = connector._list_models()

    assert "deepseek-flash" in models
    assert "deepseek-reasoner" in models

@patch(
    "avise.connectors.languagemodel.deepseek.DeepSeekLMConnector._list_models"
)
def test_status_check_success(mock_list_models):

    mock_list_models.return_value = [
        "deepseek-flash"
    ]

    connector = DeepSeekLMConnector(TEST_CONFIG)

    assert connector.status_check() is True

@patch(
    "avise.connectors.languagemodel.deepseek.DeepSeekLMConnector._list_models"
)
def test_status_check_connection_error(mock_list_models):

    mock_list_models.side_effect = Exception(
        "connection failed"
    )

    connector = DeepSeekLMConnector(TEST_CONFIG)

    with pytest.raises(ConnectionError):
        connector.status_check()

def test_missing_prompt():

    connector = DeepSeekLMConnector(TEST_CONFIG)

    with pytest.raises(KeyError):
        connector.generate({})

def test_invalid_message_type():

    connector = DeepSeekLMConnector(TEST_CONFIG)

    with pytest.raises(ValueError):
        connector.generate(
            {
                "messages": ["invalid"]
            },
            multi_turn=True,
        )
