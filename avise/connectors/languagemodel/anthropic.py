"""Connector for Anthropic Claude Messages API communication."""

from __future__ import annotations

import os
import logging
from typing import Any, Optional

import requests

from .base import BaseLMConnector, Message
from ...registry import connector_registry
from ...utils import ansi_colors

logger = logging.getLogger(__name__)


@connector_registry.register("anthropic-lm")
class AnthropicLMConnector(BaseLMConnector):
    """Connector for communicating with the Anthropic Claude Messages API.

    Used by Security Evaluation Tests (SETs) for sending single-turn and
    multi-turn prompts to Claude models and collecting their responses.
    """

    name = "anthropic-lm"
    DEFAULT_URL = "https://api.anthropic.com/v1/messages"
    DEFAULT_MODEL = "claude-haiku-4-5"
    ANTHROPIC_VERSION = "2023-06-01"

    def __init__(self, config: dict, evaluation: bool = False) -> None:
        """Initialize the Anthropic Claude connector.

        Args:
            config: Dictionary containing data from Connector configuration JSON.
            evaluation: Whether initializing for eval_model instead of target_model.

        Raises:
            TypeError: If config is not a dictionary.
            KeyError: If the required model section is missing.
            ValueError: If no API key is configured or set in ANTHROPIC_API_KEY.
        """
        if not isinstance(config, dict):
            raise TypeError("Anthropic connector config must be a dictionary.")

        self.config = config
        section_key = "eval_model" if evaluation else "target_model"
        if section_key not in config:
            raise KeyError(
                f'Anthropic connector configuration requires a "{section_key}" field.'
            )

        section: dict[str, Any] = config[section_key]
        self.model: str = section.get("name") or self.DEFAULT_MODEL
        self.api_url: str = (section.get("api_url") or self.DEFAULT_URL).rstrip("/")
        self.api_key: Optional[str] = section.get("api_key") or os.environ.get(
            "ANTHROPIC_API_KEY"
        )
        self.max_tokens: int = (
            section["max_tokens"]
            if section.get("max_tokens") is not None
            else 1024
        )
        self.timeout: float = (
            section["time_out"] if section.get("time_out") is not None else 60.0
        )
        self.temperature: Optional[float] = section.get("temperature")

        if not self.api_key:
            raise ValueError(
                f'Anthropic connector requires an "api_key" in "{section_key}", '
                "via --api-key, or via the ANTHROPIC_API_KEY environment variable."
            )

        role_label = (
            f"Evaluation model: {self.model}"
            if evaluation
            else f"Target model: {self.model}"
        )
        logger.info(f"  Anthropic Connector Initialized for {role_label}")
        logger.info(f"  Base URL: {self.api_url}")

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key or "",
            "anthropic-version": self.ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    def generate(self, data: dict, multi_turn: bool = False) -> dict:
        """Generate a response from Claude via POST /v1/messages.

        Args:
            data: Dictionary containing generation inputs:
                - prompt (str): Required when multi_turn=False.
                - messages (list[Message]): Required when multi_turn=True.
                - system_prompt (str): Optional system instruction.
                - max_tokens (int): Optional max output tokens override.
                - temperature (float): Optional sampling temperature.
            multi_turn: Engage a multi-turn conversation. Default False.

        Returns:
            Dictionary in format: {"response": str}

        Raises:
            KeyError: If a required key is missing from data.
            ValueError: If a value in data has an invalid type.
            RuntimeError: If the Anthropic API request fails.
        """
        system_prompt = data.get("system_prompt")
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise ValueError(
                'If using "system_prompt" in data, it needs to be a string.'
            )

        system_parts: list[str] = [system_prompt] if system_prompt else []
        formatted_messages: list[dict[str, str]] = []

        if multi_turn:
            if "messages" not in data or not isinstance(data["messages"], list):
                raise KeyError(
                    'Multi-turn conversation requires a "messages" key in data, '
                    "containing a list of Message objects."
                )
            for msg in data["messages"]:
                if isinstance(msg, Message):
                    role, content = msg.role, msg.content
                elif isinstance(msg, dict) and "role" in msg and "content" in msg:
                    role, content = msg["role"], msg["content"]
                else:
                    raise ValueError(
                        'Multi-turn "messages" must contain Message objects or '
                        'dicts with "role" and "content".'
                    )
                if role == "system":
                    system_parts.append(content)
                elif formatted_messages and formatted_messages[-1]["role"] == role:
                    formatted_messages[-1]["content"] += f"\n\n{content}"
                else:
                    formatted_messages.append({"role": role, "content": content})
        else:
            if "prompt" not in data:
                raise KeyError(
                    'Single-turn generation requires a "prompt" key in data.'
                )
            if not isinstance(data["prompt"], str):
                raise ValueError('Single-turn "prompt" must be a string.')
            formatted_messages.append({"role": "user", "content": data["prompt"]})

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": data.get("max_tokens", self.max_tokens),
            "messages": formatted_messages,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)

        temperature = data.get("temperature", self.temperature)
        if temperature is not None:
            payload["temperature"] = temperature

        try:
            response = requests.post(
                self.api_url,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            logger.error(
                f"{ansi_colors['red']}ERROR while generating response from "
                f"Anthropic: {exc}{ansi_colors['reset']}"
            )
            raise RuntimeError(
                f"Failed to generate response from Anthropic: {exc}"
            ) from exc

        if response.status_code != 200:
            error_detail = f"{response.status_code}: {response.text[:300]}"
            logger.error(
                f"{ansi_colors['red']}ERROR from Anthropic API: "
                f"{error_detail}{ansi_colors['reset']}"
            )
            raise RuntimeError(
                f"Failed to generate response from Anthropic: {error_detail}"
            )

        body = response.json()
        text_blocks = [
            block.get("text", "")
            for block in body.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return {"response": "".join(text_blocks)}

    def status_check(self) -> bool:
        """Verify that the Anthropic API is reachable and credentials are valid.

        Returns:
            True if the API responds with HTTP 200.

        Raises:
            ConnectionError: If the endpoint cannot be reached or returns non-200.
        """
        models_url = (
            self.api_url[: -len("/v1/messages")] + "/v1/models"
            if self.api_url.endswith("/v1/messages")
            else f"{self.api_url}/v1/models"
        )
        try:
            response = requests.get(
                models_url, headers=self._headers(), timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise ConnectionError(
                f"Cannot connect to Anthropic API at {models_url}: {exc}"
            ) from exc

        if response.status_code != 200:
            raise ConnectionError(
                f"Anthropic status check failed at {models_url}: "
                f"{response.status_code} {response.text[:200]}"
            )

        try:
            model_ids = [m.get("id") for m in response.json().get("data", [])]
            if self.model and model_ids and self.model not in model_ids:
                logger.warning(
                    f"Model '{self.model}' not found in listed Anthropic models; "
                    "proceeding anyway."
                )
        except (ValueError, AttributeError):
            pass

        return True
    