"""Connector for DeepSeek API communication.

Supports DeepSeek models served through the DeepSeek API,
including DeepSeek-Flash and DeepSeek-Reasoner models.
"""

import logging
from typing import List

import requests

from .base import BaseLMConnector, Message
from ...registry import connector_registry
from ...utils import ansi_colors

logger = logging.getLogger(__name__)


@connector_registry.register("deepseek-lm")
class DeepSeekLMConnector(BaseLMConnector):
    """Connector for communicating with the DeepSeek API.

    Used by SETs for sending prompts to DeepSeek models and collecting
    their responses.

    Supports both simple generation and generation with system prompts.

    Requires an API key, which can be passed via:
    - Model config file (api_key field)
    - Command line (--apikey argument)
    """

    name = "deepseek-lm"

    # Default model
    DEFAULT_MODEL = "deepseek-flash"

    # Default DeepSeek API endpoint
    DEFAULT_BASE_URL = "https://api.deepseek.com"

    def __init__(self, config: dict, evaluation: bool = False):
        """Initialize the DeepSeek connector.

        Args:
            config: Connector configuration data.
            evaluation: Boolean flag indicating whether initializing
                a connector for the target model or the evaluation model.

        Raises:
            KeyError: If required fields are missing from configuration data.
            TypeError: If configuration data is of a wrong type.
            SystemError: If failed to initialize the connector.
        """

        self.config = config

        section_name = "eval_model" if evaluation else "target_model"

        if section_name not in config:
            raise KeyError(
                f'DeepSeek Connector configuration file requires a "{section_name}" field. '
                "Refer to Connector documentations on how to configure connectors."
            )

        section = config[section_name]

        if "name" not in section:
            raise KeyError(
                f'DeepSeek connector requires a model name. Add "{section_name}": '
                '{"name"} to connector configuration file as a string.'
            )

        if not isinstance(section["name"], str):
            raise TypeError(
                f'DeepSeek connector requires a model "name" for the '
                f"{section_name} as a STRING."
            )

        if "api_key" not in section:
            raise KeyError(
                f"DeepSeek connector requires an API key for the {section_name}. "
                "Add 'api_key' to connector configuration file as a string."
            )

        if not isinstance(section["api_key"], str):
            raise TypeError(
                f"DeepSeek connector requires an API key for the "
                f"{section_name} as a STRING."
            )

        api_url = section.get("api_url")

        if api_url is not None and not isinstance(api_url, str):
            raise TypeError(
                f"DeepSeek connector requires an API URL for the "
                f"{section_name} as a STRING or null."
            )

        self.model = section["name"]
        self.api_key = section["api_key"]

        self.base_url = (
            api_url
            if api_url not in (None, "")
            else self.DEFAULT_BASE_URL
        )

        self.headers = section.get("headers")
        self.completion_kwargs = section.get("completion_kwargs", {})

        if (
            "max_tokens" in section
            and section["max_tokens"] is not None
        ):
            self.max_tokens = section["max_tokens"]
        else:
            self.max_tokens = 512

        if (
            "time_out" in section
            and section["time_out"] is not None
        ):
            self.time_out = section["time_out"]
        else:
            self.time_out = 60

        try:
            default_headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            if self.headers is not None:
                default_headers.update(self.headers)

            self.headers = default_headers

        except Exception as e:
            logger.error("Failed to initialize DeepSeek connector.")
            raise SystemError from e

        logger.info("  DeepSeek Connector Initialized")
        logger.info(f"  Model: {self.model}")
        logger.info(f"  Base URL: {self.base_url}")
        logger.info(
            f"  API Key: {'*' * 8}...{self.api_key[-4:] if len(self.api_key) > 4 else '****'}"
        )

    def generate(self, data: dict, multi_turn: bool = False) -> dict:
        """Generate a response from the target model via the DeepSeek API.

        Arguments:
            data: Dictionary containing data required for the generation API request.

                Valid Keys:
                    - prompt : str
                        Prompt for single turn generation.
                        Required for single turn generation.

                    - messages: list[Message]
                        List of Message objects representing the conversation history.

                        Message objects contain 'role' and 'content' attributes.

                        Required for multi-turn conversation.

                    - system_prompt : str
                        Optional system prompt that defines the model's
                        behavior, role, or constraints.

                    - temperature : float
                        Optional temperature setting for the target model.
                        Defaults to 0.5 if not set.

                    - max_tokens : int
                        Optional setting for maximum generated tokens.

            multi_turn:
                Boolean flag to indicate if engaging in a multi turn conversation
                with the target model. Default False.

        Returns:
            Generated response in format:

                {"response": str}

        Raises:
            KeyError: If a required key is missing from data.
            ValueError: If a value in data is of a wrong type.
            RuntimeError: If the API call fails.
        """

        if "temperature" not in data:
            data["temperature"] = 0.5

        data["max_tokens"] = self.max_tokens

        if "system_prompt" in data:
            if not isinstance(data["system_prompt"], str):
                raise ValueError(
                    'If using "system_prompt" in data, it needs to be a string.'
                )

        if multi_turn:
            if "messages" not in data:
                raise KeyError(
                    'Multi-turn conversation requires a "messages" key in '
                    "data variable."
                )

            if not isinstance(data["messages"], list):
                raise ValueError(
                    'Multi-turn conversation requires a "messages" key '
                    "containing a list of Message objects."
                )

            for message in data["messages"]:
                if not isinstance(message, Message):
                    raise ValueError(
                        'Multi-turn conversation requires Message objects.'
                    )

            return self._multi_turn(data)

        if "prompt" not in data:
            raise KeyError(
                'Single-turn conversation requires a "prompt" key.'
            )

        if not isinstance(data["prompt"], str):
            raise ValueError(
                'Single-turn conversation requires a prompt as a string.'
            )

        return self._single_turn(data)

    def _single_turn(self, data: dict) -> dict:
        """Make a single-turn generation.

        Arguments:
            data: Dictionary with required data for API request.

        Returns:
            {"response": str}
        """

        if "system_prompt" in data:
            messages = [
                {"role": "system", "content": data["system_prompt"]},
                {"role": "user", "content": data["prompt"]},
            ]
        else:
            messages = [
                {"role": "user", "content": data["prompt"]}
            ]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": data["temperature"],
            "max_tokens": data["max_tokens"],
            **self.completion_kwargs,
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=self.time_out,
            )

            response.raise_for_status()

            result = response.json()

            if (
                "choices" not in result
                or not result["choices"]
            ):
                raise RuntimeError(
                    "DeepSeek API returned an invalid response."
                )

            return {
                "response": (
                    result["choices"][0]["message"].get("content", "")
                )
            }

        except Exception as e:
            logger.error(
                f"{ansi_colors['red']}ERROR while generating "
                f"response from DeepSeek: {e}{ansi_colors['reset']}"
            )
            raise RuntimeError(
                "Failed to generate response from DeepSeek."
            ) from e

    def _multi_turn(self, data: dict) -> dict:
        """Make a multi-turn generation.

        Arguments:
            data: Dictionary with required conversation data.

        Returns:
            {"response": str}
        """

        messages = []

        if "system_prompt" in data:
            messages.append(
                {
                    "role": "system",
                    "content": data["system_prompt"],
                }
            )

        messages.extend(
            [
                {
                    "role": msg.role,
                    "content": msg.content,
                }
                for msg in data["messages"]
            ]
        )

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": data["temperature"],
            "max_tokens": data["max_tokens"],
            **self.completion_kwargs,
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=self.time_out,
            )

            response.raise_for_status()

            result = response.json()

            if (
                "choices" not in result
                or not result["choices"]
            ):
                raise RuntimeError(
                    "DeepSeek API returned an invalid response."
                )

            return {
                "response": (
                    result["choices"][0]["message"].get("content", "")
                )
            }

        except Exception as e:
            logger.error(
                f"{ansi_colors['red']}ERROR while generating "
                f"DeepSeek response: {e}{ansi_colors['reset']}"
            )
            raise RuntimeError(
                "Failed to generate response from DeepSeek."
            ) from e

    def status_check(self) -> bool:
        """Check if the connector can reach the DeepSeek API endpoint
        and the target model is available.

        Returns:
            True if API is reachable and the target model exists.

        Raises:
            ConnectionError: If the API is not reachable.
        """

        try:
            model_ids = self._list_models()

        except Exception as e:
            base_url_info = f" at {self.base_url}" if self.base_url else ""

            raise ConnectionError(
                f"Cannot connect to DeepSeek API{base_url_info}: {e}"
            )

        logger.info(f"Available models found: {len(model_ids)} models")

        if self.model in model_ids:
            logger.info(f"Model '{self.model}' is available.")
            return True

        logger.warning(
            f"Model '{self.model}' not found in available models list. "
            f"Proceeding anyway as some models may not be listed."
        )

        return True

    def _list_models(self) -> List[str]:
        """Helper function, used by status_check() to verify model availability.

        Returns:
            List of model names.
        """

        response = requests.get(
            f"{self.base_url}/models",
            headers=self.headers,
            timeout=self.time_out,
        )

        response.raise_for_status()

        data = response.json()

        return [model["id"] for model in data["data"]]
    