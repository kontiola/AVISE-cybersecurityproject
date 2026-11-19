"""Connector for Google Gemini API communication.

Supports Gemini models served through the Google GenAI API.
"""

import logging
from typing import List, Optional

from .base import BaseLMConnector, Message
from ...registry import connector_registry
from ...utils import ansi_colors

logger = logging.getLogger(__name__)


@connector_registry.register("gemini-lm")
class GeminiLMConnector(BaseLMConnector):
    """Connector for communicating with the Google Gemini API.

    Used by SETs for sending prompts to Gemini models and collecting their responses.
    Supports both simple generation and generation with system prompts.

    Requires an API key, which can be passed via:
    - Model config file (api_key field)
    - GEMINI_API_KEY environment variable (when api_key is null in the config)
    """

    name = "gemini-lm"

    # Default model (fast and cost-effective tier, good for development runs)
    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, config: dict, evaluation: bool = False):
        """Initialize the Gemini connector.

        Args:
            config: Connector configuration data.
            evaluation: Boolean flag indicating whether initializing a connector
                for the target model or the evaluation model.

        Raises:
            KeyError: If required fields are missing from configuration data.
            TypeError: If configuration data is of a wrong type.
            SystemError: If failed to initialize the Google GenAI client.
        """
        section_name = "eval_model" if evaluation else "target_model"

        if section_name not in config:
            raise KeyError(
                f'Gemini Connector configuration file requires a "{section_name}" field. '
                "Refer to Connector documentations on how to configure connectors."
            )
        section = config[section_name]

        if "name" not in section:
            raise KeyError(
                f'Gemini connector requires a model name. Add "{section_name}": {{"name"}} '
                "to connector configuration file as a string."
            )
        if not isinstance(section["name"], str):
            raise TypeError(
                f'Gemini connector requires a model "name" for the {section_name} as a STRING.'
            )

        # api_key may be a string, or null/missing to fall back on GEMINI_API_KEY
        api_key = section.get("api_key")
        if api_key is not None and not isinstance(api_key, str):
            raise TypeError(
                f"Gemini connector requires an API key for the {section_name} "
                "as a STRING or null (null uses the GEMINI_API_KEY environment variable)."
            )

        # api_url is optional; null/missing uses the SDK default endpoint
        api_url = section.get("api_url")
        if api_url is not None and not isinstance(api_url, str):
            raise TypeError(
                f"Gemini connector requires an API URL for the {section_name} as a STRING or null."
            )

        self.model = section["name"]
        self.api_key = api_key
        self.base_url = api_url
        self.headers = section.get("headers")
        # Extra keyword arguments passed to GenerateContentConfig (e.g. {"top_p": 0.9}).
        # Set for both target and eval models so generate() works for either.
        self.completion_kwargs = section.get("completion_kwargs", {})
        if section.get("max_tokens") is not None:
            self.max_tokens = section["max_tokens"]
        else:
            self.max_tokens = 512

        # Imported here so that importing AVISE does not load the Google GenAI SDK
        from google import genai
        from google.genai import types

        try:
            client_kwargs = {}
            if self.api_key:
                client_kwargs["api_key"] = self.api_key

            http_options = {}
            if self.base_url:
                http_options["base_url"] = self.base_url
            if self.headers is not None:
                http_options["headers"] = self.headers
            if http_options:
                client_kwargs["http_options"] = types.HttpOptions(**http_options)

            self.client = genai.Client(**client_kwargs)
        except Exception as e:
            logger.error("Failed to initialize Google GenAI client.")
            raise SystemError from e

        logger.info("  Gemini Connector Initialized")
        logger.info(f"  Model: {self.model}")
        logger.info(f"  Base URL: {self.base_url or 'default'}")
        if self.api_key:
            logger.info(
                f"  API Key: {'*' * 8}...{self.api_key[-4:] if len(self.api_key) > 4 else '****'}"
            )
        else:
            logger.info("  API Key: taken from GEMINI_API_KEY environment variable")

    def generate(self, data: dict, multi_turn: bool = False) -> dict:
        """Generate a response from the target model via the Google Gemini API.

        Arguments:
            data: Dictionary containing data required for the generation API request.
                Valid Keys:
                    - prompt : str
                        Prompt for single turn generation. Required for single turn generation.
                    - messages: list[Message]
                        List of Message objects representing the conversation history.
                        Message objects contain 'role' and 'content' attributes.
                        Required for multi-turn conversation.
                    - system_prompt : str
                        Optional system prompt that defines the model's behavior, role, or constraints.
                    - temperature : float [0, 2]
                        Optional temperature setting for the target model. Defaults to 0.5 if not set.
                    - max_tokens : int
                        Optional setting for maximum generated tokens. Defaults to 512 if not set.
            multi_turn: Boolean flag to indicate if engaging in a multi turn conversation
                with the target model. Default False.

        Returns:
            Generated response in format: {"response": str}

        Raises:
            KeyError: If a required key is missing from data.
            ValueError: If a value in data is of a wrong type.
            RuntimeError: If the API call fails.
        """
        if "temperature" not in data:
            data["temperature"] = 0.5
        if "max_tokens" not in data:
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
                    "data variable, which contains a List of Message objects "
                    "representing the conversation history."
                )
            if not isinstance(data["messages"], list):
                raise ValueError(
                    'Multi-turn conversation requires a "messages" key in '
                    "data variable, which contains a List of Message objects "
                    "representing the conversation history."
                )
            for message in data["messages"]:
                if not isinstance(message, Message):
                    raise ValueError(
                        'Multi-turn conversation requires a "messages" key in '
                        "data variable, which contains a List of Message objects "
                        "representing the conversation history."
                    )
            return self._multi_turn(data=data)
        else:
            if "prompt" not in data:
                raise KeyError(
                    'Single-turn conversation requires a "prompt" key in '
                    "data variable, which contains a prompt as a string."
                )
            if not isinstance(data["prompt"], str):
                raise ValueError(
                    'Single-turn conversation requires a "prompt" key in '
                    "data variable, which contains a prompt as a string."
                )
            return self._single_turn(data=data)

    def _single_turn(self, data: dict) -> dict:
        """Make a single-turn generation.

        Arguments:
            data: Dictionary with required data for API request.

        Returns:
            {"response": str}
        """
        from google.genai import types

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=data["prompt"])],
            )
        ]
        return self._request(
            contents=contents, system=data.get("system_prompt"), data=data
        )

    def _multi_turn(self, data: dict) -> dict:
        """Make a multi-turn generation.

        Arguments:
            data: Dictionary with required data for API request.

        Returns:
            {"response": str}
        """
        from google.genai import types

        # Gemini takes the system prompt via GenerateContentConfig.system_instruction,
        # not as a message in contents, so any system-role Messages are moved out.
        system_parts = []
        if "system_prompt" in data:
            system_parts.append(data["system_prompt"])

        gemini_contents = []
        for msg in data["messages"]:
            if msg.role == "system":
                system_parts.append(msg.content)
            else:
                # Gemini uses "model" instead of "assistant" for model responses
                gemini_role = "model" if msg.role == "assistant" else msg.role
                gemini_contents.append(
                    types.Content(
                        role=gemini_role,
                        parts=[types.Part.from_text(text=msg.content)],
                    )
                )

        system = "\n\n".join(system_parts) if system_parts else None
        return self._request(contents=gemini_contents, system=system, data=data)

    def _request(self, contents: list, system: Optional[str], data: dict) -> dict:
        """Send a request to the Google Gemini API.

        Arguments:
            contents: Conversation contents in Gemini's format (user/model roles only).
            system: Optional system instruction.
            data: Dictionary containing temperature and max_tokens.

        Returns:
            {"response": str}

        Raises:
            RuntimeError: If the API call fails.
        """
        from google.genai import types

        config_kwargs = {
            "max_output_tokens": data["max_tokens"],
            "temperature": data["temperature"],
            **self.completion_kwargs,
        }
        if system:
            config_kwargs["system_instruction"] = system

        config = types.GenerateContentConfig(**config_kwargs)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            logger.error(
                f"{ansi_colors['red']}ERROR while generating response from Gemini: {e}{ansi_colors['reset']}"
            )
            raise RuntimeError("Failed to generate response from Gemini.") from e

        text = response.text or ""
        if not text:
            finish_reason = (
                response.candidates[0].finish_reason
                if getattr(response, "candidates", None)
                else None
            )
            logger.warning(f"Gemini returned no text (finish_reason={finish_reason}).")
        return {"response": text}

    def status_check(self) -> bool:
        """Check if the connector can reach the Gemini API endpoint and the target model is available.

        Returns:
            True if API is reachable (and the target model exists, or may exist).

        Raises:
            ConnectionError: If the API is not reachable or the API key is invalid.
        """
        # Step 1: Check API connectivity and API key validity
        try:
            model_ids = self._list_models()
        except Exception as e:
            base_url_info = f" at {self.base_url}" if self.base_url else ""
            raise ConnectionError(f"Cannot connect to Gemini API{base_url_info}: {e}")

        # Step 2: Check if model exists
        logger.info(f"Available models found: {len(model_ids)} models")

        # Gemini API returns model names prefixed with "models/" (e.g. "models/gemini-2.5-flash")
        normalized_target = (
            self.model if self.model.startswith("models/") else f"models/{self.model}"
        )
        if self.model in model_ids or normalized_target in model_ids:
            logger.info(f"Model '{self.model}' is available.")
            return True

        # Note: Model aliases may not appear in the list but still work
        logger.warning(
            f"Model '{self.model}' not found in available models list. "
            f"Proceeding anyway as some models (e.g. aliases) may not be listed."
        )
        return True

    def _list_models(self) -> List[str]:
        """Helper method, used by status_check() to verify model availability.

        Returns:
            List of model IDs.
        """
        models = self.client.models.list()
        return [m.name for m in models if getattr(m, "name", None)]
