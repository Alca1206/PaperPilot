"""A thin wrapper around the Ollama chat model.

One place for: the model name, think=False for qwen3, temperature,
timeouts, and turning Ollama failures into user-readable errors.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import httpx
import ollama

from paperpilot.config import Settings

Message = Mapping[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}

# Some models emit their reasoning inside <think>...</think> even with think=False.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class LLMError(Exception):
    """The chat model call failed. The message is safe to show to the user."""


class ChatClient(Protocol):
    """The one method we need from ollama.Client. Lets tests pass a fake."""

    def chat(self, **kwargs: Any) -> Any: ...


class LLM:
    def __init__(self, settings: Settings, client: ChatClient | None = None) -> None:
        self.model = settings.chat_model
        self.temperature = settings.chat_temperature
        self._host = settings.ollama_host
        self._client = client or ollama.Client(
            host=settings.ollama_host, timeout=settings.model_timeout_seconds
        )

    def chat(
        self,
        messages: Sequence[Message],
        *,
        format: dict[str, Any] | None = None,
        num_ctx: int | None = None,
    ) -> str:
        """Send a conversation, return the model's reply as plain text."""
        options: dict[str, Any] = {"temperature": self.temperature}
        if num_ctx is not None:
            options["num_ctx"] = num_ctx

        try:
            response = self._client.chat(
                model=self.model,
                messages=list(messages),
                format=format,
                think=False,  # qwen3 would otherwise "think" at length before answering
                options=options,
            )
        except ConnectionError as exc:
            raise LLMError(f"Could not connect to Ollama at {self._host}. Is Ollama running?") from exc
        except httpx.TimeoutException as exc:
            raise LLMError(
                "The model took too long to answer. Try again, or raise "
                "PAPERPILOT_MODEL_TIMEOUT_SECONDS."
            ) from exc
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                raise LLMError(
                    f"Chat model '{self.model}' is not installed. Run: ollama pull {self.model}"
                ) from exc
            raise LLMError(f"Ollama returned an error: {exc.error}") from exc

        content = getattr(getattr(response, "message", None), "content", None) or ""
        return _THINK_BLOCK.sub("", content).strip()