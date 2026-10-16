"""Health checks for the local Ollama server and required models.

This module has no Streamlit code, so it can be tested on its own.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
import ollama

from paperpilot.config import Settings

DEFAULT_TAG = "latest"


class OllamaListClient(Protocol):
    """The one method we need from ollama.Client. Lets tests pass a fake."""

    def list(self) -> Any: ...


@dataclass(frozen=True)
class HealthReport:
    """Result of a health check, ready to display in the UI."""

    ollama_host: str
    ollama_reachable: bool
    required_models: tuple[str, ...] = ()
    installed_models: tuple[str, ...] = ()
    missing_models: tuple[str, ...] = ()
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True only when Ollama is reachable and every required model exists."""
        return self.ollama_reachable and not self.missing_models

    def is_model_available(self, name: str) -> bool:
        """True if Ollama is reachable and this required model is installed."""
        missing = {normalize_model_name(m) for m in self.missing_models}
        return self.ollama_reachable and normalize_model_name(name) not in missing

    @property
    def pull_commands(self) -> list[str]:
        """Commands the user can copy to install missing models."""
        return [f"ollama pull {name}" for name in self.missing_models]


def normalize_model_name(name: str) -> str:
    """Normalize a model name so equivalent names compare equal.

    Ollama adds ':latest' when no tag is given, so 'embeddinggemma' and
    'embeddinggemma:latest' refer to the same model.
    """
    cleaned = name.strip().lower()
    if not cleaned:
        return cleaned
    return cleaned if ":" in cleaned else f"{cleaned}:{DEFAULT_TAG}"


def find_missing_models(
    required: Iterable[str], installed: Iterable[str]
) -> tuple[str, ...]:
    """Return the required models that are not installed, in their original spelling."""
    installed_normalized = {normalize_model_name(name) for name in installed}
    return tuple(
        name for name in required if normalize_model_name(name) not in installed_normalized
    )


def _extract_model_names(list_response: Any) -> tuple[str, ...]:
    """Read model names from the response of ollama.Client.list().

    Newer ollama clients return objects with a ``.model`` attribute; older
    ones returned dicts with a ``name`` key. Both are handled.
    """
    models = getattr(list_response, "models", None)
    if models is None and isinstance(list_response, dict):
        models = list_response.get("models", [])

    names: list[str] = []
    for item in models or []:
        if isinstance(item, dict):
            name = item.get("model") or item.get("name")
        else:
            name = getattr(item, "model", None) or getattr(item, "name", None)
        if name:
            names.append(str(name))
    return tuple(names)


def check_ollama_health(
    settings: Settings, client: OllamaListClient | None = None
) -> HealthReport:
    """Check that Ollama is reachable and that the required models are installed.

    Never raises: every failure is turned into a HealthReport with an error
    message, so the UI can always show something useful.
    """
    if client is None:
        client = ollama.Client(
            host=settings.ollama_host, timeout=settings.request_timeout_seconds
        )

    required = settings.required_models

    def unreachable(message: str) -> HealthReport:
        return HealthReport(
            ollama_host=settings.ollama_host,
            ollama_reachable=False,
            required_models=required,
            error=message,
        )

    try:
        response = client.list()
    except ConnectionError:
        return unreachable(
            f"Could not connect to Ollama at {settings.ollama_host}. "
            "Is Ollama installed and running?"
        )
    except httpx.TimeoutException:
        return unreachable(
            f"Ollama at {settings.ollama_host} did not respond within "
            f"{settings.request_timeout_seconds:g} seconds."
        )
    except ollama.ResponseError as exc:
        return unreachable(f"Ollama returned an error: {exc.error}")
    except Exception as exc:  # noqa: BLE001 - a health check must never crash the UI
        return unreachable(f"Unexpected error while contacting Ollama: {exc}")

    installed = _extract_model_names(response)
    return HealthReport(
        ollama_host=settings.ollama_host,
        ollama_reachable=True,
        required_models=required,
        installed_models=installed,
        missing_models=find_missing_models(required, installed),
    )