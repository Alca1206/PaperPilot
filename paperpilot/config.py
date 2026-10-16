"""Central, non-secret configuration for PaperPilot.

Every setting has a sensible default and can be overridden with an
environment variable prefixed with ``PAPERPILOT_`` (for example
``PAPERPILOT_CHAT_MODEL``). No secrets belong in this module.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The project root is the folder that contains the ``paperpilot`` package.
# config.py -> paperpilot/ -> PaperPilot/
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

ENV_PREFIX = "PAPERPILOT_"

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_CHAT_MODEL = "qwen3:8b"
DEFAULT_EMBEDDING_MODEL = "embeddinggemma"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 10.0
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CHUNK_MAX_CHARS = 800
DEFAULT_CHUNK_OVERLAP_SENTENCES = 1
DEFAULT_MODEL_TIMEOUT_SECONDS = 300.0
DEFAULT_EMBED_BATCH_SIZE = 32
DEFAULT_EMBED_USE_PREFIXES = True
DEFAULT_QA_TOP_K = 4
DEFAULT_QA_MIN_SCORE = 0.35
DEFAULT_CHAT_TEMPERATURE = 0.1

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    ollama_host: str = DEFAULT_OLLAMA_HOST
    chat_model: str = DEFAULT_CHAT_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    data_dir: Path = DEFAULT_DATA_DIR
    chunk_max_chars: int = DEFAULT_CHUNK_MAX_CHARS
    chunk_overlap_sentences: int = DEFAULT_CHUNK_OVERLAP_SENTENCES
    # Model calls (embedding, chat) can take minutes on CPU; the health check stays quick.
    model_timeout_seconds: float = DEFAULT_MODEL_TIMEOUT_SECONDS
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE
    embed_use_prefixes: bool = DEFAULT_EMBED_USE_PREFIXES
    qa_top_k: int = DEFAULT_QA_TOP_K
    # Gate 1: if the best chunk scores below this, answer "not found" without the LLM.
    qa_min_score: float = DEFAULT_QA_MIN_SCORE
    chat_temperature: float = DEFAULT_CHAT_TEMPERATURE

    def __post_init__(self) -> None:
        if not self.ollama_host.startswith(("http://", "https://")):
            raise ValueError(
                "ollama_host must start with http:// or https://, "
                f"got {self.ollama_host!r}"
            )
        if not self.chat_model.strip():
            raise ValueError("chat_model must not be empty")
        if not self.embedding_model.strip():
            raise ValueError("embedding_model must not be empty")
        if self.request_timeout_seconds <= 0:
            raise ValueError(
                "request_timeout_seconds must be greater than 0, "
                f"got {self.request_timeout_seconds}"
            )
        if self.chunk_max_chars < 50:
            raise ValueError(f"chunk_max_chars must be at least 50, got {self.chunk_max_chars}")
        if self.chunk_overlap_sentences < 0:
            raise ValueError(
                f"chunk_overlap_sentences must be >= 0, got {self.chunk_overlap_sentences}"
            )
        if self.model_timeout_seconds <= 0:
            raise ValueError(
                f"model_timeout_seconds must be greater than 0, got {self.model_timeout_seconds}"
            )
        if self.embed_batch_size < 1:
            raise ValueError(f"embed_batch_size must be at least 1, got {self.embed_batch_size}")
        if self.qa_top_k < 1:
            raise ValueError(f"qa_top_k must be at least 1, got {self.qa_top_k}")
        if not -1.0 <= self.qa_min_score <= 1.0:
            raise ValueError(f"qa_min_score must be between -1 and 1, got {self.qa_min_score}")
        if not 0.0 <= self.chat_temperature <= 2.0:
            raise ValueError(
                f"chat_temperature must be between 0 and 2, got {self.chat_temperature}"
            )

    # Storage locations are derived from data_dir so they can never drift apart.
    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def required_models(self) -> tuple[str, str]:
        """Models that must be installed in Ollama for the app to work."""
        return (self.chat_model, self.embedding_model)

    def ensure_directories(self) -> None:
        """Create the local storage folders if they do not exist yet."""
        for directory in (self.uploads_dir, self.chroma_dir, self.exports_dir):
            directory.mkdir(parents=True, exist_ok=True)


def _resolve_path(raw: str) -> Path:
    """Turn a user-supplied path into an absolute path.

    Relative paths are resolved against the project root, not the current
    working directory, so the app behaves the same wherever it is started from.
    """
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build Settings from environment variables, falling back to defaults.

    ``env`` defaults to ``os.environ``. Tests pass a plain dict instead, so
    they never depend on the variables set on the developer's machine.
    """
    if env is None:
        env = os.environ

    def get(name: str, default: str) -> str:
        value = env.get(ENV_PREFIX + name, "").strip()
        return value or default  # blank values fall back to the default

    def get_number(name: str, default: float, cast: type) -> Any:
        raw = get(name, str(default))
        try:
            return cast(raw)
        except ValueError as exc:
            kind = "a whole number" if cast is int else "a number"
            raise ValueError(f"{ENV_PREFIX}{name} must be {kind}, got {raw!r}") from exc

    def get_bool(name: str, default: bool) -> bool:
        raw = get(name, str(default)).lower()
        if raw in _TRUE_VALUES:
            return True
        if raw in _FALSE_VALUES:
            return False
        raise ValueError(f"{ENV_PREFIX}{name} must be true or false, got {raw!r}")

    data_dir_raw = get("DATA_DIR", "")
    data_dir = _resolve_path(data_dir_raw) if data_dir_raw else DEFAULT_DATA_DIR

    return Settings(
        ollama_host=get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/"),
        chat_model=get("CHAT_MODEL", DEFAULT_CHAT_MODEL),
        embedding_model=get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        request_timeout_seconds=get_number(
            "REQUEST_TIMEOUT_SECONDS", DEFAULT_REQUEST_TIMEOUT_SECONDS, float
        ),
        data_dir=data_dir,
        chunk_max_chars=get_number("CHUNK_MAX_CHARS", DEFAULT_CHUNK_MAX_CHARS, int),
        chunk_overlap_sentences=get_number(
            "CHUNK_OVERLAP_SENTENCES", DEFAULT_CHUNK_OVERLAP_SENTENCES, int
        ),
        model_timeout_seconds=get_number(
            "MODEL_TIMEOUT_SECONDS", DEFAULT_MODEL_TIMEOUT_SECONDS, float
        ),
        embed_batch_size=get_number("EMBED_BATCH_SIZE", DEFAULT_EMBED_BATCH_SIZE, int),
        embed_use_prefixes=get_bool("EMBED_USE_PREFIXES", DEFAULT_EMBED_USE_PREFIXES),
        qa_top_k=get_number("QA_TOP_K", DEFAULT_QA_TOP_K, int),
        qa_min_score=get_number("QA_MIN_SCORE", DEFAULT_QA_MIN_SCORE, float),
        chat_temperature=get_number("CHAT_TEMPERATURE", DEFAULT_CHAT_TEMPERATURE, float),
    )