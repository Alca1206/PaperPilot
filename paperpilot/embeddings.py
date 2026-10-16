"""Turn text into vectors with the Ollama embedding model.

The Embedder class owns one rule from the concept notebook: documents and
queries must be embedded THE SAME WAY (same model, same prefix setting).
Keeping both methods on one object makes it impossible to mix them up.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

import httpx
import ollama

from paperpilot.config import Settings

# EmbeddingGemma was trained with these task prefixes (see its model card).
QUERY_PREFIX = "task: search result | query: "
DOCUMENT_PREFIX = "title: none | text: "

ProgressCallback = Callable[[int, int], None]  # (done, total)


class EmbeddingError(Exception):
    """Embedding failed. The message is safe to show to the user."""


class EmbedClient(Protocol):
    """The one method we need from ollama.Client. Lets tests pass a fake."""

    def embed(self, model: str, input: Sequence[str]) -> Any: ...


class Embedder:
    def __init__(self, settings: Settings, client: EmbedClient | None = None) -> None:
        self.model = settings.embedding_model
        self.use_prefixes = settings.embed_use_prefixes
        self.batch_size = settings.embed_batch_size
        self._host = settings.ollama_host
        self._client = client or ollama.Client(
            host=settings.ollama_host, timeout=settings.model_timeout_seconds
        )

    def embed_documents(
        self, texts: Sequence[str], progress: ProgressCallback | None = None
    ) -> list[list[float]]:
        """Embed many texts in batches. Calls ``progress(done, total)`` after each batch."""
        prepared = [self._prepare(text, DOCUMENT_PREFIX) for text in texts]
        vectors: list[list[float]] = []
        for start in range(0, len(prepared), self.batch_size):
            vectors.extend(self._embed_batch(prepared[start : start + self.batch_size]))
            if progress:
                progress(len(vectors), len(prepared))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([self._prepare(text, QUERY_PREFIX)])[0]

    def _prepare(self, text: str, prefix: str) -> str:
        return f"{prefix}{text}" if self.use_prefixes else text

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        try:
            response = self._client.embed(model=self.model, input=batch)
        except ConnectionError as exc:
            raise EmbeddingError(
                f"Could not connect to Ollama at {self._host}. Is Ollama running?"
            ) from exc
        except httpx.TimeoutException as exc:
            raise EmbeddingError(
                "Ollama took too long to embed the text. Try again, or raise "
                "PAPERPILOT_MODEL_TIMEOUT_SECONDS."
            ) from exc
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                raise EmbeddingError(
                    f"Embedding model '{self.model}' is not installed. "
                    f"Run: ollama pull {self.model}"
                ) from exc
            raise EmbeddingError(f"Ollama returned an error: {exc.error}") from exc

        vectors = [list(v) for v in getattr(response, "embeddings", None) or []]
        if len(vectors) != len(batch):
            raise EmbeddingError(
                f"Expected {len(batch)} embeddings from Ollama but got {len(vectors)}."
            )
        if any(len(v) == 0 for v in vectors):
            raise EmbeddingError("Ollama returned an empty embedding.")
        return vectors