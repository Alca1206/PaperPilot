"""Tests for paperpilot.embeddings, using a fake Ollama client (no Ollama needed)."""

import httpx
import ollama
import pytest

from paperpilot.config import Settings
from paperpilot.embeddings import (
    DOCUMENT_PREFIX,
    QUERY_PREFIX,
    Embedder,
    EmbeddingError,
)
from tests.fakes import FakeEmbedClient


def make_embedder(client, **overrides):
    return Embedder(Settings(**overrides), client=client)


def test_documents_are_embedded_in_batches():
    client = FakeEmbedClient()
    embedder = make_embedder(client, embed_batch_size=32)

    vectors = embedder.embed_documents([f"text {i}" for i in range(70)])

    assert len(vectors) == 70
    assert [len(call["input"]) for call in client.calls] == [32, 32, 6]
    assert all(call["model"] == "embeddinggemma" for call in client.calls)


def test_prefixes_are_added_when_enabled():
    client = FakeEmbedClient()
    embedder = make_embedder(client, embed_use_prefixes=True)

    embedder.embed_documents(["a chunk"])
    embedder.embed_query("a question")

    assert client.calls[0]["input"] == [DOCUMENT_PREFIX + "a chunk"]
    assert client.calls[1]["input"] == [QUERY_PREFIX + "a question"]


def test_no_prefixes_when_disabled():
    client = FakeEmbedClient()
    embedder = make_embedder(client, embed_use_prefixes=False)

    embedder.embed_documents(["a chunk"])
    embedder.embed_query("a question")

    assert client.calls[0]["input"] == ["a chunk"]
    assert client.calls[1]["input"] == ["a question"]


def test_progress_is_reported_after_each_batch():
    seen = []
    embedder = make_embedder(FakeEmbedClient(), embed_batch_size=2)

    embedder.embed_documents(["a", "b", "c", "d", "e"], progress=lambda done, total: seen.append((done, total)))

    assert seen == [(2, 5), (4, 5), (5, 5)]


def test_empty_input_makes_no_calls():
    client = FakeEmbedClient()

    assert make_embedder(client).embed_documents([]) == []
    assert client.calls == []


def test_query_returns_a_single_vector():
    vector = make_embedder(FakeEmbedClient(), embed_use_prefixes=False).embed_query("abc")

    assert vector == [3.0, 1.0, 0.0]


@pytest.mark.parametrize(
    ("error", "expected_text"),
    [
        (ConnectionError("refused"), "Could not connect"),
        (httpx.ReadTimeout("slow"), "took too long"),
        (ollama.ResponseError("model not found", 404), "ollama pull embeddinggemma"),
        (ollama.ResponseError("boom", 500), "boom"),
    ],
)
def test_ollama_errors_become_friendly_embedding_errors(error, expected_text):
    embedder = make_embedder(FakeEmbedClient(error=error))

    with pytest.raises(EmbeddingError, match=expected_text):
        embedder.embed_documents(["text"])


def test_wrong_number_of_vectors_is_detected():
    embedder = make_embedder(FakeEmbedClient(drop_one=True))

    with pytest.raises(EmbeddingError, match="Expected 2 embeddings"):
        embedder.embed_documents(["one", "two"])


# ---------- integration (real Ollama) ----------


@pytest.mark.integration
def test_real_embeddings_are_normalized():
    embedder = Embedder(Settings())
    try:
        vector = embedder.embed_query("Transformers use self-attention.")
    except EmbeddingError as exc:
        pytest.skip(f"Ollama embedding not available: {exc}")

    length = sum(x * x for x in vector) ** 0.5
    assert len(vector) > 100
    assert abs(length - 1.0) < 0.01  # Ollama returns unit-length vectors