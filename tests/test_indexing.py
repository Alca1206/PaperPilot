"""Tests for paperpilot.indexing: the chunks -> embeddings -> Chroma pipeline."""

import ollama
import pytest

from paperpilot.config import Settings
from paperpilot.embeddings import Embedder, EmbeddingError
from paperpilot.indexing import index_paper, open_index
from paperpilot.schemas import Chunk, ExtractedDocument, PageText
from paperpilot.vector_store import make_chroma_client
from tests.fakes import FakeEmbedClient

DOCUMENT = ExtractedDocument(
    source_name="paper.pdf",
    file_hash="ab" * 32,
    page_count=2,
    pages=[PageText(page=1, text="One."), PageText(page=2, text="Two.")],
)
CHUNKS = [
    Chunk(chunk_id="p1-c1", page=1, text="First chunk of text."),
    Chunk(chunk_id="p2-c1", page=2, text="Second chunk, a bit longer."),
]


@pytest.fixture
def chroma(tmp_path):
    return make_chroma_client(tmp_path / "chroma")


def test_first_run_embeds_and_stores(chroma):
    fake = FakeEmbedClient()
    settings = Settings()

    result = index_paper(DOCUMENT, CHUNKS, settings, Embedder(settings, client=fake), chroma)

    assert result.reused is False
    assert result.chunk_count == 2
    assert len(fake.calls) == 1
    assert open_index(DOCUMENT, settings, chroma).is_complete()


def test_second_run_reuses_index_without_embedding(chroma):
    settings = Settings()
    index_paper(DOCUMENT, CHUNKS, settings, Embedder(settings, client=FakeEmbedClient()), chroma)

    fake = FakeEmbedClient()
    result = index_paper(DOCUMENT, CHUNKS, settings, Embedder(settings, client=fake), chroma)

    assert result.reused is True
    assert fake.calls == []  # nothing was re-embedded


def test_changed_chunk_settings_build_a_separate_index(chroma):
    first = index_paper(
        DOCUMENT, CHUNKS, Settings(), Embedder(Settings(), client=FakeEmbedClient()), chroma
    )
    other_settings = Settings(chunk_max_chars=400)
    second = index_paper(
        DOCUMENT, CHUNKS, other_settings, Embedder(other_settings, client=FakeEmbedClient()), chroma
    )

    assert first.collection_name != second.collection_name
    assert second.reused is False


def test_embedding_failure_leaves_no_partial_index(chroma):
    settings = Settings()
    broken = Embedder(settings, client=FakeEmbedClient(error=ollama.ResponseError("boom", 500)))

    with pytest.raises(EmbeddingError):
        index_paper(DOCUMENT, CHUNKS, settings, broken, chroma)

    index = open_index(DOCUMENT, settings, chroma)
    assert index.count == 0
    assert not index.is_complete()