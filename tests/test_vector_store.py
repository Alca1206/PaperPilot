"""Tests for paperpilot.vector_store, using a real ChromaDB in a temp folder.

Vectors are tiny hand-made 3-number lists, so no Ollama is needed and the
expected similarity is easy to reason about.
"""

import re

import pytest

from paperpilot.config import Settings
from paperpilot.schemas import Chunk
from paperpilot.vector_store import (
    PaperIndex,
    collection_name,
    index_key,
    make_chroma_client,
)

CHUNKS = [
    Chunk(chunk_id="p1-c1", page=1, text="About cats."),
    Chunk(chunk_id="p2-c1", page=2, text="About dogs."),
    Chunk(chunk_id="p3-c1", page=3, text="About taxes."),
]
VECTORS = [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]]


@pytest.fixture
def client(tmp_path):
    return make_chroma_client(tmp_path / "chroma")


@pytest.fixture
def index(client):
    idx = PaperIndex(client, "paper_test")
    idx.add_chunks(CHUNKS, VECTORS, source_name="test.pdf")
    return idx


# ---------- naming ----------


def test_collection_name_uses_only_allowed_characters():
    name = collection_name("ab" * 32, Settings())

    assert re.fullmatch(r"[a-zA-Z0-9_-]{3,512}", name)
    assert name.startswith("paper_abababababababab_")


@pytest.mark.parametrize(
    "change",
    [
        {"embedding_model": "nomic-embed-text"},
        {"embed_use_prefixes": False},
        {"chunk_max_chars": 500},
        {"chunk_overlap_sentences": 0},
    ],
)
def test_any_vector_affecting_setting_changes_the_name(change):
    assert index_key(Settings(**change)) != index_key(Settings())


def test_unrelated_setting_keeps_the_name():
    assert index_key(Settings(chat_model="llama3.2:3b")) == index_key(Settings())


# ---------- storing and searching ----------


def test_query_returns_closest_first_with_page_and_score(index):
    results = index.query([1.0, 0.0, 0.0], top_k=3)

    assert [r.chunk_id for r in results] == ["p1-c1", "p2-c1", "p3-c1"]
    assert results[0].page == 1
    assert results[0].score == pytest.approx(1.0, abs=1e-3)   # identical direction
    assert results[1].score == pytest.approx(0.8, abs=1e-3)   # cosine of the two vectors
    assert results[2].score == pytest.approx(0.0, abs=1e-3)   # unrelated


def test_top_k_larger_than_collection_is_fine(index):
    assert len(index.query([1.0, 0.0, 0.0], top_k=50)) == 3


def test_where_filter_restricts_pages(index):
    results = index.query([1.0, 0.0, 0.0], top_k=3, where={"page": 3})

    assert [r.page for r in results] == [3]


def test_empty_index_returns_no_results(client):
    assert PaperIndex(client, "paper_empty").query([1.0, 0.0, 0.0]) == []


# ---------- completeness and persistence ----------


def test_new_index_is_not_complete(client):
    assert not PaperIndex(client, "paper_new").is_complete()


def test_index_is_complete_after_add(index):
    assert index.is_complete()
    assert index.count == 3


def test_data_survives_a_new_client(tmp_path):
    path = tmp_path / "chroma"
    first = PaperIndex(make_chroma_client(path), "paper_persist")
    first.add_chunks(CHUNKS, VECTORS, source_name="test.pdf")

    reopened = PaperIndex(make_chroma_client(path), "paper_persist")

    assert reopened.is_complete()
    assert reopened.query([0.0, 0.0, 1.0], top_k=1)[0].chunk_id == "p3-c1"


def test_adding_twice_does_not_duplicate(index):
    index.add_chunks(CHUNKS, VECTORS, source_name="test.pdf")

    assert index.count == 3


def test_delete_removes_everything(client, index):
    index.delete()

    assert not PaperIndex(client, "paper_test").is_complete()


# ---------- bad input ----------


def test_mismatched_lengths_raise(client):
    with pytest.raises(ValueError, match="3 chunks but 2 embeddings"):
        PaperIndex(client, "paper_bad").add_chunks(CHUNKS, VECTORS[:2], source_name="x.pdf")


def test_no_chunks_raise(client):
    with pytest.raises(ValueError, match="no chunks"):
        PaperIndex(client, "paper_none").add_chunks([], [], source_name="x.pdf")