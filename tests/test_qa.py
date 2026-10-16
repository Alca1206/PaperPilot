"""Tests for paperpilot.retrieval and paperpilot.qa.

A real ChromaDB (temp folder) + a keyword-based fake embedder + a fake LLM,
so every gate can be tested without Ollama.
"""

import pytest

from paperpilot.config import Settings
from paperpilot.embeddings import Embedder
from paperpilot.indexing import index_paper, open_index
from paperpilot.llm import LLM
from paperpilot.qa import (
    NOT_FOUND,
    answer_question,
    build_context,
    extract_citations,
    is_not_found,
)
from paperpilot.retrieval import retrieve
from paperpilot.schemas import Chunk, ExtractedDocument, PageText, RetrievedChunk
from paperpilot.vector_store import make_chroma_client
from tests.fakes import FakeChatClient, KeywordEmbedClient

VOCABULARY = ["gpu", "attention", "bleu", "translation", "heads"]

CHUNKS = [
    Chunk(chunk_id="p3-c1", page=3, text="Multi-head attention uses several attention heads in parallel."),
    Chunk(chunk_id="p7-c2", page=7, text="We trained the model on one machine with 8 GPUs (each GPU a P100)."),
    Chunk(chunk_id="p8-c1", page=8, text="The model reaches 28.4 BLEU on English-German translation."),
]
DOCUMENT = ExtractedDocument(
    source_name="attention.pdf",
    file_hash="cd" * 32,
    page_count=8,
    pages=[PageText(page=i, text="x") for i in range(1, 9)],
)


@pytest.fixture
def settings():
    return Settings(qa_top_k=2, qa_min_score=0.35)


@pytest.fixture
def embedder(settings):
    return Embedder(settings, client=KeywordEmbedClient(VOCABULARY))


@pytest.fixture
def index(tmp_path, settings, embedder):
    chroma = make_chroma_client(tmp_path / "chroma")
    index_paper(DOCUMENT, CHUNKS, settings, embedder, chroma)
    return open_index(DOCUMENT, settings, chroma)


def ask(question, index, embedder, settings, reply):
    fake_llm_client = FakeChatClient(replies=[reply])
    answer = answer_question(question, index, embedder, LLM(settings, client=fake_llm_client), settings)
    return answer, fake_llm_client


# ---------- retrieval ----------


def test_retrieve_finds_the_matching_chunk_first(index, embedder):
    results = retrieve("How many GPUs were used?", index, embedder, top_k=2)

    assert results[0].chunk_id == "p7-c2"
    assert results[0].page == 7
    assert results[0].score > results[1].score


def test_retrieve_rejects_empty_question(index, embedder):
    with pytest.raises(ValueError, match="empty"):
        retrieve("   ", index, embedder, top_k=2)


# ---------- building the prompt ----------


def test_build_context_numbers_sources_with_pages():
    results = [
        RetrievedChunk(chunk_id="p7-c2", page=7, text="GPU text", score=0.9),
        RetrievedChunk(chunk_id="p3-c1", page=3, text="Attention text", score=0.5),
    ]

    assert build_context(results) == "[1] (page 7)\nGPU text\n\n[2] (page 3)\nAttention text"


# ---------- citation parsing (Gate 2) ----------

RESULTS = [
    RetrievedChunk(chunk_id="p7-c2", page=7, text="a", score=0.9),
    RetrievedChunk(chunk_id="p8-c1", page=8, text="b", score=0.8),
]


@pytest.mark.parametrize(
    ("answer", "valid", "invalid"),
    [
        ("Eight GPUs [1].", [1], []),
        ("It took 3.5 days [1, 2].", [1, 2], []),
        ("Both [2][1].", [1, 2], []),
        ("Made up [5].", [], [5]),
        ("Zero is not a source [0].", [], [0]),
        ("No brackets at all.", [], []),
    ],
)
def test_extract_citations(answer, valid, invalid):
    citations, bad = extract_citations(answer, RESULTS)

    assert [c.number for c in citations] == valid
    assert bad == invalid


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (NOT_FOUND, True),
        ("not found in the paper", True),
        ("  Not found in the paper. The sources discuss other things.", True),
        ("The paper says it was not found in earlier work [1].", False),
    ],
)
def test_is_not_found(reply, expected):
    assert is_not_found(reply) is expected


# ---------- answer_question: the full flow ----------


def test_good_answer_has_citations_with_pages_from_metadata(index, embedder, settings):
    answer, llm_client = ask(
        "How many GPUs were used?", index, embedder, settings, "They used 8 GPUs [1]."
    )

    assert answer.status == "ok"
    assert answer.answer == "They used 8 GPUs [1]."
    assert [(c.number, c.page, c.chunk_id) for c in answer.citations] == [(1, 7, "p7-c2")]
    # The model saw numbered sources with page labels
    user_prompt = llm_client.calls[0]["messages"][1]["content"]
    assert "[1] (page 7)" in user_prompt
    assert "Question: How many GPUs were used?" in user_prompt


def test_gate1_stops_irrelevant_questions_before_the_llm(index, embedder, settings):
    answer, llm_client = ask(
        "What is a good recipe for pizza dough?", index, embedder, settings, "should not be used"
    )

    assert answer.status == "no_relevant_chunks"
    assert answer.answer == NOT_FOUND
    assert llm_client.calls == []  # the LLM was never asked
    assert answer.best_score is not None and answer.best_score < settings.qa_min_score


def test_model_saying_not_found_is_respected(index, embedder, settings):
    answer, _ = ask("How many GPUs were used?", index, embedder, settings, NOT_FOUND)

    assert answer.status == "model_said_not_found"
    assert answer.citations == []


def test_invented_source_number_is_flagged(index, embedder, settings):
    answer, _ = ask("How many GPUs were used?", index, embedder, settings, "8 GPUs [1], see also [9].")

    assert answer.status == "invalid_citations"
    assert answer.invalid_citations == [9]
    assert [c.number for c in answer.citations] == [1]


def test_answer_without_citations_is_flagged(index, embedder, settings):
    answer, _ = ask("How many GPUs were used?", index, embedder, settings, "They used 8 GPUs.")

    assert answer.status == "no_citations"


def test_page_comes_from_metadata_not_from_model_text(index, embedder, settings):
    answer, _ = ask("How many GPUs were used?", index, embedder, settings, "On page 99 it says 8 GPUs [1].")

    assert answer.citations[0].page == 7


def test_retrieved_chunks_are_kept_for_debugging(index, embedder, settings):
    answer, _ = ask("How many GPUs were used?", index, embedder, settings, "8 GPUs [1].")

    assert len(answer.retrieved) == settings.qa_top_k
    assert answer.best_score == answer.retrieved[0].score