"""Tests for paperpilot.chunking."""

import pytest

from paperpilot.chunking import (
    chunk_document,
    chunk_page,
    split_into_sentences,
    split_long_piece,
)
from paperpilot.schemas import ExtractedDocument, PageText


def make_sentences(count, page_label=""):
    return " ".join(
        f"Sentence {i}{page_label} discusses an idea in moderate detail here." for i in range(count)
    )


def make_document(*page_texts):
    return ExtractedDocument(
        source_name="test.pdf",
        file_hash="0" * 64,
        page_count=len(page_texts),
        pages=[PageText(page=i, text=t) for i, t in enumerate(page_texts, start=1)],
    )


# ---------- split_into_sentences ----------


def test_split_into_sentences_keeps_punctuation():
    assert split_into_sentences("One. Two! Three? Four") == ["One.", "Two!", "Three?", "Four"]


def test_split_into_sentences_ignores_empty_input():
    assert split_into_sentences("   ") == []


# ---------- split_long_piece ----------


def test_split_long_piece_respects_limit():
    pieces = split_long_piece("word " * 300, max_chars=100)

    assert len(pieces) > 1
    assert all(len(p) <= 100 for p in pieces)


def test_split_long_piece_cuts_a_single_huge_word():
    pieces = split_long_piece("x" * 250, max_chars=100)

    assert [len(p) for p in pieces] == [100, 100, 50]


# ---------- chunk_page ----------


@pytest.mark.parametrize("max_chars", [50, 120, 300, 800])
def test_chunks_never_exceed_max_chars(max_chars):
    text = make_sentences(40) + " " + "y" * 2000 + " " + "ref " * 400
    chunks = chunk_page(PageText(page=1, text=text), max_chars=max_chars)

    assert chunks
    assert max(c.char_count for c in chunks) <= max_chars


def test_chunk_ids_and_page_numbers():
    chunks = chunk_page(PageText(page=4, text=make_sentences(30)), max_chars=200)

    assert [c.chunk_id for c in chunks] == [f"p4-c{i}" for i in range(1, len(chunks) + 1)]
    assert all(c.page == 4 for c in chunks)


def test_neighbouring_chunks_overlap_by_one_sentence():
    chunks = chunk_page(PageText(page=1, text=make_sentences(20)), max_chars=200)

    for first, second in zip(chunks, chunks[1:]):
        last_sentence = split_into_sentences(first.text)[-1]
        assert second.text.startswith(last_sentence)


def test_zero_overlap_means_no_repeated_sentences():
    chunks = chunk_page(PageText(page=1, text=make_sentences(20)), max_chars=200, overlap_sentences=0)

    all_sentences = [s for c in chunks for s in split_into_sentences(c.text)]
    assert len(all_sentences) == len(set(all_sentences)) == 20


def test_empty_page_gives_no_chunks():
    assert chunk_page(PageText(page=1, text=""), max_chars=200) == []


@pytest.mark.parametrize(("max_chars", "overlap"), [(10, 1), (49, 0), (200, -1)])
def test_invalid_parameters_raise(max_chars, overlap):
    with pytest.raises(ValueError):
        chunk_page(PageText(page=1, text="Text."), max_chars=max_chars, overlap_sentences=overlap)


# ---------- chunk_document ----------


def test_chunks_never_cross_pages():
    document = make_document(make_sentences(15, "a"), make_sentences(15, "b"))

    chunks = chunk_document(document, max_chars=200)

    for chunk in chunks:
        expected_label = "a" if chunk.page == 1 else "b"
        other_label = "b" if expected_label == "a" else "a"
        assert f"{expected_label} discusses" in chunk.text
        assert f"{other_label} discusses" not in chunk.text


def test_chunk_ids_restart_on_each_page():
    document = make_document(make_sentences(10), make_sentences(10))

    chunks = chunk_document(document, max_chars=200)
    ids = [c.chunk_id for c in chunks]

    assert "p1-c1" in ids and "p2-c1" in ids
    assert len(ids) == len(set(ids))  # all unique


def test_tiny_chunks_are_dropped():
    document = make_document(make_sentences(5), "12", "")

    chunks = chunk_document(document, max_chars=800, min_chunk_chars=30)

    assert {c.page for c in chunks} == {1}