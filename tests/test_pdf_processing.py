"""Tests for paperpilot.pdf_processing."""

import pytest

from paperpilot.pdf_processing import (
    PDFExtractionError,
    clean_text,
    compute_file_hash,
    extract_pages,
    extract_pages_from_bytes,
)

# ---------- clean_text ----------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("trans-\nduction models", "transduction models"),
        ("line one\nline two", "line one line two"),
        ("para one.\n\npara two.", "para one.\n\npara two."),
        ("too    many   spaces", "too many spaces"),
        ("  \n padded \n  ", "padded"),
        ("", ""),
    ],
)
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


def test_clean_text_known_tradeoff_joins_real_hyphens():
    # Documented limitation: a real hyphen at a line end is lost.
    assert clean_text("self-\nattention") == "selfattention"


# ---------- compute_file_hash ----------


def test_hash_is_deterministic_and_content_based():
    assert compute_file_hash(b"abc") == compute_file_hash(b"abc")
    assert compute_file_hash(b"abc") != compute_file_hash(b"abd")
    assert len(compute_file_hash(b"abc")) == 64


# ---------- extract_pages_from_bytes: success ----------


def test_extracts_pages_with_one_indexed_numbers(make_pdf):
    data = make_pdf(["First page text.", "Second page text.", "Third page text."])

    document = extract_pages_from_bytes(data, "paper.pdf")

    assert document.page_count == 3
    assert [p.page for p in document.pages] == [1, 2, 3]
    assert "First page" in document.pages[0].text
    assert "Third page" in document.pages[2].text
    assert document.source_name == "paper.pdf"
    assert document.file_hash == compute_file_hash(data)


def test_hyphenated_line_break_is_rejoined(make_pdf):
    data = make_pdf(["The dominant sequence trans-\nduction models."])

    document = extract_pages_from_bytes(data, "paper.pdf")

    assert "transduction" in document.pages[0].text


def test_blank_page_is_kept_and_flagged(make_pdf):
    data = make_pdf(["Some real text on page one.", ""])

    document = extract_pages_from_bytes(data, "paper.pdf")

    assert document.page_count == 2
    assert document.pages[1].text == ""
    assert document.low_text_pages(min_chars=20) == [2]


def test_metadata_is_read(make_pdf):
    data = make_pdf(["Text."], metadata={"title": "My Paper", "author": "Doe"})

    document = extract_pages_from_bytes(data, "paper.pdf")

    assert document.metadata["title"] == "My Paper"
    assert document.metadata["author"] == "Doe"


# ---------- extract_pages_from_bytes: failures ----------


@pytest.mark.parametrize(
    ("data", "expected_text"),
    [
        (b"", "empty"),
        (b"this is not a pdf", "could not be opened"),
        (b"%PDF-1.4 truncated garbage", "could not be opened"),
    ],
)
def test_bad_files_raise_friendly_error(data, expected_text):
    with pytest.raises(PDFExtractionError, match=expected_text):
        extract_pages_from_bytes(data, "bad.pdf")


def test_password_protected_pdf_raises(make_pdf):
    data = make_pdf(["Secret text."], user_password="secret")

    with pytest.raises(PDFExtractionError, match="password"):
        extract_pages_from_bytes(data, "locked.pdf")


def test_pdf_without_any_text_raises_scanned_hint(make_pdf):
    data = make_pdf(["", ""])  # like a scanned PDF: pages but no text layer

    with pytest.raises(PDFExtractionError, match="scanned"):
        extract_pages_from_bytes(data, "scan.pdf")


# ---------- extract_pages (from disk) ----------


def test_extract_pages_from_disk(make_pdf, tmp_path):
    path = tmp_path / "paper.pdf"
    path.write_bytes(make_pdf(["Hello from disk."]))

    document = extract_pages(path)

    assert document.source_name == "paper.pdf"
    assert "Hello from disk" in document.pages[0].text


def test_missing_file_raises(tmp_path):
    with pytest.raises(PDFExtractionError, match="not found"):
        extract_pages(tmp_path / "missing.pdf")