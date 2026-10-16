"""PDF text extraction with page numbers (PyMuPDF).

No Streamlit code here. Failures raise PDFExtractionError with a message
that is safe to show to the user.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pymupdf

from paperpilot.schemas import ExtractedDocument, PageText


class PDFExtractionError(Exception):
    """The PDF could not be turned into usable text. The message is user-facing."""


def clean_text(text: str) -> str:
    """Turn PDF layout text into readable prose.

    PDFs store text line by line as laid out on the page. This re-joins
    hyphenated line breaks, turns single line breaks into spaces, keeps
    paragraph breaks and collapses repeated spaces.

    Known trade-off: a genuinely hyphenated word broken at a line end
    ("self-\\nattention") becomes "selfattention".
    """
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{2,}", "\x00", text)  # protect paragraph breaks
    text = text.replace("\n", " ")
    text = text.replace("\x00", "\n\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n\n ?", "\n\n", text)
    return text.strip()


def compute_file_hash(data: bytes) -> str:
    """SHA-256 fingerprint of the file contents (same bytes -> same hash)."""
    return hashlib.sha256(data).hexdigest()


def extract_pages_from_bytes(data: bytes, source_name: str) -> ExtractedDocument:
    """Extract cleaned text for every page of a PDF given as bytes.

    Raises PDFExtractionError for empty, corrupt, password-protected or
    text-less (e.g. scanned) PDFs.
    """
    if not data:
        raise PDFExtractionError(f"'{source_name}' is empty (0 bytes).")

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except pymupdf.FileDataError as exc:
        raise PDFExtractionError(
            f"'{source_name}' could not be opened. The file may be corrupt or not a PDF."
        ) from exc

    with doc:
        if doc.needs_pass:
            raise PDFExtractionError(
                f"'{source_name}' is password-protected. Remove the password and upload it again."
            )
        if doc.page_count == 0:
            raise PDFExtractionError(f"'{source_name}' has no pages.")

        pages = [
            # PyMuPDF counts from 0; humans and citations count from 1.
            PageText(page=index + 1, text=clean_text(page.get_text("text", sort=True)))
            for index, page in enumerate(doc)
        ]
        metadata = {key: str(value) for key, value in (doc.metadata or {}).items() if value}
        page_count = doc.page_count

    document = ExtractedDocument(
        source_name=source_name,
        file_hash=compute_file_hash(data),
        page_count=page_count,
        pages=pages,
        metadata=metadata,
    )

    if document.total_chars == 0:
        raise PDFExtractionError(
            f"No text could be extracted from '{source_name}'. It may be a scanned "
            "document (images only), which PaperPilot cannot read yet."
        )
    return document


def extract_pages(path: str | Path) -> ExtractedDocument:
    """Convenience wrapper: extract from a file on disk."""
    path = Path(path)
    try:
        data = path.read_bytes()
    except FileNotFoundError as exc:
        raise PDFExtractionError(f"File not found: {path}") from exc
    return extract_pages_from_bytes(data, source_name=path.name)