"""Split pages into chunks for retrieval.

Rules:
- A chunk never spans two pages, so every chunk has exactly one page number.
- Chunks are built from whole sentences, up to ``max_chars`` (a hard limit).
- Neighbouring chunks share ``overlap_sentences`` sentences, so a fact that
  sits on a boundary appears whole in at least one chunk.
"""

from __future__ import annotations

import re

from paperpilot.schemas import Chunk, ExtractedDocument, PageText

MIN_MAX_CHARS = 50

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_into_sentences(text: str) -> list[str]:
    """Split on whitespace that follows ., ! or ? (punctuation is kept).

    Not grammatically perfect ("et al.", "Fig. 3"), and it doesn't need to be:
    a chunker only needs reasonable cut points.
    """
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]


def split_long_piece(piece: str, max_chars: int) -> list[str]:
    """Split text with no sentence breaks (references, tables, URLs).

    Splits by words; a single word longer than ``max_chars`` is cut into slices.
    Every returned piece is at most ``max_chars`` long.
    """
    words: list[str] = []
    for word in piece.split():
        words.extend(word[i : i + max_chars] for i in range(0, len(word), max_chars))

    pieces: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars and current:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _validate_params(max_chars: int, overlap_sentences: int) -> None:
    if max_chars < MIN_MAX_CHARS:
        raise ValueError(f"max_chars must be at least {MIN_MAX_CHARS}, got {max_chars}")
    if overlap_sentences < 0:
        raise ValueError(f"overlap_sentences must be >= 0, got {overlap_sentences}")


def chunk_page(page: PageText, max_chars: int = 800, overlap_sentences: int = 1) -> list[Chunk]:
    """Chunk a single page. Chunk ids restart per page: p{page}-c1, p{page}-c2, ..."""
    _validate_params(max_chars, overlap_sentences)

    sentences: list[str] = []
    for sentence in split_into_sentences(page.text):
        if len(sentence) > max_chars:
            sentences.extend(split_long_piece(sentence, max_chars))
        else:
            sentences.append(sentence)

    texts: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if current and len(" ".join([*current, sentence])) > max_chars:
            texts.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            # If overlap + new sentence is still too long, drop the overlap.
            if len(" ".join([*current, sentence])) > max_chars:
                current = []
        current.append(sentence)
    if current:
        texts.append(" ".join(current))

    return [
        Chunk(chunk_id=f"p{page.page}-c{i}", page=page.page, text=text)
        for i, text in enumerate(texts, start=1)
    ]


def chunk_document(
    document: ExtractedDocument,
    max_chars: int = 800,
    overlap_sentences: int = 1,
    min_chunk_chars: int = 30,
) -> list[Chunk]:
    """Chunk every page and drop tiny junk chunks (e.g. a lone page number)."""
    chunks: list[Chunk] = []
    for page in document.pages:
        for chunk in chunk_page(page, max_chars, overlap_sentences):
            if chunk.char_count >= min_chunk_chars:
                chunks.append(chunk)
    return chunks