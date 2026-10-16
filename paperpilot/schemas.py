"""Data shapes shared across PaperPilot modules.

Every piece of text carries a 1-indexed page number, so citations stay
possible all the way through the pipeline.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PageText(BaseModel):
    """Cleaned text of one PDF page."""

    model_config = ConfigDict(frozen=True)

    page: int = Field(ge=1, description="1-indexed page number, as a human reads it")
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)


class Chunk(BaseModel):
    """A piece of one page. A chunk never spans two pages."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(description="Stable id like 'p4-c2' (page 4, chunk 2)")
    page: int = Field(ge=1)
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)


class ExtractedDocument(BaseModel):
    """Everything extracted from one PDF."""

    model_config = ConfigDict(frozen=True)

    source_name: str
    file_hash: str = Field(description="SHA-256 of the file contents")
    page_count: int = Field(ge=1)
    pages: list[PageText]
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def total_chars(self) -> int:
        return sum(page.char_count for page in self.pages)

    def low_text_pages(self, min_chars: int) -> list[int]:
        """Pages with almost no text: blank pages or scanned images."""
        return [page.page for page in self.pages if page.char_count < min_chars]

class RetrievedChunk(BaseModel):
    """A chunk returned by a similarity search, with its score."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    page: int = Field(ge=1)
    text: str
    score: float = Field(description="Cosine similarity: 1 = identical meaning")