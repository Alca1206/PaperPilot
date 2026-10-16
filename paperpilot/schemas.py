"""Data shapes shared across PaperPilot modules.

Every piece of text carries a 1-indexed page number, so citations stay
possible all the way through the pipeline.
"""

from __future__ import annotations

from typing import Literal

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




class Finding(BaseModel):
    """One key result of a paper, with the pages that support it."""

    statement: str = Field(description="One key result, stated precisely")
    pages: list[int] = Field(description="Page numbers where this is supported")


class PaperSummary(BaseModel):
    """The structured summary the model must fill in (sent to Ollama as a JSON schema)."""

    title: str
    one_sentence_summary: str
    research_problem: str
    methodology: str
    key_findings: list[Finding] = Field(min_length=1, max_length=6)
    datasets: list[str] = Field(description="Datasets used; empty list if none are named")
    limitations: list[str] = Field(
        description="Limitations stated by the authors; empty list if none are stated"
    )
    future_work: list[str] = Field(description="Future work stated by the authors; empty list if none")





class Author(BaseModel):
    given: str = Field(description="First and middle names, e.g. 'Aidan N.'")
    family: str = Field(description="Last name, e.g. 'Gomez'")


class PaperMetadata(BaseModel):
    """Bibliographic data for one paper. Missing fields stay None: never guessed."""

    title: str
    authors: list[Author]
    year: int | None = None
    kind: Literal["journal", "conference", "preprint"] = "journal"
    venue: str | None = Field(None, description="Journal or conference name")
    volume: str | None = None
    issue: str | None = None
    pages: str | None = Field(None, description="Page range like '5998-6008'")
    doi: str | None = None
    arxiv_id: str | None = None