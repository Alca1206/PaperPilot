"""Glue: chunks -> embeddings -> vector store, skipping work already done."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from paperpilot.config import Settings
from paperpilot.embeddings import Embedder, ProgressCallback
from paperpilot.schemas import Chunk, ExtractedDocument
from paperpilot.vector_store import PaperIndex, collection_name


@dataclass(frozen=True)
class IndexResult:
    collection_name: str
    chunk_count: int
    reused: bool  # True = found a complete index on disk, nothing was embedded


def open_index(document: ExtractedDocument, settings: Settings, chroma_client: Any) -> PaperIndex:
    return PaperIndex(chroma_client, collection_name(document.file_hash, settings))


def index_paper(
    document: ExtractedDocument,
    chunks: Sequence[Chunk],
    settings: Settings,
    embedder: Embedder,
    chroma_client: Any,
    progress: ProgressCallback | None = None,
) -> IndexResult:
    """Embed and store a paper's chunks, unless a complete index already exists.

    Embeddings are computed for ALL chunks before anything is written, so an
    Ollama failure halfway through leaves no half-built index behind.
    """
    index = open_index(document, settings, chroma_client)
    if index.is_complete():
        return IndexResult(index.name, index.count, reused=True)

    vectors = embedder.embed_documents([c.text for c in chunks], progress=progress)
    index.add_chunks(chunks, vectors, source_name=document.source_name)
    return IndexResult(index.name, index.count, reused=False)