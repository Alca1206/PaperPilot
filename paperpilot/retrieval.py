"""Find the chunks of a paper that best match a question."""

from __future__ import annotations

from paperpilot.embeddings import Embedder
from paperpilot.schemas import RetrievedChunk
from paperpilot.vector_store import PaperIndex


def retrieve(question: str, index: PaperIndex, embedder: Embedder, top_k: int) -> list[RetrievedChunk]:
    """Embed the question the same way as the chunks, return the top_k closest chunks.

    Results are sorted best first and always carry their page number.
    """
    question = question.strip()
    if not question:
        raise ValueError("The question is empty.")
    return index.query(embedder.embed_query(question), top_k=top_k)