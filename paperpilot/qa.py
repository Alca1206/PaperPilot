"""Grounded question answering with page-level citations.

Flow (from the concept notebook, Part 6):
  question -> retrieve top chunks
  GATE 1: best score below threshold -> "Not found in the paper." (LLM never called)
  -> numbered sources [1]..[k] -> LLM answers ONLY from them, citing [n]
  GATE 2: check every [n] the model cited points to a real source

Page numbers always come from our stored metadata, never from the model.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from paperpilot.config import Settings
from paperpilot.embeddings import Embedder
from paperpilot.llm import LLM, Message
from paperpilot.retrieval import retrieve
from paperpilot.schemas import RetrievedChunk
from paperpilot.vector_store import PaperIndex

NOT_FOUND = "Not found in the paper."

SYSTEM_PROMPT = f"""You are PaperPilot, an assistant that answers questions about ONE research paper.

Rules:
1. Use ONLY the numbered sources provided. Do not use outside knowledge.
2. After every sentence that uses a source, cite it like [1] or [2][3].
3. If the sources do not contain the answer, reply with exactly: {NOT_FOUND}
4. Be concise: at most 4 sentences.
5. Never invent numbers, names or results that are not in the sources."""

# Matches [1], [2, 3] and [1,2] (several numbers inside one pair of brackets).
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

AnswerStatus = Literal[
    "ok",                    # answered, every citation valid
    "no_relevant_chunks",    # Gate 1 stopped it; the LLM was not called
    "model_said_not_found",  # the LLM read the sources and said "not found"
    "no_citations",          # answered, but cited nothing -> treat with caution
    "invalid_citations",     # cited a source number that doesn't exist
]


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    number: int  # the [n] used in the answer
    chunk_id: str
    page: int
    text: str
    score: float


class Answer(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str
    answer: str
    status: AnswerStatus
    citations: list[Citation] = []
    invalid_citations: list[int] = []
    best_score: float | None = None
    retrieved: list[RetrievedChunk] = []  # everything retrieved, for debugging


def build_context(results: Sequence[RetrievedChunk]) -> str:
    """Number the sources [1]..[k] and label each with its page."""
    return "\n\n".join(
        f"[{i}] (page {r.page})\n{r.text}" for i, r in enumerate(results, start=1)
    )


def build_messages(question: str, results: Sequence[RetrievedChunk]) -> list[Message]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sources:\n\n{build_context(results)}\n\nQuestion: {question}"},
    ]


def extract_citations(
    answer: str, results: Sequence[RetrievedChunk]
) -> tuple[list[Citation], list[int]]:
    """Find [n] markers in the answer. Return (valid citations, invalid numbers)."""
    numbers: set[int] = set()
    for group in _CITATION.findall(answer):
        numbers.update(int(n) for n in group.split(","))

    valid = sorted(n for n in numbers if 1 <= n <= len(results))
    invalid = sorted(n for n in numbers if n not in valid)
    citations = [
        Citation(
            number=n,
            chunk_id=results[n - 1].chunk_id,  # [1] is list index 0
            page=results[n - 1].page,
            text=results[n - 1].text,
            score=results[n - 1].score,
        )
        for n in valid
    ]
    return citations, invalid


def is_not_found(answer: str) -> bool:
    return answer.strip().lower().startswith(NOT_FOUND.lower().rstrip("."))


def answer_question(
    question: str,
    index: PaperIndex,
    embedder: Embedder,
    llm: LLM,
    settings: Settings,
) -> Answer:
    """Answer a question using only the paper. Raises EmbeddingError / LLMError on Ollama failures."""
    question = question.strip()
    results = retrieve(question, index, embedder, top_k=settings.qa_top_k)
    best = results[0].score if results else None

    # GATE 1: nothing relevant enough -> don't let the model guess.
    if best is None or best < settings.qa_min_score:
        return Answer(
            question=question,
            answer=NOT_FOUND,
            status="no_relevant_chunks",
            best_score=best,
            retrieved=results,
        )

    reply = llm.chat(build_messages(question, results))

    # GATE 2: check the model's work.
    if is_not_found(reply):
        return Answer(
            question=question,
            answer=NOT_FOUND,
            status="model_said_not_found",
            best_score=best,
            retrieved=results,
        )

    citations, invalid = extract_citations(reply, results)
    if invalid:
        status: AnswerStatus = "invalid_citations"
    elif not citations:
        status = "no_citations"
    else:
        status = "ok"

    return Answer(
        question=question,
        answer=reply,
        status=status,
        citations=citations,
        invalid_citations=invalid,
        best_score=best,
        retrieved=results,
    )