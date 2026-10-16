"""Streamlit components for the Ask tab: grounded Q&A with page citations."""

from __future__ import annotations

import streamlit as st

from paperpilot.config import Settings
from paperpilot.embeddings import Embedder, EmbeddingError
from paperpilot.indexing import open_index
from paperpilot.llm import LLM, LLMError
from paperpilot.qa import Answer, Citation, answer_question
from paperpilot.schemas import ExtractedDocument
from paperpilot.ui.paper_view import get_chroma_client


def escape_markdown(text: str) -> str:
    """Stop Streamlit from treating $...$ in papers as LaTeX maths."""
    return text.replace("$", r"\$")


def citation_label(citation: Citation) -> str:
    return f"[{citation.number}] Page {citation.page} · {citation.chunk_id} · similarity {citation.score:.2f}"


def status_note(answer: Answer, min_score: float) -> tuple[str, str] | None:
    """(level, message) explaining a non-ok answer, or None. level: 'info' | 'warning'."""
    if answer.status == "no_relevant_chunks":
        best = "none" if answer.best_score is None else f"{answer.best_score:.2f}"
        return (
            "info",
            f"No passage was similar enough to the question (best similarity {best}, "
            f"threshold {min_score:.2f}), so the model was not asked.",
        )
    if answer.status == "model_said_not_found":
        return ("info", "The model read the most relevant passages and found no answer in them.")
    if answer.status == "no_citations":
        return ("warning", "This answer cites no sources. Treat it with caution.")
    if answer.status == "invalid_citations":
        numbers = ", ".join(f"[{n}]" for n in answer.invalid_citations)
        return ("warning", f"The model cited {numbers}, which is not one of the sources. Treat it with caution.")
    return None


def retrieved_rows(answer: Answer) -> list[dict[str, object]]:
    return [
        {"Rank": i, "Page": r.page, "Chunk": r.chunk_id, "Similarity": round(r.score, 3), "Preview": r.text[:100]}
        for i, r in enumerate(answer.retrieved, start=1)
    ]


def render_ask_tab(
    settings: Settings,
    document: ExtractedDocument | None,
    models_ready: bool,
) -> None:
    if document is None:
        st.info("Upload a paper in the 📄 Paper tab first.", icon="📄")
        return

    index = open_index(document, settings, get_chroma_client(str(settings.chroma_dir)))
    if not index.is_complete():
        st.info("Build the search index in the 📄 Paper tab first.", icon="🔎")
        return
    if not models_ready:
        st.warning(
            "Asking questions needs Ollama with both models installed. "
            "Check the System status tab, then click Re-check.",
            icon="⚠️",
        )
        return

    st.caption(f"Answers come only from **{document.source_name}**, with page citations.")

    # One conversation per paper + index settings, kept for this browser session.
    history: list[Answer] = st.session_state.setdefault(f"qa_history_{index.name}", [])

    question = st.chat_input("Ask a question about this paper")
    if question and question.strip():
        with st.spinner("Searching the paper and writing an answer..."):
            try:
                answer = answer_question(question, index, Embedder(settings), LLM(settings), settings)
            except (EmbeddingError, LLMError) as exc:
                st.error(str(exc))
                answer = None
        if answer is not None:
            history.append(answer)

    for answer in reversed(history):  # newest first, right under the input box
        _render_answer(answer, settings)

    if history and st.button("Clear conversation"):
        history.clear()
        st.rerun()


def _render_answer(answer: Answer, settings: Settings) -> None:
    with st.chat_message("user"):
        st.markdown(escape_markdown(answer.question))

    with st.chat_message("assistant"):
        st.markdown(escape_markdown(answer.answer))

        note = status_note(answer, settings.qa_min_score)
        if note is not None:
            level, message = note
            (st.warning if level == "warning" else st.caption)(message)

        for citation in answer.citations:
            with st.expander(citation_label(citation)):
                st.markdown("> " + escape_markdown(citation.text))

        if answer.retrieved:
            with st.expander("All retrieved passages"):
                st.dataframe(retrieved_rows(answer), hide_index=True, width="stretch")