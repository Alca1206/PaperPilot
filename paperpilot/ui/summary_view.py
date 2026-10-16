"""Streamlit components for the Summary tab: structured summary + Markdown download."""

from __future__ import annotations

from pathlib import PurePath

import streamlit as st

from paperpilot.config import Settings
from paperpilot.llm import LLM, LLMError
from paperpilot.schemas import ExtractedDocument
from paperpilot.summarizer import SummaryError, SummaryResult, summarize_document, to_markdown
from paperpilot.ui.ask_view import escape_markdown


def summary_cache_key(document: ExtractedDocument, settings: Settings) -> str:
    """Same paper + same chat model -> reuse the summary for this browser session."""
    return f"summary_{document.file_hash[:16]}_{settings.chat_model}"


def summary_filename(source_name: str) -> str:
    return f"{PurePath(source_name).stem}_summary.md"


def render_summary_tab(
    settings: Settings,
    document: ExtractedDocument | None,
    chat_model_ready: bool,
) -> None:
    if document is None:
        st.info("Upload a paper in the 📄 Paper tab first.", icon="📄")
        return

    key = summary_cache_key(document, settings)
    result: SummaryResult | None = st.session_state.get(key)

    if result is None:
        if not chat_model_ready:
            st.warning(
                f"Summaries need Ollama with `{settings.chat_model}`. "
                "Check the System status tab, then click Re-check.",
                icon="⚠️",
            )
            return
        st.caption(
            f"Reads the whole paper with `{settings.chat_model}` and fills in a fixed structure. "
            "This can take a few minutes on a laptop."
        )
        if not st.button("Generate summary", type="primary"):
            return

        progress = st.empty()
        try:
            result = summarize_document(
                document, LLM(settings), settings, status=lambda msg: progress.info(msg, icon="⏳")
            )
        except (SummaryError, LLMError) as exc:
            progress.empty()
            st.error(str(exc))
            return
        st.session_state[key] = result
        st.rerun()  # redraw from the top so the page layout is the same every time

    _render_summary(result, document)

    if st.button("Regenerate summary"):
        del st.session_state[key]
        st.rerun()


def _render_summary(result: SummaryResult, document: ExtractedDocument) -> None:
    summary = result.summary
    st.subheader(escape_markdown(summary.title))
    st.markdown(f"**In one sentence:** {escape_markdown(summary.one_sentence_summary)}")

    for problem in result.page_problems:
        st.warning(f"Page check: {problem}", icon="⚠️")

    st.markdown("#### Research problem")
    st.markdown(escape_markdown(summary.research_problem))
    st.markdown("#### Methodology")
    st.markdown(escape_markdown(summary.methodology))

    st.markdown("#### Key findings")
    for finding in summary.key_findings:
        pages = ", ".join(map(str, finding.pages)) or "?"
        st.markdown(f"- {escape_markdown(finding.statement)} *(p. {pages})*")

    for heading, items in [
        ("Datasets", summary.datasets),
        ("Limitations", summary.limitations),
        ("Future work", summary.future_work),
    ]:
        st.markdown(f"#### {heading}")
        if items:
            st.markdown("\n".join(f"- {escape_markdown(item)}" for item in items))
        else:
            st.caption("Not stated in the paper.")

    st.caption(
        f"Strategy: {result.strategy} · valid JSON on attempt {result.attempts} · "
        "AI-generated: verify important claims against the paper."
    )
    st.download_button(
        "Download summary (.md)",
        data=to_markdown(summary, document.source_name),
        file_name=summary_filename(document.source_name),
        mime="text/markdown",
        type="primary",
        on_click="ignore",  # downloading should not rerun the whole app
    )