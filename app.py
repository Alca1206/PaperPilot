"""PaperPilot: Streamlit entry point.

Run with:  streamlit run app.py

This file only wires the UI together. All logic lives in the paperpilot package.
"""

import streamlit as st

from paperpilot.config import load_settings
from paperpilot.ui.ask_view import render_ask_tab
from paperpilot.ui.health_view import (
    get_health_report,
    render_health_details,
    render_health_summary,
)
from paperpilot.ui.paper_view import render_paper_tab
from paperpilot.ui.references_view import render_references_tab
from paperpilot.ui.summary_view import render_summary_tab


def main() -> None:
    st.set_page_config(page_title="PaperPilot", page_icon="📄", layout="wide")

    try:
        settings = load_settings()
    except ValueError as exc:
        st.error(f"Invalid configuration: {exc}")
        st.stop()

    settings.ensure_directories()
    report = get_health_report(settings)

    with st.sidebar:
        st.title("📄 PaperPilot")
        st.caption("Local-first AI research paper assistant")
        st.divider()
        render_health_summary(report)
        st.divider()
        st.caption(f"Chat model: `{settings.chat_model}`")
        st.caption(f"Embedding model: `{settings.embedding_model}`")

    st.title("PaperPilot")
    st.write(
        "Upload academic papers, get structured summaries, ask questions "
        "grounded in the text, and export references. Everything runs locally."
    )

    paper_tab, summary_tab, ask_tab, refs_tab, status_tab = st.tabs(
        ["📄 Paper", "📝 Summary", "💬 Ask", "📚 References", "🩺 System status"]
    )
    with paper_tab:
        document = render_paper_tab(
            settings,
            embedding_model_ready=report.is_model_available(settings.embedding_model),
        )
    with summary_tab:
        render_summary_tab(
            settings,
            document,
            chat_model_ready=report.is_model_available(settings.chat_model),
        )
    with ask_tab:
        render_ask_tab(settings, document, models_ready=report.ok)
    with refs_tab:
        render_references_tab(
            settings,
            document,
            chat_model_ready=report.is_model_available(settings.chat_model),
        )
    with status_tab:
        render_health_details(report, settings)


if __name__ == "__main__":
    main()