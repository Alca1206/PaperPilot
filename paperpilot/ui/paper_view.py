"""Streamlit components for the Paper tab: upload, extraction and chunk inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from paperpilot.chunking import chunk_document
from paperpilot.config import Settings
from paperpilot.embeddings import Embedder, EmbeddingError
from paperpilot.indexing import index_paper, open_index
from paperpilot.pdf_processing import PDFExtractionError, extract_pages_from_bytes
from paperpilot.schemas import Chunk, ExtractedDocument
from paperpilot.vector_store import make_chroma_client

LOW_TEXT_PAGE_CHARS = 20


@st.cache_data(show_spinner=False, max_entries=10)
def process_pdf(
    data: bytes, source_name: str, max_chars: int, overlap_sentences: int
) -> tuple[ExtractedDocument, list[Chunk]]:
    """Extract and chunk a PDF. Cached: the same file + settings is processed once."""
    document = extract_pages_from_bytes(data, source_name)
    chunks = chunk_document(document, max_chars=max_chars, overlap_sentences=overlap_sentences)
    return document, chunks


@st.cache_resource
def get_chroma_client(chroma_dir: str) -> Any:
    """One shared database connection for the whole app (not one per rerun)."""
    return make_chroma_client(Path(chroma_dir))


def chunk_table_rows(chunks: list[Chunk], preview_chars: int = 120) -> list[dict[str, object]]:
    """Rows for the chunk overview table."""
    return [
        {
            "Chunk": chunk.chunk_id,
            "Page": chunk.page,
            "Characters": chunk.char_count,
            "Preview": chunk.text[:preview_chars] + ("…" if chunk.char_count > preview_chars else ""),
        }
        for chunk in chunks
    ]


def render_paper_tab(settings: Settings, embedding_model_ready: bool) -> None:
    uploaded_files = st.file_uploader(
        "Upload research papers (PDF)", type=["pdf"], accept_multiple_files=True
    )
    if not uploaded_files:
        st.info("Upload a PDF to get started.", icon="📄")
        return

    # Several files can be uploaded, but one paper is processed at a time.
    names = [f.name for f in uploaded_files]
    selected_name = st.selectbox("Paper to work on", names) if len(names) > 1 else names[0]
    uploaded = uploaded_files[names.index(selected_name)]

    try:
        with st.spinner(f"Reading {uploaded.name}..."):
            document, chunks = process_pdf(
                uploaded.getvalue(),
                uploaded.name,
                settings.chunk_max_chars,
                settings.chunk_overlap_sentences,
            )
    except PDFExtractionError as exc:
        st.error(str(exc))
        return

    _render_overview(document, chunks)
    _render_pages(document)
    _render_chunks(chunks, settings)
    _render_index(document, chunks, settings, embedding_model_ready)


def _render_overview(document: ExtractedDocument, chunks: list[Chunk]) -> None:
    st.subheader(document.metadata.get("title") or document.source_name)
    col1, col2, col3 = st.columns(3)
    col1.metric("Pages", document.page_count)
    col2.metric("Chunks", len(chunks))
    col3.metric("Characters", f"{document.total_chars:,}")

    low_text = document.low_text_pages(LOW_TEXT_PAGE_CHARS)
    if low_text:
        st.warning(
            f"Little or no text found on page(s) {', '.join(map(str, low_text))}. "
            "They may be blank, figures only, or scanned images.",
            icon="⚠️",
        )


def _render_pages(document: ExtractedDocument) -> None:
    with st.expander("Extracted text by page"):
        page_number = st.number_input(
            "Page", min_value=1, max_value=document.page_count, value=1, step=1
        )
        page = document.pages[int(page_number) - 1]
        st.caption(f"{page.char_count:,} characters")
        st.text(page.text or "(no text on this page)")


def _render_chunks(chunks: list[Chunk], settings: Settings) -> None:
    with st.expander(f"Chunks ({len(chunks)})"):
        st.caption(
            f"Up to {settings.chunk_max_chars} characters each, "
            f"{settings.chunk_overlap_sentences} sentence(s) of overlap, never crossing a page."
        )
        st.dataframe(chunk_table_rows(chunks), hide_index=True, width="stretch")


def _render_index(
    document: ExtractedDocument,
    chunks: list[Chunk],
    settings: Settings,
    embedding_model_ready: bool,
) -> None:
    st.subheader("Search index")
    chroma = get_chroma_client(str(settings.chroma_dir))
    index = open_index(document, settings, chroma)

    if index.is_complete():
        st.success(f"Indexed: {index.count} chunks saved on disk. Ready for questions.", icon="✅")
        return
    if not chunks:
        st.warning("This paper produced no chunks, so there is nothing to index.", icon="⚠️")
        return
    if not embedding_model_ready:
        st.warning(
            f"Building the index needs Ollama with `{settings.embedding_model}`. "
            "Check the System status tab, then click Re-check.",
            icon="⚠️",
        )
        return

    st.caption(
        f"Embeds {len(chunks)} chunks with `{settings.embedding_model}` and saves them to "
        "disk. This happens once per paper; afterwards it loads instantly."
    )
    if not st.button("Build search index", type="primary"):
        return

    bar = st.progress(0.0, text="Embedding chunks...")

    def on_progress(done: int, total: int) -> None:
        bar.progress(done / total, text=f"Embedding chunks... {done}/{total}")

    try:
        index_paper(document, chunks, settings, Embedder(settings), chroma, on_progress)
    except EmbeddingError as exc:
        bar.empty()
        st.error(str(exc))
        return

    st.rerun()  # redraw: the index is now complete, so the success message shows