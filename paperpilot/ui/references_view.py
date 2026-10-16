"""Streamlit components for the References tab.

Flow: draft metadata (AI from page 1, or blank) -> user edits + confirms in a form
-> formatted references in the chosen style -> downloads.
"""

from __future__ import annotations

from pathlib import PurePath

import streamlit as st
from pydantic import ValidationError

from paperpilot.bibliography import (
    FORMATTERS,
    MetadataError,
    extract_metadata,
    find_identifiers,
    first_page_text,
    format_bibtex,
    missing_fields,
    references_markdown,
    verify_against_page,
)
from paperpilot.config import Settings
from paperpilot.llm import LLM, LLMError
from paperpilot.schemas import Author, ExtractedDocument, PaperMetadata

KINDS = ["journal", "conference", "preprint"]


def parse_authors(text: str) -> list[Author]:
    """One author per line: 'Family, Given' or 'Given Family'."""
    authors = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "," in line:
            family, given = (part.strip() for part in line.split(",", 1))
        else:
            *given_parts, family = line.split()
            given = " ".join(given_parts)
        authors.append(Author(given=given, family=family))
    return authors


def authors_to_text(authors: list[Author]) -> str:
    return "\n".join(f"{a.family}, {a.given}" for a in authors)


def parse_year(text: str) -> int | None:
    """Blank -> None. Anything else must be a plausible 4-digit year."""
    text = text.strip()
    if not text:
        return None
    if not (text.isdigit() and len(text) == 4):
        raise ValueError(f"Year must be 4 digits, got '{text}'")
    return int(text)


def blank_metadata(document: ExtractedDocument) -> PaperMetadata:
    """Starting point for manual entry: only what is certain (regex IDs, PDF title)."""
    ids = find_identifiers(first_page_text(document))
    return PaperMetadata(
        title=document.metadata.get("title", ""),
        authors=[],
        doi=ids["doi"],
        arxiv_id=ids["arxiv_id"],
    )


def references_filename(source_name: str, extension: str) -> str:
    return f"{PurePath(source_name).stem}_references.{extension}"


def render_references_tab(
    settings: Settings,
    document: ExtractedDocument | None,
    chat_model_ready: bool,
) -> None:
    if document is None:
        st.info("Upload a paper in the 📄 Paper tab first.", icon="📄")
        return

    draft_key = f"ref_draft_{document.file_hash[:16]}"
    confirmed_key = f"ref_confirmed_{document.file_hash[:16]}"

    confirmed: PaperMetadata | None = st.session_state.get(confirmed_key)
    if confirmed is not None:
        _render_references(confirmed, document, draft_key, confirmed_key)
        return

    draft: PaperMetadata | None = st.session_state.get(draft_key)
    if draft is None:
        _render_start(settings, document, chat_model_ready, draft_key)
        return

    _render_form(draft, document, draft_key, confirmed_key)


def _render_start(
    settings: Settings, document: ExtractedDocument, chat_model_ready: bool, draft_key: str
) -> None:
    st.write(
        "References are built **only** from page 1 of the paper and from what you confirm. "
        "Nothing is filled in from the model's memory."
    )
    col1, col2 = st.columns(2)
    if col1.button("Extract from page 1 (AI)", type="primary", disabled=not chat_model_ready):
        with st.spinner("Reading page 1..."):
            try:
                st.session_state[draft_key] = extract_metadata(document, LLM(settings))
            except (MetadataError, LLMError) as exc:
                st.error(str(exc))
                return
        st.rerun()
    if col2.button("Fill in manually"):
        st.session_state[draft_key] = blank_metadata(document)
        st.rerun()
    if not chat_model_ready:
        st.caption(f"AI extraction needs `{settings.chat_model}` (see System status).")


def _render_form(
    draft: PaperMetadata, document: ExtractedDocument, draft_key: str, confirmed_key: str
) -> None:
    st.subheader("Check and confirm the metadata")
    st.caption("Compare every field with page 1 of the PDF. Only confirmed data is formatted.")

    for warning in verify_against_page(draft, first_page_text(document)):
        st.warning(warning + " (from the model's memory, or a typo?)", icon="⚠️")

    with st.form("metadata_form"):
        title = st.text_input("Title", value=draft.title)
        authors_text = st.text_area(
            "Authors (one per line: Family, Given)", value=authors_to_text(draft.authors), height=180
        )
        col1, col2 = st.columns(2)
        year_text = col1.text_input("Year (publication year, not the arXiv date)", value=str(draft.year or ""))
        kind = col2.selectbox("Type", KINDS, index=KINDS.index(draft.kind))
        venue = st.text_input("Journal / conference", value=draft.venue or "")
        col3, col4, col5 = st.columns(3)
        volume = col3.text_input("Volume", value=draft.volume or "")
        issue = col4.text_input("Issue", value=draft.issue or "")
        pages = col5.text_input("Pages", value=draft.pages or "")
        col6, col7 = st.columns(2)
        doi = col6.text_input("DOI", value=draft.doi or "")
        arxiv_id = col7.text_input("arXiv ID", value=draft.arxiv_id or "")
        submitted = st.form_submit_button("Confirm metadata", type="primary")

    if st.button("Start over"):
        del st.session_state[draft_key]
        st.rerun()

    if not submitted:
        return
    try:
        metadata = PaperMetadata(
            title=title.strip(),
            authors=parse_authors(authors_text),
            year=parse_year(year_text),
            kind=kind,
            venue=venue.strip() or None,
            volume=volume.strip() or None,
            issue=issue.strip() or None,
            pages=pages.strip() or None,
            doi=doi.strip() or None,
            arxiv_id=arxiv_id.strip() or None,
        )
        if not metadata.title:
            raise ValueError("Title is required.")
    except (ValueError, ValidationError) as exc:
        st.error(f"Please fix the form: {exc}")
        return

    st.session_state[confirmed_key] = metadata
    st.rerun()


def _render_references(
    metadata: PaperMetadata, document: ExtractedDocument, draft_key: str, confirmed_key: str
) -> None:
    st.success("Metadata confirmed.", icon="✅")
    missing = missing_fields(metadata)
    if missing:
        st.caption(f"Not provided: {', '.join(missing)}. These parts are left out, never invented.")

    style = st.radio("Citation style", list(FORMATTERS), horizontal=True)
    formatted = FORMATTERS[style](metadata)
    if style == "BibTeX":
        st.code(formatted, language="latex")
    else:
        st.markdown(formatted.replace("$", r"\$"))
        st.code(formatted.replace("*", ""), language=None)  # plain text, easy to copy

    col1, col2, col3 = st.columns(3)
    col1.download_button(
        "Download .bib",
        data=format_bibtex(metadata),
        file_name=references_filename(document.source_name, "bib"),
        mime="application/x-bibtex",
        on_click="ignore",
    )
    col2.download_button(
        "Download all styles (.md)",
        data=references_markdown(metadata),
        file_name=references_filename(document.source_name, "md"),
        mime="text/markdown",
        on_click="ignore",
    )
    if col3.button("Edit metadata"):
        st.session_state[draft_key] = metadata  # edit what was confirmed, not the old draft
        del st.session_state[confirmed_key]
        st.rerun()