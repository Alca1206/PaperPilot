"""Tests for the Streamlit UI layer."""

import pytest
from streamlit.testing.v1 import AppTest

from paperpilot.config import PROJECT_ROOT, Settings
from paperpilot.health import HealthReport
from paperpilot.qa import Answer, Citation
from paperpilot.schemas import Author, Chunk, RetrievedChunk
from paperpilot.ui.ask_view import citation_label, escape_markdown, retrieved_rows, status_note
from paperpilot.ui.health_view import model_status_rows
from paperpilot.ui.paper_view import chunk_table_rows, process_pdf
from paperpilot.ui.summary_view import summary_cache_key, summary_filename
from paperpilot.ui.references_view import (
    authors_to_text,
    blank_metadata,
    parse_authors,
    parse_year,
    references_filename,
)

REQUIRED = ("qwen3:8b", "embeddinggemma")


def test_rows_when_all_models_installed():
    report = HealthReport(
        ollama_host="http://localhost:11434",
        ollama_reachable=True,
        required_models=REQUIRED,
        installed_models=("qwen3:8b", "embeddinggemma:latest"),
    )

    statuses = [row["Status"] for row in model_status_rows(report)]

    assert statuses == ["✅ Installed", "✅ Installed"]


def test_rows_mark_missing_model():
    report = HealthReport(
        ollama_host="http://localhost:11434",
        ollama_reachable=True,
        required_models=REQUIRED,
        installed_models=("qwen3:8b",),
        missing_models=("embeddinggemma",),
    )

    rows = model_status_rows(report)

    assert rows[0] == {"Model": "qwen3:8b", "Status": "✅ Installed"}
    assert rows[1] == {"Model": "embeddinggemma", "Status": "❌ Missing"}


def test_rows_unknown_when_ollama_unreachable():
    report = HealthReport(
        ollama_host="http://localhost:11434",
        ollama_reachable=False,
        required_models=REQUIRED,
        error="Could not connect",
    )

    statuses = [row["Status"] for row in model_status_rows(report)]

    assert all(status.startswith("❔") for status in statuses)


def test_app_starts_without_errors():
    """Smoke test: the whole app script runs, whether or not Ollama is up."""
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()

    assert not app.exception
    titles = [title.value for title in app.title]
    assert "PaperPilot" in titles          # main page
    assert "📄 PaperPilot" in titles       # sidebar
    assert len(app.tabs) == 5


# ---------- Phase 2: Paper tab ----------


def test_chunk_table_rows_shorten_long_previews():
    chunks = [
        Chunk(chunk_id="p1-c1", page=1, text="short text"),
        Chunk(chunk_id="p2-c1", page=2, text="x" * 300),
    ]

    rows = chunk_table_rows(chunks, preview_chars=120)

    assert rows[0] == {"Chunk": "p1-c1", "Page": 1, "Characters": 10, "Preview": "short text"}
    assert rows[1]["Preview"].endswith("…")
    assert len(rows[1]["Preview"]) == 121


def test_process_pdf_extracts_and_chunks(make_pdf):
    data = make_pdf([
        "Page one has a sentence. And another one.",
        "Page two has enough text to survive the junk filter.",
    ])

    document, chunks = process_pdf(data, "paper.pdf", 800, 1)

    assert document.page_count == 2
    assert {c.page for c in chunks} == {1, 2}


def test_paper_tab_shows_upload_prompt_before_upload():
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()

    assert not app.exception
    assert any("Upload a PDF" in info.value for info in app.info)


# ---------- Phase 4: Ask tab ----------


def make_answer(**overrides):
    fields = {"question": "Q?", "answer": "A [1].", "status": "ok", "best_score": 0.8}
    fields.update(overrides)
    return Answer(**fields)


def test_escape_markdown_protects_dollar_signs():
    assert escape_markdown("costs $5 and $x$") == r"costs \$5 and \$x\$"


def test_citation_label_shows_number_page_and_score():
    citation = Citation(number=2, chunk_id="p7-c2", page=7, text="t", score=0.734)

    assert citation_label(citation) == "[2] Page 7 · p7-c2 · similarity 0.73"


def test_status_note_ok_is_silent():
    assert status_note(make_answer(), min_score=0.35) is None


@pytest.mark.parametrize(
    ("overrides", "level", "snippet"),
    [
        ({"status": "no_relevant_chunks", "best_score": 0.12}, "info", "best similarity 0.12"),
        ({"status": "model_said_not_found"}, "info", "found no answer"),
        ({"status": "no_citations"}, "warning", "cites no sources"),
        ({"status": "invalid_citations", "invalid_citations": [9]}, "warning", "[9]"),
    ],
)
def test_status_note_explains_each_problem(overrides, level, snippet):
    note = status_note(make_answer(**overrides), min_score=0.35)

    assert note[0] == level
    assert snippet in note[1]


def test_retrieved_rows_are_ranked():
    answer = make_answer(
        retrieved=[
            RetrievedChunk(chunk_id="p7-c2", page=7, text="GPU text", score=0.81234),
            RetrievedChunk(chunk_id="p3-c1", page=3, text="Other", score=0.4),
        ]
    )

    rows = retrieved_rows(answer)

    assert [(r["Rank"], r["Page"], r["Similarity"]) for r in rows] == [(1, 7, 0.812), (2, 3, 0.4)]


def test_ask_tab_asks_for_a_paper_first():
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()

    assert not app.exception
    assert any("Upload a paper in the" in info.value for info in app.info)




# ---------- Phase 5: Summary tab ----------


@pytest.mark.parametrize(
    ("source", "expected"),
    [("attention.pdf", "attention_summary.md"), ("my paper.v2.pdf", "my paper.v2_summary.md")],
)
def test_summary_filename(source, expected):
    assert summary_filename(source) == expected


def test_summary_cache_key_changes_with_paper_and_model(make_pdf):
    document_a, _ = process_pdf(make_pdf(["Paper A has some text here."]), "a.pdf", 800, 1)
    document_b, _ = process_pdf(make_pdf(["Paper B has other text here."]), "b.pdf", 800, 1)
    settings = Settings()

    assert summary_cache_key(document_a, settings) != summary_cache_key(document_b, settings)
    assert summary_cache_key(document_a, settings) != summary_cache_key(
        document_a, Settings(chat_model="llama3.2:3b")
    )




# ---------- Phase 6: References tab ----------


def test_parse_authors_accepts_both_name_orders():
    authors = parse_authors("Vaswani, Ashish\n\nAidan N. Gomez\n  Kaiser, Łukasz  ")

    assert [(a.given, a.family) for a in authors] == [
        ("Ashish", "Vaswani"),
        ("Aidan N.", "Gomez"),
        ("Łukasz", "Kaiser"),
    ]


def test_authors_round_trip_through_the_form():
    authors = [Author(given="Aidan N.", family="Gomez"), Author(given="Ashish", family="Vaswani")]

    assert parse_authors(authors_to_text(authors)) == authors


@pytest.mark.parametrize(("text", "expected"), [("2017", 2017), ("  ", None), ("", None)])
def test_parse_year(text, expected):
    assert parse_year(text) == expected


@pytest.mark.parametrize("bad", ["17", "twenty", "2017a"])
def test_parse_year_rejects_bad_input(bad):
    with pytest.raises(ValueError, match="4 digits"):
        parse_year(bad)


def test_blank_metadata_only_contains_certain_facts(make_pdf):
    data = make_pdf(["My Title. arXiv:1706.03762v7 and some more text."], metadata={"title": "My Title"})
    document, _ = process_pdf(data, "paper.pdf", 800, 1)

    metadata = blank_metadata(document)

    assert metadata.title == "My Title"
    assert metadata.arxiv_id == "1706.03762"
    assert metadata.authors == [] and metadata.year is None and metadata.venue is None


def test_references_filename():
    assert references_filename("attention.pdf", "bib") == "attention_references.bib"


def test_app_has_no_placeholders_left():
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()

    assert not app.exception
    assert not any("arrives in Phase" in info.value for info in app.info)