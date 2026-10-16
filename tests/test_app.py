"""Tests for the Streamlit UI layer."""

import pytest
from streamlit.testing.v1 import AppTest

from paperpilot.config import PROJECT_ROOT
from paperpilot.health import HealthReport
from paperpilot.qa import Answer, Citation
from paperpilot.schemas import Chunk, RetrievedChunk
from paperpilot.ui.ask_view import citation_label, escape_markdown, retrieved_rows, status_note
from paperpilot.ui.health_view import model_status_rows
from paperpilot.ui.paper_view import chunk_table_rows, process_pdf

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