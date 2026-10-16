"""Tests for the Streamlit UI layer."""

from streamlit.testing.v1 import AppTest

from paperpilot.config import PROJECT_ROOT
from paperpilot.health import HealthReport
from paperpilot.schemas import Chunk
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