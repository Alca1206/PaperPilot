"""Tests for paperpilot.summarizer, with a fake LLM (no Ollama needed)."""

import json

import pytest
from pydantic import ValidationError

from paperpilot.config import Settings
from paperpilot.llm import LLM
from paperpilot.schemas import ExtractedDocument, PageText, PaperSummary
from paperpilot.summarizer import (
    SummaryError,
    check_pages,
    estimate_tokens,
    pages_to_text,
    summarize_document,
    to_markdown,
)
from tests.fakes import FakeChatClient


def summary_json(**overrides):
    data = {
        "title": "Attention Is All You Need",
        "one_sentence_summary": "A model built only on attention.",
        "research_problem": "Recurrent models are slow to train.",
        "methodology": "The Transformer.",
        "key_findings": [{"statement": "28.4 BLEU on EN-DE", "pages": [8]}],
        "datasets": ["WMT 2014 English-German"],
        "limitations": [],
        "future_work": ["Apply to images"],
    }
    data.update(overrides)
    return json.dumps(data)


def make_document(page_texts):
    return ExtractedDocument(
        source_name="paper.pdf",
        file_hash="ef" * 32,
        page_count=len(page_texts),
        pages=[PageText(page=i, text=t) for i, t in enumerate(page_texts, start=1)],
    )


SHORT_DOC = make_document(["Intro text.", "", "Results text."])
# ~10,000 tokens of text: too long for the direct strategy with default settings
LONG_DOC = make_document(["word " * 4000 for _ in range(10)])


# ---------- schema ----------


def test_valid_json_is_accepted():
    summary = PaperSummary.model_validate_json(summary_json())

    assert summary.key_findings[0].pages == [8]


@pytest.mark.parametrize(
    "raw",
    [
        '{"title": "only a title"}',              # missing fields
        summary_json(key_findings=[]),             # needs at least 1 finding
        "Sure! Here is the summary: ...",          # not JSON at all
    ],
)
def test_invalid_json_is_rejected(raw):
    with pytest.raises(ValidationError):
        PaperSummary.model_validate_json(raw)


# ---------- helpers ----------


def test_estimate_tokens_is_chars_divided_by_four():
    assert estimate_tokens("x" * 400) == 100


def test_pages_to_text_marks_pages_and_skips_empty_ones():
    text = pages_to_text(SHORT_DOC.pages)

    assert text == "--- Page 1 ---\nIntro text.\n\n--- Page 3 ---\nResults text."


def test_check_pages_flags_missing_and_impossible_pages():
    summary = PaperSummary.model_validate_json(
        summary_json(
            key_findings=[
                {"statement": "fine", "pages": [2]},
                {"statement": "impossible", "pages": [99]},
                {"statement": "unreferenced", "pages": []},
            ]
        )
    )

    problems = check_pages(summary, page_count=15)

    assert len(problems) == 2
    assert "[99]" in problems[0] and "15 pages" in problems[0]
    assert "no page reference" in problems[1]


# ---------- summarize_document ----------


def test_short_paper_uses_direct_strategy_in_one_call():
    client = FakeChatClient(replies=[summary_json()])

    result = summarize_document(SHORT_DOC, LLM(Settings(), client=client), Settings())

    assert result.strategy == "direct"
    assert result.attempts == 1
    assert result.summary.title == "Attention Is All You Need"
    call = client.calls[0]
    assert call["format"] == PaperSummary.model_json_schema()
    assert call["options"]["num_ctx"] == 16384
    assert "--- Page 1 ---" in call["messages"][1]["content"]


def test_invalid_answer_is_retried_with_the_error_shown_to_the_model():
    client = FakeChatClient(replies=['{"title": 1}', summary_json()])

    result = summarize_document(SHORT_DOC, LLM(Settings(), client=client), Settings())

    assert result.attempts == 2
    retry_messages = client.calls[1]["messages"]
    assert retry_messages[-2] == {"role": "assistant", "content": '{"title": 1}'}
    assert "That JSON was invalid" in retry_messages[-1]["content"]


def test_gives_up_after_max_attempts():
    client = FakeChatClient(replies=["nope", "still nope"])
    settings = Settings(summary_max_attempts=2)

    with pytest.raises(SummaryError, match="2 attempts"):
        summarize_document(SHORT_DOC, LLM(settings, client=client), settings)
    assert len(client.calls) == 2


def test_long_paper_uses_map_reduce():
    settings = Settings(summary_pages_per_group=4)
    # 10 pages / 4 per group = 3 note calls, then 1 summary call
    client = FakeChatClient(replies=["notes A (p. 1)", "notes B (p. 5)", "notes C (p. 9)", summary_json()])
    messages_seen = []

    result = summarize_document(LONG_DOC, LLM(settings, client=client), settings, status=messages_seen.append)

    assert result.strategy == "map-reduce"
    assert len(client.calls) == 4
    final_prompt = client.calls[3]["messages"][1]["content"]
    assert "notes A (p. 1)" in final_prompt and "notes C (p. 9)" in final_prompt
    assert "word word" not in final_prompt  # the raw pages were NOT sent in the final call
    assert any("pages 1-4" in m for m in messages_seen)


def test_page_problems_are_reported():
    client = FakeChatClient(replies=[summary_json(key_findings=[{"statement": "x", "pages": [42]}])])

    result = summarize_document(SHORT_DOC, LLM(Settings(), client=client), Settings())

    assert result.page_problems and "[42]" in result.page_problems[0]


# ---------- markdown export ----------


def test_markdown_has_all_sections_and_marks_empty_lists():
    summary = PaperSummary.model_validate_json(summary_json())

    markdown = to_markdown(summary, "attention.pdf")

    assert markdown.startswith("# Attention Is All You Need")
    assert "`attention.pdf`" in markdown
    assert "- 28.4 BLEU on EN-DE (p. 8)" in markdown
    assert "## Limitations\n_Not stated in the paper._" in markdown
    for heading in ["Research problem", "Methodology", "Key findings", "Datasets", "Future work"]:
        assert f"## {heading}" in markdown