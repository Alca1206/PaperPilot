"""Structured paper summaries (concept notebook, Part 7).

- Pydantic schema -> Ollama `format=` -> JSON in exactly that shape
- Validate with Pydantic; on failure, show the model its error and retry
- Paper too long for the context window -> map-reduce:
  notes per group of pages (map), then the summary from the notes (reduce)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import ValidationError

from paperpilot.config import Settings
from paperpilot.llm import LLM, Message
from paperpilot.schemas import ExtractedDocument, PageText, PaperSummary

StatusCallback = Callable[[str], None]  # receives short progress messages for the UI

NOTES_PROMPT = """You are taking notes on part of a research paper.
Write concise bullet-point notes covering: the problem, methods, datasets, results (with exact numbers),
limitations and future work, but ONLY what appears in this text.
End every bullet with the page it came from, like (p. 3). Do not add outside knowledge."""

SUMMARY_PROMPT = """You summarize ONE research paper into JSON.
Rules:
- Use ONLY the provided text. Never use outside knowledge.
- For each key finding, list the page numbers where it is stated (pages are marked '--- Page N ---' or '(p. N)').
- If the paper does not state datasets, limitations or future work, return an empty list. Do NOT guess.
- Keep numbers exactly as written in the paper."""


class SummaryError(Exception):
    """No valid summary could be produced. The message is safe to show to the user."""


@dataclass(frozen=True)
class SummaryResult:
    summary: PaperSummary
    strategy: Literal["direct", "map-reduce"]
    attempts: int
    page_problems: list[str]


def estimate_tokens(text: str) -> int:
    """Rough rule of thumb: about 4 characters per token for English text."""
    return len(text) // 4


def pages_to_text(pages: Sequence[PageText]) -> str:
    """Join pages with '--- Page N ---' markers so the model can see where text came from."""
    return "\n\n".join(f"--- Page {p.page} ---\n{p.text}" for p in pages if p.text)


def make_notes(
    pages: Sequence[PageText],
    llm: LLM,
    settings: Settings,
    status: StatusCallback | None = None,
) -> str:
    """MAP step: summarize groups of pages into short notes that keep (p. N) markers."""
    size = settings.summary_pages_per_group
    text_pages = [p for p in pages if p.text]
    groups = [text_pages[i : i + size] for i in range(0, len(text_pages), size)]

    notes = []
    for number, group in enumerate(groups, start=1):
        if status:
            status(f"Taking notes on pages {group[0].page}-{group[-1].page} ({number}/{len(groups)})...")
        notes.append(
            llm.chat(
                [
                    {"role": "system", "content": NOTES_PROMPT},
                    {"role": "user", "content": pages_to_text(group)},
                ],
                num_ctx=settings.summary_num_ctx,
            )
        )
    return "\n\n".join(notes)


def check_pages(summary: PaperSummary, page_count: int) -> list[str]:
    """Problems Pydantic can't see: pages that don't exist in THIS paper, or no page at all."""
    problems = []
    for finding in summary.key_findings:
        short = finding.statement[:60]
        bad = [p for p in finding.pages if not 1 <= p <= page_count]
        if bad:
            problems.append(f"'{short}' cites page(s) {bad}, but the paper has {page_count} pages.")
        if not finding.pages:
            problems.append(f"'{short}' has no page reference.")
    return problems


def summarize_document(
    document: ExtractedDocument,
    llm: LLM,
    settings: Settings,
    status: StatusCallback | None = None,
) -> SummaryResult:
    """Produce a validated PaperSummary. Raises SummaryError, or LLMError if Ollama fails."""
    full_text = pages_to_text(document.pages)
    if estimate_tokens(full_text) <= settings.summary_direct_limit_tokens:
        strategy: Literal["direct", "map-reduce"] = "direct"
        source = full_text
    else:
        strategy = "map-reduce"
        source = make_notes(document.pages, llm, settings, status)

    messages: list[Message] = [
        {"role": "system", "content": SUMMARY_PROMPT},
        {"role": "user", "content": f"Paper text:\n\n{source}"},
    ]
    schema = PaperSummary.model_json_schema()

    for attempt in range(1, settings.summary_max_attempts + 1):
        if status:
            status(f"Writing the structured summary (attempt {attempt})...")
        raw = llm.chat(messages, format=schema, num_ctx=settings.summary_num_ctx)
        try:
            summary = PaperSummary.model_validate_json(raw)
        except ValidationError as exc:
            # Show the model its own answer and exactly what was wrong, then ask again.
            messages = [
                *messages,
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"That JSON was invalid:\n{exc}\nReturn corrected JSON only."},
            ]
            continue
        return SummaryResult(
            summary=summary,
            strategy=strategy,
            attempts=attempt,
            page_problems=check_pages(summary, document.page_count),
        )

    raise SummaryError(
        f"The model did not produce a valid summary after {settings.summary_max_attempts} "
        "attempts. Try again; results vary between runs."
    )


def _bullets(items: Sequence[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "_Not stated in the paper._"


def to_markdown(summary: PaperSummary, source_name: str) -> str:
    """Render a summary as a Markdown document for download."""
    findings = "\n".join(
        f"- {f.statement} (p. {', '.join(map(str, f.pages)) or '?'})" for f in summary.key_findings
    )
    return f"""# {summary.title}

*Summary generated by PaperPilot from `{source_name}`. Verify against the original paper.*

**In one sentence:** {summary.one_sentence_summary}

## Research problem
{summary.research_problem}

## Methodology
{summary.methodology}

## Key findings
{findings}

## Datasets
{_bullets(summary.datasets)}

## Limitations
{_bullets(summary.limitations)}

## Future work
{_bullets(summary.future_work)}
"""