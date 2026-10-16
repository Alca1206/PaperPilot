"""Bibliographic metadata and reference formatting (concept notebook, Part 8).

Rule: every field comes from the PDF or from the user, never from the model's memory.
- Regex finds DOI / arXiv ID (deterministic, exact)
- The LLM extracts title/authors/year/venue from PAGE 1 ONLY
- Every field is checked against the page-1 text; mismatches become warnings
- The user confirms or edits the metadata (in the UI)
- APA / MLA / IEEE / BibTeX are produced by plain Python, never by the LLM
"""

from __future__ import annotations

import re
from collections.abc import Callable

from pydantic import ValidationError

from paperpilot.llm import LLM, Message
from paperpilot.schemas import ExtractedDocument, PaperMetadata

PAGE1_MAX_CHARS = 4000
MAX_ATTEMPTS = 3
NUM_CTX = 8192

DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"<>]+", re.IGNORECASE)
ARXIV_RE = re.compile(r"arXiv:\s?(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)

METADATA_PROMPT = """Extract bibliographic metadata for this research paper from the text of its FIRST PAGE.
Rules:
- Use ONLY what is written in the text. Do NOT use your own knowledge of the paper, even if you recognize it.
- If a field is not written in the text, use null. Never guess volume, issue, pages or DOI.
- Split each author into given names and family name. Ignore emails, affiliations and symbols like * † ‡.
- year = the publication year, NOT an arXiv submission or revision date.
- kind = "conference" for proceedings, "journal" for journals, "preprint" if only arXiv/preprint."""


class MetadataError(Exception):
    """Metadata could not be extracted. The message is safe to show to the user."""


# ---------- extraction ----------


def find_identifiers(text: str) -> dict[str, str | None]:
    """DOI and arXiv ID by regex: if they are on the page, they are found exactly."""
    doi = DOI_RE.search(text)
    arxiv = ARXIV_RE.search(text)
    return {
        "doi": doi.group(0).rstrip(".,;)") if doi else None,
        "arxiv_id": arxiv.group(1) if arxiv else None,
    }


def first_page_text(document: ExtractedDocument) -> str:
    return document.pages[0].text if document.pages else ""


def extract_metadata(document: ExtractedDocument, llm: LLM) -> PaperMetadata:
    """Ask the model for metadata from page 1 only; regex identifiers override the model."""
    page1 = first_page_text(document)
    messages: list[Message] = [
        {"role": "system", "content": METADATA_PROMPT},
        {
            "role": "user",
            "content": f"PDF metadata: {document.metadata}\n\nFirst page:\n{page1[:PAGE1_MAX_CHARS]}",
        },
    ]
    schema = PaperMetadata.model_json_schema()

    for _ in range(MAX_ATTEMPTS):
        raw = llm.chat(messages, format=schema, num_ctx=NUM_CTX)
        try:
            metadata = PaperMetadata.model_validate_json(raw)
        except ValidationError as exc:
            messages = [
                *messages,
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"That JSON was invalid:\n{exc}\nReturn corrected JSON only."},
            ]
            continue
        # Regex results beat the model: they are exact copies from the page.
        found = {key: value for key, value in find_identifiers(page1).items() if value}
        return metadata.model_copy(update=found)

    raise MetadataError(
        "The model could not extract metadata. Fill in the form manually instead."
    )


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def verify_against_page(metadata: PaperMetadata, page_text: str) -> list[str]:
    """Warn about every field that does not appear on page 1 (possible model memory or typo).

    Limits: short values like volume "3" can match by accident, and a found year is
    not proof it is the PUBLICATION year. That is why the user confirms.
    """
    page = _normalize(page_text)
    digits_only = re.sub(r"\D", "", page_text)
    warnings = []
    if metadata.title and _normalize(metadata.title) not in page:
        warnings.append(f"Title not found on page 1: '{metadata.title}'")
    for author in metadata.authors:
        if author.family.lower() not in page:
            warnings.append(f"Author not found on page 1: '{author.given} {author.family}'")
    if metadata.year and str(metadata.year) not in page_text:
        warnings.append(f"Year {metadata.year} not found on page 1")
    for field in ("venue", "volume", "issue", "doi"):
        value = getattr(metadata, field)
        if value and _normalize(value) not in page:
            warnings.append(f"{field.capitalize()} '{value}' not found on page 1")
    if metadata.pages and re.sub(r"\D", "", metadata.pages) not in digits_only:
        warnings.append(f"Pages '{metadata.pages}' not found on page 1")
    return warnings


def missing_fields(metadata: PaperMetadata) -> list[str]:
    checks = {
        "authors": metadata.authors,
        "year": metadata.year,
        "venue or arXiv ID": metadata.venue or metadata.arxiv_id,
        "DOI or arXiv ID": metadata.doi or metadata.arxiv_id,
    }
    return [name for name, value in checks.items() if not value]


# ---------- formatting (plain Python, no AI) ----------


def initials(given: str) -> str:
    """'Aidan N.' -> 'A. N.'   'Jean-Pierre' -> 'J.-P.'"""
    return " ".join(
        "-".join(part[0].upper() + "." for part in name.split("-") if part) for name in given.split()
    )


def doi_url(metadata: PaperMetadata) -> str | None:
    if metadata.doi:
        return f"https://doi.org/{metadata.doi}"
    if metadata.arxiv_id:
        return f"https://doi.org/10.48550/arXiv.{metadata.arxiv_id}"  # arXiv's official DOI form
    return None


def page_range(pages: str | None, dash: str = "–") -> str | None:
    return re.sub(r"\s*[-–—]+\s*", dash, pages) if pages else None


def format_apa(m: PaperMetadata) -> str:
    """APA 7th edition.

    Built from three parts:  WHO (WHEN). WHAT. WHERE. DOI
    With no authors, APA moves WHAT to the front and does not repeat it.
    """
    names = [f"{a.family}, {initials(a.given)}" for a in m.authors]
    if len(names) > 20:  # APA 7: first 19, ellipsis, last
        names = [*names[:19], "... " + names[-1]]
    when = f"({m.year or 'n.d.'})"

    # WHAT: standalone works (preprints, unknown venue) have an italic title
    standalone = m.kind == "preprint" or not m.venue
    what = f"*{m.title}*" if standalone else m.title
    if standalone and m.arxiv_id:
        what += " [Preprint]"

    # WHERE
    if standalone:
        where = "arXiv." if m.arxiv_id else ""
    elif m.kind == "conference":
        where = f"In *{m.venue}*" + (f" (pp. {page_range(m.pages)})" if m.pages else "") + "."
    else:
        where = f"*{m.venue}*"
        if m.volume:
            where += f", *{m.volume}*" + (f"({m.issue})" if m.issue else "")
        if m.pages:
            where += f", {page_range(m.pages)}"
        where += "."

    if names:
        who = names[0] if len(names) == 1 else ", ".join(names[:-1]) + ", & " + names[-1]
        parts = [f"{who} {when}.", f"{what}."]
    else:
        parts = [f"{what}. {when}."]
    if where:
        parts.append(where)
    url = doi_url(m)
    if url:
        parts.append(url)
    return " ".join(parts)


def format_mla(m: PaperMetadata) -> str:
    """MLA 9th edition."""
    a = m.authors
    if not a:
        authors = ""
    elif len(a) == 1:
        authors = f"{a[0].family}, {a[0].given}"
    elif len(a) == 2:
        authors = f"{a[0].family}, {a[0].given}, and {a[1].given} {a[1].family}"
    else:
        authors = f"{a[0].family}, {a[0].given}, et al"

    out = (f"{authors.rstrip('.')}. " if authors else "") + f"\u201c{m.title}.\u201d"
    container = m.venue or ("arXiv" if m.arxiv_id else None)
    details = [f"*{container}*"] if container else []
    if m.volume:
        details.append(f"vol. {m.volume}")
    if m.issue:
        details.append(f"no. {m.issue}")
    if m.year:
        details.append(str(m.year))
    if m.pages:
        details.append(f"pp. {page_range(m.pages)}")
    url = doi_url(m)
    if url:
        details.append(url)
    return out + (" " + ", ".join(details) + "." if details else "")


def format_ieee(m: PaperMetadata) -> str:
    """IEEE reference style."""
    names = [f"{initials(a.given)} {a.family}" for a in m.authors]
    if len(names) > 6:
        authors = f"{names[0]} et al."
    elif len(names) <= 1:
        authors = names[0] if names else ""
    elif len(names) == 2:
        authors = f"{names[0]} and {names[1]}"
    else:
        authors = ", ".join(names[:-1]) + ", and " + names[-1]

    details = []
    if m.kind == "conference" and m.venue:
        details.append(f"in *{m.venue}*")
    elif m.venue:
        details.append(f"*{m.venue}*")
    elif m.arxiv_id:
        details.append(f"arXiv:{m.arxiv_id}")
    if m.volume:
        details.append(f"vol. {m.volume}")
    if m.issue:
        details.append(f"no. {m.issue}")
    if m.pages:
        details.append(f"pp. {page_range(m.pages)}")
    if m.year:
        details.append(str(m.year))

    out = (f"{authors}, " if authors else "") + f"\u201c{m.title},\u201d " + ", ".join(details)
    out = out.rstrip(", ") + "."
    return out + (f" doi: {m.doi}." if m.doi else "")


_STOPWORDS = {"a", "an", "the", "on", "of", "for", "in", "and", "to", "is"}


def bibtex_key(m: PaperMetadata) -> str:
    """e.g. vaswani2017attention"""
    family = re.sub(r"[^a-z]", "", m.authors[0].family.lower()) if m.authors else "anon"
    words = [w for w in re.findall(r"[a-z]+", m.title.lower()) if w not in _STOPWORDS]
    return f"{family or 'anon'}{m.year or ''}{words[0] if words else ''}"


def format_bibtex(m: PaperMetadata) -> str:
    entry_type = {"journal": "article", "conference": "inproceedings", "preprint": "misc"}[m.kind]
    venue_field = {"journal": "journal", "conference": "booktitle", "preprint": "howpublished"}[m.kind]
    fields = {
        "title": "{" + m.title + "}",  # double braces keep capitalisation in LaTeX
        "author": " and ".join(f"{a.family}, {a.given}" for a in m.authors),
    }
    if m.year:
        fields["year"] = str(m.year)
    if m.venue:
        fields[venue_field] = m.venue
    if m.volume:
        fields["volume"] = m.volume
    if m.issue:
        fields["number"] = m.issue
    if m.pages:
        fields["pages"] = page_range(m.pages, "--")
    if m.doi:
        fields["doi"] = m.doi
    if m.arxiv_id:
        fields["eprint"] = m.arxiv_id
        fields["archivePrefix"] = "arXiv"
    body = ",\n".join(f"  {key} = {{{value}}}" for key, value in fields.items() if value)
    return f"@{entry_type}{{{bibtex_key(m)},\n{body}\n}}"


FORMATTERS: dict[str, Callable[[PaperMetadata], str]] = {
    "APA 7": format_apa,
    "MLA 9": format_mla,
    "IEEE": format_ieee,
    "BibTeX": format_bibtex,
}


def references_markdown(m: PaperMetadata) -> str:
    """All styles in one Markdown document for download."""
    parts = [f"# References: {m.title}\n"]
    for style, formatter in FORMATTERS.items():
        if style == "BibTeX":
            parts.append(f"## {style}\n\n```bibtex\n{formatter(m)}\n```\n")
        else:
            parts.append(f"## {style}\n\n{formatter(m)}\n")
    return "\n".join(parts)