"""Tests for paperpilot.bibliography. Expected outputs were checked by hand against the style rules."""

import json

import pytest

from paperpilot.bibliography import (
    MetadataError,
    bibtex_key,
    extract_metadata,
    find_identifiers,
    format_apa,
    format_bibtex,
    format_ieee,
    format_mla,
    initials,
    missing_fields,
    references_markdown,
    verify_against_page,
)
from paperpilot.config import Settings
from paperpilot.llm import LLM
from paperpilot.schemas import Author, ExtractedDocument, PageText, PaperMetadata
from tests.fakes import FakeChatClient

TRANSFORMER_AUTHORS = [
    Author(given=g, family=f)
    for g, f in [
        ("Ashish", "Vaswani"), ("Noam", "Shazeer"), ("Niki", "Parmar"), ("Jakob", "Uszkoreit"),
        ("Llion", "Jones"), ("Aidan N.", "Gomez"), ("Łukasz", "Kaiser"), ("Illia", "Polosukhin"),
    ]
]
CONFERENCE = PaperMetadata(
    title="Attention Is All You Need",
    authors=TRANSFORMER_AUTHORS,
    year=2017,
    kind="conference",
    venue="31st Conference on Neural Information Processing Systems (NIPS 2017)",
    arxiv_id="1706.03762",
)
PREPRINT = PaperMetadata(
    title="Some Preprint",
    kind="preprint",
    authors=[Author(given="Jane", family="Doe"), Author(given="Max", family="Mustermann")],
    arxiv_id="2101.00001",
)
JOURNAL = PaperMetadata(
    title="Deep Things",
    authors=[Author(given="Jane Q.", family="Doe")],
    year=2020,
    venue="Journal of Stuff",
    volume="12",
    issue="3",
    pages="100 - 120",
    doi="10.1000/xyz123",
)

PAGE1 = (
    "Attention Is All You Need Ashish Vaswani∗ Google Brain avaswani@google.com Noam Shazeer∗ "
    "Niki Parmar Jakob Uszkoreit Llion Jones Aidan N. Gomez Łukasz Kaiser Illia Polosukhin "
    "Abstract The dominant sequence transduction models ... 31st Conference on Neural Information "
    "Processing Systems (NIPS 2017), Long Beach, CA, USA. arXiv:1706.03762v7 [cs.CL] 2 Aug 2023"
)


# ---------- identifiers ----------


def test_find_identifiers():
    found = find_identifiers("see arXiv:1706.03762v7 [cs.CL] and doi 10.1145/3292500.3330701.")

    assert found == {"doi": "10.1145/3292500.3330701", "arxiv_id": "1706.03762"}


def test_find_identifiers_none():
    assert find_identifiers("nothing here") == {"doi": None, "arxiv_id": None}


# ---------- formatting ----------


@pytest.mark.parametrize(
    ("given", "expected"),
    [("Aidan N.", "A. N."), ("Jean-Pierre", "J.-P."), ("Łukasz", "Ł.")],
)
def test_initials(given, expected):
    assert initials(given) == expected


def test_apa_conference_with_eight_authors():
    assert format_apa(CONFERENCE) == (
        "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., "
        "Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. In *31st Conference on "
        "Neural Information Processing Systems (NIPS 2017)*. https://doi.org/10.48550/arXiv.1706.03762"
    )


def test_apa_preprint_without_year_uses_nd():
    assert format_apa(PREPRINT) == (
        "Doe, J., & Mustermann, M. (n.d.). *Some Preprint* [Preprint]. arXiv. "
        "https://doi.org/10.48550/arXiv.2101.00001"
    )


def test_apa_journal():
    assert format_apa(JOURNAL) == (
        "Doe, J. Q. (2020). Deep Things. *Journal of Stuff*, *12*(3), 100–120. "
        "https://doi.org/10.1000/xyz123"
    )


def test_mla():
    assert format_mla(CONFERENCE).startswith("Vaswani, Ashish, et al. \u201cAttention Is All You Need.\u201d")
    assert format_mla(PREPRINT) == (
        "Doe, Jane, and Max Mustermann. \u201cSome Preprint.\u201d *arXiv*, "
        "https://doi.org/10.48550/arXiv.2101.00001."
    )
    # "Jane Q." must not produce a double full stop
    assert format_mla(JOURNAL).startswith("Doe, Jane Q. \u201cDeep Things.\u201d")


def test_ieee():
    assert format_ieee(CONFERENCE) == (
        "A. Vaswani et al., \u201cAttention Is All You Need,\u201d in *31st Conference on Neural "
        "Information Processing Systems (NIPS 2017)*, 2017."
    )
    assert format_ieee(PREPRINT) == "J. Doe and M. Mustermann, \u201cSome Preprint,\u201d arXiv:2101.00001."
    assert format_ieee(JOURNAL) == (
        "J. Q. Doe, \u201cDeep Things,\u201d *Journal of Stuff*, vol. 12, no. 3, pp. 100–120, 2020. "
        "doi: 10.1000/xyz123."
    )


def test_bibtex_entry_types_and_fields():
    conference = format_bibtex(CONFERENCE)
    assert conference.startswith("@inproceedings{vaswani2017attention,")
    assert "booktitle = {31st Conference" in conference
    assert "Kaiser, Łukasz and Polosukhin, Illia" in conference
    assert "archivePrefix = {arXiv}" in conference

    assert format_bibtex(PREPRINT).startswith("@misc{doesome,")
    journal = format_bibtex(JOURNAL)
    assert journal.startswith("@article{doe2020deep,")
    assert "pages = {100--120}" in journal
    assert "title = {{Deep Things}}" in journal


def test_missing_values_are_left_out_never_invented():
    minimal = PaperMetadata(title="Only A Title", authors=[])

    assert format_apa(minimal) == "*Only A Title*. (n.d.)."  # APA: no author -> title first, once
    assert "vol." not in format_ieee(minimal)
    assert bibtex_key(minimal) == "anononly"
    assert missing_fields(minimal) == ["authors", "year", "venue or arXiv ID", "DOI or arXiv ID"]


def test_references_markdown_contains_every_style():
    markdown = references_markdown(JOURNAL)

    for heading in ["## APA 7", "## MLA 9", "## IEEE", "## BibTeX", "```bibtex"]:
        assert heading in markdown


# ---------- verification against page 1 ----------


def test_verify_accepts_fields_that_are_on_the_page():
    assert verify_against_page(CONFERENCE, PAGE1) == []


def test_verify_flags_fields_from_model_memory():
    remembered = CONFERENCE.model_copy(
        update={
            "authors": [*TRANSFORMER_AUTHORS, Author(given="Invented", family="Person")],
            "venue": "Advances in Neural Information Processing Systems",
            "volume": "30",
            "pages": "5998-6008",
        }
    )

    warnings = verify_against_page(remembered, PAGE1)

    assert len(warnings) == 4
    assert any("Invented Person" in w for w in warnings)
    assert any("Venue" in w for w in warnings)
    assert any("Volume '30'" in w for w in warnings)
    assert any("Pages '5998-6008'" in w for w in warnings)


# ---------- extraction with a fake LLM ----------

DOCUMENT = ExtractedDocument(
    source_name="attention.pdf",
    file_hash="ab" * 32,
    page_count=2,
    pages=[PageText(page=1, text=PAGE1), PageText(page=2, text="Body text that must not be sent.")],
    metadata={"title": "Attention Is All You Need"},
)


def model_reply(**overrides):
    data = {"title": "Attention Is All You Need", "authors": [{"given": "Ashish", "family": "Vaswani"}],
            "year": 2017, "kind": "conference", "venue": None, "volume": None, "issue": None,
            "pages": None, "doi": None, "arxiv_id": "9999.99999"}
    data.update(overrides)
    return json.dumps(data)


def test_extraction_sends_only_page_one_and_regex_overrides_model():
    client = FakeChatClient(replies=[model_reply()])

    metadata = extract_metadata(DOCUMENT, LLM(Settings(), client=client))

    prompt = client.calls[0]["messages"][1]["content"]
    assert "Attention Is All You Need" in prompt
    assert "must not be sent" not in prompt
    assert client.calls[0]["format"] == PaperMetadata.model_json_schema()
    assert metadata.arxiv_id == "1706.03762"  # from the page, not the model's 9999.99999
    assert metadata.year == 2017


def test_extraction_retries_then_gives_up():
    ok_after_retry = FakeChatClient(replies=['{"title": 5}', model_reply()])
    assert extract_metadata(DOCUMENT, LLM(Settings(), client=ok_after_retry)).title == "Attention Is All You Need"

    always_bad = FakeChatClient(replies=["nope", "nope", "nope"])
    with pytest.raises(MetadataError, match="manually"):
        extract_metadata(DOCUMENT, LLM(Settings(), client=always_bad))