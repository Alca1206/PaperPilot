"""Shared test fixtures.

PDFs are generated in code, so the repo needs no binary test files and
every test shows exactly what its PDF contains.
"""

import pymupdf
import pytest


def build_pdf(page_texts, metadata=None, user_password=None):
    """Return the bytes of a PDF with one page per string ("" = blank page)."""
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=10)
    if metadata:
        doc.set_metadata(metadata)
    if user_password:
        data = doc.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            user_pw=user_password,
            owner_pw=user_password + "-owner",
        )
    else:
        data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def make_pdf():
    """Fixture form of build_pdf, so tests can request it by name."""
    return build_pdf