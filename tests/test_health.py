"""Tests for paperpilot.health.

Unit tests use a fake client, so they run without Ollama. The single
integration test at the bottom talks to a real Ollama server and is skipped
when none is running.
"""

import httpx
import ollama
import pytest

from paperpilot.config import Settings
from paperpilot.health import (
    check_ollama_health,
    find_missing_models,
    normalize_model_name,
)


class FakeClient:
    """Stands in for ollama.Client: returns a canned response or raises an error."""

    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error

    def list(self):
        if self._error is not None:
            raise self._error
        return self._response


def make_list_response(*names):
    """Build a response with the same type the real ollama client returns."""
    return ollama.ListResponse(models=[ollama.ListResponse.Model(model=n) for n in names])


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path)


# ---------- normalize_model_name ----------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("embeddinggemma", "embeddinggemma:latest"),
        ("embeddinggemma:latest", "embeddinggemma:latest"),
        ("qwen3:8b", "qwen3:8b"),
        ("  Qwen3:8B  ", "qwen3:8b"),
        ("", ""),
    ],
)
def test_normalize_model_name(raw, expected):
    assert normalize_model_name(raw) == expected


# ---------- find_missing_models ----------

def test_nothing_missing_when_all_installed():
    missing = find_missing_models(
        ["qwen3:8b", "embeddinggemma"], ["qwen3:8b", "embeddinggemma:latest"]
    )
    assert missing == ()


def test_reports_missing_models_in_original_spelling():
    missing = find_missing_models(["qwen3:8b", "embeddinggemma"], ["llama3.2:3b"])
    assert missing == ("qwen3:8b", "embeddinggemma")


def test_different_tag_does_not_count_as_installed():
    # qwen3:4b is a different model from qwen3:8b
    assert find_missing_models(["qwen3:8b"], ["qwen3:4b"]) == ("qwen3:8b",)


# ---------- check_ollama_health ----------

def test_healthy_when_reachable_and_all_models_installed(settings):
    client = FakeClient(make_list_response("qwen3:8b", "embeddinggemma:latest"))

    report = check_ollama_health(settings, client=client)

    assert report.ok
    assert report.ollama_reachable
    assert report.missing_models == ()
    assert report.error is None
    assert report.installed_models == ("qwen3:8b", "embeddinggemma:latest")


def test_reports_missing_models_and_pull_commands(settings):
    client = FakeClient(make_list_response("qwen3:8b"))

    report = check_ollama_health(settings, client=client)

    assert report.ollama_reachable
    assert not report.ok
    assert report.missing_models == ("embeddinggemma",)
    assert report.pull_commands == ["ollama pull embeddinggemma"]


def test_reachable_with_no_models_installed(settings):
    report = check_ollama_health(settings, client=FakeClient(make_list_response()))

    assert report.ollama_reachable
    assert report.missing_models == ("qwen3:8b", "embeddinggemma")


def test_handles_older_dict_style_response(settings):
    response = {"models": [{"name": "qwen3:8b"}, {"name": "embeddinggemma:latest"}]}

    report = check_ollama_health(settings, client=FakeClient(response))

    assert report.ok


@pytest.mark.parametrize(
    ("error", "expected_text"),
    [
        (ConnectionError("refused"), "Could not connect"),
        (httpx.ConnectTimeout("timed out"), "did not respond"),
        (ollama.ResponseError("server exploded", 500), "server exploded"),
        (RuntimeError("weird"), "Unexpected error"),
    ],
)
def test_errors_become_unreachable_report_instead_of_crashing(settings, error, expected_text):
    report = check_ollama_health(settings, client=FakeClient(error=error))

    assert not report.ok
    assert not report.ollama_reachable
    assert expected_text in report.error
    assert report.required_models == ("qwen3:8b", "embeddinggemma")


# ---------- integration (real Ollama) ----------

@pytest.mark.integration
def test_real_ollama_health_check(settings):
    report = check_ollama_health(settings)
    if not report.ollama_reachable:
        pytest.skip(f"Ollama not reachable: {report.error}")

    # We can't know which models the developer has, only that the call worked.
    assert isinstance(report.installed_models, tuple)


# ---------- is_model_available (Phase 3) ----------


def test_is_model_available(settings):
    report = check_ollama_health(settings, client=FakeClient(make_list_response("embeddinggemma:latest")))

    assert report.is_model_available("embeddinggemma")
    assert not report.is_model_available("qwen3:8b")


def test_no_model_is_available_when_ollama_is_down(settings):
    report = check_ollama_health(settings, client=FakeClient(error=ConnectionError("down")))

    assert not report.is_model_available("embeddinggemma")