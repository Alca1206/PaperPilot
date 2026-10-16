"""Tests for paperpilot.llm, using a fake chat client (no Ollama needed)."""

import httpx
import ollama
import pytest

from paperpilot.config import Settings
from paperpilot.llm import LLM, LLMError
from tests.fakes import FakeChatClient

MESSAGES = [{"role": "user", "content": "Hello"}]


def test_sends_model_messages_and_settings():
    client = FakeChatClient(replies=["Hi there"])
    llm = LLM(Settings(chat_temperature=0.2), client=client)

    reply = llm.chat(MESSAGES)

    call = client.calls[0]
    assert reply == "Hi there"
    assert call["model"] == "qwen3:8b"
    assert call["messages"] == MESSAGES
    assert call["think"] is False
    assert call["options"] == {"temperature": 0.2}
    assert call["format"] is None


def test_passes_format_and_context_window():
    client = FakeChatClient(replies=["{}"])
    schema = {"type": "object"}

    LLM(Settings(), client=client).chat(MESSAGES, format=schema, num_ctx=16384)

    assert client.calls[0]["format"] == schema
    assert client.calls[0]["options"]["num_ctx"] == 16384


def test_think_blocks_are_removed():
    client = FakeChatClient(replies=["<think>long reasoning...</think>\n\nThe answer [1]."])

    assert LLM(Settings(), client=client).chat(MESSAGES) == "The answer [1]."


def test_empty_reply_becomes_empty_string():
    assert LLM(Settings(), client=FakeChatClient(replies=[None])).chat(MESSAGES) == ""


@pytest.mark.parametrize(
    ("error", "expected_text"),
    [
        (ConnectionError("refused"), "Could not connect"),
        (httpx.ReadTimeout("slow"), "took too long"),
        (ollama.ResponseError("model not found", 404), "ollama pull qwen3:8b"),
        (ollama.ResponseError("out of memory", 500), "out of memory"),
    ],
)
def test_ollama_errors_become_llm_errors(error, expected_text):
    llm = LLM(Settings(), client=FakeChatClient(error=error))

    with pytest.raises(LLMError, match=expected_text):
        llm.chat(MESSAGES)


@pytest.mark.integration
def test_real_chat_model_answers():
    llm = LLM(Settings())
    try:
        reply = llm.chat([{"role": "user", "content": "Reply with exactly the word: pong"}])
    except LLMError as exc:
        pytest.skip(f"Chat model not available: {exc}")

    assert "pong" in reply.lower()