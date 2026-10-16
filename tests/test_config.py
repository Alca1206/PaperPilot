"""Tests for paperpilot.config."""

from dataclasses import FrozenInstanceError

import pytest

from paperpilot.config import PROJECT_ROOT, Settings, load_settings


def test_defaults_when_no_env_vars():
    settings = load_settings(env={})

    assert settings.ollama_host == "http://localhost:11434"
    assert settings.chat_model == "qwen3:8b"
    assert settings.embedding_model == "embeddinggemma"
    assert settings.request_timeout_seconds == 10.0
    assert settings.data_dir == PROJECT_ROOT / "data"


def test_required_models_are_chat_and_embedding():
    settings = load_settings(env={})

    assert settings.required_models == ("qwen3:8b", "embeddinggemma")


def test_env_vars_override_defaults(tmp_path):
    env = {
        "PAPERPILOT_OLLAMA_HOST": "http://192.168.1.50:11434",
        "PAPERPILOT_CHAT_MODEL": "llama3.2:3b",
        "PAPERPILOT_EMBEDDING_MODEL": "nomic-embed-text",
        "PAPERPILOT_REQUEST_TIMEOUT_SECONDS": "2.5",
        "PAPERPILOT_DATA_DIR": str(tmp_path),
    }

    settings = load_settings(env=env)

    assert settings.ollama_host == "http://192.168.1.50:11434"
    assert settings.chat_model == "llama3.2:3b"
    assert settings.embedding_model == "nomic-embed-text"
    assert settings.request_timeout_seconds == 2.5
    assert settings.data_dir == tmp_path


def test_trailing_slash_is_removed_from_host():
    settings = load_settings(env={"PAPERPILOT_OLLAMA_HOST": "http://localhost:11434/"})

    assert settings.ollama_host == "http://localhost:11434"


def test_blank_env_values_fall_back_to_defaults():
    settings = load_settings(env={"PAPERPILOT_CHAT_MODEL": "   "})

    assert settings.chat_model == "qwen3:8b"


def test_relative_data_dir_is_resolved_against_project_root():
    settings = load_settings(env={"PAPERPILOT_DATA_DIR": "my_data"})

    assert settings.data_dir == (PROJECT_ROOT / "my_data").resolve()


def test_storage_paths_live_under_data_dir(tmp_path):
    settings = Settings(data_dir=tmp_path)

    assert settings.uploads_dir == tmp_path / "uploads"
    assert settings.chroma_dir == tmp_path / "chroma"
    assert settings.exports_dir == tmp_path / "exports"


def test_ensure_directories_creates_folders_and_is_repeatable(tmp_path):
    settings = Settings(data_dir=tmp_path / "data")

    settings.ensure_directories()
    settings.ensure_directories()  # second call must not fail

    assert settings.uploads_dir.is_dir()
    assert settings.chroma_dir.is_dir()
    assert settings.exports_dir.is_dir()


@pytest.mark.parametrize("bad_value", ["abc", "ten seconds"])
def test_non_numeric_timeout_raises_clear_error(bad_value):
    with pytest.raises(ValueError, match="REQUEST_TIMEOUT_SECONDS"):
        load_settings(env={"PAPERPILOT_REQUEST_TIMEOUT_SECONDS": bad_value})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ollama_host": "localhost:11434"},  # missing http://
        {"chat_model": ""},
        {"embedding_model": "   "},
        {"request_timeout_seconds": 0},
        {"request_timeout_seconds": -1},
    ],
)
def test_invalid_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)


def test_settings_are_immutable():
    settings = Settings()

    with pytest.raises(FrozenInstanceError):
        settings.chat_model = "something-else"

# ---------- Phase 2: chunk settings ----------


def test_chunk_setting_defaults():
    settings = load_settings(env={})

    assert settings.chunk_max_chars == 800
    assert settings.chunk_overlap_sentences == 1


def test_chunk_settings_from_env():
    settings = load_settings(
        env={"PAPERPILOT_CHUNK_MAX_CHARS": "500", "PAPERPILOT_CHUNK_OVERLAP_SENTENCES": "0"}
    )

    assert settings.chunk_max_chars == 500
    assert settings.chunk_overlap_sentences == 0


@pytest.mark.parametrize("bad_value", ["abc", "8.5"])
def test_non_integer_chunk_size_raises_clear_error(bad_value):
    with pytest.raises(ValueError, match="CHUNK_MAX_CHARS must be a whole number"):
        load_settings(env={"PAPERPILOT_CHUNK_MAX_CHARS": bad_value})


@pytest.mark.parametrize(
    "kwargs", [{"chunk_max_chars": 10}, {"chunk_overlap_sentences": -1}]
)
def test_invalid_chunk_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)

# ---------- Phase 3: embedding settings ----------


def test_embedding_setting_defaults():
    settings = load_settings(env={})

    assert settings.model_timeout_seconds == 300.0
    assert settings.embed_batch_size == 32
    assert settings.embed_use_prefixes is True


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("true", True), ("1", True), ("YES", True), ("false", False), ("0", False), ("off", False)],
)
def test_prefix_flag_accepts_common_spellings(raw, expected):
    settings = load_settings(env={"PAPERPILOT_EMBED_USE_PREFIXES": raw})

    assert settings.embed_use_prefixes is expected


def test_invalid_bool_raises_clear_error():
    with pytest.raises(ValueError, match="EMBED_USE_PREFIXES must be true or false"):
        load_settings(env={"PAPERPILOT_EMBED_USE_PREFIXES": "maybe"})


@pytest.mark.parametrize(
    "kwargs", [{"model_timeout_seconds": 0}, {"embed_batch_size": 0}]
)
def test_invalid_embedding_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)

# ---------- Phase 4: Q&A settings ----------


def test_qa_setting_defaults():
    settings = load_settings(env={})

    assert settings.qa_top_k == 4
    assert settings.qa_min_score == 0.35
    assert settings.chat_temperature == 0.1


def test_qa_settings_from_env():
    settings = load_settings(
        env={
            "PAPERPILOT_QA_TOP_K": "6",
            "PAPERPILOT_QA_MIN_SCORE": "0.5",
            "PAPERPILOT_CHAT_TEMPERATURE": "0",
        }
    )

    assert (settings.qa_top_k, settings.qa_min_score, settings.chat_temperature) == (6, 0.5, 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"qa_top_k": 0},
        {"qa_min_score": 1.5},
        {"qa_min_score": -2},
        {"chat_temperature": -0.1},
        {"chat_temperature": 3},
    ],
)
def test_invalid_qa_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)