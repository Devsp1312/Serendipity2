"""
Tests for src/llm_client.py

Covers: call_llm retry logic, exponential backoff, prompt loading/caching,
connection error handling, and model discovery.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from src.llm_client import (
    call_llm,
    load_prompt,
    get_extraction_prompt,
    get_gatekeeper_prompt,
    list_models,
    check_connection,
    OllamaUnavailableError,
    _prompt_cache,
)
from src.schemas import LLMSchemaError


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_ollama_response(content: str):
    """Builds a mock ollama.chat() response object."""
    msg = MagicMock()
    msg.content = content
    resp = MagicMock()
    resp.message = msg
    return resp


# ─── call_llm — success ───────────────────────────────────────────────────────

def test_call_llm_success_first_attempt():
    valid_json = '{"core_values": ["discipline"]}'
    with patch("src.llm_client.ollama.chat", return_value=_make_ollama_response(valid_json)):
        result, raw = call_llm("sys", "user", "llama3", max_retries=0)
    assert result == {"core_values": ["discipline"]}
    assert raw == valid_json


# ─── call_llm — retry on bad JSON ────────────────────────────────────────────

def test_call_llm_retries_on_invalid_json():
    bad  = "not json at all"
    good = '{"key": "value"}'

    responses = [
        _make_ollama_response(bad),
        _make_ollama_response(good),
    ]
    with patch("src.llm_client.ollama.chat", side_effect=responses), \
         patch("src.llm_client.time.sleep"):  # Skip actual sleep in tests
        result, raw = call_llm("sys", "user", "llama3", max_retries=1)
    assert result == {"key": "value"}


def test_call_llm_raises_after_max_retries():
    bad = "not json"
    responses = [_make_ollama_response(bad)] * 4  # more than max_retries

    with patch("src.llm_client.ollama.chat", side_effect=responses), \
         patch("src.llm_client.time.sleep"):
        with pytest.raises(LLMSchemaError, match="invalid JSON"):
            call_llm("sys", "user", "llama3", max_retries=2)


# ─── call_llm — backoff ───────────────────────────────────────────────────────

def test_call_llm_backoff_sleeps_between_retries():
    bad  = "not json"
    good = '{"ok": true}'

    responses = [
        _make_ollama_response(bad),
        _make_ollama_response(good),
    ]
    with patch("src.llm_client.ollama.chat", side_effect=responses), \
         patch("src.llm_client.time.sleep") as mock_sleep, \
         patch("src.llm_client.random.uniform", return_value=0.0):
        call_llm("sys", "user", "llama3", max_retries=1)

    # sleep should have been called once (after the first failed attempt)
    assert mock_sleep.call_count == 1
    sleep_duration = mock_sleep.call_args[0][0]
    # Base delay is 1.0 + jitter (0.0 in this test)
    assert sleep_duration == pytest.approx(1.0, abs=0.01)


# ─── call_llm — connection errors ────────────────────────────────────────────

def test_call_llm_raises_on_connection_error():
    with patch("src.llm_client.ollama.chat", side_effect=ConnectionError("refused")):
        with pytest.raises(OllamaUnavailableError, match="ollama serve"):
            call_llm("sys", "user", "llama3")


def test_call_llm_raises_on_os_error():
    with patch("src.llm_client.ollama.chat", side_effect=OSError("no socket")):
        with pytest.raises(OllamaUnavailableError):
            call_llm("sys", "user", "llama3")


def test_call_llm_raises_on_ollama_response_error():
    import ollama
    with patch("src.llm_client.ollama.chat", side_effect=ollama.ResponseError("model not found")):
        with pytest.raises(OllamaUnavailableError, match="ollama pull"):
            call_llm("sys", "user", "llama3")


# ─── load_prompt ─────────────────────────────────────────────────────────────

def test_load_prompt_loads_file(tmp_path, monkeypatch):
    import src.llm_client as lc
    # Clear cache to avoid stale entries from other tests
    lc._prompt_cache.clear()

    monkeypatch.setattr(lc, "PROMPTS_DIR", tmp_path)
    (tmp_path / "test_prompt.txt").write_text(
        "# version: 1.0\nYou are a helpful assistant.",
        encoding="utf-8",
    )

    prompt = load_prompt("test_prompt")
    assert prompt == "You are a helpful assistant."
    assert "# version" not in prompt


def test_load_prompt_strips_comment_lines(tmp_path, monkeypatch):
    import src.llm_client as lc
    lc._prompt_cache.clear()

    monkeypatch.setattr(lc, "PROMPTS_DIR", tmp_path)
    (tmp_path / "p.txt").write_text(
        "# comment 1\n# comment 2\nActual prompt content.\nMore content.",
        encoding="utf-8",
    )

    prompt = load_prompt("p")
    assert "comment" not in prompt
    assert "Actual prompt content." in prompt


def test_load_prompt_caches_result(tmp_path, monkeypatch):
    import src.llm_client as lc
    lc._prompt_cache.clear()

    monkeypatch.setattr(lc, "PROMPTS_DIR", tmp_path)
    prompt_file = tmp_path / "cached.txt"
    prompt_file.write_text("Prompt text.", encoding="utf-8")

    with patch.object(Path, "read_text", wraps=prompt_file.read_text) as mock_read:
        load_prompt("cached")
        load_prompt("cached")  # second call should use cache

    # read_text should only be called once
    assert mock_read.call_count == 1


def test_load_prompt_raises_for_missing_file(tmp_path, monkeypatch):
    import src.llm_client as lc
    lc._prompt_cache.clear()

    monkeypatch.setattr(lc, "PROMPTS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="prompts/nonexistent.txt"):
        load_prompt("nonexistent")


# ─── list_models / check_connection ──────────────────────────────────────────

def test_list_models_returns_empty_on_failure():
    with patch("src.llm_client.ollama.list", side_effect=Exception("unreachable")):
        result = list_models()
    assert result == []


def test_check_connection_true_when_reachable():
    with patch("src.llm_client.ollama.list", return_value=MagicMock()):
        assert check_connection() is True


def test_check_connection_false_when_unreachable():
    with patch("src.llm_client.ollama.list", side_effect=Exception("unreachable")):
        assert check_connection() is False
