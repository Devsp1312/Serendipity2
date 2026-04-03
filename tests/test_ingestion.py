"""
Tests for src/ingestion.py

Covers: transcript parsing, speaker normalization, input validation,
multi-line turns, and turn statistics.
"""

import pytest

from src.ingestion import (
    Turn,
    parse_transcript,
    format_transcript_for_llm,
    get_turn_stats,
    validate_transcript_input,
)
from src.config import MAX_TRANSCRIPT_BYTES, MAX_SPEAKER_LABEL_LENGTH


# ─── validate_transcript_input ────────────────────────────────────────────────

def test_validate_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        validate_transcript_input("")


def test_validate_whitespace_only_raises():
    with pytest.raises(ValueError, match="empty"):
        validate_transcript_input("   \n\t  ")


def test_validate_oversized_raises():
    oversized = "A" * (MAX_TRANSCRIPT_BYTES + 1)
    with pytest.raises(ValueError, match="too large"):
        validate_transcript_input(oversized)


def test_validate_binary_content_raises():
    with pytest.raises(ValueError, match="binary"):
        validate_transcript_input("User: hello\x00world")


def test_validate_valid_input_passes():
    # Should not raise
    validate_transcript_input("User: hello there\nSpeaker B: hi!")


# ─── parse_transcript ─────────────────────────────────────────────────────────

def test_parse_basic_transcript():
    text = "User: Hello!\nSpeaker B: Hi there."
    turns = parse_transcript(text)
    assert len(turns) == 2
    assert turns[0].speaker == "User"
    assert turns[0].text    == "Hello!"
    assert turns[1].speaker == "Speaker B"
    assert turns[1].text    == "Hi there."


def test_speaker_case_normalization_lowercase():
    text = "alice: How are you?\nbob: Fine thanks."
    turns = parse_transcript(text)
    assert turns[0].speaker == "Alice"
    assert turns[1].speaker == "Bob"


def test_speaker_case_normalization_uppercase():
    text = "ALICE: How are you?\nBOB: Fine thanks."
    turns = parse_transcript(text)
    assert turns[0].speaker == "Alice"
    assert turns[1].speaker == "Bob"


def test_speaker_case_normalization_mixed():
    """Alice, alice, ALICE should all normalize to the same speaker."""
    text = "Alice: First.\nalice: Second.\nALICE: Third."
    turns = parse_transcript(text)
    speakers = {t.speaker for t in turns}
    assert speakers == {"Alice"}


def test_multiline_turns_joined():
    text = (
        "User: This is the first line\n"
        "and this continues the thought.\n"
        "Speaker B: Got it."
    )
    turns = parse_transcript(text)
    assert len(turns) == 2
    assert "first line" in turns[0].text
    assert "continues" in turns[0].text


def test_no_speaker_labels_falls_back_to_user_turn():
    turns = parse_transcript("Just some plain text without labels.")
    assert len(turns) == 1
    assert turns[0].speaker == "User"
    assert turns[0].text == "Just some plain text without labels."


def test_blank_lines_ignored():
    text = "\n\nUser: Hello.\n\nSpeaker B: Hi.\n\n"
    turns = parse_transcript(text)
    assert len(turns) == 2


def test_speaker_label_too_long_truncated(caplog):
    long_name = "A" * (MAX_SPEAKER_LABEL_LENGTH + 10)
    text = f"{long_name}: Hello there."
    import logging
    with caplog.at_level(logging.WARNING, logger="src.ingestion"):
        turns = parse_transcript(text)
    assert len(turns[0].speaker) == MAX_SPEAKER_LABEL_LENGTH
    assert "truncated" in caplog.text.lower() or "exceeds" in caplog.text.lower()


# ─── format_transcript_for_llm ────────────────────────────────────────────────

def test_format_transcript_for_llm():
    turns = [Turn("Alice", "Hello!"), Turn("Bob", "Hi!")]
    result = format_transcript_for_llm(turns)
    assert result == "Alice: Hello!\nBob: Hi!"


# ─── get_turn_stats ───────────────────────────────────────────────────────────

def test_get_turn_stats():
    turns = [
        Turn("Alice", "First"),
        Turn("Bob", "Second"),
        Turn("Alice", "Third"),
    ]
    stats = get_turn_stats(turns)
    assert stats["Alice"] == 2
    assert stats["Bob"]   == 1


def test_get_turn_stats_empty():
    assert get_turn_stats([]) == {}
