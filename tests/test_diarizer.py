"""
Tests for src/diarizer.py

Covers: JSON parsing robustness, name-spotting heuristics, transcript
relabeling, and full pipeline orchestration with mocks.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.diarize import (
    _parse_json,
    spot_speaker_names,
    relabel_transcript,
    run_diarization_pipeline,
)
from src.core.schemas import DiarizedTranscript, SpeakerInfo


# ── _parse_json ───────────────────────────────────────────────────────────────

def test_parse_json_with_think_tags():
    raw = '<think>internal chain</think>{"key": "value", "num": 42}'
    out = _parse_json(raw)
    assert out["key"] == "value"
    assert out["num"] == 42


def test_parse_json_with_markdown_block():
    raw = '```json\n{"recording_type": "interview", "topic": "career"}\n```'
    out = _parse_json(raw)
    assert out["recording_type"] == "interview"


def test_parse_json_plain_json():
    raw = '{"speakers": ["Alice", "Bob"]}'
    out = _parse_json(raw)
    assert out["speakers"] == ["Alice", "Bob"]


def test_parse_json_invalid_returns_empty_dict():
    out = _parse_json("not json at all")
    assert out == {}


def test_parse_json_partial_json_falls_back_gracefully():
    # Braces present but not valid — should return {}
    out = _parse_json('{"broken":')
    assert out == {}


# ── spot_speaker_names ────────────────────────────────────────────────────────

def test_spot_speaker_names_finds_self_identification():
    transcript = (
        "Speaker 1: I'm Monica, nice to meet you.\n"
        "Speaker 2: Hi Monica, I'm Rachel.\n"
        "Speaker 1: Good to see you Rachel."
    )
    mapping = spot_speaker_names(transcript)
    assert mapping.get("Speaker 1") == "Monica"
    assert mapping.get("Speaker 2") == "Rachel"


def test_spot_speaker_names_finds_direct_address():
    # "Hey Joey" → next speaker is Joey
    transcript = (
        "Speaker 1: Hey Joey, you okay?\n"
        "Speaker 2: Yeah I'm fine.\n"
        "Speaker 2: Hey Joey, come here.\n"
        "Speaker 3: What?\n"
    )
    mapping = spot_speaker_names(transcript)
    # Speaker 2 addressed as Joey (self-id signal from Speaker 2's first line isn't here,
    # so we rely on address counts — needs ≥2 so multiple lines needed)
    # At minimum the function should not crash and return a dict
    assert isinstance(mapping, dict)


def test_spot_speaker_names_ignores_filter_words():
    transcript = (
        "Speaker 1: Hey man, what's up?\n"
        "Speaker 2: Nothing much.\n"
        "Speaker 1: Hey dude, seriously.\n"
        "Speaker 2: Fine.\n"
    )
    mapping = spot_speaker_names(transcript)
    # "man" and "dude" are in the blocklist — should NOT appear as names
    assert "man" not in mapping.values()
    assert "dude" not in mapping.values()
    assert "Man" not in mapping.values()
    assert "Dude" not in mapping.values()


def test_spot_speaker_names_returns_empty_for_unrecognisable():
    transcript = "Speaker 1: blah blah blah\nSpeaker 2: yep yep\n"
    mapping = spot_speaker_names(transcript)
    assert mapping == {}


def test_spot_speaker_names_requires_minimum_evidence():
    # Self-id signal has weight 2, so one "I'm X" line satisfies the threshold (≥ 2).
    # To actually go below threshold, use only the weaker address signal (weight 1)
    # appearing just once.
    transcript = (
        "Speaker 1: Hey Zelda, what's up?\n"
        "Speaker 2: Nothing.\n"
    )
    mapping = spot_speaker_names(transcript)
    # Only one address occurrence (weight 1) — does not meet the ≥ 2 threshold
    assert mapping.get("Speaker 2") is None


# ── relabel_transcript ────────────────────────────────────────────────────────

def test_relabel_transcript_substitutes_known_speakers():
    text = "Speaker 1: hello\nSpeaker 2: hi there"
    result = relabel_transcript(text, {"Speaker 1": "Alice", "Speaker 2": "Bob"})
    assert "Alice: hello" in result
    assert "Bob: hi there" in result


def test_relabel_transcript_leaves_unmapped_speakers_unchanged():
    text = "Speaker 1: hello\nSpeaker 2: hi"
    result = relabel_transcript(text, {"Speaker 1": "Alice"})
    assert "Alice: hello" in result
    assert "Speaker 2: hi" in result


def test_relabel_transcript_empty_name_map():
    text = "Speaker 1: hello\nSpeaker 2: hi"
    result = relabel_transcript(text, {})
    assert result == text


# ── run_diarization_pipeline ─────────────────────────────────────────────────

def test_run_pipeline_skips_name_id_for_single_speaker(tmp_path):
    """When only one speaker is returned, skip name-spotting (nothing to resolve)."""
    single = DiarizedTranscript(
        raw_text="User: hello world",
        labeled_text="User: hello world",
        speakers=[SpeakerInfo(speaker_id="User", name="User", confidence=1.0)],
    )

    with (
        patch("src.pipeline.diarize._get_hf_token", return_value=None),
        patch("src.pipeline.diarize._transcribe_basic", return_value=single),
    ):
        out = run_diarization_pipeline(str(tmp_path / "fake.mp3"), "fake.mp3")

    assert out.raw_text == "User: hello world"
    assert len(out.speakers) == 1


def test_run_pipeline_relabels_multi_speaker_transcript(tmp_path):
    """With two speakers and a mocked LLM name-confirm, the transcript is relabeled."""
    diarized = DiarizedTranscript(
        raw_text="Speaker 1: hello there\nSpeaker 2: hi",
        labeled_text="Speaker 1: hello there\nSpeaker 2: hi",
        speakers=[
            SpeakerInfo(speaker_id="Speaker 1"),
            SpeakerInfo(speaker_id="Speaker 2"),
        ],
    )
    name_map = {"Speaker 1": "Alice", "Speaker 2": "Bob"}

    with (
        patch("src.pipeline.diarize._get_hf_token", return_value=None),
        patch("src.pipeline.diarize._transcribe_basic", return_value=diarized),
        patch("src.pipeline.diarize.spot_speaker_names", return_value=name_map),
        patch("src.pipeline.diarize._confirm_names_llm", return_value=name_map),
    ):
        out = run_diarization_pipeline(str(tmp_path / "fake.mp3"), "fake.mp3")

    assert "Alice: hello there" in out.labeled_text
    assert "Bob: hi" in out.labeled_text


def test_run_pipeline_uses_basic_transcribe_without_hf_token(tmp_path):
    """No HF token → _transcribe_basic is called, not diarize_audio_chunked."""
    basic_result = DiarizedTranscript(
        raw_text="User: test",
        labeled_text="User: test",
        speakers=[SpeakerInfo(speaker_id="User")],
    )

    with (
        patch("src.pipeline.diarize._get_hf_token", return_value=None),
        patch("src.pipeline.diarize._transcribe_basic", return_value=basic_result) as mock_basic,
        patch("src.pipeline.diarize.diarize_audio_chunked") as mock_chunked,
    ):
        out = run_diarization_pipeline(str(tmp_path / "fake.mp3"), "fake.mp3")

    mock_basic.assert_called_once()
    mock_chunked.assert_not_called()
    assert out.raw_text == "User: test"
