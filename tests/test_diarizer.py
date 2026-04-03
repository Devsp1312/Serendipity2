"""
Tests for src/diarizer.py

Covers: JSON extraction robustness, transcript relabeling, speaker-map creation,
and full pipeline orchestration behavior with mocks.
"""

from src.diarizer import (
    _extract_json,
    build_speaker_map,
    relabel_transcript,
    run_diarization_pipeline,
)
from src.schemas import DiarizedTranscript, SpeakerInfo


def test_extract_json_with_think_tags():
    raw = '<think>internal chain</think>{"speakers": [], "recording_type": "podcast", "topic": "money"}'
    out = _extract_json(raw)
    assert out["recording_type"] == "podcast"
    assert out["topic"] == "money"


def test_extract_json_with_markdown_block():
    raw = """Here is the result:\n```json\n{\n  \"speakers\": [],\n  \"recording_type\": \"interview\",\n  \"topic\": \"career\"\n}\n```"""
    out = _extract_json(raw)
    assert out["recording_type"] == "interview"


def test_extract_json_invalid_returns_safe_default():
    out = _extract_json("not json at all")
    assert out == {"speakers": [], "recording_type": "unknown", "topic": ""}


def test_build_speaker_map_prefers_name_then_role_then_id():
    result = {
        "speakers": [
            {"speaker_id": "SPEAKER_00", "name": "Alice", "role": "Host"},
            {"speaker_id": "SPEAKER_01", "name": None, "role": "Guest"},
            {"speaker_id": "SPEAKER_02", "name": None, "role": ""},
        ]
    }
    mapping = build_speaker_map(result)
    assert mapping["SPEAKER_00"] == "Alice"
    assert mapping["SPEAKER_01"] == "Guest"
    assert mapping["SPEAKER_02"] == "SPEAKER_02"


def test_relabel_transcript_substitutes_known_speakers():
    diarized_text = "SPEAKER_00: hello\nSPEAKER_01: hi"
    mapped = relabel_transcript(diarized_text, {"SPEAKER_00": "Alice"})
    assert "Alice: hello" in mapped
    assert "SPEAKER_01: hi" in mapped


def test_run_pipeline_skips_id_for_single_user(monkeypatch):
    single = DiarizedTranscript(
        raw_text="User: hello",
        labeled_text="User: hello",
        speakers=[SpeakerInfo(speaker_id="User", name="User", confidence=1.0)],
    )

    monkeypatch.setattr("src.diarizer.diarize_audio", lambda *args, **kwargs: single)

    called = {"value": False}

    def _identify(*args, **kwargs):
        called["value"] = True
        return {}

    monkeypatch.setattr("src.diarizer.identify_speakers", _identify)

    out = run_diarization_pipeline("/tmp/fake.wav", "fake.wav")
    assert out.raw_text == "User: hello"
    assert called["value"] is False


def test_run_pipeline_maps_and_relabels(monkeypatch):
    diarized = DiarizedTranscript(
        raw_text="SPEAKER_00: hello there\nSPEAKER_01: hi",
        speakers=[
            SpeakerInfo(speaker_id="SPEAKER_00"),
            SpeakerInfo(speaker_id="SPEAKER_01"),
        ],
    )

    monkeypatch.setattr("src.diarizer.diarize_audio", lambda *args, **kwargs: diarized)
    monkeypatch.setattr(
        "src.diarizer.identify_speakers",
        lambda *args, **kwargs: {
            "speakers": [
                {"speaker_id": "SPEAKER_00", "name": "Alice", "role": "Host", "confidence": 0.9},
                {"speaker_id": "SPEAKER_01", "name": None, "role": "Guest", "confidence": 0.7},
            ],
            "recording_type": "interview",
            "topic": "finance",
        },
    )

    out = run_diarization_pipeline("/tmp/fake.wav", "fake.wav")
    assert out.recording_type == "interview"
    assert out.topic == "finance"
    assert out.labeled_text.startswith("Alice:")
    assert "Guest:" in out.labeled_text
    assert out.speakers[0].name == "Alice"
    assert out.speakers[1].role == "Guest"
