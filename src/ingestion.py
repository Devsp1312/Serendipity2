"""
Phase 1 — Transcript ingestion and parsing.
Converts raw speaker-labeled text into structured Turn objects.

Expected format:  "SpeakerName: text content"
Speaker names are normalized to Title Case so "alice", "ALICE", and "Alice"
all map to the same speaker — preventing duplicate graph nodes downstream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.config import MAX_TRANSCRIPT_BYTES, MAX_SPEAKER_LABEL_LENGTH
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Turn:
    speaker: str
    text: str

    def __str__(self) -> str:
        return f"{self.speaker}: {self.text}"


def validate_transcript_input(raw_text: str) -> None:
    """Raises ValueError for empty input, oversized files, or binary content."""
    if not raw_text or not raw_text.strip():
        raise ValueError("Transcript is empty. Please provide speaker-labeled text.")

    byte_size = len(raw_text.encode("utf-8"))
    if byte_size > MAX_TRANSCRIPT_BYTES:
        raise ValueError(
            f"Transcript is too large ({byte_size / 1024 / 1024:.1f} MB). "
            f"Maximum is {MAX_TRANSCRIPT_BYTES / 1024 / 1024:.0f} MB."
        )

    if "\x00" in raw_text:
        raise ValueError("Transcript contains binary content (null bytes). Upload a plain-text file.")


def _normalize_speaker(name: str) -> str:
    """Strips whitespace, applies Title Case, and truncates overly long labels."""
    name = name.strip().title()
    if len(name) > MAX_SPEAKER_LABEL_LENGTH:
        logger.warning("Speaker label %r exceeds %d chars — truncating.", name, MAX_SPEAKER_LABEL_LENGTH)
        name = name[:MAX_SPEAKER_LABEL_LENGTH]
    return name


def parse_transcript(raw_text: str) -> list[Turn]:
    """
    Splits raw transcript on lines matching '^SpeakerName: text'.
    Multi-line turns are joined under the same speaker.

    If no speaker labels are found, the whole text is treated as a single
    'User' turn so free-form text still flows through the pipeline.

    Raises ValueError if the input fails validation.
    """
    validate_transcript_input(raw_text)

    lines   = raw_text.strip().splitlines()
    turns:  list[Turn] = []
    speaker: str | None = None
    parts:   list[str] = []
    pattern = re.compile(r'^([\w][\w\s]*?):\s+(.*)')

    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if m:
            if speaker is not None:
                turns.append(Turn(speaker=speaker, text=" ".join(parts)))
            speaker = _normalize_speaker(m.group(1))
            parts   = [m.group(2).strip()]
        elif speaker is not None:
            parts.append(line)

    if speaker is not None and parts:
        turns.append(Turn(speaker=speaker, text=" ".join(parts)))

    if not turns:
        # No speaker labels detected — wrap the whole text as a single User turn
        fallback = " ".join(l.strip() for l in lines if l.strip())
        turns    = [Turn(speaker="User", text=fallback)]
        logger.info("No speaker labels found; falling back to single User turn.")

    unique_speakers = {t.speaker for t in turns}
    logger.info("Ingestion complete: %d turns, %d speaker(s): %s",
                len(turns), len(unique_speakers), sorted(unique_speakers))
    return turns


def format_transcript_for_llm(turns: list[Turn]) -> str:
    """Reassembles turns into a plain-text block for LLM prompt injection."""
    return "\n".join(str(t) for t in turns)


def get_turn_stats(turns: list[Turn]) -> dict:
    """Returns {speaker: turn_count} for display in the UI."""
    stats: dict[str, int] = {}
    for turn in turns:
        stats[turn.speaker] = stats.get(turn.speaker, 0) + 1
    return stats


def transcribe_audio(audio_path: str, hf_token: str, device: str | None = None) -> str:
    """
    Transcribes an audio file using WhisperX with full speaker diarization.

    Steps:
      1. Load WhisperX base model (auto-selects CUDA or CPU)
      2. Transcribe to word-level segments
      3. Align words for precise timestamps
      4. Run pyannote diarization pipeline via HuggingFace token
      5. Assign speaker IDs to each word segment
      6. Collapse consecutive segments into one line per speaker turn

    Returns a diarized string where each line is:
        SPEAKER_XX: "spoken text"

    Requires: pip install whisperx
    Requires: a HuggingFace token with access to pyannote/speaker-diarization-3.1
    """
    import torch
    import whisperx

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    compute_type = "float16" if device == "cuda" else "int8"

    logger.info("transcribe_audio: loading model  device=%s  compute_type=%s", device, compute_type)
    model = whisperx.load_model("base", device=device, compute_type=compute_type)

    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16)
    logger.info("transcribe_audio: transcription complete  language=%s  segments=%d",
                result.get("language"), len(result.get("segments", [])))

    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(result["segments"], model_a, metadata, audio, device)

    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    logger.info("transcribe_audio: diarization complete")

    # Collapse consecutive segments by the same speaker into one quoted line
    lines: list[str] = []
    current_speaker: str | None = None
    current_parts: list[str] = []

    for seg in result["segments"]:
        speaker = seg.get("speaker", "SPEAKER_00")
        text = seg["text"].strip()
        if not text:
            continue
        if speaker != current_speaker:
            if current_speaker is not None:
                lines.append(f'{current_speaker}: "{" ".join(current_parts)}"')
            current_speaker = speaker
            current_parts = [text]
        else:
            current_parts.append(text)

    if current_speaker is not None:
        lines.append(f'{current_speaker}: "{" ".join(current_parts)}"')

    diarized = "\n".join(lines)
    unique_speakers = {seg.get("speaker", "SPEAKER_00") for seg in result["segments"]}
    logger.info("transcribe_audio: formatted  lines=%d  speakers=%s", len(lines), sorted(unique_speakers))
    return diarized
