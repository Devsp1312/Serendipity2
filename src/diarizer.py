"""
Speaker diarization and identification pipeline.

Combines three local-only tools:
  1. WhisperX  — transcription + word-level alignment
  2. pyannote  — speaker separation (SPEAKER_00, SPEAKER_01, ...)
  3. Ollama    — LLM identifies who each speaker actually is

Falls back gracefully:
  - No HF token / pyannote unavailable → faster-whisper (single "User:" speaker)
  - Ollama unreachable → keeps SPEAKER_XX labels as-is

Everything runs on-device. The HuggingFace token is only used once to download
the pyannote model (~300 MB), which is then cached locally forever.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path

import ollama

from src.config import (
    HF_TOKEN_PATH,
    SPEAKER_ID_MODEL,
    WHISPER_MODEL,
    WHISPER_DEVICE,
    WHISPER_COMPUTE,
    WHISPER_BEAM_SIZE,
)
from src.llm_client import get_speaker_id_prompt
from src.logger import get_logger
from src.schemas import DiarizedTranscript, SpeakerInfo

logger = get_logger(__name__)


# ─── Token helpers ───────────────────────────────────────────────────────────

def _get_hf_token() -> str | None:
    """
    Reads the HuggingFace token from (in priority order):
      1. HF_TOKEN environment variable
      2. ~/.huggingface/token file
    Returns None if neither is found.
    """
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip()
    if HF_TOKEN_PATH.exists():
        return HF_TOKEN_PATH.read_text().strip()
    return None


# ─── Core diarization ───────────────────────────────────────────────────────

def diarize_audio(audio_path: str, hf_token: str | None = None) -> DiarizedTranscript:
    """
    Full local pipeline: audio file → diarized transcript with speaker labels.

    Uses WhisperX for transcription + alignment, and pyannote for speaker
    separation. Falls back to faster-whisper (single-speaker) if pyannote
    is unavailable.

    Args:
        audio_path: Path to audio file (.mp3, .wav, .m4a)
        hf_token:   HuggingFace token (optional — auto-detected from env/file)

    Returns:
        DiarizedTranscript with raw_text containing "SPEAKER_XX: text" lines
    """
    token = hf_token or _get_hf_token()

    if token:
        try:
            return _diarize_with_whisperx(audio_path, token)
        except Exception as e:
            logger.warning("WhisperX diarization failed: %s — falling back to basic transcription", e)

    # Fallback: faster-whisper without speaker separation
    return _transcribe_basic(audio_path)


def _diarize_with_whisperx(audio_path: str, hf_token: str) -> DiarizedTranscript:
    """WhisperX + pyannote diarization pipeline."""
    import torch
    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    logger.info("Diarization started  path=%s  device=%s", audio_path, device)
    t0 = time.monotonic()

    # Step 1: Transcribe
    model = whisperx.load_model(WHISPER_MODEL, device=device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16)
    logger.info("Transcription complete  language=%s  segments=%d",
                result.get("language"), len(result.get("segments", [])))

    # Step 2: Align words for precise timestamps
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(result["segments"], model_a, metadata, audio, device)

    # Step 3: Diarize — assign speaker IDs to segments
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # Step 4: Collapse consecutive same-speaker segments into lines
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
                lines.append(f'{current_speaker}: {" ".join(current_parts)}')
            current_speaker = speaker
            current_parts = [text]
        else:
            current_parts.append(text)

    if current_speaker is not None:
        lines.append(f'{current_speaker}: {" ".join(current_parts)}')

    diarized_text = "\n".join(lines)
    unique_speakers = sorted({seg.get("speaker", "SPEAKER_00") for seg in result["segments"]})
    elapsed = time.monotonic() - t0

    logger.info("Diarization complete  speakers=%s  lines=%d  elapsed=%.1fs",
                unique_speakers, len(lines), elapsed)

    return DiarizedTranscript(
        raw_text=diarized_text,
        speakers=[SpeakerInfo(speaker_id=s) for s in unique_speakers],
    )


def _transcribe_basic(audio_path: str) -> DiarizedTranscript:
    """Fallback: faster-whisper transcription without speaker separation."""
    from faster_whisper import WhisperModel

    logger.info("Basic transcription (no diarization)  path=%s", audio_path)
    t0 = time.monotonic()

    model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    segments, info = model.transcribe(str(audio_path), beam_size=WHISPER_BEAM_SIZE)
    full_text = " ".join(seg.text.strip() for seg in segments)
    elapsed = time.monotonic() - t0

    logger.info("Basic transcription complete  chars=%d  elapsed=%.1fs", len(full_text), elapsed)

    raw_text = f"User: {full_text}"
    return DiarizedTranscript(
        raw_text=raw_text,
        labeled_text=raw_text,
        speakers=[SpeakerInfo(speaker_id="User", name="User", confidence=1.0)],
        recording_type="monologue",
    )


# ─── Speaker identification (LLM) ───────────────────────────────────────────

def identify_speakers(
    diarized_text: str,
    filename: str,
    model: str | None = None,
) -> dict:
    """
    Uses a local Ollama LLM to identify who each SPEAKER_XX is.

    Sends the first ~600 words of the diarized transcript + filename to the model.
    Returns the parsed JSON response with speaker identities.

    Args:
        diarized_text: Diarized transcript with SPEAKER_XX labels
        filename:      Original audio filename (contains useful context)
        model:         Ollama model to use (defaults to SPEAKER_ID_MODEL from config)

    Returns:
        Dict with "speakers", "recording_type", "topic" keys
    """
    model = model or SPEAKER_ID_MODEL

    # Sample first ~600 words
    words = diarized_text.split()
    sample = " ".join(words[:600])
    total_words = len(words)

    user_prompt = (
        f'Audio filename: "{filename}"\n'
        f"Total words in transcript: {total_words}\n\n"
        f"=== TRANSCRIPT (first 600 words) ===\n\n"
        f"{sample}\n\n"
        f"=== END TRANSCRIPT ===\n\n"
        f"Based on the transcript and filename, identify all speakers. "
        f"Return ONLY a JSON object."
    )

    logger.info("Speaker identification started  model=%s  filename=%s", model, filename)
    t0 = time.monotonic()

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": get_speaker_id_prompt()},
                {"role": "user", "content": user_prompt},
            ],
            format="json",
        )
        raw = response.message.content
        elapsed = time.monotonic() - t0
        logger.info("Speaker identification complete  elapsed=%.1fs", elapsed)

        return _extract_json(raw)

    except Exception as e:
        logger.warning("Speaker identification failed: %s", e)
        return {"speakers": [], "recording_type": "unknown", "topic": ""}


def _extract_json(text: str) -> dict:
    """Extract a JSON object from LLM output, handling thinking tags and markdown."""
    # Strip <think>...</think> blocks (qwen-style)
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try markdown code blocks
    md_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
    if md_match:
        try:
            return json.loads(md_match.group(1))
        except json.JSONDecodeError:
            pass

    # Find outermost braces
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse speaker ID JSON from LLM response (len=%d)", len(cleaned))
    return {"speakers": [], "recording_type": "unknown", "topic": ""}


# ─── Transcript relabeling ───────────────────────────────────────────────────

def relabel_transcript(diarized_text: str, speaker_map: dict[str, str]) -> str:
    """
    Replaces SPEAKER_XX labels with real names throughout the transcript.

    Args:
        diarized_text: Text with "SPEAKER_00: ..." lines
        speaker_map:   {"SPEAKER_00": "LeBron James", "SPEAKER_01": "JJ Redick"}

    Returns:
        Transcript with real names: "LeBron James: ..."
    """
    result = diarized_text
    for speaker_id, name in speaker_map.items():
        if name:
            result = result.replace(f"{speaker_id}:", f"{name}:")
    return result


def build_speaker_map(speaker_id_result: dict) -> dict[str, str]:
    """
    Converts the LLM's speaker identification result into a simple mapping.
    Returns: {"SPEAKER_00": "LeBron James", ...}
    """
    mapping = {}
    for speaker in speaker_id_result.get("speakers", []):
        sid = speaker.get("speaker_id", "")
        name = speaker.get("name")
        role = speaker.get("role", "")
        # Use name if available, otherwise use role, otherwise keep original ID
        label = name or role or sid
        if sid:
            mapping[sid] = label
    return mapping


# ─── Full pipeline ───────────────────────────────────────────────────────────

def run_diarization_pipeline(
    audio_path: str,
    filename: str,
    speaker_id_model: str | None = None,
    hf_token: str | None = None,
) -> DiarizedTranscript:
    """
    Complete audio → identified-speaker transcript pipeline.

    1. Diarize audio (WhisperX + pyannote, or fallback to faster-whisper)
    2. Identify speakers via LLM
    3. Relabel transcript with real names

    Args:
        audio_path:       Path to audio file
        filename:         Original filename (for context)
        speaker_id_model: Ollama model for speaker ID (default: config.SPEAKER_ID_MODEL)
        hf_token:         HuggingFace token (default: auto-detected)

    Returns:
        DiarizedTranscript with labeled_text containing real speaker names
    """
    # Step 1: Diarize
    transcript = diarize_audio(audio_path, hf_token=hf_token)

    # If basic fallback was used (single "User:" speaker), skip identification
    if len(transcript.speakers) <= 1 and transcript.speakers[0].speaker_id == "User":
        logger.info("Single speaker (basic mode) — skipping speaker identification")
        return transcript

    # Step 2: Identify speakers via LLM
    speaker_result = identify_speakers(transcript.raw_text, filename, model=speaker_id_model)

    # Update transcript metadata
    transcript.recording_type = speaker_result.get("recording_type", "conversation")
    transcript.topic = speaker_result.get("topic", "")

    # Update speaker info
    for speaker_data in speaker_result.get("speakers", []):
        sid = speaker_data.get("speaker_id", "")
        for s in transcript.speakers:
            if s.speaker_id == sid:
                s.name = speaker_data.get("name")
                s.role = speaker_data.get("role")
                s.confidence = speaker_data.get("confidence", 0.0)
                break

    # Step 3: Relabel transcript with real names
    speaker_map = build_speaker_map(speaker_result)
    transcript.labeled_text = relabel_transcript(transcript.raw_text, speaker_map)

    logger.info("Diarization pipeline complete  speakers=%d  type=%s  topic=%s",
                len(transcript.speakers), transcript.recording_type, transcript.topic)

    return transcript
