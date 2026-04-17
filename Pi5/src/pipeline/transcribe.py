"""
Pipeline Step 1 — Audio Transcription.

Input:  audio file  (.mp3 / .wav / .m4a / .flac / .ogg / .aac / .wma)
Output: plain-text transcript  (no speaker labels yet — that's Step 2)

Converts audio files to plain-text transcripts using faster-whisper.
No speaker labels, no timestamps — raw transcribed speech only.

Audio is decoded to 16 kHz mono float32, peak-normalized to -3 dBFS,
then passed through faster-whisper's built-in silero-VAD to skip silence
before transcription (greedy decoding, beam_size=1 for speed).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

from src.core.config import WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE
from src.core.logger import get_logger

logger = get_logger(__name__)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}

# Peak normalization target: -3 dBFS
_TARGET_PEAK: float = 10 ** (-3 / 20)   # ≈ 0.7079
_SAMPLE_RATE: int = 16_000


@dataclass
class TranscriptResult:
    """Timing and content result from a single transcription run."""
    text: str
    audio_duration_sec: float
    transcription_sec: float
    language: str
    rtf: float  # real-time factor = transcription_sec / audio_duration_sec


def load_model(
    model_size: str = WHISPER_MODEL,
    device: str = WHISPER_DEVICE,
    compute_type: str = WHISPER_COMPUTE,
) -> WhisperModel:
    """
    Load the faster-whisper model. Call once and reuse across files.

    Args:
        model_size:   tiny | base | small | medium | large-v3
        device:       cpu | cuda
        compute_type: int8 | float16 | float32
    """
    logger.info(
        "Loading faster-whisper model: %s on %s (%s)", model_size, device, compute_type
    )
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def decode_and_normalize(audio_path: str | Path) -> tuple[np.ndarray, float]:
    """
    Decode audio to 16 kHz mono float32 and peak-normalize to -3 dBFS.

    Uses faster-whisper's internal decode_audio (backed by PyAV) so all
    common formats — MP3, WAV, M4A, FLAC, OGG, AAC, WMA — are supported.

    Returns:
        (audio, duration_sec) where audio is a 1-D float32 array at 16000 Hz.
    """
    audio: np.ndarray = decode_audio(str(audio_path), sampling_rate=_SAMPLE_RATE)
    duration_sec = len(audio) / _SAMPLE_RATE

    peak = float(np.abs(audio).max())
    if peak > 0:
        audio = audio * (_TARGET_PEAK / peak)

    return audio, duration_sec


def transcribe_file(
    audio_path: str | Path,
    model: WhisperModel,
    *,
    vad: bool = True,
    beam_size: int = 1,
) -> TranscriptResult:
    """
    Transcribe a single audio file to plain text.

    Args:
        audio_path: Path to any supported audio file.
        model:      Loaded WhisperModel (call load_model() first).
        vad:        Strip silence/music via silero-VAD before transcription.
        beam_size:  1 = greedy decoding (fastest). Higher = more accurate but slower.

    Returns:
        TranscriptResult with text, timing, language, and RTF.
    """
    audio_path = Path(audio_path)
    logger.info("Transcribing: %s", audio_path.name)

    audio, duration_sec = decode_and_normalize(audio_path)

    t0 = time.perf_counter()
    segments_gen, info = model.transcribe(
        audio,
        beam_size=beam_size,
        vad_filter=vad,
        vad_parameters={"min_silence_duration_ms": 500},
        language=None,  # auto-detect
    )

    parts: list[str] = []
    for segment in segments_gen:
        text = segment.text.strip()
        if text:
            parts.append(text)

    transcription_sec = time.perf_counter() - t0
    transcript = " ".join(parts)

    rtf = transcription_sec / duration_sec if duration_sec > 0 else 0.0

    logger.info(
        "Done: %s | audio=%.1fs | transcription=%.1fs | RTF=%.2f× | lang=%s | chars=%d",
        audio_path.name,
        duration_sec,
        transcription_sec,
        rtf,
        info.language,
        len(transcript),
    )

    return TranscriptResult(
        text=transcript,
        audio_duration_sec=duration_sec,
        transcription_sec=transcription_sec,
        language=info.language or "unknown",
        rtf=rtf,
    )


def transcribe_to_file(
    audio_path: str | Path,
    model: WhisperModel,
    output_path: Optional[str | Path] = None,
    **kwargs,
) -> tuple[Path, TranscriptResult]:
    """
    Transcribe audio and write the plain-text transcript to a .txt file.

    If output_path is None, saves alongside the audio with the same stem
    (e.g. 'day 1.mp3' → 'day 1.txt').

    Returns:
        (output_path, TranscriptResult)
    """
    audio_path = Path(audio_path)
    if output_path is None:
        output_path = audio_path.with_suffix(".txt")
    output_path = Path(output_path)

    result = transcribe_file(audio_path, model, **kwargs)
    output_path.write_text(result.text, encoding="utf-8")
    logger.info("Transcript saved: %s", output_path)

    return output_path, result
