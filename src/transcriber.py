"""
Audio transcription using faster-whisper (local, CPU-friendly, no API key).
Falls back gracefully if faster-whisper is not installed.

Output is plain text prefixed with 'User:' so it flows through the pipeline
as a single speaker turn. Add your own speaker labels before running if needed.
Whisper settings are configured in src/config.py (model size, device, etc.).
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from src.config import WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE, WHISPER_BEAM_SIZE
from src.logger import get_logger

logger = get_logger(__name__)


def is_available() -> bool:
    """Returns True if faster-whisper is installed."""
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def transcribe(audio_bytes: bytes, filename: str, whisper_model: str | None = None) -> str:
    """
    Transcribes audio bytes to plain text using faster-whisper.
    Writes to a temp file, runs WhisperModel, returns 'User: <transcript>'.

    Raises ImportError if faster-whisper isn't installed.
    Raises RuntimeError if transcription fails.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError("faster-whisper is not installed. Run: pip install faster-whisper")

    model_size = whisper_model or WHISPER_MODEL
    suffix     = Path(filename).suffix or ".wav"
    logger.info("Transcription started  filename=%s  model=%s  device=%s", filename, model_size, WHISPER_DEVICE)
    t0 = time.monotonic()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        model       = WhisperModel(model_size, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
        segments, _ = model.transcribe(tmp_path, beam_size=WHISPER_BEAM_SIZE)
        full_text   = " ".join(seg.text.strip() for seg in segments)
        logger.info("Transcription complete  chars=%d  elapsed=%.2fs", len(full_text), round(time.monotonic() - t0, 2))
        return f"User: {full_text}"
    except Exception as e:
        logger.error("Transcription failed for %s: %s", filename, e)
        raise RuntimeError(f"Transcription failed: {e}") from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def get_install_instructions() -> str:
    return (
        "Audio transcription requires faster-whisper.\n"
        "Install: pip install faster-whisper\n\n"
        "Downloads the Whisper 'base' model (~150 MB) on first use.\n"
        "Alternatively, upload a .txt file with speaker-labeled transcript."
    )
