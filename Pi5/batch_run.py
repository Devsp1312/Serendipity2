#!/usr/bin/env python3
"""
Serendipity batch runner — full pipeline for a folder of audio files.

For each audio file in order:
  1. Stage 1 — Transcribe audio → plain text  (saves .txt alongside the audio)
  2. Stage 2+ — Parse → Multi-pass extraction → Gatekeeper → graph update

Crash-safe: progress is saved to batch_state.json after every file.
If the process dies mid-run, just re-run the same command — already-completed
files are skipped automatically.

Usage:
    python batch_run.py spongebob/
    python batch_run.py spongebob/ --profile spongebob --llm llama3 --whisper base
    python batch_run.py spongebob/ --transcribe-only  # Stage 1 only, no LLM graph
    python batch_run.py spongebob/ --reset            # clear saved state, start fresh
"""

from src.core.logger import setup_logging
setup_logging()

import argparse
import gc
import json
import re
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.core.config import DEFAULT_MODEL, DEFAULT_PROFILE, WHISPER_MODEL
from src.core.llm_client import check_connection, list_models, OllamaUnavailableError
from src.core.logger import get_logger
from src.core.schemas import LLMSchemaError
from src.pipeline.transcribe import AUDIO_EXTENSIONS, load_model, transcribe_to_file

logger = get_logger(__name__)

SEP  = "=" * 64
SEP2 = "-" * 64

# How long to pause between files (seconds) — lets Pi5 cool down and flush caches
_COOLDOWN_SEC = 30


def _natural_key(p: Path) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def _discover_audio(folder: Path) -> list[Path]:
    files = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return sorted(files, key=_natural_key)


# ── State file (crash-safe progress tracking) ────────────────────────────────

def _state_path(folder: Path) -> Path:
    return folder / "batch_state.json"


def _load_state(folder: Path, audio_files: list[Path]) -> dict:
    """Load existing state or initialise fresh state for all files."""
    sp = _state_path(folder)
    if sp.exists():
        try:
            saved = json.loads(sp.read_text(encoding="utf-8"))
            # Validate that the saved file list matches current files
            saved_names = {e["file"] for e in saved.get("files", [])}
            current_names = {f.name for f in audio_files}
            if saved_names == current_names:
                done = sum(1 for e in saved["files"] if e["status"] == "done")
                print(f"  Resuming from saved state ({done}/{len(audio_files)} already done)")
                return saved
            else:
                print("  Saved state doesn't match current files — starting fresh")
        except Exception as e:
            print(f"  Could not read batch_state.json ({e}) — starting fresh")

    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "files": [
            {"file": f.name, "status": "pending", "started_at": None, "finished_at": None, "error": None}
            for f in audio_files
        ]
    }


def _save_state(folder: Path, state: dict) -> None:
    try:
        _state_path(folder).write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Could not save batch state: %s", e)


def _get_entry(state: dict, filename: str) -> dict:
    for entry in state["files"]:
        if entry["file"] == filename:
            return entry
    raise KeyError(filename)


def _mark(state: dict, folder: Path, filename: str, status: str, error: str = None) -> None:
    entry = _get_entry(state, filename)
    entry["status"] = status
    if status in ("done", "failed"):
        entry["finished_at"] = datetime.now(timezone.utc).isoformat()
    if error:
        entry["error"] = error
    _save_state(folder, state)


# ── Graceful Ctrl+C ──────────────────────────────────────────────────────────

_interrupted = False

def _handle_sigint(sig, frame):
    global _interrupted
    print("\n\n  [Ctrl+C] Stopping after current file completes. State saved — re-run to resume.")
    _interrupted = True


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    global _interrupted
    signal.signal(signal.SIGINT, _handle_sigint)

    parser = argparse.ArgumentParser(description="Serendipity batch audio → knowledge graph runner")
    parser.add_argument("folder",              type=str,  help="Folder containing audio files")
    parser.add_argument("--profile",           type=str,  default=None,          help="Profile name (default: folder name)")
    parser.add_argument("--llm",               type=str,  default=None,          help="Ollama model (default: auto-detect)")
    parser.add_argument("--whisper",           type=str,  default=WHISPER_MODEL,  help="Whisper model size (default: base)")
    parser.add_argument("--overwrite",         action="store_true",              help="Re-transcribe even if .txt exists")
    parser.add_argument("--transcribe-only",   action="store_true",              help="Stage 1 only — skip LLM pipeline")
    parser.add_argument("--max-chars",         type=int,  default=0,             help="Max transcript chars sent to LLM (default: 0 = unlimited)")
    parser.add_argument("--cooldown",          type=int,  default=_COOLDOWN_SEC, help=f"Seconds to pause between files (default: {_COOLDOWN_SEC})")
    parser.add_argument("--reset",             action="store_true",              help="Clear saved batch state and start fresh")
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        print(f"[ERROR] Not a directory: {folder}")
        sys.exit(1)

    profile = (args.profile or folder.name).strip().lower()

    print(f"\n{SEP}")
    print("  SERENDIPITY — Batch Pipeline Runner")
    print(SEP)
    print(f"  Folder  : {folder}")
    print(f"  Profile : {profile}")
    print(f"  Whisper : {args.whisper}")
    if args.max_chars:
        print(f"  Max chars: {args.max_chars:,}")
    else:
        print(f"  Max chars: unlimited")

    # ── Discover files ────────────────────────────────────────────────────────
    audio_files = _discover_audio(folder)
    if not audio_files:
        print(f"\n[ERROR] No audio files found in {folder}")
        sys.exit(1)

    total = len(audio_files)
    print(f"  Files   : {total} audio file(s)")

    # ── State file ────────────────────────────────────────────────────────────
    if args.reset and _state_path(folder).exists():
        _state_path(folder).unlink()
        print("  Saved state cleared — starting fresh")

    state = _load_state(folder, audio_files)
    _save_state(folder, state)  # write immediately so the file always exists
    print(SEP)

    # ── LLM check (skip if transcribe-only) ───────────────────────────────────
    llm_model = None
    if not args.transcribe_only:
        if not check_connection():
            print("\n[FATAL] Cannot reach Ollama. Start it with: ollama serve")
            sys.exit(1)

        if args.llm:
            llm_model = args.llm
        else:
            available = list_models()
            if not available:
                print("\n[FATAL] No Ollama models found. Pull one: ollama pull llama3")
                sys.exit(1)
            llm_model = available[0]

        print(f"  LLM     : {llm_model}")
        print(SEP)

    # ── Load Whisper model once ───────────────────────────────────────────────
    print(f"\nLoading Whisper '{args.whisper}' model…")
    whisper_model = load_model(args.whisper)
    print("Model loaded.\n")

    # ── Main loop ─────────────────────────────────────────────────────────────
    session_start = time.monotonic()
    transcribed = 0
    pipeline_ran = 0
    errors = 0
    skipped = 0

    for idx, audio_path in enumerate(audio_files):
        if _interrupted:
            break

        entry = _get_entry(state, audio_path.name)

        # Skip files already completed in a previous run
        if entry["status"] == "done":
            skipped += 1
            print(f"  [{idx+1}/{total}]  {audio_path.name}  — already done, skipping")
            continue

        file_num = idx + 1
        print(f"\n{SEP}")
        print(f"  [{file_num}/{total}]  {audio_path.name}")
        print(SEP2)

        entry["started_at"] = datetime.now(timezone.utc).isoformat()
        entry["status"] = "in_progress"
        _save_state(folder, state)

        # ── Stage 1: Transcription ────────────────────────────────────────────
        txt_path = audio_path.with_suffix(".txt")
        skip_transcription = txt_path.exists() and not args.overwrite

        if skip_transcription:
            print(f"  ✓ Transcript exists — skipping Stage 1  ({txt_path.name})")
            transcript_text = txt_path.read_text(encoding="utf-8")
            print(f"    Chars: {len(transcript_text):,}")
        else:
            t0 = time.monotonic()
            print(f"  Stage 1: Transcribing {audio_path.name}…")
            try:
                out_path, result = transcribe_to_file(
                    audio_path, whisper_model, vad=True, beam_size=1
                )
                elapsed = time.monotonic() - t0
                speed = result.audio_duration_sec / max(result.transcription_sec, 0.01)
                print(
                    f"  ✓ Transcribed  "
                    f"audio={result.audio_duration_sec/60:.1f}min  "
                    f"time={result.transcription_sec:.1f}s  "
                    f"speed={speed:.0f}×  "
                    f"lang={result.language}  "
                    f"chars={len(result.text):,}"
                )
                transcript_text = result.text
                transcribed += 1
            except Exception as e:
                msg = str(e)
                print(f"  ✗ Transcription failed: {msg}")
                logger.error("Transcription failed for %s: %s", audio_path.name, e)
                _mark(state, folder, audio_path.name, "failed", error=msg)
                errors += 1
                continue

        if args.transcribe_only:
            _mark(state, folder, audio_path.name, "done")
            continue

        # ── Stage 2+: Full pipeline ───────────────────────────────────────────
        if args.max_chars and len(transcript_text) > args.max_chars:
            transcript_text = transcript_text[:args.max_chars].rsplit(" ", 1)[0]
            print(f"  ↩ Transcript capped at {len(transcript_text):,} chars (--max-chars)")
        else:
            print(f"  Transcript: {len(transcript_text):,} chars (full, no cap)")

        print(f"\n  Stage 2+: Running extraction → gatekeeper…")

        from main import run_pipeline

        t_pipe = time.monotonic()
        try:
            run_pipeline(transcript_text, model=llm_model, profile=profile)
            pipe_elapsed = time.monotonic() - t_pipe
            print(f"\n  ✓ Pipeline done in {pipe_elapsed:.1f}s")
            _mark(state, folder, audio_path.name, "done")
            pipeline_ran += 1
        except (OllamaUnavailableError, LLMSchemaError) as e:
            msg = str(e)
            print(f"  ✗ Pipeline error: {msg}")
            logger.error("Pipeline failed for %s: %s", audio_path.name, e)
            _mark(state, folder, audio_path.name, "failed", error=msg)
            errors += 1
        except Exception as e:
            msg = str(e)
            print(f"  ✗ Unexpected error: {msg}")
            logger.exception("Unexpected pipeline error for %s", audio_path.name)
            _mark(state, folder, audio_path.name, "failed", error=msg)
            errors += 1

        # ── Memory cleanup between files ──────────────────────────────────────
        gc.collect()

        # ── Cooldown (lets Pi5 cool down between heavy jobs) ──────────────────
        remaining = [
            f for f in audio_files[idx+1:]
            if _get_entry(state, f.name)["status"] != "done"
        ]
        if remaining and not _interrupted and args.cooldown > 0:
            print(f"\n  Cooling down {args.cooldown}s before next file…")
            time.sleep(args.cooldown)

    # ── Final summary ─────────────────────────────────────────────────────────
    total_elapsed = time.monotonic() - session_start

    done_count  = sum(1 for e in state["files"] if e["status"] == "done")
    fail_count  = sum(1 for e in state["files"] if e["status"] == "failed")
    still_todo  = sum(1 for e in state["files"] if e["status"] in ("pending", "in_progress"))

    print(f"\n{SEP}")
    if _interrupted:
        print("  BATCH PAUSED (Ctrl+C)")
    else:
        print("  BATCH COMPLETE" if still_todo == 0 else "  BATCH FINISHED WITH REMAINING FILES")
    print(SEP2)
    print(f"  Done             : {done_count}/{total}")
    if skipped:
        print(f"  Already done     : {skipped} (skipped)")
    if transcribed:
        print(f"  Transcribed      : {transcribed}")
    if pipeline_ran:
        print(f"  Pipeline runs    : {pipeline_ran}")
    if fail_count:
        print(f"  Failed           : {fail_count}")
        for e in state["files"]:
            if e["status"] == "failed":
                print(f"    - {e['file']}: {e.get('error', '')[:80]}")
    if still_todo:
        print(f"  Still to process : {still_todo}")
        print(f"  Re-run the same command to continue from where this stopped.")
    print(f"  Total time       : {total_elapsed/60:.1f} min ({total_elapsed:.0f}s)")
    print(f"  Profile          : {profile}  (view in web UI → Profile tab)")
    print(f"  State file       : {_state_path(folder)}")
    print(SEP + "\n")


if __name__ == "__main__":
    main()
