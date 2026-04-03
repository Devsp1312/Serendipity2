"""
Speaker identification helper script.

Uses the shared diarization pipeline in src/diarizer.py so behavior stays
consistent with the CLI and Streamlit app.

Usage:
    python scripts/identify_speakers.py
    python scripts/identify_speakers.py "path/to/specific/file.mp3"
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diarizer import run_diarization_pipeline

AUDIO_DIR = Path(__file__).parent.parent / "audio files"


def process_file(audio_path: Path, reuse_transcript: bool = True):
    """Run diarization + speaker identification for one audio file."""
    print(f"\n{'='*60}")
    print(f"  {audio_path.name}")
    print(f"  Size: {audio_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"{'='*60}")

    transcript = run_diarization_pipeline(str(audio_path), filename=audio_path.name)

    result = {
        "speakers": [
            {
                "speaker_id": s.speaker_id,
                "name": s.name,
                "role": s.role,
                "confidence": s.confidence,
            }
            for s in transcript.speakers
        ],
        "recording_type": transcript.recording_type,
        "topic": transcript.topic,
    }

    txt_path = audio_path.with_suffix(".transcript.txt")
    out_text = transcript.labeled_text or transcript.raw_text
    txt_path.write_text(out_text, encoding="utf-8")
    print(f"  Transcript saved: {txt_path.name}")

    # Display results
    print(f"\n  RESULTS:")
    print(f"  Recording type: {result.get('recording_type', 'unknown')}")
    print(f"  Topic: {result.get('topic', 'unknown')}")
    print(f"  Speakers found: {len(result.get('speakers', []))}")
    for s in result.get("speakers", []):
        conf = s.get("confidence", 0)
        print(f"    -> {s.get('name', '?') or s.get('speaker_id', '?')} ({s.get('role', '?')}) "
              f"[confidence: {conf}]")

    return result


def main():
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if not path.exists():
            print(f"File not found: {path}")
            sys.exit(1)
        results = {path.name: process_file(path)}
    else:
        if not AUDIO_DIR.exists():
            print(f"Audio directory not found: {AUDIO_DIR}")
            sys.exit(1)

        audio_files = sorted(
            p for p in AUDIO_DIR.iterdir()
            if p.suffix.lower() in (".mp3", ".wav", ".m4a", ".flac", ".ogg")
        )

        if not audio_files:
            print("No audio files found.")
            sys.exit(1)

        print(f"Found {len(audio_files)} audio file(s):\n")
        for i, f in enumerate(audio_files, 1):
            print(f"  {i}. {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")

        results = {}
        for audio_path in audio_files:
            results[audio_path.name] = process_file(audio_path)

    # Save all results
    out_path = AUDIO_DIR / "speaker_identification.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"All results saved to: {out_path}")


if __name__ == "__main__":
    main()
