"""
Batch pipeline runner — processes multiple transcript files in sequence.

Handles SRT subtitle files (the standard Friends / TV episode format) as well
as plain speaker-labelled transcripts. Each file is run through the full
Serendipity pipeline and updates the knowledge graph cumulatively.

Usage:
    python3 batch_run.py                          # process friends txt/ folder
    python3 batch_run.py --folder "friends txt"   # explicit folder
    python3 batch_run.py --folder "friends txt" --model mistral:7b
    python3 batch_run.py --folder "friends txt" --dry-run  # parse only, no LLM
    python3 batch_run.py --folder "friends txt" --start-day 5  # resume from day 5
"""

from src.logger import setup_logging
setup_logging()

import argparse
import re
import sys
import time
from pathlib import Path

from src.config import DEFAULT_MODEL, DEFAULT_PROFILE
from src.graph_store import (
    get_graph_path,
    load_graph,
    save_graph,
    save_snapshot,
    get_profile_summary,
)
from src.ingestion import parse_transcript, get_turn_stats, format_transcript_for_llm
from src.extraction import run_extraction
from src.gatekeeper import run_gatekeeper
from src.llm_client import check_connection, list_models, OllamaUnavailableError
from src.logger import get_logger
from src.schemas import LLMSchemaError
from src.telemetry import get_collector

logger = get_logger(__name__)

BANNER    = "=" * 62
THIN_LINE = "─" * 62


# ─── SRT converter ────────────────────────────────────────────────────────────

def srt_to_transcript(raw: str) -> str:
    """
    Converts an SRT subtitle file to the speaker-labelled format expected
    by the pipeline: "SpeakerName: text content".

    Strategy:
    - Parse SRT blocks (index / timestamp / text)
    - Lines already containing "SPEAKER: text" are preserved
    - Lines starting with "- " indicate overlapping dialogue (two speakers
      in the same frame) — each half is labelled "Speaker:"
    - Pure sound effects [IN BRACKETS] are skipped
    - Everything else is grouped under "Speaker:" labels, merging adjacent
      lines that belong to the same subtitle block

    The resulting transcript is suitable for passing straight to parse_transcript().
    """
    # ── 1. Split into blocks ─────────────────────────────────────────────────
    blocks: list[str] = []
    current: list[str] = []

    for raw_line in raw.splitlines():
        stripped = raw_line.strip()
        if stripped == "":
            if current:
                blocks.append("\n".join(current))
                current = []
        else:
            current.append(stripped)
    if current:
        blocks.append("\n".join(current))

    # ── 2. Convert each block to labelled turns ───────────────────────────────
    output_lines: list[str] = []
    _SRT_INDEX     = re.compile(r"^\d+$")
    _SRT_TIMESTAMP = re.compile(r"\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}")
    _SPEAKER_LABEL = re.compile(r"^([A-Z][A-Z .'\-]{0,30}):\s*(.*)")
    _SOUND_EFFECT  = re.compile(r"^\[.*\]$")
    _OVERLAP_LINE  = re.compile(r"^-\s+(.+)")

    for block in blocks:
        lines = block.splitlines()

        # Skip pure index or timestamp blocks
        if len(lines) == 1 and (_SRT_INDEX.match(lines[0]) or _SRT_TIMESTAMP.match(lines[0])):
            continue
        if len(lines) >= 2 and _SRT_INDEX.match(lines[0]) and _SRT_TIMESTAMP.match(lines[1]):
            lines = lines[2:]   # strip the index + timestamp header

        text_lines: list[str] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if _SRT_INDEX.match(line) or _SRT_TIMESTAMP.match(line):
                continue
            if _SOUND_EFFECT.match(line):
                continue
            text_lines.append(line)

        if not text_lines:
            continue

        # Try to parse each text line for speaker labels / overlap markers
        for tl in text_lines:
            speaker_match  = _SPEAKER_LABEL.match(tl)
            overlap_match  = _OVERLAP_LINE.match(tl)

            if speaker_match:
                name, rest = speaker_match.group(1).strip(), speaker_match.group(2).strip()
                name = name.title()
                if rest:
                    output_lines.append(f"{name}: {rest}")
                # If the label has no following text, the next block will continue it
            elif overlap_match:
                # "- text" format — just emit as Speaker:
                output_lines.append(f"Speaker: {overlap_match.group(1).strip()}")
            else:
                # Plain dialogue with no speaker label — check if last line has same speaker
                # to continue it, otherwise emit as Speaker:
                if output_lines:
                    last = output_lines[-1]
                    last_speaker = last.split(":")[0]
                    output_lines[-1] = last + " " + tl
                else:
                    output_lines.append(f"Speaker: {tl}")

    return "\n".join(output_lines)


def is_srt_file(text: str) -> bool:
    """Returns True if the text looks like an SRT subtitle file."""
    timestamp_pattern = re.compile(r"\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}")
    matches = timestamp_pattern.findall(text[:2000])
    return len(matches) >= 3


# ─── File discovery ───────────────────────────────────────────────────────────

def find_transcripts(folder: Path) -> list[Path]:
    """
    Returns all .txt files in the folder, sorted numerically by day number.
    Files are expected to be named something like "day 1.txt", "day 2.txt" etc.
    Falls back to alphabetical sort if no day numbers are found.
    """
    files = list(folder.glob("*.txt"))
    if not files:
        return []

    def sort_key(p: Path) -> int:
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else 999

    return sorted(files, key=sort_key)


# ─── Pretty printing ──────────────────────────────────────────────────────────

def _print_banner(text: str) -> None:
    print(f"\n{BANNER}")
    print(f"  {text}")
    print(BANNER)


def _print_result_line(icon: str, label: str, value) -> None:
    print(f"  {icon}  {label:<28} {value}")


# ─── Single-file pipeline ─────────────────────────────────────────────────────

def run_one(
    filepath: Path,
    model: str,
    day_num: int,
    total: int,
    dry_run: bool,
    profile: str,
) -> dict:
    """
    Runs the full pipeline on a single transcript file.
    Returns a result dict with counts and timings.
    """
    label = f"Day {day_num}/{total} — {filepath.name}"
    _print_banner(label)

    raw = filepath.read_text(encoding="utf-8", errors="replace")

    # ── Convert SRT if needed ──────────────────────────────────────────────────
    if is_srt_file(raw):
        print("  ℹ  SRT format detected — converting to speaker format…")
        raw = srt_to_transcript(raw)

    # ── Phase 1: Ingestion ─────────────────────────────────────────────────────
    try:
        turns = parse_transcript(raw)
    except ValueError as e:
        print(f"  ✗  Ingestion failed: {e}")
        return {"day": day_num, "file": filepath.name, "status": "skip", "error": str(e)}

    stats = get_turn_stats(turns)
    speaker_summary = "  ·  ".join(f"{s}: {c}" for s, c in list(stats.items())[:6])
    print(f"  ✓  Parsed {len(turns)} turns  —  {speaker_summary}")

    if dry_run:
        print("  ⏭  Dry-run mode — skipping LLM phases")
        return {"day": day_num, "file": filepath.name, "status": "dry-run", "turns": len(turns)}

    # ── Phase 2: Extraction ────────────────────────────────────────────────────
    run = get_collector().new_run(model=model)
    run.turns_parsed    = len(turns)
    run.unique_speakers = len(stats)

    print(f"  ⟳  Extraction…")
    t2 = time.monotonic()
    try:
        extraction, _ = run_extraction(format_transcript_for_llm(turns), model=model)
    except (OllamaUnavailableError, LLMSchemaError) as e:
        print(f"  ✗  Extraction failed: {e}")
        return {"day": day_num, "file": filepath.name, "status": "error", "error": str(e)}
    run.extraction_latency_sec = round(time.monotonic() - t2, 2)
    run.core_values_found      = len(extraction.core_values)
    run.goals_found            = len(extraction.long_term_goals)
    run.states_found           = len(extraction.short_term_values)
    run.relationships_found    = len(extraction.relationships)

    print(f"     values={run.core_values_found}  goals={run.goals_found}  "
          f"states={run.states_found}  rels={run.relationships_found}  "
          f"({run.extraction_latency_sec}s)")

    # ── Phase 3: Gatekeeper ────────────────────────────────────────────────────
    print(f"  ⟳  Gatekeeper…")
    G = load_graph(profile=profile)
    run.nodes_before = G.number_of_nodes()
    t3 = time.monotonic()
    try:
        G, _ = run_gatekeeper(G, extraction, model=model)
    except (OllamaUnavailableError, LLMSchemaError) as e:
        print(f"  ✗  Gatekeeper failed: {e}")
        return {"day": day_num, "file": filepath.name, "status": "error", "error": str(e)}
    run.gatekeeper_latency_sec = round(time.monotonic() - t3, 2)
    run.nodes_after = G.number_of_nodes()

    save_graph(G, model=model, profile=profile)
    snap = save_snapshot(G, model=model, profile=profile)

    node_delta = run.nodes_after - run.nodes_before
    delta_str  = f"+{node_delta}" if node_delta >= 0 else str(node_delta)
    total_sec  = round(run.extraction_latency_sec + run.gatekeeper_latency_sec, 2)

    print(f"     nodes: {run.nodes_before} → {run.nodes_after} ({delta_str})  "
          f"gatekeeper: {run.gatekeeper_latency_sec}s  total: {total_sec}s")
    print(f"  💾  {snap.name}")

    return {
        "day":        day_num,
        "file":       filepath.name,
        "status":     "ok",
        "turns":      len(turns),
        "values":     run.core_values_found,
        "goals":      run.goals_found,
        "states":     run.states_found,
        "rels":       run.relationships_found,
        "nodes_before": run.nodes_before,
        "nodes_after":  run.nodes_after,
        "elapsed_sec":  total_sec,
    }


# ─── Batch runner ─────────────────────────────────────────────────────────────

def run_batch(folder: Path, model: str, dry_run: bool, start_day: int, profile: str) -> None:
    files = find_transcripts(folder)
    if not files:
        print(f"\n✗  No .txt files found in: {folder}")
        sys.exit(1)

    # Filter to start_day if requested
    if start_day > 1:
        files = [f for i, f in enumerate(files, 1) if i >= start_day]
        if not files:
            print(f"\n✗  No files at or after day {start_day}")
            sys.exit(1)

    total = len(files)

    _print_banner("SERENDIPITY — Batch Runner")
    print(f"  Folder:    {folder}")
    print(f"  Files:     {total} transcript(s)")
    print(f"  Model:     {model}")
    print(f"  Profile:   {profile}")
    if dry_run:
        print(f"  Mode:      DRY RUN (no LLM calls)")
    if start_day > 1:
        print(f"  Starting:  day {start_day}")

    if not dry_run:
        print(f"\n  Checking Ollama connection…", end=" ", flush=True)
        if not check_connection():
            print("✗")
            print("\n  [FATAL] Ollama is not running. Start it with: ollama serve")
            sys.exit(1)
        print("✓")

    # ── Run each file ─────────────────────────────────────────────────────────
    batch_start = time.monotonic()
    results = []
    for i, filepath in enumerate(files, start=start_day if start_day > 1 else 1):
        result = run_one(
            filepath,
            model,
            i,
            start_day - 1 + total if start_day > 1 else total,
            dry_run,
            profile,
        )
        results.append(result)

        # Pause between episodes so the model doesn't get hammered
        if not dry_run and i < (start_day - 1 + total if start_day > 1 else total):
            time.sleep(1.0)

    # ── Final summary ─────────────────────────────────────────────────────────
    batch_elapsed = round(time.monotonic() - batch_start, 1)
    ok_results = [r for r in results if r["status"] == "ok"]
    skip_results = [r for r in results if r["status"] in ("skip", "error")]

    _print_banner("Batch Complete")

    print(f"  Files processed:  {len(results)}")
    if ok_results:
        print(f"  Successful:       {len(ok_results)}")
        total_turns  = sum(r.get("turns", 0) for r in ok_results)
        total_values = sum(r.get("values", 0) for r in ok_results)
        total_goals  = sum(r.get("goals", 0) for r in ok_results)
        total_rels   = sum(r.get("rels", 0) for r in ok_results)
        print(f"  Total turns:      {total_turns}")
        print(f"  Values extracted: {total_values}")
        print(f"  Goals extracted:  {total_goals}")
        print(f"  Rels extracted:   {total_rels}")

    if skip_results:
        print(f"  Skipped/errored:  {len(skip_results)}")
        for r in skip_results:
            print(f"    ✗  {r['file']}: {r.get('error', r['status'])}")

    if not dry_run and ok_results:
        G_final = load_graph(profile=profile)
        summary = get_profile_summary(G_final)
        print(f"\n  Final graph:  {G_final.number_of_nodes()} nodes")
        print(f"  Graph path:   {get_graph_path(profile)}")

        if summary["core_values"]:
            top = sorted(summary["core_values"], key=lambda x: -x["confidence"])[:5]
            print(f"\n  Top core values:")
            for v in top:
                bar = "█" * int(v["confidence"] * 10) + "░" * (10 - int(v["confidence"] * 10))
                print(f"    {bar}  {v['label'].title():<30} {v['confidence']:.0%}")

        if summary["relationships"]:
            print(f"\n  Relationships mapped: {len(summary['relationships'])}")
            for r in summary["relationships"][:8]:
                print(f"    • {r['name'].title():<20} {r['relationship_type']:<12} {r['tone']}")

        if summary["long_term_goals"]:
            print(f"\n  Long-term goals:")
            for g in summary["long_term_goals"][:5]:
                print(f"    → {g['label'].capitalize()}")

    print(f"\n  Total time: {batch_elapsed}s")
    print(f"\n  View the full profile: streamlit run app.py")
    print(BANNER + "\n")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Serendipity pipeline on a folder of transcript files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 batch_run.py
  python3 batch_run.py --folder "friends txt" --model mistral:7b
  python3 batch_run.py --folder "friends txt" --dry-run
  python3 batch_run.py --folder "friends txt" --start-day 10
        """,
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="friends txt",
        help='Folder containing transcript files (default: "friends txt")',
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ollama model name (default: first available)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse transcripts only — no LLM calls, no graph updates",
    )
    parser.add_argument(
        "--start-day",
        type=int,
        default=1,
        help="Resume from this day number (skips earlier files)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=DEFAULT_PROFILE,
        help="Profile name to update (default: ted)",
    )

    args = parser.parse_args()
    folder = Path(args.folder)

    if not folder.exists():
        print(f"\n✗  Folder not found: {folder}")
        print(f"   Looking from: {Path.cwd()}")
        sys.exit(1)

    # Resolve model
    model = args.model
    if model is None and not args.dry_run:
        available = list_models()
        if not available:
            print("\n✗  No Ollama models found. Pull one first: ollama pull llama3")
            sys.exit(1)
        model = available[0]
    elif model is None:
        model = DEFAULT_MODEL

    run_batch(
        folder=folder,
        model=model,
        dry_run=args.dry_run,
        start_day=args.start_day,
        profile=args.profile.strip().lower() if args.profile else DEFAULT_PROFILE,
    )


if __name__ == "__main__":
    main()
