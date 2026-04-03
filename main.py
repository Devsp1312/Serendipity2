"""
Serendipity CLI — headless pipeline runner.

This is the command-line version of the app. It runs the same three-phase
pipeline as the web interface but prints everything to the terminal instead.

Usage:
    python main.py                        # Uses built-in mock transcript
    python main.py --transcript path.txt  # Uses your own transcript file
    python main.py --model llama3.2       # Override which Ollama model to use
"""

# Set up logging before any other imports so all modules get the handlers
from src.logger import setup_logging
setup_logging()

import argparse
import json
import sys
import time
from pathlib import Path

from src.config import DEFAULT_MODEL, DEFAULT_PROFILE
from src.ingestion import parse_transcript, get_turn_stats, format_transcript_for_llm
from src.diarizer import run_diarization_pipeline
from src.extraction import run_extraction
from src.gatekeeper import run_gatekeeper, apply_identity
from src.multi_extract import run_multi_pass_pipeline
from src.graph_store import (
    get_graph_path,
    get_profile_summary,
    load_graph,
    save_graph,
    save_snapshot,
)
from src.llm_client import check_connection, list_models, OllamaUnavailableError
from src.logger import get_logger
from src.mock_data import MOCK_TRANSCRIPT
from src.schemas import LLMSchemaError
from src.telemetry import get_collector

logger = get_logger(__name__)

BANNER = "=" * 60


def _section(title: str) -> None:
    """Prints a visual separator with a title — just for readability in the terminal."""
    print(f"\n{BANNER}")
    print(f"  {title}")
    print(BANNER)


def run_pipeline(transcript_text: str, model: str, profile: str) -> None:
    """
    Runs the multi-pass pipeline on a single transcript.

    Phase 1 — Ingestion:    Parse the raw text into structured speaker turns.
    Phase 2 — Multi-pass:   Pre-pass + focused extractions + dedup + identity.
    Phase 3 — Gatekeeper:   Ask the LLM what to add/update/remove in the knowledge graph.
    """
    pipeline_start = time.monotonic()
    run = get_collector().new_run(model=model)

    # ── Phase 1: Ingestion ────────────────────────────────────────────────────
    _section("Phase 1 — Transcript Ingestion")
    turns = parse_transcript(transcript_text)
    stats = get_turn_stats(turns)

    run.transcript_chars = len(transcript_text)
    run.turns_parsed     = len(turns)
    run.unique_speakers  = len(stats)

    print(f"  Turns parsed: {len(turns)}")
    for speaker, count in stats.items():
        print(f"    {speaker}: {count} turn(s)")

    # ── Phase 2: Multi-pass extraction ────────────────────────────────────────
    _section("Phase 2 — Multi-Pass Focused Extraction")
    print(f"  Model: {model}")
    diarized_text = format_transcript_for_llm(turns)

    G = load_graph(profile=profile)
    run.nodes_before = G.number_of_nodes()
    print(f"  Loaded graph ({profile}): {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    t2 = time.monotonic()
    dedup_output, identity_output, raw_logs = run_multi_pass_pipeline(
        diarized_text=diarized_text,
        model=model,
        existing_graph=G,
    )
    run.extraction_latency_sec = round(time.monotonic() - t2, 2)

    run.core_values_found   = len(dedup_output.core_values)
    run.goals_found         = len(dedup_output.long_term_goals)
    run.states_found        = len(dedup_output.short_term_values)
    run.relationships_found = len(dedup_output.relationships)

    for log_entry in raw_logs:
        print(f"\n{log_entry[:500]}")

    print(f"\n  Core values found:      {run.core_values_found}")
    print(f"  Long-term goals found:  {run.goals_found}")
    print(f"  Short-term values:      {run.states_found}")
    print(f"  Interests found:        {len(dedup_output.interests)}")
    print(f"  Relationships mapped:   {run.relationships_found}")
    custom_count = sum(len(items) for items in dedup_output.custom_categories.values())
    if custom_count:
        print(f"  Custom category items:  {custom_count}")
    if dedup_output.cross_category_insights:
        print(f"  Cross-category insights: {len(dedup_output.cross_category_insights)}")
    print(f"  Extraction time:        {run.extraction_latency_sec}s")

    identity = identity_output.identity
    if identity.name or identity.age or identity.occupation or identity.location:
        print(f"\n  Identity:")
        if identity.name:       print(f"    Name:       {identity.name} ({identity.name_confidence:.1f})")
        if identity.age:        print(f"    Age:        {identity.age} ({identity.age_confidence:.1f})")
        if identity.occupation: print(f"    Occupation: {identity.occupation} ({identity.occupation_confidence:.1f})")
        if identity.location:   print(f"    Location:   {identity.location} ({identity.location_confidence:.1f})")

    # ── Phase 3: Gatekeeper ───────────────────────────────────────────────────
    _section("Phase 3 — Gatekeeper / Delta Update")

    t3 = time.monotonic()
    G, raw_gatekeeper = run_gatekeeper(G, dedup_output, model=model)
    run.gatekeeper_latency_sec = round(time.monotonic() - t3, 2)

    print("\n--- LLM REASONING (Gatekeeper) ---")
    try:
        print(json.dumps(json.loads(raw_gatekeeper), indent=2))
    except Exception:
        print(raw_gatekeeper)
    print("-----------------------------------")

    # Apply identity info to user node
    apply_identity(G, identity_output)
    run.nodes_after = G.number_of_nodes()

    save_graph(G, model=model, profile=profile)
    snapshot_path = save_snapshot(G, model=model, profile=profile)
    graph_path = get_graph_path(profile)

    print(f"\n  Updated graph:  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Profile:        {profile}")
    print(f"  Saved to:       {graph_path}")
    print(f"  Snapshot:       {snapshot_path}")
    print(f"  Gatekeeper time: {run.gatekeeper_latency_sec}s")

    # ── Summary ───────────────────────────────────────────────────────────────
    _section("Profile Summary")
    summary = get_profile_summary(G)
    print(json.dumps(summary, indent=2))

    total_elapsed = round(time.monotonic() - pipeline_start, 2)
    print(f"\n  Total pipeline time: {total_elapsed}s")
    if run.coercions_triggered:
        print(f"  ⚠ Enum coercions:  {run.coercions_triggered} (check logs/serendipity.log)")

    logger.info(
        "Pipeline complete  model=%s  elapsed=%.2fs  nodes=%d->%d  coercions=%d",
        model, total_elapsed, run.nodes_before, run.nodes_after, run.coercions_triggered,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Serendipity pipeline CLI")
    parser.add_argument("--transcript", type=str, default=None, help="Path to a .txt transcript file")
    parser.add_argument("--audio",      type=str, default=None, help="Path to an audio file for diarization + speaker ID")
    parser.add_argument("--hf-token",   type=str, default=None, help="Optional HuggingFace token for pyannote diarization")
    parser.add_argument("--model",      type=str, default=None, help="Ollama model name")
    parser.add_argument("--profile",    type=str, default=DEFAULT_PROFILE, help="Profile name (e.g. ted, sal, dev)")
    args = parser.parse_args()

    print("\n" + BANNER)
    print("  SERENDIPITY — Personal Knowledge Builder")
    print(BANNER)

    logger.info("CLI started")

    if not check_connection():
        msg = "Cannot reach Ollama. Ensure it is running: ollama serve"
        logger.critical(msg)
        print(f"\n[FATAL] {msg}")
        sys.exit(1)

    if args.model:
        model = args.model
    else:
        available = list_models()
        if not available:
            msg = "No Ollama models found. Pull one first: ollama pull llama3"
            logger.critical(msg)
            print(f"\n[FATAL] {msg}")
            sys.exit(1)
        model = available[0]
        print(f"\n  Using model: {model}")
        if len(available) > 1:
            print(f"  Other available: {', '.join(available[1:])}")

    if args.audio:
        try:
            print(f"\n  Audio: {args.audio}")
            print("  Running diarization + speaker identification…")
            diarized = run_diarization_pipeline(
                audio_path=args.audio,
                filename=Path(args.audio).name,
                hf_token=args.hf_token,
            )
            transcript_text = diarized.labeled_text or diarized.raw_text
            print(f"  Speakers detected: {len(diarized.speakers)}")
            print(f"  Transcript turns:  {len(transcript_text.splitlines())}")
        except FileNotFoundError:
            msg = f"Audio file not found: {args.audio}"
            logger.critical(msg)
            print(f"\n[FATAL] {msg}")
            sys.exit(1)
    elif args.transcript:
        try:
            with open(args.transcript, "r", encoding="utf-8") as f:
                transcript_text = f.read()
            print(f"\n  Transcript: {args.transcript}")
        except FileNotFoundError:
            msg = f"Transcript file not found: {args.transcript}"
            logger.critical(msg)
            print(f"\n[FATAL] {msg}")
            sys.exit(1)
    else:
        transcript_text = MOCK_TRANSCRIPT
        print("\n  Using built-in mock transcript.")

    profile = args.profile.strip().lower() if args.profile else DEFAULT_PROFILE
    print(f"\n  Profile: {profile}")

    try:
        run_pipeline(transcript_text, model=model, profile=profile)
    except OllamaUnavailableError as e:
        logger.critical("OllamaUnavailableError: %s", e)
        print(f"\n[FATAL] {e}")
        sys.exit(1)
    except LLMSchemaError as e:
        logger.critical("LLMSchemaError: %s  raw=%s", e, getattr(e, "raw_response", "")[:200])
        print(f"\n[FATAL] LLM returned invalid schema: {e}")
        if hasattr(e, "raw_response") and e.raw_response:
            print(f"  Raw response: {e.raw_response[:500]}")
        sys.exit(2)

    print(f"\n{BANNER}")
    print("  Pipeline complete.")
    print(BANNER + "\n")


if __name__ == "__main__":
    main()
