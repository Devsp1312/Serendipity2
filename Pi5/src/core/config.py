"""
Centralized configuration for Serendipity.

All tunable constants live here. Override any value via environment variables —
no code changes needed for local customization.

WHY this file exists in the pipeline:
  Every stage of the pipeline (transcription, diarization, extraction, gatekeeper,
  graph storage, visualizer) needs shared constants — model names, file paths,
  confidence thresholds, visual colors. Centralizing them here means:
    - A single place to tune behavior for a demo or deployment
    - No magic numbers scattered across modules
    - Environment-variable overrides work uniformly (no per-module env parsing)
    - Tests can monkeypatch a single module rather than hunting constants everywhere

Sections (in pipeline order):
  Project root      — base path for all relative lookups
  Data paths        — where profiles, audio, logs, and prompts live on disk
  LLM defaults      — model name + retry/timeout tuning for all LLM calls
  Gatekeeper        — confidence starting value and how fast it grows
  Multi-pass ext.   — how many extraction passes to run and their categories
  Speaker diarize   — pyannote model download token + speaker-ID model choice
  Ingestion         — safety limits on transcript size / speaker labels
  Transcriber       — Whisper model variant and inference settings
  Snapshot          — how many graph history rows to keep
  Logging           — log verbosity, rotation size, and backup count
  Node type / edge  — ID-prefix and relation-name mappings shared by gatekeeper & graph
  Visualizer        — Three.js layout constants and per-type node colors
  Streamlit keys    — session_state key names (prevents silent typo bugs)

Usage:
    from src.core.config import DEFAULT_MODEL, PROFILES_DIR
"""

import os
from pathlib import Path


# ── Project root ───────────────────────────────────────────────────────────────
# Computed from __file__ so it works regardless of working directory (important
# for pytest runs from any location).
PROJECT_ROOT = Path(__file__).parent.parent.parent  # src/core/config.py → src/core/ → src/ → project root


# ── Data paths ─────────────────────────────────────────────────────────────────
# These paths define where pipeline artifacts land on disk.
# DATA_DIR   — root for all generated data (graphs, snapshots)
# PROFILES_DIR — one subdirectory per named profile (e.g. data/profiles/friends/)
# AUDIO_DIR  — where raw audio files are read from by the batch runner
# PROMPTS_DIR — prompt text files read by llm_client.py at call time
DATA_DIR      = PROJECT_ROOT / os.environ.get("SERENDIPITY_DATA_DIR", "data")
PROFILES_DIR  = DATA_DIR / "profiles"
AUDIO_DIR     = PROJECT_ROOT / os.environ.get("SERENDIPITY_AUDIO_DIR", "Friends")
DEFAULT_PROFILE = os.environ.get("SERENDIPITY_DEFAULT_PROFILE", "friends")
PROFILE_PRESETS = tuple(
    p.strip().lower()
    for p in os.environ.get("SERENDIPITY_PROFILE_PRESETS", "friends").split(",")
    if p.strip()
)
LOGS_DIR      = PROJECT_ROOT / os.environ.get("SERENDIPITY_LOGS_DIR", "logs")
PROMPTS_DIR   = PROJECT_ROOT / "prompts"


# ── LLM defaults ───────────────────────────────────────────────────────────────
# Used by llm_client.call_llm() for every LLM call in the pipeline.
# LLM_MAX_RETRIES + exponential-backoff settings guard against transient Ollama
# timeouts which are common when the model is being loaded for the first time.
DEFAULT_MODEL      = os.environ.get("SERENDIPITY_DEFAULT_MODEL", "qwen3.5")
LLM_MAX_RETRIES    = int(os.environ.get("SERENDIPITY_MAX_RETRIES", "3"))
LLM_BASE_DELAY_SEC = float(os.environ.get("SERENDIPITY_BASE_DELAY", "1.0"))
LLM_MAX_DELAY_SEC  = float(os.environ.get("SERENDIPITY_MAX_DELAY", "30.0"))
LLM_BACKOFF_FACTOR = float(os.environ.get("SERENDIPITY_BACKOFF_FACTOR", "2.0"))
LLM_TIMEOUT_SEC    = float(os.environ.get("SERENDIPITY_LLM_TIMEOUT", "180.0"))


# ── Gatekeeper ─────────────────────────────────────────────────────────────────
# Controls how graph confidence values evolve over time.
# DEFAULT_CONFIDENCE — starting weight when a new node is first added to the graph
# STRENGTHEN_INCREMENT — how much confidence increases each time the same trait is
#   seen again in a new recording (capped at CONFIDENCE_CAP = 1.0)
DEFAULT_CONFIDENCE   = float(os.environ.get("SERENDIPITY_DEFAULT_CONFIDENCE", "0.6"))
STRENGTHEN_INCREMENT = float(os.environ.get("SERENDIPITY_STRENGTHEN_INC", "0.1"))
CONFIDENCE_CAP       = 1.0


# ── Multi-pass extraction ─────────────────────────────────────────────────────
# The extraction pipeline runs multiple focused LLM passes instead of one big call.
# MAX_EXTRA_PASSES — the pre-pass LLM can invent custom categories; this caps how
#   many it can schedule beyond the 3 mandatory core categories below.
# CORE_EXTRACTION_CATEGORIES — always run regardless of what the pre-pass decides;
#   these map directly to node types in the graph.
MAX_EXTRA_PASSES          = int(os.environ.get("SERENDIPITY_MAX_EXTRA_PASSES", "2"))
IDENTITY_MIN_CONFIDENCE   = float(os.environ.get("SERENDIPITY_IDENTITY_MIN_CONFIDENCE", "0.3"))
CORE_EXTRACTION_CATEGORIES = ("short_term_state", "long_term_goal", "core_value")


# ── Speaker diarization ───────────────────────────────────────────────────────
# Model for identifying who each SPEAKER_XX is (must produce clean JSON, no thinking tags)
SPEAKER_ID_MODEL = os.environ.get("SERENDIPITY_SPEAKER_ID_MODEL", "mistral:7b")
# HuggingFace token for one-time pyannote model download (read from file or env)
HF_TOKEN_PATH    = Path.home() / ".huggingface" / "token"


# ── Ingestion ──────────────────────────────────────────────────────────────────
# Hard limits that protect the pipeline from excessively large inputs.
# MAX_TRANSCRIPT_BYTES — rejects files larger than 5 MB before any LLM call is made
# MAX_SPEAKER_LABEL_LENGTH — truncates abnormally long speaker labels from diarization
MAX_TRANSCRIPT_BYTES     = int(os.environ.get("SERENDIPITY_MAX_TRANSCRIPT_BYTES", str(5 * 1024 * 1024)))  # 5 MB
MAX_SPEAKER_LABEL_LENGTH = int(os.environ.get("SERENDIPITY_MAX_SPEAKER_LEN", "64"))


# ── Transcriber ────────────────────────────────────────────────────────────────
# faster-whisper settings. "base" is fast enough for real-time demo use;
# switch to "large-v3" for production accuracy at the cost of speed.
WHISPER_MODEL    = os.environ.get("SERENDIPITY_WHISPER_MODEL", "base")
WHISPER_DEVICE   = os.environ.get("SERENDIPITY_WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE  = os.environ.get("SERENDIPITY_WHISPER_COMPUTE", "int8")
WHISPER_BEAM_SIZE = int(os.environ.get("SERENDIPITY_BEAM_SIZE", "5"))


# ── Snapshot retention ─────────────────────────────────────────────────────────
# The graph DB stores a full snapshot row after every pipeline run (for rollback).
# MAX_SNAPSHOTS caps how many rows to keep so the DB doesn't grow unboundedly.
MAX_SNAPSHOTS = int(os.environ.get("SERENDIPITY_MAX_SNAPSHOTS", "100"))


# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL        = os.environ.get("SERENDIPITY_LOG_LEVEL", "INFO")
LOG_MAX_BYTES    = int(os.environ.get("SERENDIPITY_LOG_MAX_BYTES", str(5 * 1024 * 1024)))  # 5 MB
LOG_BACKUP_COUNT = int(os.environ.get("SERENDIPITY_LOG_BACKUP_COUNT", "5"))


# ── Graph node type / edge relation mappings ───────────────────────────────────
# Centralised here so both graph_store.py and gatekeeper.py import from one place.

# Used by make_node_id() to create collision-free IDs across node types.
# e.g. "discipline" as a core_value -> "cv_discipline", as a goal -> "ltg_discipline".
NODE_TYPE_PREFIX = {
    "core_value":       "cv",
    "long_term_goal":   "ltg",
    "short_term_state": "sts",
    "person":           "person",
}

# Maps LLM-facing node types to edge relation names.
# Note: the key "relationship" is what the LLM outputs, but the graph stores these
# as node_type="person". The translation happens in gatekeeper.py's apply_actions().
RELATION_FOR_TYPE = {
    "core_value":       "holds_value",
    "long_term_goal":   "pursues_goal",
    "short_term_state": "experiencing",
    "relationship":     "knows",
}


# ── Visualizer ─────────────────────────────────────────────────────────────────
# All Three.js / 3d-force-graph visual parameters are here so the HTML template
# in visualizer.py never needs to be edited for visual tweaks.
# VIZ_NODE_RADIUS_* — controls how far nodes sit from the centre (inverse of confidence)
# VIZ_NODE_SIZE_*   — sphere diameter scales linearly with confidence
# VIZ_LINK_WIDTH_*  — beam thickness scales with confidence
# VIZ_PARTICLE_SPEED_* — animated directional particles along each beam
VIZ_DEFAULT_HEIGHT       = int(os.environ.get("SERENDIPITY_VIZ_HEIGHT", "700"))
VIZ_STAR_COUNT           = 180
VIZ_NODE_RADIUS_MIN      = 40.0
VIZ_NODE_RADIUS_SPAN     = 100.0
VIZ_NODE_SIZE_BASE       = 3.0
VIZ_NODE_SIZE_SPAN       = 5.0
VIZ_LINK_WIDTH_BASE      = 2.0
VIZ_LINK_WIDTH_SPAN      = 6.0
VIZ_PARTICLE_SPEED_BASE  = 0.002
VIZ_PARTICLE_SPEED_SPAN  = 0.005
VIZ_CAMERA_X             = 80
VIZ_CAMERA_Y             = 90
VIZ_CAMERA_Z             = 200

# Per-type node colors rendered in the 3D scene and the legend overlay.
# These are also used by build_graph_data() to color-code nodes by what they represent.
NODE_COLORS = {
    "user":             "#FFFFFF",
    "core_value":       "#00B4D8",   # cyan
    "long_term_goal":   "#06D6A0",   # teal-green
    "short_term_state": "#EF476F",   # coral-red
    "person":           "#FFD166",   # gold
}
VIZ_DEFAULT_COLOR  = "#AAAAAA"
VIZ_CUSTOM_COLOR   = "#B388FF"  # light purple for model-invented custom categories


# ── Streamlit session-state keys ───────────────────────────────────────────────
# Defined as constants to prevent silent typo bugs (wrong key → None in st.session_state).
SS_MODEL             = "model"
SS_LOADED_TRANSCRIPT = "loaded_transcript"
SS_TRANSCRIPT_SOURCE = "transcript_source"
SS_LAST_DIFF         = "last_diff"
SS_PIPELINE_LOG      = "pipeline_log"
SS_LAST_RUN_METRICS  = "last_run_metrics"
SS_PROFILE           = "profile"
