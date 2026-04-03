# Serendipity — Codebase Map

A quick guide to every file, how they connect, and where to look when you want to change something.

---

## Project at a glance

```
transcript (text or audio)
        │
        ▼
  ┌─────────────┐
  │  ingestion  │  splits into speaker turns, normalises names
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ extraction  │  LLM pass 1 — "what did I learn about this person?"
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ gatekeeper  │  LLM pass 2 — "what should change in the graph?"
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ graph_store │  save to disk, snapshot, diff, rollback
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  visualizer │  NetworkX → 3D HTML/JS sphere
  └─────────────┘
```

---

## File tree

```
Serendipity/
│
├── app.py                  ← Streamlit UI  (the web app)
├── main.py                 ← CLI runner    (headless, same pipeline)
│
├── src/
│   ├── config.py           ← ALL constants live here (paths, thresholds, colours…)
│   ├── logger.py           ← sets up logging → logs/serendipity.log
│   ├── telemetry.py        ← in-memory session metrics (latency, coercions…)
│   │
│   ├── ingestion.py        ← Phase 1: text → list[Turn]
│   ├── extraction.py       ← Phase 2: turns → ExtractionOutput  (LLM call)
│   ├── gatekeeper.py       ← Phase 3: extraction → graph delta  (LLM call)
│   │
│   ├── llm_client.py       ← Ollama wrapper, retry + backoff, prompt loader
│   ├── schemas.py          ← Pydantic models for everything the LLM returns
│   ├── graph_store.py      ← load/save/snapshot/diff/rollback the graph
│   ├── visualizer.py       ← builds the self-contained 3D HTML widget
│   ├── transcriber.py      ← audio → text via faster-whisper
│   └── mock_data.py        ← built-in demo transcript
│
├── prompts/
│   ├── extraction.txt      ← system prompt for Phase 2 (edit to tune)
│   └── gatekeeper.txt      ← system prompt for Phase 3 (edit to tune)
│
├── tests/
│   ├── conftest.py         ← shared fixtures (sample graphs, mock LLM responses)
│   ├── test_ingestion.py
│   ├── test_schemas.py
│   ├── test_llm_client.py
│   ├── test_graph_store.py
│   ├── test_gatekeeper.py
│   └── test_telemetry.py
│
├── data/                   ← auto-created on first run
│   ├── profile_graph.json  ← the live knowledge graph
│   └── snapshots/          ← timestamped history
│
├── logs/                   ← auto-created on first run
│   └── serendipity.log     ← rotating log file (5 MB × 5 backups)
│
├── START.md                ← how to install and use the app
├── ARCHITECTURE.md         ← deep technical reference
└── CODEBASE.md             ← this file
```

---

## Module responsibilities (one line each)

| File | One-liner |
|------|-----------|
| `config.py` | Every magic number / path in the project. Change things here. |
| `logger.py` | Call `setup_logging()` once at startup; call `get_logger(__name__)` in each module. |
| `telemetry.py` | Tracks per-run metrics (latency, coercions, node delta) in memory. |
| `ingestion.py` | Regex-parses `"Speaker: text"` lines into `Turn` objects; normalises speaker names. |
| `extraction.py` | Formats turns as text, fires an LLM call, returns `ExtractionOutput`. |
| `gatekeeper.py` | Summarises the existing graph, fires a second LLM call, applies the resulting actions. |
| `llm_client.py` | `call_llm()` — handles Ollama chat, JSON retry with exponential backoff, prompt caching. |
| `schemas.py` | Pydantic models for `ExtractionOutput` and `GatekeeperOutput`. Invalid enums → warning + coerce. |
| `graph_store.py` | NetworkX graph ↔ JSON on disk. Snapshots, diffs, and rollback. |
| `visualizer.py` | `build_visualizer_html(G)` → self-contained HTML string (Three.js + 3d-force-graph). |
| `transcriber.py` | Wraps faster-whisper. Gracefully unavailable if not installed. |

---

## "Where do I go to change…"

| I want to change… | Go to |
|-------------------|-------|
| The Ollama model used by default | `config.py` → `DEFAULT_MODEL` |
| How many times the LLM retries bad JSON | `config.py` → `LLM_MAX_RETRIES` |
| How much confidence increases on "strengthen" | `config.py` → `STRENGTHEN_INCREMENT` |
| The maximum transcript file size | `config.py` → `MAX_TRANSCRIPT_BYTES` |
| The Whisper model size (audio transcription) | `config.py` → `WHISPER_MODEL` |
| What the LLM is asked to extract | `prompts/extraction.txt` |
| How the LLM decides to update the graph | `prompts/gatekeeper.txt` |
| Node colours in the 3D graph | `config.py` → `NODE_COLORS` |
| Camera position / star count in the 3D graph | `config.py` → `VIZ_CAMERA_*` / `VIZ_STAR_COUNT` |
| The graph file location | `config.py` → `DATA_DIR` (or env var `SERENDIPITY_DATA_DIR`) |
| Where logs are written | `config.py` → `LOGS_DIR` (or env var `SERENDIPITY_LOGS_DIR`) |
| The valid relationship types | `schemas.py` → `VALID_RELATIONSHIP_TYPES` |
| The valid gatekeeper operations | `schemas.py` → `VALID_OPERATIONS` |
| The Streamlit tab layout | `app.py` → the `st.tabs([…])` block |
| The rollback UI | `app.py` → `# ── Rollback` section in `tab_history` |
| Session metrics display | `app.py` → `# ── Session telemetry` section in `tab_log` |

---

## How the graph is structured

Everything hangs off a single `"user"` anchor node:

```
user ──[holds_value]──► cv_discipline          (core_value)
user ──[pursues_goal]──► ltg_finish_phd        (long_term_goal)
user ──[experiencing]──► sts_stressed_today    (short_term_state)
user ──[knows]──────────► person_alice         (person)
```

**Confidence** lives on the **edge**, not the node:
- `core_value` / `long_term_goal` → `weight`
- `short_term_state` → `intensity`
- `person` → `strength`

**Node IDs** are deterministic slugs: `make_node_id("core_value", "Academic Rigor")` → `"cv_academic_rigor"`. Same label always = same node, no duplicates.

**Short-term states** are wiped on every run — they reflect *today*, not history.

---

## How retries work

```
call_llm() receives bad JSON
        │
        ├── attempt 1 failed → sleep ~1s  → retry
        ├── attempt 2 failed → sleep ~2s  → retry
        ├── attempt 3 failed → sleep ~4s  → retry
        └── attempt 4 failed → raise LLMSchemaError
```

Jitter (random 0–0.5s) is added to each sleep to avoid synchronized retries.
Connection errors (`ollama serve` not running) raise immediately — no retry.

---

## How to run the tests

```bash
python3 -m pytest tests/ -v
```

Expected output: **98 passed** in ~0.1s (all tests mock the LLM — no Ollama needed).

```bash
# Run a single file
python3 -m pytest tests/test_ingestion.py -v

# Show log output during tests
python3 -m pytest tests/ -v -s
```

---

## Environment variable overrides

Set any of these to customise behaviour without editing code:

```bash
export SERENDIPITY_DEFAULT_MODEL=mistral:7b
export SERENDIPITY_MAX_RETRIES=5
export SERENDIPITY_DATA_DIR=/my/custom/data/path
export SERENDIPITY_LOG_LEVEL=DEBUG
export SERENDIPITY_WHISPER_MODEL=small
```

---

## Adding a new node type (recipe)

1. `config.py` — add to `NODE_TYPE_PREFIX` and `RELATION_FOR_TYPE`
2. `config.py` — add a colour to `NODE_COLORS`
3. `schemas.py` — add the new type to `VALID_NODE_TYPES`
4. `graph_store.py` — handle it in `get_profile_summary()`
5. `prompts/extraction.txt` + `prompts/gatekeeper.txt` — mention the new type
6. Write tests in `tests/test_gatekeeper.py` and `tests/test_graph_store.py`

---

## Key data types

```python
Turn               ingestion.py    speaker: str, text: str
ExtractionOutput   schemas.py      core_values, long_term_goals, short_term_states, relationships
GatekeeperAction   schemas.py      operation, node_type, label, confidence, metadata
GatekeeperOutput   schemas.py      actions: list[GatekeeperAction]
NodeDiff           graph_store.py  node_id, label, node_type, old/new confidence, delta
GraphDiff          graph_store.py  added, removed, strengthened, unchanged
RunMetrics         telemetry.py    per-run counters for latency, coercions, node delta
```
