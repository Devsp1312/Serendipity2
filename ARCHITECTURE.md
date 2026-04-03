# Serendipity — Architecture

## Overview

Serendipity is a **local AI-powered personal knowledge graph builder**.
It analyzes conversation transcripts to extract psychological insights (core values, long-term goals,
current states, relationships) and stores them as an evolving 3D knowledge graph — all running
entirely on your machine, with no cloud APIs and no data leaving your device.

---

## System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      Entry Points                               │
│   app.py  (Streamlit web UI)      main.py  (headless CLI)      │
└────────────────────────┬────────────────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │     Phase 1: Ingestion      │  src/ingestion.py
          │  raw text → list[Turn]      │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │    Phase 2: Extraction      │  src/extraction.py
          │  turns → ExtractionOutput   │──── Ollama LLM ──► src/llm_client.py
          └──────────────┬──────────────┘         │
                         │                  prompts/extraction.txt
          ┌──────────────▼──────────────┐
          │    Phase 3: Gatekeeper      │  src/gatekeeper.py
          │  extraction → graph delta   │──── Ollama LLM ──► src/llm_client.py
          └──────────────┬──────────────┘         │
                         │                  prompts/gatekeeper.txt
          ┌──────────────▼──────────────┐
          │      Graph Store            │  src/graph_store.py
          │  NetworkX DiGraph ↔ JSON    │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │      Visualizer             │  src/visualizer.py
          │  DiGraph → 3D HTML/JS       │
          └─────────────────────────────┘
```

---

## Component Reference

### Entry Points

| File | Role |
|------|------|
| `app.py` | Streamlit web UI. Six tabs: 3D Graph, Profile, Upload & Process, History, Raw Data, Log. |
| `main.py` | Headless CLI. Runs the same pipeline and prints results to the terminal. |

### Source Modules (`src/`)

| Module | Role | Key Public Functions |
|--------|------|----------------------|
| `config.py` | All tunable constants. Override via env vars. | — (constants only) |
| `logger.py` | Application-wide structured logging. | `setup_logging()`, `get_logger(name)` |
| `telemetry.py` | In-memory session metrics (no disk writes). | `get_collector()`, `TelemetryCollector` |
| `ingestion.py` | Parses speaker-labeled text into `Turn` objects. Speaker names normalized to Title Case. | `parse_transcript()`, `validate_transcript_input()` |
| `extraction.py` | Phase 2 LLM call. Returns `ExtractionOutput`. | `run_extraction(turns, model)` |
| `gatekeeper.py` | Phase 3 LLM call. Applies graph delta actions. | `run_gatekeeper(G, extraction, model)`, `apply_actions()` |
| `llm_client.py` | Ollama wrapper. Loads prompts from files. Exponential backoff retry. | `call_llm()`, `load_prompt()`, `check_connection()` |
| `schemas.py` | Pydantic models for LLM JSON contracts. Coerces invalid enums with `logger.warning`. | `validate_extraction()`, `validate_gatekeeper()` |
| `graph_store.py` | NetworkX graph I/O, snapshots, diffs, rollback. | `load_graph()`, `save_graph()`, `rollback_to_snapshot()`, `diff_graphs()` |
| `visualizer.py` | Converts graph to self-contained HTML/JS (Three.js + 3d-force-graph). | `build_visualizer_html(G, height)` |
| `transcriber.py` | Optional audio→text using faster-whisper (CPU, no API key). | `transcribe(audio_bytes, filename)`, `is_available()` |
| `mock_data.py` | Built-in demo transcript for first-run experience. | `MOCK_TRANSCRIPT` |

---

## Configuration

All tunable values live in `src/config.py`. Every constant can be overridden with an environment variable — no code changes needed.

| Constant | Env Var | Default | Notes |
|----------|---------|---------|-------|
| `DEFAULT_MODEL` | `SERENDIPITY_DEFAULT_MODEL` | `llama3` | Fallback when no model is specified |
| `LLM_MAX_RETRIES` | `SERENDIPITY_MAX_RETRIES` | `3` | Max JSON retry attempts |
| `LLM_BASE_DELAY_SEC` | `SERENDIPITY_BASE_DELAY` | `1.0` | Backoff base delay |
| `LLM_MAX_DELAY_SEC` | `SERENDIPITY_MAX_DELAY` | `30.0` | Backoff cap |
| `GRAPH_PATH` | `SERENDIPITY_DATA_DIR` | `data/profile_graph.json` | Live graph |
| `SNAPSHOTS_DIR` | `SERENDIPITY_DATA_DIR` | `data/snapshots/` | History |
| `LOGS_DIR` | `SERENDIPITY_LOGS_DIR` | `logs/` | Rotating log files |
| `DEFAULT_CONFIDENCE` | `SERENDIPITY_DEFAULT_CONFIDENCE` | `0.6` | Fallback confidence |
| `STRENGTHEN_INCREMENT` | `SERENDIPITY_STRENGTHEN_INC` | `0.1` | Confidence boost per strengthen |
| `MAX_TRANSCRIPT_BYTES` | `SERENDIPITY_MAX_TRANSCRIPT_BYTES` | 5 MB | Upload size limit |
| `WHISPER_MODEL` | `SERENDIPITY_WHISPER_MODEL` | `base` | Whisper model size |

### Prompts

LLM system prompts live in `prompts/`:
- `prompts/extraction.txt` — Phase 2 extraction instructions
- `prompts/gatekeeper.txt` — Phase 3 gatekeeper instructions

Lines beginning with `#` are treated as metadata comments and stripped before sending to the LLM.
Edit prompt files to tune extraction quality without touching Python code.

---

## Data Flow

### Phase 1 — Ingestion (`src/ingestion.py`)
- Input: raw transcript text (uploaded `.txt` or transcribed audio)
- Validates: non-empty, ≤5 MB, no binary content
- Parses regex `^SpeakerName: text` into `list[Turn]`
- Normalizes speaker names to Title Case → prevents duplicate graph nodes
- Output: `list[Turn]`

### Phase 2 — Extraction (`src/extraction.py` + `src/llm_client.py`)
- Input: `list[Turn]`
- Formats transcript as "Speaker: text\n" string
- Sends to Ollama with `prompts/extraction.txt` system prompt
- LLM returns JSON: `{core_values, long_term_goals, short_term_states, relationships}`
- Validated via `ExtractionOutput` Pydantic model (invalid enums coerced + warned)
- Output: `ExtractionOutput`

### Phase 3 — Gatekeeper (`src/gatekeeper.py` + `src/llm_client.py`)
- Input: current `nx.DiGraph` + `ExtractionOutput`
- Summarizes existing graph via `get_profile_summary()`
- Sends to Ollama with `prompts/gatekeeper.txt` system prompt
- LLM returns JSON: `{actions: [{operation, node_type, label, confidence, metadata}]}`
- Validated via `GatekeeperOutput` Pydantic model
- `apply_actions()` mutates the graph in-place
- Output: updated `nx.DiGraph`

---

## Data Storage

### Live Graph: `data/profile_graph.json`
NetworkX `node_link_data` format. Contains:
- Nodes with `node_type`, `label`/`name`, `evidence_count`
- Edges with `relation`, confidence fields (`weight`, `intensity`, or `strength`)
- Graph-level metadata: `created`, `last_updated`, `last_model`

### Snapshots: `data/snapshots/`
Timestamped copies of the live graph. Filename: `profile_YYYYMMDD_HHMMSS_model.json`.
Snapshots are never deleted automatically (configurable via `MAX_SNAPSHOTS`).

### Logs: `logs/serendipity.log`
Rotating log file (5 MB, 5 backups). Contains all INFO+ events. Useful for:
- Debugging LLM retry loops
- Auditing enum coercions (invalid LLM outputs)
- Performance profiling (per-phase latency)

---

## Graph Schema

### Node Types

| `node_type` | ID Prefix | Confidence Edge Field | Example |
|-------------|-----------|----------------------|---------|
| `core_value` | `cv_` | `weight` | `cv_discipline` |
| `long_term_goal` | `ltg_` | `weight` | `ltg_finish_phd` |
| `short_term_state` | `sts_` | `intensity` | `sts_stressed_about_deadline` |
| `person` | `person_` | `strength` | `person_alice` |

All non-user nodes connect to the `user` anchor node via a single directed edge.

### Edge Types

| `relation` | Connects | Notes |
|------------|----------|-------|
| `holds_value` | user → core_value | |
| `pursues_goal` | user → long_term_goal | |
| `experiencing` | user → short_term_state | Cleared on every pipeline run |
| `knows` | user → person | Also stores `relationship_type`, `tone` |

### Deterministic Node IDs
`make_node_id(node_type, label)` produces stable slugs:
- Unicode normalization (NFKD) → ASCII
- Lowercase, spaces → underscores
- Example: `("core_value", "Academic Rigor")` → `"cv_academic_rigor"`

This prevents duplicate nodes when the LLM uses slightly different casing across runs.

---

## LLM Integration

### Retry Strategy
`call_llm()` uses exponential backoff with jitter on `json.JSONDecodeError`:
- Attempt 1: immediate
- Attempt 2: sleep ~1s
- Attempt 3: sleep ~2s
- Attempt 4: sleep ~4s (capped at 30s)
- After max retries: raise `LLMSchemaError`

Connection errors (`ConnectionError`, `OSError`, `ollama.ResponseError`) raise `OllamaUnavailableError` immediately — no retry, since Ollama is not running.

### Prompt Management
Prompts are loaded from `prompts/*.txt` and cached in memory for the session.
Version metadata at the top of each file (lines starting with `#`) is stripped automatically.

---

## Testing

```bash
# Install test dependencies
pip install pytest pytest-mock

# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_ingestion.py -v

# Run with logging output visible
pytest tests/ -v -s
```

### Test Structure

| File | What it tests |
|------|---------------|
| `tests/conftest.py` | Shared fixtures (graphs, mock LLM responses) |
| `tests/test_ingestion.py` | Parsing, speaker normalization, input validation |
| `tests/test_schemas.py` | Pydantic validation, coercion warnings |
| `tests/test_llm_client.py` | Retry logic, backoff, prompt loading |
| `tests/test_graph_store.py` | Load/save, snapshots, rollback, diff |
| `tests/test_gatekeeper.py` | Action application (add/strengthen/update/remove) |
| `tests/test_telemetry.py` | Metric collection and aggregation |

---

## Adding a New Node Type

1. **Add to `src/config.py`:**
   ```python
   NODE_TYPE_PREFIX["habit"] = "habit"
   RELATION_FOR_TYPE["habit"] = "practices"
   ```

2. **Add to `src/schemas.py`:**
   - Add `"habit"` to `VALID_NODE_TYPES`

3. **Update `src/visualizer.py` config:**
   ```python
   # In src/config.py
   NODE_COLORS["habit"] = "#B5838D"
   ```

4. **Update the LLM prompts** in `prompts/extraction.txt` and `prompts/gatekeeper.txt` to include the new type.

5. **Update `src/graph_store.py`'s `get_profile_summary()`** to include the new type in the summary dict.

6. **Add tests** in `tests/test_gatekeeper.py` and `tests/test_graph_store.py`.

---

## Known Limitations

- **Single-user only:** The graph has one hardcoded `user` anchor node. Multi-user support would require namespaced graphs.
- **Ollama-only:** The LLM client is tightly coupled to Ollama. Swapping in the Anthropic API would require a new client module.
- **No prompt injection protection:** Transcript text is sent to the LLM unmodified. Partially mitigated by local-only LLM, but adversarial transcripts could manipulate extraction.
- **`relationship_type: "expert"` in older snapshots:** Snapshots created before v0.2 may contain `"expert"` as a relationship_type (not in the valid enum set). The coercion warning in schemas.py will surface this on next pipeline run.
- **Snapshot growth:** Snapshots accumulate indefinitely. Set `SERENDIPITY_MAX_SNAPSHOTS` or periodically clean `data/snapshots/`.
