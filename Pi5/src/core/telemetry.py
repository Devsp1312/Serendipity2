"""
In-memory pipeline telemetry.

Tracks per-run metrics (extraction counts, LLM latency, enum coercions, graph growth).
All data is session-scoped and never written to disk — consistent with the local-first approach.

Usage:
    from src.core.telemetry import get_collector
    run = get_collector().new_run(model="llama3")
    run.turns_parsed = 12
    summary = get_collector().get_summary()
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RunMetrics:
    """Metrics captured from a single end-to-end pipeline run."""
    run_id: str   # ISO-format UTC timestamp
    model:  str

    # Ingestion
    transcript_chars: int = 0
    turns_parsed:     int = 0
    unique_speakers:  int = 0

    # Extraction phase
    core_values_found:      int   = 0
    goals_found:            int   = 0
    states_found:           int   = 0
    relationships_found:    int   = 0
    coercions_triggered:    int   = 0   # enum values silently coerced to safe defaults
    extraction_latency_sec: float = 0.0

    # Gatekeeper phase
    actions_add:            int   = 0
    actions_strengthen:     int   = 0
    actions_update:         int   = 0
    actions_remove:         int   = 0
    gatekeeper_latency_sec: float = 0.0

    # Graph state after run
    nodes_before:   int = 0
    nodes_after:    int = 0
    avg_confidence: Optional[float] = None
    min_confidence: Optional[float] = None
    max_confidence: Optional[float] = None


class TelemetryCollector:
    """Accumulates RunMetrics across the session. Obtain the singleton via get_collector()."""

    def __init__(self) -> None:
        self._runs: list[RunMetrics] = []

    def new_run(self, model: str) -> RunMetrics:
        """Creates, registers, and returns a new RunMetrics for the caller to populate."""
        run = RunMetrics(run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"), model=model)
        self._runs.append(run)
        logger.info("Telemetry: new run %s  model=%s", run.run_id, model)
        return run

    def record_coercion(self) -> None:
        """Increments the coercion counter on the most recent run (called by schemas.py validators)."""
        if self._runs:
            self._runs[-1].coercions_triggered += 1

    def get_all_runs(self) -> list[RunMetrics]:
        return list(self._runs)

    def get_summary(self) -> dict:
        """Aggregated stats across all runs. Returns {} if no runs yet."""
        if not self._runs:
            return {}
        avg = sum(r.extraction_latency_sec + r.gatekeeper_latency_sec for r in self._runs) / len(self._runs)
        return {
            "total_runs":       len(self._runs),
            "avg_pipeline_sec": round(avg, 2),
            "total_coercions":  sum(r.coercions_triggered for r in self._runs),
            "models_used":      sorted({r.model for r in self._runs}),
        }


# ─── Module-level singleton ───────────────────────────────────────────────────

_collector: Optional[TelemetryCollector] = None


def get_collector() -> TelemetryCollector:
    """Returns the global TelemetryCollector, creating it on first call."""
    global _collector
    if _collector is None:
        _collector = TelemetryCollector()
    return _collector
