"""
Tests for src/telemetry.py

Covers: RunMetrics creation, coercion counting, session accumulation,
and summary statistics.
"""

import pytest

from src.telemetry import TelemetryCollector, RunMetrics, get_collector


# ─── TelemetryCollector ───────────────────────────────────────────────────────

def test_new_run_creates_run_metrics():
    collector = TelemetryCollector()
    run = collector.new_run(model="llama3")
    assert isinstance(run, RunMetrics)
    assert run.model == "llama3"
    assert run.run_id != ""


def test_new_run_auto_generates_run_id():
    collector = TelemetryCollector()
    r1 = collector.new_run("m1")
    r2 = collector.new_run("m2")
    assert r1.run_id != r2.run_id


def test_get_all_runs_accumulates():
    collector = TelemetryCollector()
    collector.new_run("m1")
    collector.new_run("m2")
    assert len(collector.get_all_runs()) == 2


def test_get_all_runs_returns_copy():
    """Mutations to the returned list should not affect the internal state."""
    collector = TelemetryCollector()
    collector.new_run("m1")
    runs = collector.get_all_runs()
    runs.clear()
    assert len(collector.get_all_runs()) == 1


def test_get_summary_empty_when_no_runs():
    collector = TelemetryCollector()
    assert collector.get_summary() == {}


def test_get_summary_calculates_totals():
    collector = TelemetryCollector()
    r1 = collector.new_run("m1")
    r1.extraction_latency_sec  = 2.0
    r1.gatekeeper_latency_sec  = 1.0
    r1.coercions_triggered     = 3

    r2 = collector.new_run("m2")
    r2.extraction_latency_sec  = 4.0
    r2.gatekeeper_latency_sec  = 2.0
    r2.coercions_triggered     = 1

    summary = collector.get_summary()
    assert summary["total_runs"] == 2
    assert summary["avg_pipeline_sec"] == pytest.approx((3.0 + 6.0) / 2, abs=0.01)
    assert summary["total_coercions"] == 4


def test_get_summary_models_used():
    collector = TelemetryCollector()
    collector.new_run("llama3")
    collector.new_run("mistral")
    collector.new_run("llama3")  # duplicate

    summary = collector.get_summary()
    assert set(summary["models_used"]) == {"llama3", "mistral"}


def test_record_coercion_increments_latest_run():
    collector = TelemetryCollector()
    run = collector.new_run("m1")
    assert run.coercions_triggered == 0

    collector.record_coercion()
    collector.record_coercion()
    assert run.coercions_triggered == 2


def test_record_coercion_no_op_when_no_runs():
    """Should not raise if called before any run is started."""
    collector = TelemetryCollector()
    collector.record_coercion()  # Should not raise


# ─── get_collector singleton ─────────────────────────────────────────────────

def test_get_collector_returns_same_instance():
    c1 = get_collector()
    c2 = get_collector()
    assert c1 is c2
