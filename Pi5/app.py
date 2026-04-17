"""
Serendipity — Streamlit Web Interface

Run with: streamlit run app.py
"""

# Set up logging before anything else so all imported modules get the handlers
from src.core.logger import setup_logging
setup_logging()
from src.core.logger import get_logger
logger = get_logger(__name__)

import datetime
import json
import re
import time
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from networkx.readwrite import json_graph as _jg

from src.core.config import (
    AUDIO_DIR,
    DEFAULT_PROFILE,
    PROFILES_DIR,
    SS_MODEL, SS_LOADED_TRANSCRIPT, SS_TRANSCRIPT_SOURCE,
    SS_LAST_DIFF, SS_PIPELINE_LOG, SS_LAST_RUN_METRICS, SS_PROFILE,
    VIZ_DEFAULT_HEIGHT,
)
from src.ui.visualizer import build_visualizer_html
from src.storage.graph import (
    GraphDiff,
    diff_graphs,
    get_profile_summary,
    list_profiles,
    list_snapshots,
    load_graph,
    normalize_profile_id,
    rollback_to_snapshot,
    save_graph,
    save_snapshot,
)
from src.pipeline.ingest import get_turn_stats, parse_transcript, format_transcript_for_llm
from src.pipeline.gatekeeper import run_gatekeeper, apply_identity
from src.pipeline.qr_export import export_day_to_qr
from src.pipeline.extract import (
    run_pre_pass, run_focused_extraction, run_dedup, run_identity_extraction,
    run_multi_pass_pipeline,
)
from src.core.llm_client import (
    OllamaUnavailableError,
    check_connection,
    list_models,
)
from src.core.schemas import LLMSchemaError
from src.core.telemetry import get_collector
from src.pipeline.diarize import run_diarization_pipeline


# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Serendipity",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Custom CSS for polished dark theme ──────────────────────────────────────

st.markdown("""
<style>
  /* ── Hide Streamlit chrome ───────────────────────────────────────────────── */
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }

  /* ── Global typography ───────────────────────────────────────────────────── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
  }

  /* ── Sidebar ─────────────────────────────────────────────────────────────── */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080c12 0%, #0b1018 50%, #0d1320 100%);
    border-right: 1px solid rgba(255,255,255,0.04);
  }
  section[data-testid="stSidebar"] .stMarkdown h1 {
    font-size: 1.4rem;
    letter-spacing: -0.02em;
  }

  /* ── Tab styling ─────────────────────────────────────────────────────────── */
  button[data-baseweb="tab"] {
    font-size: 0.82rem;
    font-weight: 500;
    padding: 10px 18px;
    letter-spacing: 0.01em;
    border-radius: 8px 8px 0 0 !important;
  }

  /* ── Metric cards ────────────────────────────────────────────────────────── */
  [data-testid="stMetric"] {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 16px 20px;
    backdrop-filter: blur(10px);
  }
  [data-testid="stMetricLabel"] {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    opacity: 0.5;
    font-weight: 500;
  }
  [data-testid="stMetricValue"] {
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -0.02em;
  }

  /* ── Progress bars ───────────────────────────────────────────────────────── */
  .stProgress > div > div {
    border-radius: 8px;
    height: 6px;
  }

  /* ── Buttons ─────────────────────────────────────────────────────────────── */
  .stButton > button {
    border-radius: 10px;
    font-weight: 600;
    letter-spacing: 0.01em;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
  }
  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00B4D8 0%, #06D6A0 100%);
    border: none;
    color: #0a0e14;
    box-shadow: 0 4px 15px rgba(0, 180, 216, 0.25);
  }
  .stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 25px rgba(0, 180, 216, 0.4);
    transform: translateY(-1px);
  }

  /* ── Alert boxes ─────────────────────────────────────────────────────────── */
  .stAlert {
    border-radius: 10px;
    border-left-width: 4px;
  }

  /* ── Dataframe ───────────────────────────────────────────────────────────── */
  [data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
  }

  /* ── Expanders ───────────────────────────────────────────────────────────── */
  .streamlit-expanderHeader {
    border-radius: 10px;
    font-weight: 500;
    font-size: 0.9rem;
  }

  /* ── Section headers ─────────────────────────────────────────────────────── */
  .section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 6px;
  }
  .section-header .icon {
    width: 34px; height: 34px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
  }
  .section-header h3 {
    margin: 0;
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: -0.01em;
  }

  /* ── Dividers ────────────────────────────────────────────────────────────── */
  hr {
    border-color: rgba(255,255,255,0.05) !important;
    margin: 1.2rem 0 !important;
  }

  /* ── Empty state messages ────────────────────────────────────────────────── */
  .empty-state {
    text-align: center;
    padding: 60px 24px;
    color: rgba(255,255,255,0.3);
    font-size: 0.95rem;
    line-height: 1.6;
  }
  .empty-state .icon {
    font-size: 3rem;
    margin-bottom: 16px;
    opacity: 0.6;
  }

  /* ── Glass card ──────────────────────────────────────────────────────────── */
  .glass-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 20px 24px;
    backdrop-filter: blur(12px);
    margin-bottom: 16px;
  }

  /* ── Identity card ───────────────────────────────────────────────────────── */
  .identity-card {
    background: linear-gradient(135deg, rgba(0,180,216,0.08) 0%, rgba(6,214,160,0.06) 100%);
    border: 1px solid rgba(0,180,216,0.15);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 24px;
  }
  .identity-card .identity-name {
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 4px;
    color: #fff;
  }
  .identity-card .identity-detail {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.55);
    margin: 2px 0;
  }
  .identity-card .identity-detail span {
    color: rgba(255,255,255,0.85);
    font-weight: 500;
  }
  .identity-badge {
    display: inline-block;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    color: rgba(255,255,255,0.7);
    margin-right: 6px;
    margin-top: 8px;
  }

  /* ── Sidebar branding ────────────────────────────────────────────────────── */
  .sidebar-brand {
    text-align: center;
    padding: 8px 0 4px 0;
  }
  .sidebar-brand .logo {
    font-size: 2rem;
    margin-bottom: 6px;
  }
  .sidebar-brand .title {
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #00B4D8, #06D6A0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .sidebar-brand .subtitle {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.35);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 2px;
  }

  /* ── Status badge ────────────────────────────────────────────────────────── */
  .status-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    border-radius: 10px;
    font-size: 0.8rem;
    font-weight: 500;
  }
  .status-badge.connected {
    background: rgba(6, 214, 160, 0.08);
    border: 1px solid rgba(6, 214, 160, 0.2);
    color: #06D6A0;
  }
  .status-badge.disconnected {
    background: rgba(239, 71, 111, 0.08);
    border: 1px solid rgba(239, 71, 111, 0.2);
    color: #EF476F;
  }
  .status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
  }
  .status-dot.green { background: #06D6A0; box-shadow: 0 0 8px rgba(6,214,160,0.5); }
  .status-dot.red { background: #EF476F; box-shadow: 0 0 8px rgba(239,71,111,0.5); }

  /* ── Profile value item ──────────────────────────────────────────────────── */
  .value-item {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
  }
  .value-item .label {
    font-weight: 600;
    font-size: 0.9rem;
    margin-bottom: 6px;
  }

  /* ── Upload drop zone ────────────────────────────────────────────────────── */
  [data-testid="stFileUploader"] {
    border-radius: 14px !important;
  }
  [data-testid="stFileUploader"] section {
    border-radius: 14px !important;
    border: 2px dashed rgba(255,255,255,0.08) !important;
    background: rgba(255,255,255,0.015) !important;
    transition: all 0.3s ease;
    padding: 24px !important;
  }
  [data-testid="stFileUploader"] section:hover {
    border-color: rgba(0,180,216,0.3) !important;
    background: rgba(0,180,216,0.03) !important;
  }

  /* ── Code blocks ─────────────────────────────────────────────────────────── */
  .stCodeBlock {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.06);
  }

  /* ── Selectbox ───────────────────────────────────────────────────────────── */
  [data-baseweb="select"] {
    border-radius: 10px;
  }

  /* ── Run Days — stepper ──────────────────────────────────────────────────── */
  .rd-stepper { display:flex; align-items:flex-start; margin:4px 0 28px 0; }
  .rd-step { display:flex; flex-direction:column; align-items:center; flex:1; }
  .rd-step-circle {
    width:40px; height:40px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:0.9rem; font-weight:700; z-index:1;
    border:2px solid rgba(255,255,255,0.08);
    background:rgba(255,255,255,0.03);
    color:rgba(255,255,255,0.2);
    transition:all 0.3s ease;
  }
  .rd-step-circle.done  { background:rgba(6,214,160,0.15); border-color:rgba(6,214,160,0.6); color:#06D6A0; }
  .rd-step-circle.active{ background:rgba(0,180,216,0.2);  border-color:#00B4D8; color:#00B4D8;
                           box-shadow:0 0 20px rgba(0,180,216,0.3); }
  .rd-step-label { font-size:0.67rem; margin-top:7px; color:rgba(255,255,255,0.22); white-space:nowrap; text-align:center; }
  .rd-step-label.done   { color:rgba(6,214,160,0.75); }
  .rd-step-label.active { color:#00B4D8; font-weight:600; }
  .rd-connector-wrap { flex:1; display:flex; align-items:center; padding:0 4px; margin-top:-22px; }
  .rd-connector { width:100%; height:2px; border-radius:2px; background:rgba(255,255,255,0.06); transition:background 0.4s; }
  .rd-connector.done { background:rgba(6,214,160,0.45); }

  /* ── Run Days — insight rows ─────────────────────────────────────────────── */
  .rd-group { background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); border-radius:14px; padding:16px 20px; margin-bottom:12px; }
  .rd-group-title { font-size:0.67rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; opacity:0.38; margin-bottom:12px; }
  .rd-insight-row { display:flex; align-items:center; gap:10px; padding:8px 0; border-bottom:1px solid rgba(255,255,255,0.04); }
  .rd-insight-row:last-child { border-bottom:none; }
  .rd-insight-text { flex:1; font-size:0.85rem; line-height:1.4; }
  .rd-conf-bar { height:4px; border-radius:4px; background:rgba(255,255,255,0.07); overflow:hidden; width:60px; flex-shrink:0; }
  .rd-conf-bar-fill { height:100%; border-radius:4px; }
  .rd-conf-pct { font-size:0.7rem; color:rgba(255,255,255,0.3); width:30px; text-align:right; flex-shrink:0; }

  /* ── Run Days — pass cards ───────────────────────────────────────────────── */
  .rd-pass-card { background:rgba(255,255,255,0.025); border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:14px 18px; margin-bottom:10px; }
  .rd-pass-head { display:flex; align-items:center; gap:10px; margin-bottom:6px; }
  .rd-pass-accent { width:3px; height:18px; border-radius:2px; flex-shrink:0; }
  .rd-pass-tag { font-weight:600; font-size:0.88rem; }
  .rd-pass-pri { margin-left:auto; opacity:0.28; font-size:0.7rem; }
  .rd-pass-focus { font-size:0.8rem; opacity:0.5; line-height:1.5; padding-left:13px; }

  /* ── Run Days — callout ──────────────────────────────────────────────────── */
  .rd-callout { background:rgba(179,136,255,0.07); border-left:3px solid rgba(179,136,255,0.4); border-radius:0 10px 10px 0; padding:10px 14px; margin-bottom:8px; font-size:0.84rem; line-height:1.55; color:rgba(255,255,255,0.72); }

  /* ── Run Days — identity fields ──────────────────────────────────────────── */
  .rd-id-field { display:flex; align-items:center; gap:14px; padding:11px 0; border-bottom:1px solid rgba(255,255,255,0.05); }
  .rd-id-field:last-child { border-bottom:none; }
  .rd-id-icon { font-size:1.05rem; width:26px; text-align:center; flex-shrink:0; }
  .rd-id-label { font-size:0.68rem; text-transform:uppercase; letter-spacing:0.07em; opacity:0.38; width:78px; flex-shrink:0; }
  .rd-id-value { font-size:0.92rem; font-weight:500; flex:1; }
  .rd-id-conf { font-size:0.7rem; opacity:0.35; }

  /* ── Run Days — file loaded banner ──────────────────────────────────────── */
  .rd-file-banner { display:flex; align-items:center; gap:14px; background:rgba(0,180,216,0.07); border:1px solid rgba(0,180,216,0.15); border-radius:12px; padding:14px 18px; margin-bottom:16px; }
  .rd-file-name { font-weight:600; font-size:0.95rem; flex:1; }
  .rd-file-meta { font-size:0.78rem; opacity:0.45; }

  /* ── Run Days — profile cards ────────────────────────────────────────────── */
  .rd-prof-card {
    border-radius:16px; padding:18px 14px; text-align:center; margin-bottom:8px;
    transition:all 0.2s ease;
  }
  .rd-prof-card.active {
    background:rgba(0,180,216,0.08); border:2px solid rgba(0,180,216,0.55);
    box-shadow:0 0 22px rgba(0,180,216,0.12);
  }
  .rd-prof-card.inactive {
    background:rgba(255,255,255,0.025); border:1.5px solid rgba(255,255,255,0.07);
  }
  .rd-prof-icon  { font-size:2.1rem; margin-bottom:8px; }
  .rd-prof-name  { font-weight:700; font-size:0.92rem; letter-spacing:-0.01em; margin-bottom:4px; }
  .rd-prof-stats { font-size:0.68rem; opacity:0.38; line-height:1.6; }

  /* ── Run Days — day tile grid ────────────────────────────────────────────── */
  .rd-day-tile {
    border-radius:12px; padding:10px 6px; text-align:center; margin-bottom:6px;
    transition:all 0.2s ease;
  }
  .rd-day-tile.done     { background:rgba(6,214,160,0.05);  border:1.5px solid rgba(6,214,160,0.35); }
  .rd-day-tile.pending  { background:rgba(255,255,255,0.02); border:1.5px solid rgba(255,255,255,0.06); }
  .rd-day-tile.selected { background:rgba(0,180,216,0.1);   border:2px solid rgba(0,180,216,0.65);
                           box-shadow:0 0 16px rgba(0,180,216,0.18); }
  .rd-day-icon  { font-size:1.0rem; margin-bottom:2px; }
  .rd-day-name  { font-size:0.75rem; font-weight:600; margin-bottom:2px; }
  .rd-day-delta { font-size:0.62rem; opacity:0.45; }

  /* ── Run Days — progress bar ─────────────────────────────────────────────── */
  .rd-progress-wrap { background:rgba(255,255,255,0.05); border-radius:6px; height:6px; margin:10px 0 18px; overflow:hidden; }
  .rd-progress-fill { height:100%; border-radius:6px; background:linear-gradient(90deg,#00B4D8,#06D6A0); transition:width 0.4s ease; }
</style>
""", unsafe_allow_html=True)


# ─── Helper functions ─────────────────────────────────────────────────────────

def section_header(icon: str, title: str, bg: str = "rgba(255,255,255,0.06)"):
    """Render a styled section header with icon."""
    st.markdown(
        f'<div class="section-header">'
        f'<div class="icon" style="background:{bg}">{icon}</div>'
        f'<h3>{title}</h3>'
        f'</div>',
        unsafe_allow_html=True,
    )


def empty_state(icon: str, message: str):
    """Render a centered empty-state placeholder."""
    st.markdown(
        f'<div class="empty-state">'
        f'<div class="icon">{icon}</div>'
        f'<div>{message}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _clear_transcript_state() -> None:
    """Removes loaded transcript from session state to free memory."""
    st.session_state.pop(SS_LOADED_TRANSCRIPT, None)
    st.session_state.pop(SS_TRANSCRIPT_SOURCE, None)


def run_pipeline(turns, model: str, profile: str) -> None:
    """Runs the multi-pass extraction pipeline and stores results in session state.

    Pipeline phases:
      1. Pre-pass (decide extraction order + custom passes)
      2. Focused extraction passes (3 core + 0-2 custom)
      3. Dedup + cross-category reasoning
      4. Identity extraction (smart update)
      5. Gatekeeper (graph delta)
      6. Apply identity + save
    """
    run = get_collector().new_run(model=model)
    diarized_text = format_transcript_for_llm(turns)

    with st.status("Running multi-pass pipeline…", expanded=True) as status:

        # ── Phases 1-4: Multi-pass extraction ────────────────────────────
        st.write(f"Multi-pass extraction  (model: `{model}`)")
        logger.info(
            "App pipeline: multi-pass extraction started  model=%s  profile=%s",
            model, profile,
        )
        t_extract = time.monotonic()
        try:
            G_before = load_graph(profile=profile)
            run.nodes_before = G_before.number_of_nodes()

            dedup_output, identity_output, raw_logs = run_multi_pass_pipeline(
                diarized_text=diarized_text,
                model=model,
                existing_graph=G_before,
                prepass_model=st.session_state.get("prepass_model"),
            )
            run.extraction_latency_sec = round(time.monotonic() - t_extract, 2)
            run.core_values_found   = len(dedup_output.core_values)
            run.goals_found         = len(dedup_output.long_term_goals)
            run.states_found        = len(dedup_output.short_term_values)
            run.relationships_found = len(dedup_output.relationships)

        except (OllamaUnavailableError, LLMSchemaError) as e:
            logger.error("Multi-pass extraction failed: %s", e)
            status.update(label="Pipeline failed", state="error")
            st.error(str(e))
            return

        # ── QR export (before merge) ─────────────────────────────────────
        try:
            _png, _json = export_day_to_qr(dedup_output, Path("data/qrcodes"))
            st.image(str(_png), caption=f"Day QR — scan to sync with matching server ({_json.name})")
        except Exception as _qr_exc:
            logger.warning("QR export failed (non-fatal): %s", _qr_exc)

        # ── Phase 5: Gatekeeper ──────────────────────────────────────────
        st.write("Gatekeeper — Delta Update")
        logger.info("App pipeline: gatekeeper phase started")
        t_gk = time.monotonic()
        try:
            G_after, raw_gatekeeper = run_gatekeeper(G_before, dedup_output, model=model)
            run.gatekeeper_latency_sec = round(time.monotonic() - t_gk, 2)

            raw_logs.append(f"=== GATEKEEPER ===\n{raw_gatekeeper}")

        except (OllamaUnavailableError, LLMSchemaError) as e:
            logger.error("Gatekeeper failed: %s", e)
            status.update(label="Pipeline failed", state="error")
            st.error(str(e))
            return

        # ── Phase 6: Apply identity + save ───────────────────────────────
        st.write("Applying identity & saving")
        apply_identity(G_after, identity_output)
        run.nodes_after = G_after.number_of_nodes()

        save_graph(G_after, model=model, profile=profile)
        snapshot_path = save_snapshot(G_after, model=model, profile=profile)
        raw_logs.append(f"\nSnapshot saved: {snapshot_path}")

        diff = diff_graphs(G_before, G_after)
        st.session_state[SS_LAST_DIFF]        = diff
        st.session_state[SS_PIPELINE_LOG]     = "\n".join(raw_logs)
        st.session_state[SS_LAST_RUN_METRICS] = run

        # Clear transcript from memory after successful processing
        _clear_transcript_state()

        total_sec = round(run.extraction_latency_sec + run.gatekeeper_latency_sec, 2)
        logger.info(
            "App pipeline complete  model=%s  profile=%s  elapsed=%.2fs  nodes=%d->%d",
            model, profile, total_sec, run.nodes_before, run.nodes_after,
        )
        status.update(label=f"Pipeline complete! ({total_sec}s)", state="complete")

    render_diff(diff)
    st.rerun()


def render_diff(diff: GraphDiff) -> None:
    """Renders an inline diff summary after a pipeline run."""
    st.divider()
    section_header("🔄", "What changed")

    if not diff.has_changes:
        st.info("No changes to the profile graph.")
        return

    col_a, col_r, col_s = st.columns(3)
    col_a.metric("Added", len(diff.added))
    col_r.metric("Removed", len(diff.removed))
    col_s.metric("Boosted", len(diff.strengthened))

    for nd in diff.added:
        conf = f"  — confidence {nd.new_confidence:.0%}" if nd.new_confidence else ""
        st.success(f"+ **{nd.label}** `{nd.node_type}`{conf}")

    for nd in diff.removed:
        st.error(f"− **{nd.label}** `{nd.node_type}`")

    for nd in diff.strengthened:
        delta = f"  (+{nd.delta:.2f})" if nd.delta else ""
        st.info(f"↑ **{nd.label}** `{nd.node_type}`{delta}")


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">'
        '<div class="logo">🎧</div>'
        '<div class="title">Serendipity</div>'
        '<div class="subtitle">Personal Knowledge Builder · v0.3</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    # ── Profile selector
    available_profiles = list_profiles()
    active_profile = normalize_profile_id(st.session_state.get(SS_PROFILE, DEFAULT_PROFILE))
    if active_profile not in available_profiles:
        available_profiles.append(active_profile)
        available_profiles = sorted(set(available_profiles))

    selected_profile = st.selectbox(
        "Profile",
        options=available_profiles,
        index=available_profiles.index(active_profile),
        help="Each profile has separate graph data and snapshots.",
    )
    st.session_state[SS_PROFILE] = selected_profile

    st.divider()

    # ── Connection status
    ollama_ok = check_connection()
    if ollama_ok:
        st.markdown(
            '<div class="status-badge connected">'
            '<span class="status-dot green"></span> Ollama connected'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-badge disconnected">'
            '<span class="status-dot red"></span> Ollama unreachable'
            '</div>',
            unsafe_allow_html=True,
        )
        st.caption("Run `ollama serve` to start")

    st.write("")

    # ── Model selector (LLMs only — filter out audio/embedding models)
    _NON_LLM_KEYWORDS = {"whisper", "embed", "clip", "stable-diffusion", "tts", "speech"}
    available_models = [
        m for m in list_models()
        if not any(kw in m.lower() for kw in _NON_LLM_KEYWORDS)
    ]
    if available_models:
        # Default to llama3.2:3b if available, else first in list
        _default_model = st.session_state.get(SS_MODEL, "")
        _default_idx = (
            available_models.index(_default_model)
            if _default_model in available_models
            else next((i for i, m in enumerate(available_models) if "llama" in m.lower()), 0)
        )
        selected_model = st.selectbox(
            "Model",
            options=available_models,
            index=_default_idx,
            help="Ollama model for extraction & gatekeeper",
        )
        st.session_state[SS_MODEL] = selected_model

        # ── Pre-pass model override (optional)
        st.caption("Advanced")
        use_separate_prepass = st.checkbox(
            "Use different model for pre-pass",
            value=False,
            help="Use a specialized model just for the extraction planning stage"
        )
        if use_separate_prepass:
            prepass_model = st.selectbox(
                "Pre-pass model",
                options=available_models,
                help="Model for extraction planning (pre-pass stage only)"
            )
            st.session_state["prepass_model"] = prepass_model
        else:
            st.session_state.pop("prepass_model", None)
    else:
        st.warning("No LLM models found — run `ollama pull llama3.2`")
        st.session_state.setdefault(SS_MODEL, "llama3.2:3b")

    st.divider()

    # ── Graph stats
    G_sidebar = load_graph(profile=selected_profile)
    graph_meta = G_sidebar.graph

    c1, c2 = st.columns(2)
    c1.metric("Nodes", G_sidebar.number_of_nodes())
    c2.metric("Edges", G_sidebar.number_of_edges())

    st.write("")
    meta_parts = []
    if graph_meta.get("last_updated"):
        ts = graph_meta["last_updated"]
        short_ts = ts[:16].replace("T", " · ") if len(ts) > 16 else ts
        meta_parts.append(f"Updated {short_ts}")
    if graph_meta.get("last_model"):
        meta_parts.append(f"Model: {graph_meta['last_model']}")
    if meta_parts:
        st.caption(" · ".join(meta_parts))

    st.divider()
    if st.button("Reset graph", use_container_width=True, type="secondary"):
        from src.storage.graph import _fresh_graph
        save_graph(_fresh_graph(), profile=selected_profile)
        logger.info("Graph reset via sidebar button for profile=%s", selected_profile)
        st.success(f"Profile '{selected_profile}' reset.")
        st.rerun()


# ─── Main panel ───────────────────────────────────────────────────────────────

tab_3d, tab_profile, tab_run, tab_stage1, tab_history, tab_raw, tab_log = st.tabs([
    "🌐  Graph",
    "🧠  Profile",
    "▶  Run Days",
    "🎙  Stage 1",
    "📊  History",
    "🗂  Raw Data",
    "📋  Log",
])


# ─── 3D Graph tab ─────────────────────────────────────────────────────────────

with tab_3d:
    G_viz = load_graph(profile=selected_profile)
    if G_viz.number_of_nodes() <= 1:
        empty_state("🌐", "No profile data yet.<br>Upload a transcript to build your 3D knowledge graph.")
    else:
        node_count = G_viz.number_of_nodes() - 1
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Insights", node_count)
        cv_count = len([n for n, d in G_viz.nodes(data=True) if d.get("node_type") == "core_value"])
        ltg_count = len([n for n, d in G_viz.nodes(data=True) if d.get("node_type") == "long_term_goal"])
        rel_count = len([n for n, d in G_viz.nodes(data=True) if d.get("node_type") == "person"])
        c2.metric("Values", cv_count)
        c3.metric("Goals", ltg_count)
        c4.metric("People", rel_count)
        st.caption("Thicker beams = stronger signal · Hover for details · Drag to rotate")
        html_str = build_visualizer_html(G_viz, height=VIZ_DEFAULT_HEIGHT)
        components.html(html_str, height=VIZ_DEFAULT_HEIGHT, scrolling=False)


# ─── Profile tab ─────────────────────────────────────────────────────────────

with tab_profile:
    G = load_graph(profile=selected_profile)
    summary = get_profile_summary(G)

    if G.number_of_nodes() <= 1:
        empty_state("🧠", "No profile data yet.<br>Upload a transcript and run the pipeline to build a knowledge graph.")
    else:
        _view_mode = st.radio(
            "View",
            ["Card", "Table"],
            horizontal=True,
            label_visibility="collapsed",
            key="profile_view_mode",
        )
        st.write("")

        # ── Identity card (if available) ─────────────────────────────────
        identity = summary.get("identity", {})
        user_data = G.nodes.get("user", {})
        has_identity = user_data.get("identity_name") or user_data.get("identity_occupation")

        if has_identity:
            name = user_data.get("identity_name", selected_profile.title())
            badges_html = ""
            if user_data.get("identity_occupation"):
                badges_html += f'<span class="identity-badge">💼 {user_data["identity_occupation"]}</span>'
            if user_data.get("identity_age"):
                badges_html += f'<span class="identity-badge">🎂 Age {user_data["identity_age"]}</span>'
            if user_data.get("identity_location"):
                badges_html += f'<span class="identity-badge">📍 {user_data["identity_location"]}</span>'

            node_count = G.number_of_nodes() - 1
            badges_html += f'<span class="identity-badge">🔗 {node_count} insights</span>'

            st.markdown(
                f'<div class="identity-card">'
                f'<div class="identity-name">{name}</div>'
                f'<div>{badges_html}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if _view_mode == "Card":
            # ── Two-column layout ────────────────────────────────────────────
            col1, col2 = st.columns(2, gap="large")

            with col1:
                section_header("💎", "Core Values", "rgba(0,180,216,0.15)")
                if summary["core_values"]:
                    for cv in sorted(summary["core_values"], key=lambda x: -x["confidence"]):
                        conf  = cv["confidence"]
                        count = cv.get("evidence_count", 1)
                        mentions = f"{count} mention{'s' if count != 1 else ''}"
                        st.markdown(
                            f'<div class="value-item"><div class="label">{cv["label"].title()}</div></div>',
                            unsafe_allow_html=True,
                        )
                        st.progress(conf, text=f"{conf:.0%}  ·  {mentions}")
                else:
                    st.caption("None detected yet.")

                st.write("")
                section_header("🎯", "Long-term Goals", "rgba(6,214,160,0.15)")
                if summary["long_term_goals"]:
                    for ltg in sorted(summary["long_term_goals"], key=lambda x: -x["confidence"]):
                        conf = ltg["confidence"]
                        st.markdown(
                            f'<div class="value-item"><div class="label">{ltg["label"].capitalize()}</div></div>',
                            unsafe_allow_html=True,
                        )
                        st.progress(conf, text=f"{conf:.0%}")
                else:
                    st.caption("None detected yet.")

            with col2:
                section_header("⚡", "Current States", "rgba(239,71,111,0.15)")
                if summary["short_term_states"]:
                    for sts in summary["short_term_states"]:
                        category  = sts.get("category", "state")
                        intensity = sts.get("intensity", 0.5)
                        icon      = {"stressor": "🔴", "craving": "🟡", "task": "🔵"}.get(category, "⚪")
                        st.markdown(
                            f'<div class="value-item"><div class="label">{icon} {sts["label"].capitalize()}'
                            f'<span style="opacity:0.4;font-size:0.75rem;margin-left:8px">{category}</span></div></div>',
                            unsafe_allow_html=True,
                        )
                        st.progress(intensity, text=f"{intensity:.0%}")
                else:
                    st.caption("None detected yet.")

                st.write("")
                section_header("👥", "Relationships", "rgba(255,209,102,0.15)")
                if summary["relationships"]:
                    rel_data = [
                        {
                            "Name":     r["name"].title(),
                            "Type":     r["relationship_type"].capitalize(),
                            "Tone":     r["tone"].capitalize(),
                            "Strength": f"{r['strength']:.0%}",
                        }
                        for r in summary["relationships"]
                    ]
                    st.dataframe(pd.DataFrame(rel_data), use_container_width=True, hide_index=True)
                else:
                    st.caption("No relationships mapped yet.")

            # ── Custom categories (if any) ───────────────────────────────────
            custom_cats = summary.get("custom_categories", {})
            if custom_cats:
                st.divider()
                for cat_name, items in custom_cats.items():
                    section_header("✨", cat_name.replace("_", " ").title(), "rgba(179,136,255,0.15)")
                    for item in items:
                        conf = item.get("confidence", 0.5)
                        st.markdown(
                            f'<div class="value-item"><div class="label">{item["label"].capitalize()}</div></div>',
                            unsafe_allow_html=True,
                        )
                        st.progress(conf, text=f"{conf:.0%}")

        else:
            # ── Table view ───────────────────────────────────────────────────
            _rows = []
            for cv in summary["core_values"]:
                _rows.append({
                    "Category":   "Core Value",
                    "Insight":    cv["label"].title(),
                    "Confidence": cv["confidence"],
                    "Seen":       cv.get("evidence_count", 1),
                })
            for ltg in summary["long_term_goals"]:
                _rows.append({
                    "Category":   "Long-term Goal",
                    "Insight":    ltg["label"].capitalize(),
                    "Confidence": ltg["confidence"],
                    "Seen":       ltg.get("evidence_count", 1),
                })
            for sts in summary["short_term_states"]:
                _rows.append({
                    "Category":   "Current State",
                    "Insight":    sts["label"].capitalize(),
                    "Confidence": sts.get("intensity", 0.5),
                    "Seen":       sts.get("evidence_count", 1),
                })
            for rel in summary["relationships"]:
                _rows.append({
                    "Category":   "Relationship",
                    "Insight":    f'{rel["name"].title()} ({rel["relationship_type"].capitalize()}, {rel["tone"].lower()} tone)',
                    "Confidence": rel["strength"],
                    "Seen":       rel.get("evidence_count", 1),
                })
            for _cat_name, _items in summary.get("custom_categories", {}).items():
                for _item in _items:
                    _rows.append({
                        "Category":   _cat_name.replace("_", " ").title(),
                        "Insight":    _item["label"].capitalize(),
                        "Confidence": _item.get("confidence", 0.5),
                        "Seen":       _item.get("evidence_count", 1),
                    })

            if _rows:
                _df = (
                    pd.DataFrame(_rows)
                    .sort_values(["Category", "Confidence"], ascending=[True, False])
                    .reset_index(drop=True)
                )
                st.dataframe(
                    _df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Confidence": st.column_config.ProgressColumn(
                            "Confidence",
                            min_value=0.0,
                            max_value=1.0,
                            format="%.0%%",
                        ),
                        "Seen": st.column_config.NumberColumn(
                            "Seen",
                            help="Number of times this insight was observed",
                        ),
                    },
                )
            else:
                st.caption("No insights yet.")


# ─── Stage 1 — Transcription tab ─────────────────────────────────────────────

_STAGE1_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}


def _natural_sort_key(p: Path) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


with tab_stage1:
    section_header("🎙", "Stage 1 — Transcription")
    st.caption(
        "Transcribe audio files to plain text — no speaker labels, no timestamps. "
        "Saves a .txt file alongside each audio file."
    )
    st.write("")

    # Discover folders that contain audio files
    _PROJECT_ROOT = Path(__file__).parent
    _audio_folders = sorted(
        p for p in _PROJECT_ROOT.iterdir()
        if p.is_dir()
        and not p.name.startswith(".")
        and any(f.suffix.lower() in _STAGE1_AUDIO_EXTS for f in p.iterdir() if f.is_file())
    )

    if not _audio_folders:
        st.info("No audio folders found in the project directory.")
    else:
        col_folder, col_model, col_beam = st.columns([2, 2, 1])

        with col_folder:
            _folder_name = st.selectbox(
                "Folder",
                [f.name for f in _audio_folders],
                key="s1_folder",
            )
        _folder = _PROJECT_ROOT / _folder_name
        _audio_files = sorted(
            [f for f in _folder.iterdir() if f.suffix.lower() in _STAGE1_AUDIO_EXTS],
            key=_natural_sort_key,
        )

        _file_labels = {
            f.name: f"{f.name}  ({f.stat().st_size / 1024 / 1024:.1f} MB)"
            for f in _audio_files
        }

        with col_model:
            _whisper_size = st.selectbox(
                "Whisper model",
                ["tiny", "base", "small", "medium", "large-v3"],
                index=1,
                key="s1_model",
            )
        with col_beam:
            _beam = st.number_input(
                "Beam",
                min_value=1,
                max_value=5,
                value=1,
                help="1 = greedy (fastest)",
                key="s1_beam",
            )

        _selected_name = st.selectbox(
            "Audio file",
            [f.name for f in _audio_files],
            format_func=lambda n: _file_labels[n],
            key="s1_file",
        )
        _audio_path = _folder / _selected_name

        if st.button("▶  Transcribe", type="primary", key="s1_run"):
            with st.status(f"Transcribing **{_selected_name}**…", expanded=True) as _status:
                st.write("Loading Whisper model…")
                from src.pipeline.transcribe import load_model as _load_model, transcribe_to_file as _transcribe_to_file
                _model = _load_model(_whisper_size)
                st.write(f"Decoding + normalizing audio…")
                st.write(f"Running VAD + transcription (beam_size={_beam})…")
                _out_path, _result = _transcribe_to_file(_audio_path, _model, beam_size=_beam)
                _status.update(label="✅ Transcription complete", state="complete")

            st.write("")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Audio duration", f"{_result.audio_duration_sec / 60:.1f} min")
            c2.metric("Transcription time", f"{_result.transcription_sec:.1f} s")
            _speed = _result.audio_duration_sec / max(_result.transcription_sec, 0.01)
            c3.metric("Transcription speed", f"{_speed:.0f}× faster")
            c4.metric("Output chars", f"{len(_result.text):,}")

            st.caption(
                f"Language detected: `{_result.language}`  ·  "
                f"Saved → `{_out_path.relative_to(_PROJECT_ROOT)}`"
            )

            with st.expander("Transcript preview", expanded=True):
                st.text(_result.text[:3000] if _result.text else "(empty — file may be silence-only)")
                if len(_result.text) > 3000:
                    st.caption("(showing first 3000 chars — full transcript saved to .txt)")


# ─── Run Days — Step-by-Step Pipeline tab ────────────────────────────────────

# ── Shared helpers for Run Days visual components ────────────────────────────

def _rd_conf_color(c: float) -> str:
    if c >= 0.75: return "#06D6A0"
    if c >= 0.50: return "#FFD166"
    return "#EF476F"

def _rd_stepper_html(stage: int) -> str:
    labels = ["Pre-pass", "Extractions", "Dedup", "Identity", "Gatekeeper"]
    icons  = ["🔍", "⚡", "🔄", "🪪", "🛡"]
    html = '<div class="rd-stepper">'
    for i, (label, icon) in enumerate(zip(labels, icons)):
        if i < stage:
            cls, inner = "done", "✓"
        elif i == stage:
            cls, inner = "active", icon
        else:
            cls, inner = "", str(i + 1)
        if i > 0:
            conn = "done" if i <= stage else ""
            html += f'<div class="rd-connector-wrap"><div class="rd-connector {conn}"></div></div>'
        label_cls = "done" if i < stage else ("active" if i == stage else "")
        html += (
            f'<div class="rd-step">'
            f'<div class="rd-step-circle {cls}">{inner}</div>'
            f'<div class="rd-step-label {label_cls}">{label}</div>'
            f'</div>'
        )
    return html + '</div>'

def _rd_insight_rows_html(items) -> str:
    if not items:
        return '<div style="opacity:0.38;font-size:0.82rem;padding:6px 0">Nothing detected.</div>'
    rows = ""
    for it in items:
        c = it.confidence
        color = _rd_conf_color(c)
        rows += (
            f'<div class="rd-insight-row">'
            f'<div class="rd-insight-text">{it.item}</div>'
            f'<div class="rd-conf-bar"><div class="rd-conf-bar-fill" style="width:{c*100:.0f}%;background:{color}"></div></div>'
            f'<div class="rd-conf-pct">{c:.0%}</div>'
            f'</div>'
        )
    return rows

def _rd_group_html(title: str, icon: str, items, accent: str = "#00B4D8") -> str:
    rows = _rd_insight_rows_html(items)
    return (
        f'<div class="rd-group">'
        f'<div class="rd-group-title">{icon} {title}</div>'
        f'{rows}'
        f'</div>'
    )

def _rd_clean_insight(s: str) -> str:
    """Strip dict-repr wrapping if LLM returned a dict coerced to string.
    Handles both {"item": "..."} and arbitrary single-key dicts like
    {"Free-text insight connecting items...": "actual text"}."""
    if s.startswith("{"):
        try:
            import ast as _ast
            parsed = _ast.literal_eval(s)
            if isinstance(parsed, dict):
                # Prefer 'item' key; otherwise take the first value that is a non-empty string
                if "item" in parsed:
                    return str(parsed["item"])
                for v in parsed.values():
                    if isinstance(v, str) and v.strip():
                        return v.strip()
        except Exception:
            pass
    return s


with tab_run:

    # ─── Helpers ─────────────────────────────────────────────────────────────
    _RD_ROOT = Path(__file__).parent

    # Emoji icon for each known profile; fallback to 👤
    _RD_PROFILE_ICONS: dict[str, str] = {
        "friends": "👥", "spongebob": "🧽", "ted": "🎤",
        "sal": "🎭", "dev": "💻", "ross": "🦕", "rachel": "💇",
        "monica": "🍳", "joey": "🍕", "chandler": "🙃", "phoebe": "🎸",
    }

    def _rd_folder_for_profile(pid: str) -> Path | None:
        """Return the transcript folder whose lowercase name matches the profile id."""
        for d in _RD_ROOT.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                if normalize_profile_id(d.name) == normalize_profile_id(pid):
                    if any(f.suffix.lower() == ".txt" for f in d.iterdir() if f.is_file()):
                        return d
        return None

    def _rd_txt_files_for_profile(pid: str) -> list[Path]:
        """All .txt files in the profile's transcript folder, naturally sorted."""
        folder = _rd_folder_for_profile(pid)
        if not folder:
            return []
        return sorted(
            [f for f in folder.iterdir() if f.suffix.lower() == ".txt"],
            key=_natural_sort_key,
        )

    def _rd_run_log_path(pid: str) -> Path:
        return PROFILES_DIR / normalize_profile_id(pid) / "run_log.json"

    def _rd_load_run_log(pid: str) -> dict:
        p = _rd_run_log_path(pid)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
        return {}

    def _rd_save_run_log(pid: str, filename: str, nodes_before: int, nodes_after: int) -> None:
        log = _rd_load_run_log(pid)
        log[filename] = {
            "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "nodes_before": nodes_before,
            "nodes_after":  nodes_after,
        }
        rp = _rd_run_log_path(pid)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(log, indent=2))

    def _rd_load_day(file_path: Path, profile: str, day_idx: int) -> None:
        """Load a day's transcript into session state and reset the pipeline."""
        raw = file_path.read_text(encoding="utf-8")
        if len(raw) > 6000:
            raw = raw[:6000].rsplit(" ", 1)[0]
        for k in ["rd_prepass", "rd_extractions", "rd_dedup", "rd_identity"]:
            st.session_state.pop(k, None)
        st.session_state["rd_stage"]           = 0
        st.session_state["rd_file"]            = file_path.name
        st.session_state["rd_text"]            = raw
        st.session_state["rd_G_before"]        = load_graph(profile=profile)
        st.session_state["rd_selected_day_idx"] = day_idx

    # ─── Header ──────────────────────────────────────────────────────────────
    section_header("▶", "Run Days")
    st.caption("Pick a profile, then walk through each day one pipeline stage at a time.")
    st.write("")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Profile Cards
    # ═══════════════════════════════════════════════════════════════════════════
    _rd_all_pids = list_profiles()
    # Also surface any folders not yet in list_profiles
    for _d in _RD_ROOT.iterdir():
        if _d.is_dir() and not _d.name.startswith("."):
            if any(f.suffix.lower() == ".txt" for f in _d.iterdir() if f.is_file()):
                _pid = normalize_profile_id(_d.name)
                if _pid not in _rd_all_pids:
                    _rd_all_pids.append(_pid)
    _rd_all_pids = sorted(set(_rd_all_pids))

    # Build profile metadata list
    _rd_profile_meta: list[dict] = []
    for _pid in _rd_all_pids:
        _G_p       = load_graph(profile=_pid)
        _folder_p  = _rd_folder_for_profile(_pid)
        _day_files = _rd_txt_files_for_profile(_pid)
        _run_log   = _rd_load_run_log(_pid)
        _last_ts   = _G_p.graph.get("last_updated", "")
        _rd_profile_meta.append({
            "id":         _pid,
            "nodes":      max(_G_p.number_of_nodes() - 1, 0),
            "edges":      _G_p.number_of_edges(),
            "last_ts":    _last_ts[:10] if _last_ts else "—",
            "day_count":  len(_day_files),
            "days_done":  sum(1 for f in _day_files if f.name in _run_log),
            "icon":       _RD_PROFILE_ICONS.get(_pid, "👤"),
        })

    st.markdown(
        '<div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;'
        'letter-spacing:0.1em;opacity:0.35;margin-bottom:12px">Choose Profile</div>',
        unsafe_allow_html=True,
    )

    _rd_prof_cols = st.columns(max(1, min(len(_rd_profile_meta), 5)))
    for _pi, _pm in enumerate(_rd_profile_meta):
        _is_active = normalize_profile_id(_pm["id"]) == normalize_profile_id(selected_profile)
        _card_cls  = "active" if _is_active else "inactive"
        _prog_pct  = int((_pm["days_done"] / _pm["day_count"] * 100) if _pm["day_count"] else 0)
        with _rd_prof_cols[_pi]:
            st.markdown(
                f'<div class="rd-prof-card {_card_cls}">'
                f'<div class="rd-prof-icon">{_pm["icon"]}</div>'
                f'<div class="rd-prof-name">{_pm["id"].title()}</div>'
                f'<div class="rd-prof-stats">'
                f'{_pm["nodes"]} nodes · {_pm["edges"]} edges<br>'
                f'{_pm["days_done"]}/{_pm["day_count"]} days · updated {_pm["last_ts"]}'
                f'</div>'
                f'</div>'
                f'<div class="rd-progress-wrap">'
                f'<div class="rd-progress-fill" style="width:{_prog_pct}%"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button(
                "✓ Active" if _is_active else "Select",
                key=f"rd_prof_btn_{_pm['id']}",
                type="primary" if _is_active else "secondary",
                use_container_width=True,
            ):
                st.session_state[SS_PROFILE] = _pm["id"]
                for _k in ["rd_stage", "rd_file", "rd_text", "rd_prepass",
                           "rd_extractions", "rd_dedup", "rd_identity",
                           "rd_G_before", "rd_selected_day_idx"]:
                    st.session_state.pop(_k, None)
                st.rerun()

    st.write("")
    st.divider()

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Day Timeline
    # ═══════════════════════════════════════════════════════════════════════════
    _rd_txt_files   = _rd_txt_files_for_profile(selected_profile)
    _rd_run_log     = _rd_load_run_log(selected_profile)
    _rd_sel_idx     = st.session_state.get("rd_selected_day_idx", None)
    _rd_days_done   = sum(1 for f in _rd_txt_files if f.name in _rd_run_log)

    if not _rd_txt_files:
        empty_state("📂", f"No .txt transcript files found for '{selected_profile}'.")
    else:
        # Section header row
        _rd_hdr_col1, _rd_hdr_col2 = st.columns([3, 1])
        with _rd_hdr_col1:
            st.markdown(
                f'<div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;'
                f'letter-spacing:0.1em;opacity:0.35;margin-bottom:4px">Days — {selected_profile.title()}</div>'
                f'<div style="font-size:0.78rem;opacity:0.5">'
                f'{_rd_days_done} of {len(_rd_txt_files)} days completed</div>',
                unsafe_allow_html=True,
            )
        with _rd_hdr_col2:
            st.markdown(
                f'<div style="text-align:right;font-size:1.6rem;font-weight:800;'
                f'color:{"#06D6A0" if _rd_days_done == len(_rd_txt_files) else "#00B4D8"}">'
                f'{int(_rd_days_done / len(_rd_txt_files) * 100)}%</div>',
                unsafe_allow_html=True,
            )

        st.write("")

        # Day tiles — 6 per row
        _TILES_PER_ROW = 6
        for _row_start in range(0, len(_rd_txt_files), _TILES_PER_ROW):
            _row_files = _rd_txt_files[_row_start:_row_start + _TILES_PER_ROW]
            _tile_cols = st.columns(len(_row_files))
            for _ti, _tf in enumerate(_row_files):
                _tidx      = _row_start + _ti
                _is_done   = _tf.name in _rd_run_log
                _is_sel    = _tidx == _rd_sel_idx
                _tile_cls  = "selected" if _is_sel else ("done" if _is_done else "pending")
                _t_icon    = "✅" if _is_done else ("▶" if _is_sel else "⬜")
                _log_entry = _rd_run_log.get(_tf.name, {})
                _delta_val = (_log_entry.get("nodes_after", 0) - _log_entry.get("nodes_before", 0)) if _is_done else None
                _delta_str = (f"+{_delta_val}" if _delta_val and _delta_val >= 0 else str(_delta_val)) if _delta_val is not None else "—"
                with _tile_cols[_ti]:
                    st.markdown(
                        f'<div class="rd-day-tile {_tile_cls}">'
                        f'<div class="rd-day-icon">{_t_icon}</div>'
                        f'<div class="rd-day-name">{_tf.stem}</div>'
                        f'<div class="rd-day-delta">{_delta_str}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button(
                        "↻" if _is_done else "Run",
                        key=f"rd_day_btn_{_tidx}",
                        type="primary" if _is_sel else "secondary",
                        use_container_width=True,
                    ):
                        _rd_load_day(_tf, selected_profile, _tidx)
                        st.rerun()

        st.write("")
        st.divider()

        # ── Pipeline UI (shown after a day tile is selected) ─────────────────
        _rd_stage = st.session_state.get("rd_stage", -1)

        if _rd_stage >= 0 and st.session_state.get("rd_file"):

            # File banner
            _rd_text  = st.session_state.get("rd_text", "")
            _rd_model = st.session_state.get(SS_MODEL, "llama3.2:3b")
            st.markdown(
                f'<div class="rd-file-banner">'
                f'<span style="font-size:1.3rem">📄</span>'
                f'<div class="rd-file-name">{st.session_state["rd_file"]}</div>'
                f'<div class="rd-file-meta">{len(_rd_text):,} chars · {selected_profile} · {_rd_model}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Stepper
            st.markdown(_rd_stepper_html(_rd_stage), unsafe_allow_html=True)

            # ── Stage 0: Ready — show transcript preview ──────────────────────
            if _rd_stage == 0:
                with st.expander("📖 Transcript preview", expanded=False):
                    st.markdown(
                        f'<div style="font-size:0.82rem;line-height:1.7;opacity:0.7;white-space:pre-wrap;'
                        f'font-family:monospace;max-height:260px;overflow:auto">'
                        f'{_rd_text[:800]}{"…" if len(_rd_text) > 800 else ""}</div>',
                        unsafe_allow_html=True,
                    )
                st.write("")
                st.button(
                    "🔍 Run Pre-pass →", type="primary", key="rd_btn_prepass",
                    use_container_width=True, disabled=not ollama_ok,
                    on_click=lambda: None,  # actual logic below
                )
                if st.session_state.get("rd_btn_prepass"):
                    with st.spinner("Analysing transcript and planning extractions…"):
                        _pp_result = run_pre_pass(_rd_text, _rd_model)
                    st.session_state["rd_prepass"] = _pp_result
                    st.session_state["rd_stage"]   = 1
                    st.rerun()

            # ── Stage 1: Pre-pass done — show extraction plan ─────────────────
            elif _rd_stage == 1:
                _rd_pp = st.session_state["rd_prepass"]

                # Subject callout
                _rd_reasoning_html = (
                    '<br><span style="opacity:0.5;font-size:0.8rem">'
                    + _rd_pp.reasoning[:220]
                    + ("…" if len(_rd_pp.reasoning) > 220 else "")
                    + "</span>"
                ) if _rd_pp.reasoning else ""
                st.markdown(
                    f'<div style="background:rgba(0,180,216,0.08);border:1px solid rgba(0,180,216,0.2);'
                    f'border-radius:10px;padding:12px 18px;margin-bottom:16px;font-size:0.88rem">'
                    f'<span style="opacity:0.5;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em">Subject detected</span><br>'
                    f'<span style="font-weight:600;color:#00B4D8">{_rd_pp.main_subject_id}</span>'
                    f'{_rd_reasoning_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Pass cards
                _cat_colors = {
                    "core_value": "#00B4D8", "long_term_goal": "#06D6A0",
                    "short_term_state": "#EF476F", "short_term_value": "#EF476F",
                }
                cards_html = ""
                for _p in sorted(_rd_pp.passes, key=lambda x: x.priority):
                    _accent = _cat_colors.get(_p.category.split("|")[0], "#B388FF")
                    _label  = _p.category.replace("|", " · ").replace("_", " ").title()
                    _focus  = _p.focus_prompt or "Standard extraction pass"
                    cards_html += (
                        f'<div class="rd-pass-card">'
                        f'<div class="rd-pass-head">'
                        f'<div class="rd-pass-accent" style="background:{_accent}"></div>'
                        f'<div class="rd-pass-tag">{_label}</div>'
                        f'<div class="rd-pass-pri">priority {_p.priority}</div>'
                        f'</div>'
                        f'<div class="rd-pass-focus">{_focus}</div>'
                        f'</div>'
                    )
                st.markdown(cards_html, unsafe_allow_html=True)
                st.write("")

                if st.button(
                    f"⚡ Run {len(_rd_pp.passes)} Extraction Pass{'es' if len(_rd_pp.passes) != 1 else ''} →",
                    type="primary", key="rd_btn_extract",
                    use_container_width=True, disabled=not ollama_ok,
                ):
                    _exts = []
                    with st.spinner(f"Running {len(_rd_pp.passes)} extraction pass(es)…"):
                        for _p in sorted(_rd_pp.passes, key=lambda x: x.priority):
                            _ext_result, _ = run_focused_extraction(
                                _rd_text, _p.category, _p.focus_prompt,
                                _rd_pp.main_subject_id, _rd_model,
                            )
                            _exts.append(_ext_result)
                    st.session_state["rd_extractions"] = _exts
                    st.session_state["rd_stage"]       = 2
                    st.rerun()

            # ── Stage 2: Extractions done — show per-category results ─────────
            elif _rd_stage == 2:
                _rd_exts = st.session_state["rd_extractions"]
                _total   = sum(len(e.items) for e in _rd_exts)

                st.markdown(
                    f'<div style="font-size:0.75rem;opacity:0.45;text-transform:uppercase;letter-spacing:0.09em;margin-bottom:14px">'
                    f'{_total} item{"s" if _total != 1 else ""} extracted across {len(_rd_exts)} pass{"es" if len(_rd_exts) != 1 else ""}</div>',
                    unsafe_allow_html=True,
                )

                _cat_icons = {
                    "core_value": "💎", "long_term_goal": "🎯",
                    "short_term_state": "⚡", "short_term_value": "⚡",
                }
                for _ext in _rd_exts:
                    _icon  = _cat_icons.get(_ext.category.split("|")[0], "✦")
                    _label = _ext.category.replace("|", " · ").replace("_", " ").title()
                    st.markdown(
                        f'<div class="rd-group">'
                        f'<div class="rd-group-title">{_icon} {_label} &nbsp;·&nbsp; {len(_ext.items)} item{"s" if len(_ext.items) != 1 else ""}</div>'
                        f'{_rd_insight_rows_html(_ext.items)}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                st.write("")
                if st.button("🔄 Run Dedup & Cross-Reason →", type="primary", key="rd_btn_dedup",
                             use_container_width=True, disabled=not ollama_ok):
                    with st.spinner("Merging, deduplicating, and cross-reasoning across all passes…"):
                        _dd_result, _ = run_dedup(_rd_exts, _rd_model)
                    st.session_state["rd_dedup"] = _dd_result
                    st.session_state["rd_stage"] = 3
                    st.rerun()

            # ── Stage 3: Dedup done — two-col layout + cross-insights ─────────
            elif _rd_stage == 3:
                _rd_dd = st.session_state["rd_dedup"]
                _col1, _col2 = st.columns(2, gap="large")
                with _col1:
                    st.markdown(_rd_group_html("Core Values", "💎", _rd_dd.core_values), unsafe_allow_html=True)
                    st.markdown(_rd_group_html("Long-term Goals", "🎯", _rd_dd.long_term_goals), unsafe_allow_html=True)
                with _col2:
                    st.markdown(_rd_group_html("Current States", "⚡", _rd_dd.short_term_values), unsafe_allow_html=True)
                    st.markdown(_rd_group_html("Interests", "💡", _rd_dd.interests), unsafe_allow_html=True)

                if _rd_dd.cross_category_insights:
                    _insights_clean = [_rd_clean_insight(s) for s in _rd_dd.cross_category_insights]
                    callouts = "".join(f'<div class="rd-callout">🔗 {s}</div>' for s in _insights_clean if s.strip())
                    st.markdown(
                        f'<div style="margin-top:6px">'
                        f'<div class="rd-group-title" style="margin-bottom:10px">CROSS-CATEGORY INSIGHTS</div>'
                        f'{callouts}</div>',
                        unsafe_allow_html=True,
                    )

                st.write("")
                if st.button("🪪 Extract Identity →", type="primary", key="rd_btn_identity",
                             use_container_width=True, disabled=not ollama_ok):
                    with st.spinner("Extracting biographical identity…"):
                        _id_result, _ = run_identity_extraction(
                            _rd_text,
                            st.session_state["rd_G_before"],
                            st.session_state["rd_prepass"].main_subject_id,
                            _rd_model,
                        )
                    st.session_state["rd_identity"] = _id_result
                    st.session_state["rd_stage"]    = 4
                    st.rerun()

            # ── Stage 4: Identity — styled fields card ────────────────────────
            elif _rd_stage == 4:
                _rd_id_obj = st.session_state["rd_identity"].identity
                _field_icons = {"name": "👤", "age": "🎂", "occupation": "💼", "location": "📍"}
                _field_rows = ""
                for _f in ("name", "age", "occupation", "location"):
                    _val = getattr(_rd_id_obj, _f)
                    if _val:
                        _c = getattr(_rd_id_obj, f"{_f}_confidence")
                        _color = _rd_conf_color(_c)
                        _field_rows += (
                            f'<div class="rd-id-field">'
                            f'<div class="rd-id-icon">{_field_icons[_f]}</div>'
                            f'<div class="rd-id-label">{_f.title()}</div>'
                            f'<div class="rd-id-value">{_val}</div>'
                            f'<div class="rd-conf-bar" style="width:52px">'
                            f'<div class="rd-conf-bar-fill" style="width:{_c*100:.0f}%;background:{_color}"></div></div>'
                            f'<div class="rd-conf-pct">{_c:.0%}</div>'
                            f'</div>'
                        )

                if _field_rows:
                    st.markdown(
                        f'<div class="rd-group">'
                        f'<div class="rd-group-title">🪪 IDENTITY</div>'
                        f'{_field_rows}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("No identity information detected in this transcript.")

                st.write("")
                if st.button("🛡 Apply to Profile →", type="primary", key="rd_btn_gatekeeper",
                             use_container_width=True, disabled=not ollama_ok):
                    with st.spinner("Running gatekeeper — analysing what to add, update, or remove…"):
                        _G_live    = load_graph(profile=selected_profile)
                        _G_before  = st.session_state["rd_G_before"]
                        _G_live, _ = run_gatekeeper(_G_live, st.session_state["rd_dedup"], _rd_model)
                        _G_live    = apply_identity(_G_live, st.session_state["rd_identity"])
                        save_graph(_G_live, profile=selected_profile)
                        save_snapshot(_G_live, model=_rd_model, profile=selected_profile)
                        _rd_diff   = diff_graphs(_G_before, _G_live)
                        st.session_state[SS_LAST_DIFF] = _rd_diff
                        # Record this day in the run log so the timeline tile turns green
                        _rd_save_run_log(
                            selected_profile,
                            st.session_state["rd_file"],
                            max(_G_before.number_of_nodes() - 1, 0),
                            max(_G_live.number_of_nodes() - 1, 0),
                        )
                    st.session_state["rd_stage"] = 5
                    st.rerun()

            # ── Stage 5: Done — diff summary + advance to next day ───────────
            elif _rd_stage == 5:
                _G_after  = load_graph(profile=selected_profile)
                _G_before = st.session_state["rd_G_before"]
                _n_before = max(_G_before.number_of_nodes() - 1, 0)
                _n_after  = max(_G_after.number_of_nodes() - 1, 0)
                _delta    = _n_after - _n_before

                # Summary stat cards
                st.markdown(
                    f'<div style="display:flex;gap:12px;margin-bottom:20px">'
                    f'<div style="flex:1;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:16px 20px;text-align:center">'
                    f'<div style="font-size:0.68rem;opacity:0.38;text-transform:uppercase;letter-spacing:0.09em;margin-bottom:6px">Insights before</div>'
                    f'<div style="font-size:1.8rem;font-weight:700">{_n_before}</div></div>'
                    f'<div style="flex:1;background:rgba(6,214,160,0.08);border:1px solid rgba(6,214,160,0.2);border-radius:12px;padding:16px 20px;text-align:center">'
                    f'<div style="font-size:0.68rem;opacity:0.38;text-transform:uppercase;letter-spacing:0.09em;margin-bottom:6px">Insights after</div>'
                    f'<div style="font-size:1.8rem;font-weight:700;color:#06D6A0">{_n_after}'
                    f'<span style="font-size:1rem;margin-left:6px">{"+" if _delta >= 0 else ""}{_delta}</span></div></div>'
                    f'<div style="flex:1;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:16px 20px;text-align:center">'
                    f'<div style="font-size:0.68rem;opacity:0.38;text-transform:uppercase;letter-spacing:0.09em;margin-bottom:6px">Graph edges</div>'
                    f'<div style="font-size:1.8rem;font-weight:700">{_G_after.number_of_edges()}</div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                if SS_LAST_DIFF in st.session_state:
                    render_diff(st.session_state[SS_LAST_DIFF])

                # Next day auto-advance
                st.write("")
                _cur_day_idx  = st.session_state.get("rd_selected_day_idx", -1)
                _next_day_idx = _cur_day_idx + 1
                _has_next     = _next_day_idx < len(_rd_txt_files)
                _next_label   = (
                    f"▶ Run {_rd_txt_files[_next_day_idx].stem} →"
                    if _has_next else "✅ All days complete"
                )
                if st.button(_next_label, type="primary", key="rd_btn_next",
                             use_container_width=True, disabled=not _has_next):
                    _rd_load_day(_rd_txt_files[_next_day_idx], selected_profile, _next_day_idx)
                    st.rerun()


# ─── History / Diff tab ───────────────────────────────────────────────────────

with tab_history:
    section_header("📊", "Snapshot History")
    snapshots = list_snapshots(profile=selected_profile)

    if len(snapshots) < 2:
        if snapshots:
            s = snapshots[0]
            st.caption(f"1 snapshot: {s['filename']}  ·  {s['node_count']} nodes  ·  model: {s['model']}")
        empty_state("📊", "Run the pipeline at least twice to compare snapshots.")
    else:
        def snap_label(s: dict) -> str:
            ts = s["timestamp"][:19].replace("T", "  ") if len(s["timestamp"]) > 19 else s["timestamp"]
            return f"{ts}  ·  {s['node_count']} nodes  ·  {s['model']}"

        # ── Diff viewer ──────────────────────────────────────────────────────
        col_a, col_b = st.columns(2)
        with col_a:
            idx_a = st.selectbox(
                "From (older)",
                range(len(snapshots)),
                format_func=lambda i: snap_label(snapshots[i]),
                index=min(1, len(snapshots) - 1),
                key="hist_from",
            )
        with col_b:
            idx_b = st.selectbox(
                "To (newer)",
                range(len(snapshots)),
                format_func=lambda i: snap_label(snapshots[i]),
                index=0,
                key="hist_to",
            )

        if idx_a == idx_b:
            st.warning("Select two different snapshots.")
        else:
            G_a = load_graph(snapshots[idx_a]["path"])
            G_b = load_graph(snapshots[idx_b]["path"])
            diff = diff_graphs(G_a, G_b)

            col1, col2, col3 = st.columns(3)
            col1.metric("Added", len(diff.added))
            col2.metric("Removed", len(diff.removed))
            col3.metric("Boosted", len(diff.strengthened))

            if diff.added:
                st.markdown("#### Added")
                for nd in diff.added:
                    conf = f"  — {nd.new_confidence:.0%}" if nd.new_confidence else ""
                    st.success(f"+ **{nd.label}** `{nd.node_type}`{conf}")

            if diff.removed:
                st.markdown("#### Removed")
                for nd in diff.removed:
                    st.error(f"− **{nd.label}** `{nd.node_type}`")

            if diff.strengthened:
                st.markdown("#### Strengthened")
                for nd in diff.strengthened:
                    delta_str = f"  +{nd.delta:.2f}" if nd.delta else ""
                    st.info(f"↑ **{nd.label}** `{nd.node_type}`{delta_str}")

            if diff.unchanged:
                with st.expander(f"Unchanged — {len(diff.unchanged)} nodes", expanded=False):
                    for nd in diff.unchanged:
                        st.caption(f"{nd.node_type}  ·  {nd.label}")

        # ── Rollback ──────────────────────────────────────────────────────────
        st.divider()
        section_header("↩️", "Rollback")
        st.caption(
            "Restore the live graph to any previous snapshot. "
            "A safety backup is automatically saved first."
        )

        rollback_idx = st.selectbox(
            "Roll back to",
            range(len(snapshots)),
            format_func=lambda i: snap_label(snapshots[i]),
            key="rollback_target",
        )

        if st.button("↩ Rollback to selected snapshot", type="secondary"):
            try:
                restored = rollback_to_snapshot(snapshots[rollback_idx]["path"], profile=selected_profile)
                logger.info("Rollback executed via UI to %s", snapshots[rollback_idx]["filename"])
                st.success(
                    f"Rolled back to **{snapshots[rollback_idx]['filename']}** "
                    f"({restored.number_of_nodes()} nodes). "
                    f"A safety backup was saved automatically."
                )
                st.rerun()
            except (FileNotFoundError, ValueError) as e:
                logger.error("Rollback failed: %s", e)
                st.error(f"Rollback failed: {e}")


# ─── Raw Data tab ─────────────────────────────────────────────────────────────

with tab_raw:
    section_header("🗂️", "Raw Graph Data")
    G_raw    = load_graph(profile=selected_profile)
    raw_data = _jg.node_link_data(G_raw)
    st.json(raw_data, expanded=False)
    st.download_button(
        "Download profile graph JSON",
        data=json.dumps(raw_data, indent=2),
        file_name=f"profile_graph_{selected_profile}.json",
        mime="application/json",
    )


# ─── Log tab ──────────────────────────────────────────────────────────────────

with tab_log:
    section_header("📋", "Pipeline Log")

    if SS_PIPELINE_LOG in st.session_state:
        st.code(st.session_state[SS_PIPELINE_LOG], language="json")
    else:
        empty_state("📋", "Run the pipeline to see LLM reasoning here.")

    # ── Active prompts (transparency) ─────────────────────────────────────────
    st.divider()
    section_header("📝", "Active Prompts")
    st.caption("The exact instructions sent to the LLM at each pipeline phase.")
    try:
        from src.core.llm_client import load_prompt
        prompt_groups = {
            "Pre-pass (extraction planning)": "pipeline/pre_pass",
            "Extraction: Short-term states": "extraction/short_term",
            "Extraction: Long-term goals": "extraction/long_term",
            "Extraction: Core values": "extraction/core_value",
            "Dedup + Cross-reasoning": "pipeline/dedup",
            "Identity extraction": "pipeline/identity",
            "Gatekeeper (graph delta)": "gatekeeper/gatekeeper",
        }
        for label, prompt_name in prompt_groups.items():
            try:
                with st.expander(label, expanded=False):
                    st.text_area(
                        prompt_name,
                        load_prompt(prompt_name),
                        height=200,
                        disabled=True,
                        label_visibility="collapsed",
                    )
            except FileNotFoundError:
                pass
    except Exception as e:
        st.error(f"Error loading prompts: {e}")

    # ── Session telemetry ─────────────────────────────────────────────────────
    st.divider()
    section_header("📈", "Session Metrics")

    collector = get_collector()
    summary   = collector.get_summary()

    if summary:
        c1, c2, c3 = st.columns(3)
        c1.metric("Runs this session", summary["total_runs"])
        c2.metric("Avg pipeline time", f"{summary['avg_pipeline_sec']}s")
        c3.metric("Enum coercions", summary["total_coercions"])

        if summary["total_coercions"] > 0:
            st.caption(
                "⚠ Coercions indicate the LLM returned unexpected enum values that were "
                "automatically corrected. Check logs/serendipity.log for details."
            )

        runs = collector.get_all_runs()
        if runs:
            run_rows = [
                {
                    "Run":            r.run_id,
                    "Model":          r.model,
                    "Turns":          r.turns_parsed,
                    "Values":         r.core_values_found,
                    "Goals":          r.goals_found,
                    "States":         r.states_found,
                    "Relationships":  r.relationships_found,
                    "Extract (s)":    round(r.extraction_latency_sec, 2),
                    "Gate (s)":       round(r.gatekeeper_latency_sec, 2),
                    "Coercions":      r.coercions_triggered,
                    "Nodes Δ":        r.nodes_after - r.nodes_before if r.nodes_after else "—",
                }
                for r in runs
            ]
            st.dataframe(
                pd.DataFrame(run_rows),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.caption("No pipeline runs recorded this session.")
