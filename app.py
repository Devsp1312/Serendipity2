"""
Serendipity — Streamlit Web Interface

Run with: streamlit run app.py
"""

# Set up logging before anything else so all imported modules get the handlers
from src.logger import setup_logging
setup_logging()
from src.logger import get_logger
logger = get_logger(__name__)

import json
import time
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from networkx.readwrite import json_graph as _jg

from src.config import (
    AUDIO_DIR,
    DEFAULT_PROFILE,
    SS_MODEL, SS_LOADED_TRANSCRIPT, SS_TRANSCRIPT_SOURCE,
    SS_LAST_DIFF, SS_PIPELINE_LOG, SS_LAST_RUN_METRICS, SS_PROFILE,
    VIZ_DEFAULT_HEIGHT,
)
from src.visualizer import build_visualizer_html
from src.graph_store import (
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
from src.ingestion import get_turn_stats, parse_transcript, format_transcript_for_llm
from src.extraction import run_extraction
from src.gatekeeper import run_gatekeeper, apply_identity
from src.multi_extract import run_multi_pass_pipeline
from src.llm_client import (
    OllamaUnavailableError,
    check_connection,
    list_models,
    get_extraction_prompt,
    get_gatekeeper_prompt,
)
from src.mock_data import MOCK_TRANSCRIPT
from src.schemas import LLMSchemaError
from src.telemetry import get_collector
from src.diarizer import run_diarization_pipeline


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
    col_s.metric("Strengthened", len(diff.strengthened))

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

    # ── Model selector
    available_models = list_models()
    if available_models:
        selected_model = st.selectbox(
            "Model",
            options=available_models,
            index=0,
            help="Ollama model for extraction & gatekeeper",
        )
        st.session_state[SS_MODEL] = selected_model
    else:
        st.warning("No models found — run `ollama pull llama3`")
        st.session_state.setdefault(SS_MODEL, "llama3")

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
        from src.graph_store import _fresh_graph
        save_graph(_fresh_graph(), profile=selected_profile)
        logger.info("Graph reset via sidebar button for profile=%s", selected_profile)
        st.success(f"Profile '{selected_profile}' reset.")
        st.rerun()


# ─── Main panel ───────────────────────────────────────────────────────────────

tab_3d, tab_profile, tab_upload, tab_history, tab_raw, tab_log = st.tabs([
    "🌐  Graph",
    "🧠  Profile",
    "📤  Upload & Process",
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


# ─── Upload & Process tab ─────────────────────────────────────────────────────

# Helper: load a local audio file from the audio files directory
def _load_local_audio(audio_path: Path) -> None:
    """Diarize a local audio file and store transcript in session state."""
    with st.spinner(f"Diarizing {audio_path.name}…"):
        try:
            diarized = run_diarization_pipeline(
                audio_path=str(audio_path),
                filename=audio_path.name,
            )
            transcript_text = diarized.labeled_text or diarized.raw_text
            st.session_state[SS_LOADED_TRANSCRIPT] = transcript_text
            st.session_state[SS_TRANSCRIPT_SOURCE] = audio_path.name
            speaker_info = f"{len(diarized.speakers)} speaker(s)"
            if diarized.recording_type:
                speaker_info += f"  ·  {diarized.recording_type}"
            if diarized.topic:
                speaker_info += f"  ·  {diarized.topic}"
            st.caption(speaker_info)
        except Exception as e:
            logger.error("Audio diarization failed for %s: %s", audio_path.name, e)
            st.error(f"Audio processing failed: {e}")


with tab_upload:
    # ── Local Audio Files panel ──────────────────────────────────────────────
    _AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
    local_audio_files = sorted(
        [p for p in AUDIO_DIR.iterdir() if p.suffix.lower() in _AUDIO_EXTS]
    ) if AUDIO_DIR.exists() else []

    if local_audio_files:
        section_header("🎵", "Local Audio Files")
        st.caption(f"{len(local_audio_files)} file(s) in `audio files/` — select one to diarize and run the pipeline.")
        st.write("")

        # Build display labels: "lebron_jj_redick_basketball.mp3  (27.0 MB)"
        file_labels = {
            p.name: f"{p.stem.replace('_', ' ').title()}  ({p.stat().st_size / 1024 / 1024:.1f} MB)"
            for p in local_audio_files
        }

        # Two columns: file selector | profile selector
        col_file, col_profile = st.columns([3, 1])
        with col_file:
            selected_filename = st.selectbox(
                "Audio file",
                options=[p.name for p in local_audio_files],
                format_func=lambda n: file_labels[n],
                label_visibility="collapsed",
            )
        with col_profile:
            # Inline profile selector for batch use (independent from sidebar)
            available_profiles = list_profiles()
            batch_profile = st.selectbox(
                "Profile",
                options=available_profiles,
                index=available_profiles.index(
                    normalize_profile_id(st.session_state.get(SS_PROFILE, DEFAULT_PROFILE))
                ) if normalize_profile_id(st.session_state.get(SS_PROFILE, DEFAULT_PROFILE)) in available_profiles else 0,
                key="batch_profile_selector",
            )

        selected_audio_path = AUDIO_DIR / selected_filename

        col_load, col_run, _ = st.columns([1, 1, 3])
        with col_load:
            if st.button("Load & diarize", use_container_width=True, type="secondary"):
                _load_local_audio(selected_audio_path)
                st.rerun()
        with col_run:
            model = st.session_state.get(SS_MODEL, "llama3")
            if st.button(
                f"Run pipeline",
                use_container_width=True,
                type="primary",
                disabled=not ollama_ok,
                key="local_run_btn",
            ):
                _load_local_audio(selected_audio_path)
                if SS_LOADED_TRANSCRIPT in st.session_state:
                    try:
                        turns = parse_transcript(st.session_state[SS_LOADED_TRANSCRIPT])
                        # Update sidebar profile to match batch selection
                        st.session_state[SS_PROFILE] = batch_profile
                        run_pipeline(turns, model, batch_profile)
                    except ValueError as e:
                        st.warning(f"Parse warning: {e}")

        st.divider()

    # ── Upload / Demo panel ──────────────────────────────────────────────────
    section_header("📤", "Upload a file")
    st.caption("Upload a conversation transcript or audio file. The multi-pass pipeline will extract insights and build your knowledge graph.")
    st.write("")

    uploaded_file = st.file_uploader(
        "Upload transcript (.txt) or audio (.mp3, .wav, .m4a)",
        type=["txt", "mp3", "wav", "m4a"],
        help="Text files should have speaker labels like 'User: ...' or 'Speaker B: ...'",
    )

    st.write("")
    use_demo = st.button("Or load demo transcript", use_container_width=False, type="secondary")

    if use_demo:
        st.session_state[SS_LOADED_TRANSCRIPT] = MOCK_TRANSCRIPT
        st.session_state[SS_TRANSCRIPT_SOURCE] = "Demo transcript"

    if uploaded_file is not None:
        filename   = uploaded_file.name
        file_bytes = uploaded_file.read()

        if filename.endswith(".txt"):
            st.session_state[SS_LOADED_TRANSCRIPT] = file_bytes.decode("utf-8", errors="replace")
            st.session_state[SS_TRANSCRIPT_SOURCE] = filename

        elif filename.endswith((".mp3", ".wav", ".m4a")):
            with st.spinner("Diarizing and identifying speakers…"):
                tmp_path = None
                try:
                    suffix = Path(filename).suffix or ".wav"
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(file_bytes)
                        tmp_path = tmp.name

                    diarized = run_diarization_pipeline(
                        audio_path=tmp_path,
                        filename=filename,
                    )
                    transcript_text = diarized.labeled_text or diarized.raw_text
                    st.session_state[SS_LOADED_TRANSCRIPT] = transcript_text
                    st.session_state[SS_TRANSCRIPT_SOURCE] = filename
                    st.caption(
                        f"Detected {len(diarized.speakers)} speaker(s)"
                        + (f"  ·  Type: {diarized.recording_type}" if diarized.recording_type else "")
                    )
                except Exception as e:
                    logger.error("Audio diarization failed for %s: %s", filename, e)
                    st.error(f"Audio processing failed: {e}")
                finally:
                    if tmp_path:
                        Path(tmp_path).unlink(missing_ok=True)

    # ── Loaded transcript section ────────────────────────────────────────────
    if SS_LOADED_TRANSCRIPT in st.session_state:
        source = st.session_state.get(SS_TRANSCRIPT_SOURCE, "")
        with st.expander(f"Transcript preview — {source}", expanded=False):
            preview = st.session_state[SS_LOADED_TRANSCRIPT]
            st.text(preview[:2000])
            if len(preview) > 2000:
                st.caption("(truncated)")

        col_clear, _ = st.columns([1, 4])
        with col_clear:
            if st.button("Clear transcript", use_container_width=True):
                _clear_transcript_state()
                st.rerun()

        turns = None
        try:
            turns = parse_transcript(st.session_state[SS_LOADED_TRANSCRIPT])
            stats = get_turn_stats(turns)
            cols  = st.columns(len(stats))
            for col, (speaker, count) in zip(cols, stats.items()):
                col.metric(f"{speaker}", f"{count} turns")
        except ValueError as e:
            st.warning(f"Parse warning: {e}")

        st.divider()
        model = st.session_state.get(SS_MODEL, "llama3")

        if st.button(
            f"Run pipeline  ·  {model}",
            disabled=(not ollama_ok or turns is None),
            type="primary",
            use_container_width=True,
        ):
            if not ollama_ok:
                st.error("Ollama is not running. Start it with: `ollama serve`")
            else:
                run_pipeline(turns, model, selected_profile)

    # Show last diff if no transcript is loaded
    if SS_LAST_DIFF in st.session_state and SS_LOADED_TRANSCRIPT not in st.session_state:
        render_diff(st.session_state[SS_LAST_DIFF])


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
            col3.metric("Strengthened", len(diff.strengthened))

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
        from src.llm_client import load_prompt
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
