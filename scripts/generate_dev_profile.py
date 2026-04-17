"""
Generate a realistic fake "dev" profile — simulating 30 days of daily audio uploads
from a developer building AI products. Creates the main graph + historical snapshots.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx
from networkx.readwrite import json_graph

from src.storage.graph import (
    get_graph_path, get_snapshots_dir, make_node_id, _fresh_graph,
)

PROFILE = "dev"
BASE_DATE = datetime(2026, 3, 1, tzinfo=timezone.utc)  # day 1 = March 1


def ts(day: int, hour: int = 20, minute: int = 30) -> str:
    """ISO timestamp for a given day offset."""
    dt = BASE_DATE + timedelta(days=day - 1, hours=hour, minutes=minute)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def add_value(G, label, weight, evidence_count=1, **meta):
    nid = make_node_id("core_value", label)
    G.add_node(nid, node_type="core_value", label=label, evidence_count=evidence_count, **meta)
    G.add_edge("user", nid, relation="holds_value", weight=weight, intensity=weight)


def add_goal(G, label, weight, **meta):
    nid = make_node_id("long_term_goal", label)
    G.add_node(nid, node_type="long_term_goal", label=label, evidence_count=1, **meta)
    G.add_edge("user", nid, relation="pursues_goal", weight=weight)


def add_state(G, label, intensity, category="stressor"):
    nid = make_node_id("short_term_state", label)
    G.add_node(nid, node_type="short_term_state", label=label, category=category)
    G.add_edge("user", nid, relation="experiencing", intensity=intensity)


def add_person(G, name, rel_type, tone, strength):
    nid = make_node_id("person", name)
    G.add_node(nid, node_type="person", name=name)
    G.add_edge("user", nid, relation="knows", relationship_type=rel_type,
               tone=tone, strength=strength)


def add_custom(G, node_type, label, weight, evidence_count=1):
    nid = make_node_id(node_type, label)
    G.add_node(nid, node_type=node_type, label=label, evidence_count=evidence_count)
    G.add_edge("user", nid, relation="has_trait", weight=weight)


def set_identity(G, name, name_conf, age, age_conf, occupation, occ_conf,
                 location, loc_conf, additional=None, additional_conf=None):
    u = G.nodes["user"]
    u["identity_name"] = name
    u["identity_name_confidence"] = name_conf
    u["identity_age"] = age
    u["identity_age_confidence"] = age_conf
    u["identity_occupation"] = occupation
    u["identity_occupation_confidence"] = occ_conf
    u["identity_location"] = location
    u["identity_location_confidence"] = loc_conf
    if additional:
        u["identity_additional"] = additional
        u["identity_additional_confidence"] = additional_conf or {}


def save(G, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_graph.node_link_data(G), f, indent=2)
    print(f"  Saved: {path}  ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")


# ═══════════════════════════════════════════════════════════════════════════════
# Day 7 snapshot — first week: basic profile forming
# ═══════════════════════════════════════════════════════════════════════════════

def build_day7():
    G = _fresh_graph()
    G.graph.update(created=ts(1), last_updated=ts(7), last_model="qwen3.5:latest", profile_id=PROFILE)

    set_identity(G, "Dev", 0.6, "25", 0.4, "Software Developer", 0.7,
                 None, 0.0)

    # Core values (just starting to emerge)
    add_value(G, "Privacy First", 0.7, evidence_count=3)
    add_value(G, "Continuous Learning", 0.7, evidence_count=3)
    add_value(G, "Building Things That Matter", 0.6, evidence_count=2)

    # Goals (early mentions)
    add_goal(G, "Build an AI-Powered Product", 0.7)
    add_goal(G, "Learn Machine Learning Deeply", 0.6)

    # Short-term states (day 7 only)
    add_state(G, "Excited About New Project Idea", 0.85, category="emotion")
    add_state(G, "Overwhelmed by Framework Choices", 0.6, category="stressor")

    # People
    add_person(G, "Alex", "colleague", "collaborative", 0.65)
    add_person(G, "Sarah", "mentor", "supportive", 0.6)
    add_person(G, "Mom", "family", "supportive", 0.7)

    return G


# ═══════════════════════════════════════════════════════════════════════════════
# Day 14 snapshot — second week: deeper patterns
# ═══════════════════════════════════════════════════════════════════════════════

def build_day14():
    G = _fresh_graph()
    G.graph.update(created=ts(1), last_updated=ts(14), last_model="qwen3.5:latest", profile_id=PROFILE)

    set_identity(G, "Dev", 0.75, "25", 0.5, "Software Developer", 0.85,
                 "San Francisco", 0.5)

    # Core values (strengthening)
    add_value(G, "Privacy First", 0.8, evidence_count=6)
    add_value(G, "Continuous Learning", 0.8, evidence_count=5)
    add_value(G, "Building Things That Matter", 0.7, evidence_count=4)
    add_value(G, "Quality Over Speed", 0.7, evidence_count=3)
    add_value(G, "Open Source Philosophy", 0.6, evidence_count=2)

    # Goals (more defined)
    add_goal(G, "Build an AI-Powered Product", 0.8)
    add_goal(G, "Learn Machine Learning Deeply", 0.7)
    add_goal(G, "Start a Tech Company", 0.6)
    add_goal(G, "Contribute to Open Source", 0.6)

    # Short-term states (day 14)
    add_state(G, "Debugging a Tricky Async Bug", 0.7, category="stressor")
    add_state(G, "Proud of First Working Prototype", 0.9, category="emotion")
    add_state(G, "Sleep Deprived From Late Coding", 0.65, category="stressor")

    # People (expanding network)
    add_person(G, "Alex", "colleague", "collaborative", 0.75)
    add_person(G, "Sarah", "mentor", "supportive", 0.7)
    add_person(G, "Mom", "family", "supportive", 0.75)
    add_person(G, "Marcus", "friend", "collaborative", 0.6)
    add_person(G, "Priya", "colleague", "collaborative", 0.55)

    # Custom categories starting to appear
    add_custom(G, "technical_skill", "Python", 0.8, evidence_count=5)
    add_custom(G, "technical_skill", "LLM APIs", 0.7, evidence_count=3)
    add_custom(G, "technical_skill", "Streamlit", 0.6, evidence_count=2)

    return G


# ═══════════════════════════════════════════════════════════════════════════════
# Day 21 snapshot — third week: rich profile
# ═══════════════════════════════════════════════════════════════════════════════

def build_day21():
    G = _fresh_graph()
    G.graph.update(created=ts(1), last_updated=ts(21), last_model="qwen3.5:latest", profile_id=PROFILE)

    set_identity(G, "Dev", 0.85, "25", 0.65, "AI Engineer", 0.9,
                 "San Francisco", 0.7,
                 additional={"education": "CS degree", "hobby": "Running"},
                 additional_conf={"education": 0.6, "hobby": 0.5})

    # Core values (well-established)
    add_value(G, "Privacy First", 0.9, evidence_count=10)
    add_value(G, "Continuous Learning", 0.9, evidence_count=9)
    add_value(G, "Building Things That Matter", 0.85, evidence_count=7)
    add_value(G, "Quality Over Speed", 0.8, evidence_count=5)
    add_value(G, "Open Source Philosophy", 0.75, evidence_count=4)
    add_value(G, "User Empowerment", 0.7, evidence_count=3)
    add_value(G, "Simplicity in Design", 0.7, evidence_count=4)

    # Goals (clearer picture)
    add_goal(G, "Build an AI-Powered Product", 0.9)
    add_goal(G, "Learn Machine Learning Deeply", 0.8)
    add_goal(G, "Start a Tech Company", 0.7)
    add_goal(G, "Contribute to Open Source", 0.7)
    add_goal(G, "Write a Technical Blog", 0.6)
    add_goal(G, "Build a Developer Community", 0.55)

    # Short-term states (day 21)
    add_state(G, "Refactoring Pipeline Architecture", 0.75, category="focus")
    add_state(G, "Frustrated With LLM Inconsistency", 0.65, category="stressor")
    add_state(G, "Energized by User Feedback", 0.8, category="emotion")

    # People (deeper relationships)
    add_person(G, "Alex", "colleague", "collaborative", 0.85)
    add_person(G, "Sarah", "mentor", "supportive", 0.8)
    add_person(G, "Mom", "family", "supportive", 0.8)
    add_person(G, "Marcus", "friend", "collaborative", 0.7)
    add_person(G, "Priya", "colleague", "collaborative", 0.65)
    add_person(G, "Jordan", "friend", "supportive", 0.6)
    add_person(G, "Professor Chen", "mentor", "supportive", 0.55)

    # Technical skills
    add_custom(G, "technical_skill", "Python", 0.9, evidence_count=10)
    add_custom(G, "technical_skill", "LLM APIs", 0.8, evidence_count=6)
    add_custom(G, "technical_skill", "Streamlit", 0.75, evidence_count=5)
    add_custom(G, "technical_skill", "NetworkX", 0.7, evidence_count=4)
    add_custom(G, "technical_skill", "Prompt Engineering", 0.75, evidence_count=5)
    add_custom(G, "technical_skill", "JavaScript", 0.6, evidence_count=3)

    # Work habits
    add_custom(G, "work_habit", "Late Night Coder", 0.8, evidence_count=8)
    add_custom(G, "work_habit", "Iterative Builder", 0.7, evidence_count=5)
    add_custom(G, "work_habit", "Rubber Duck Debugger", 0.6, evidence_count=3)

    return G


# ═══════════════════════════════════════════════════════════════════════════════
# Day 30 — full profile (current state)
# ═══════════════════════════════════════════════════════════════════════════════

def build_day30():
    G = _fresh_graph()
    G.graph.update(created=ts(1), last_updated=ts(30), last_model="qwen3.5:latest", profile_id=PROFILE)

    set_identity(G, "Dev", 0.9, "25", 0.75, "AI Engineer & Founder", 0.95,
                 "San Francisco", 0.85,
                 additional={
                     "education": "CS degree",
                     "hobby": "Running",
                     "side_project": "Serendipity",
                     "favorite_language": "Python",
                 },
                 additional_conf={
                     "education": 0.8,
                     "hobby": 0.7,
                     "side_project": 0.95,
                     "favorite_language": 0.9,
                 })

    # ── Core Values (30 days of reinforcement) ──────────────────────────────
    add_value(G, "Privacy First", 1.0, evidence_count=16)
    add_value(G, "Continuous Learning", 1.0, evidence_count=14)
    add_value(G, "Building Things That Matter", 0.95, evidence_count=11)
    add_value(G, "Quality Over Speed", 0.9, evidence_count=8)
    add_value(G, "Open Source Philosophy", 0.85, evidence_count=7)
    add_value(G, "User Empowerment", 0.8, evidence_count=5)
    add_value(G, "Simplicity in Design", 0.8, evidence_count=6)
    add_value(G, "Persistence Through Setbacks", 0.85, evidence_count=6)
    add_value(G, "Creative Problem Solving", 0.9, evidence_count=9)
    add_value(G, "Local-First Architecture", 0.85, evidence_count=5)

    # ── Long-Term Goals ─────────────────────────────────────────────────────
    add_goal(G, "Build an AI-Powered Product", 0.95)
    add_goal(G, "Learn Machine Learning Deeply", 0.85)
    add_goal(G, "Start a Tech Company", 0.8)
    add_goal(G, "Contribute to Open Source", 0.8)
    add_goal(G, "Write a Technical Blog", 0.7)
    add_goal(G, "Build a Developer Community", 0.65)
    add_goal(G, "Create Tools for Personal AI", 0.9)
    add_goal(G, "Give a Conference Talk", 0.55)

    # ── Short-Term States (day 30 — latest session) ─────────────────────────
    add_state(G, "Excited About Multi-Pass Pipeline", 0.9, category="emotion")
    add_state(G, "Planning the UI Redesign", 0.75, category="focus")
    add_state(G, "Tired From Late Night Coding", 0.6, category="stressor")
    add_state(G, "Motivated by Progress This Month", 0.85, category="emotion")
    add_state(G, "Thinking About Monetization", 0.5, category="focus")

    # ── Relationships ───────────────────────────────────────────────────────
    add_person(G, "Alex", "colleague", "collaborative", 0.9)
    add_person(G, "Sarah", "mentor", "supportive", 0.85)
    add_person(G, "Mom", "family", "supportive", 0.9)
    add_person(G, "Dad", "family", "supportive", 0.7)
    add_person(G, "Marcus", "friend", "collaborative", 0.8)
    add_person(G, "Priya", "colleague", "collaborative", 0.75)
    add_person(G, "Jordan", "friend", "supportive", 0.7)
    add_person(G, "Professor Chen", "mentor", "supportive", 0.65)
    add_person(G, "Omar", "friend", "collaborative", 0.6)
    add_person(G, "Lisa", "colleague", "neutral", 0.55)

    # ── Technical Skills (custom category) ──────────────────────────────────
    add_custom(G, "technical_skill", "Python", 1.0, evidence_count=18)
    add_custom(G, "technical_skill", "LLM APIs & Prompt Engineering", 0.9, evidence_count=12)
    add_custom(G, "technical_skill", "Streamlit", 0.85, evidence_count=8)
    add_custom(G, "technical_skill", "NetworkX & Graph Theory", 0.8, evidence_count=6)
    add_custom(G, "technical_skill", "JavaScript & Web Dev", 0.7, evidence_count=5)
    add_custom(G, "technical_skill", "Docker & DevOps", 0.6, evidence_count=3)
    add_custom(G, "technical_skill", "Pydantic & Data Validation", 0.75, evidence_count=5)
    add_custom(G, "technical_skill", "Audio Processing", 0.55, evidence_count=2)

    # ── Work Habits (custom category) ───────────────────────────────────────
    add_custom(G, "work_habit", "Late Night Coder", 0.9, evidence_count=14)
    add_custom(G, "work_habit", "Iterative Builder", 0.85, evidence_count=9)
    add_custom(G, "work_habit", "Rubber Duck Debugger", 0.7, evidence_count=5)
    add_custom(G, "work_habit", "Whiteboard First", 0.65, evidence_count=4)
    add_custom(G, "work_habit", "Music While Coding", 0.8, evidence_count=7)

    # ── Interests (custom category) ─────────────────────────────────────────
    add_custom(G, "interest", "AI Safety & Alignment", 0.8, evidence_count=6)
    add_custom(G, "interest", "Knowledge Graphs", 0.85, evidence_count=7)
    add_custom(G, "interest", "Running & Fitness", 0.7, evidence_count=5)
    add_custom(G, "interest", "Indie Hacking & Startups", 0.75, evidence_count=5)
    add_custom(G, "interest", "Podcast Listening", 0.6, evidence_count=4)
    add_custom(G, "interest", "Mechanical Keyboards", 0.55, evidence_count=2)

    return G


# ═══════════════════════════════════════════════════════════════════════════════
# Generate everything
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\nGenerating dev profile (30 days of simulated data)...\n")

    # Build all snapshots
    snapshots = {
        7:  build_day7(),
        14: build_day14(),
        21: build_day21(),
        30: build_day30(),
    }

    # Save snapshots
    snap_dir = get_snapshots_dir(PROFILE)
    snap_dir.mkdir(parents=True, exist_ok=True)

    for day, G in snapshots.items():
        dt = BASE_DATE + timedelta(days=day - 1, hours=20, minutes=30)
        ts_str = dt.strftime("%Y%m%d_%H%M%S")
        snap_path = snap_dir / f"profile_{ts_str}_qwen3_5_latest.json"
        print(f"Day {day:2d}:")
        save(G, snap_path)

    # Save current graph (day 30)
    current = snapshots[30]
    graph_path = get_graph_path(PROFILE)
    print(f"\nCurrent profile:")
    save(current, graph_path)

    # Summary
    G = current
    n_values = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "core_value")
    n_goals = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "long_term_goal")
    n_states = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "short_term_state")
    n_people = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "person" and d.get("name") != "User")
    n_custom = G.number_of_nodes() - 1 - n_values - n_goals - n_states - n_people  # minus "user"

    print(f"""
Profile Summary (Day 30):
  Core Values:      {n_values}
  Long-Term Goals:  {n_goals}
  Short-Term States:{n_states}
  Relationships:    {n_people}
  Custom Traits:    {n_custom}
  Total Nodes:      {G.number_of_nodes()}
  Total Edges:      {G.number_of_edges()}
""")

    print("Done! Switch to 'dev' profile in the Streamlit app to see it.\n")
