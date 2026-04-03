"""
Generate rich demo profile data for Serendipity.

Creates three profiles (ted, sal, dev) that share a common foundation of values
but have diverged over time — each with unique goals, states, and relationships.

Each profile gets 5 snapshots simulating ~5 sessions of pipeline runs, so the
History tab shows real evolution: new nodes appearing, confidence growing,
relationships shifting, and states changing day to day.

Run once:  .venv/bin/python3 generate_demo_profiles.py
"""

import sys
import shutil
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.logger import setup_logging
setup_logging()

import networkx as nx
from src.graph_store import (
    _fresh_graph,
    make_node_id,
    save_graph,
    save_snapshot,
    get_graph_path,
    get_snapshots_dir,
)
from src.config import PROFILES_DIR


# ── Helper builders ──────────────────────────────────────────────────────────

def add_value(G, label, weight, evidence=1):
    nid = make_node_id("core_value", label)
    G.add_node(nid, node_type="core_value", label=label, evidence_count=evidence)
    G.add_edge("user", nid, relation="holds_value", weight=weight, intensity=weight)

def add_goal(G, label, weight):
    nid = make_node_id("long_term_goal", label)
    G.add_node(nid, node_type="long_term_goal", label=label)
    G.add_edge("user", nid, relation="pursues_goal", weight=weight, intensity=weight)

def add_state(G, label, intensity, category="stressor"):
    nid = make_node_id("short_term_state", label)
    G.add_node(nid, node_type="short_term_state", label=label, category=category)
    G.add_edge("user", nid, relation="experiencing", intensity=intensity)

def add_person(G, name, rel_type, tone, strength):
    nid = make_node_id("person", name)
    G.add_node(nid, node_type="person", name=name)
    G.add_edge(
        "user", nid, relation="knows",
        relationship_type=rel_type, tone=tone, strength=strength,
    )

def clear_states(G):
    """Remove all short-term state nodes (simulates the gatekeeper wiping them)."""
    to_remove = [n for n, d in G.nodes(data=True) if d.get("node_type") == "short_term_state"]
    G.remove_nodes_from(to_remove)

def strengthen(G, label, node_type, bump=0.08):
    """Bump confidence on an existing node to simulate the strengthen operation."""
    nid = make_node_id(node_type, label)
    if not G.has_edge("user", nid):
        return
    edge = G["user"][nid]
    for key in ("weight", "intensity", "strength"):
        if key in edge:
            edge[key] = round(min(1.0, edge[key] + bump), 3)
    if "evidence_count" in G.nodes.get(nid, {}):
        G.nodes[nid]["evidence_count"] += 1


# ═════════════════════════════════════════════════════════════════════════════
# TED — PhD student in NLP / machine learning
# 5 sessions tracking his thesis crunch, lab relationships, and identity
# ═════════════════════════════════════════════════════════════════════════════

def ted_snapshots():
    snapshots = []

    # ── Session 1: First conversation — baseline profile
    G = _fresh_graph()
    add_value(G, "Discipline", 0.62, evidence=1)
    add_value(G, "Intellectual Curiosity", 0.70, evidence=1)
    add_value(G, "Empathy", 0.55, evidence=1)
    add_value(G, "Academic Rigor", 0.60, evidence=1)
    add_goal(G, "Complete PhD thesis", 0.72)
    add_goal(G, "Build systems that understand people", 0.58)
    add_state(G, "Nervous about upcoming defense", 0.65, "stressor")
    add_state(G, "Adjusting to new lab schedule", 0.50, "task")
    add_person(G, "Dr. Patel", "mentor", "supportive", 0.60)
    add_person(G, "Alice", "colleague", "neutral", 0.50)
    G.graph["profile_id"] = "ted"
    snapshots.append(("llama3", G.copy()))

    # ── Session 2: Thesis pressure building, Alice becoming closer
    strengthen(G, "Discipline", "core_value", 0.08)
    strengthen(G, "Intellectual Curiosity", "core_value", 0.07)
    strengthen(G, "Academic Rigor", "core_value", 0.10)
    add_value(G, "Deep Work", 0.65, evidence=1)
    add_goal(G, "Publish at NeurIPS", 0.55)
    clear_states(G)
    add_state(G, "Stressed about methodology chapter", 0.78, "stressor")
    add_state(G, "Running BERT experiments late at night", 0.62, "task")
    add_state(G, "Skipping meals", 0.48, "stressor")
    # Alice relationship upgrades
    G["user"][make_node_id("person", "Alice")]["tone"] = "supportive"
    G["user"][make_node_id("person", "Alice")]["relationship_type"] = "friend"
    G["user"][make_node_id("person", "Alice")]["strength"] = 0.65
    add_person(G, "Speaker B", "friend", "collaborative", 0.55)
    snapshots.append(("llama3", G.copy()))

    # ── Session 3: Breakthrough on BERT, but burnout creeping in
    strengthen(G, "Discipline", "core_value", 0.10)
    strengthen(G, "Intellectual Curiosity", "core_value", 0.08)
    strengthen(G, "Deep Work", "core_value", 0.09)
    add_value(G, "Human-Centered AI", 0.68, evidence=1)
    add_value(G, "Physical Fitness", 0.45, evidence=1)
    strengthen(G, "Dr. Patel", "person", 0.10)
    strengthen(G, "Alice", "person", 0.12)
    add_goal(G, "Resume daily running habit", 0.50)
    add_goal(G, "Attend ML reading group", 0.45)
    clear_states(G)
    add_state(G, "Excited about 94% BERT accuracy", 0.82, "task")
    add_state(G, "Exhausted from all-nighters", 0.72, "stressor")
    add_state(G, "Missing exercise routine", 0.58, "craving")
    add_state(G, "Feeling disconnected from research community", 0.50, "stressor")
    add_person(G, "Marcus", "colleague", "collaborative", 0.48)
    snapshots.append(("llama3", G.copy()))

    # ── Session 4: The big conversation — identity and purpose
    strengthen(G, "Empathy", "core_value", 0.12)
    strengthen(G, "Human-Centered AI", "core_value", 0.12)
    strengthen(G, "Intellectual Curiosity", "core_value", 0.06)
    add_value(G, "Authenticity", 0.62, evidence=1)
    add_value(G, "Mentorship", 0.50, evidence=1)
    strengthen(G, "Alice", "person", 0.10)
    strengthen(G, "Speaker B", "friend", 0.08)
    strengthen(G, "Marcus", "person", 0.07)
    add_goal(G, "Bridge AI research and human understanding", 0.65)
    clear_states(G)
    add_state(G, "Stressed about thesis deadline", 0.88, "stressor")
    add_state(G, "Reconsidering career direction", 0.55, "task")
    add_state(G, "Grateful for Alice's support", 0.72, "craving")
    add_state(G, "Wanting to reconnect with running", 0.60, "craving")
    add_person(G, "Prof. Okonkwo", "mentor", "neutral", 0.42)
    snapshots.append(("mistral:7b", G.copy()))

    # ── Session 5: Current state — final thesis push
    strengthen(G, "Discipline", "core_value", 0.07)
    strengthen(G, "Academic Rigor", "core_value", 0.06)
    strengthen(G, "Human-Centered AI", "core_value", 0.05)
    strengthen(G, "Authenticity", "core_value", 0.10)
    strengthen(G, "Physical Fitness", "core_value", 0.08)
    strengthen(G, "Dr. Patel", "person", 0.08)
    strengthen(G, "Alice", "person", 0.05)
    add_value(G, "Resilience", 0.58, evidence=1)
    add_goal(G, "Take a real vacation after defense", 0.55)
    clear_states(G)
    add_state(G, "Final thesis push — 3 days left", 0.92, "stressor")
    add_state(G, "Excited about BERT results", 0.72, "task")
    add_state(G, "Considering ML reading group this weekend", 0.45, "task")
    add_state(G, "Craving a long run outdoors", 0.65, "craving")
    add_state(G, "Sleep deprived but hopeful", 0.60, "stressor")
    add_person(G, "Lab Group", "colleague", "supportive", 0.55)
    snapshots.append(("llama3", G.copy()))

    return snapshots


# ═════════════════════════════════════════════════════════════════════════════
# SAL — UX designer transitioning to freelance
# 5 sessions tracking her creative journey, client stress, and relationships
# ═════════════════════════════════════════════════════════════════════════════

def sal_snapshots():
    snapshots = []

    # ── Session 1: Just starting out — first freelance gigs
    G = _fresh_graph()
    add_value(G, "Empathy", 0.72, evidence=2)
    add_value(G, "Creative Expression", 0.68, evidence=1)
    add_value(G, "Discipline", 0.48, evidence=1)
    add_value(G, "Intellectual Curiosity", 0.55, evidence=1)
    add_goal(G, "Launch freelance design studio", 0.62)
    add_goal(G, "Build a design system library", 0.50)
    add_state(G, "Nervous about first solo client", 0.70, "stressor")
    add_state(G, "Excited to choose own projects", 0.65, "craving")
    add_person(G, "Lena", "mentor", "supportive", 0.65)
    add_person(G, "Jordan", "friend", "supportive", 0.72)
    add_person(G, "Mom", "family", "supportive", 0.80)
    G.graph["profile_id"] = "sal"
    snapshots.append(("llama3", G.copy()))

    # ── Session 2: First big client, discovering design values
    strengthen(G, "Empathy", "core_value", 0.08)
    strengthen(G, "Creative Expression", "core_value", 0.10)
    add_value(G, "User Advocacy", 0.65, evidence=1)
    add_value(G, "Work-Life Balance", 0.58, evidence=1)
    strengthen(G, "Lena", "person", 0.08)
    add_person(G, "Priya", "colleague", "collaborative", 0.55)
    add_goal(G, "Learn motion design", 0.48)
    clear_states(G)
    add_state(G, "Juggling two client projects", 0.72, "stressor")
    add_state(G, "Inspired by new Figma features", 0.60, "task")
    add_state(G, "Need to update portfolio", 0.55, "task")
    snapshots.append(("llama3", G.copy()))

    # ── Session 3: Growing pains — client conflict, financial stress
    strengthen(G, "Empathy", "core_value", 0.07)
    strengthen(G, "User Advocacy", "core_value", 0.12)
    strengthen(G, "Discipline", "core_value", 0.06)
    add_value(G, "Mindfulness", 0.52, evidence=1)
    add_value(G, "Integrity", 0.60, evidence=1)
    strengthen(G, "Jordan", "person", 0.10)
    strengthen(G, "Priya", "person", 0.10)
    add_person(G, "Marcus", "colleague", "tense", 0.40)
    add_goal(G, "Speak at a design conference", 0.45)
    add_goal(G, "Set up proper invoicing system", 0.55)
    clear_states(G)
    add_state(G, "Client wants scope changes without paying", 0.82, "stressor")
    add_state(G, "Anxious about financial stability", 0.68, "stressor")
    add_state(G, "Craving a creative side project", 0.62, "craving")
    add_state(G, "Considering therapy for work stress", 0.45, "task")
    snapshots.append(("mistral:7b", G.copy()))

    # ── Session 4: Finding community, building confidence
    strengthen(G, "Creative Expression", "core_value", 0.08)
    strengthen(G, "Mindfulness", "core_value", 0.12)
    strengthen(G, "Work-Life Balance", "core_value", 0.15)
    add_value(G, "Community Building", 0.55, evidence=1)
    add_value(G, "Teaching", 0.48, evidence=1)
    add_value(G, "Vulnerability", 0.45, evidence=1)
    strengthen(G, "Lena", "person", 0.10)
    strengthen(G, "Mom", "person", 0.08)
    add_person(G, "Design Collective", "colleague", "collaborative", 0.52)
    add_person(G, "Kai", "friend", "supportive", 0.58)
    add_goal(G, "Mentor junior designers", 0.50)
    add_goal(G, "Start a design podcast", 0.42)
    clear_states(G)
    add_state(G, "Joined local design meetup — loved it", 0.72, "task")
    add_state(G, "Still stressed about Marcus situation", 0.55, "stressor")
    add_state(G, "Planning first portfolio redesign", 0.65, "task")
    add_state(G, "Feeling more grounded after meditation", 0.50, "craving")
    snapshots.append(("llama3", G.copy()))

    # ── Session 5: Current state — hitting stride
    strengthen(G, "Empathy", "core_value", 0.06)
    strengthen(G, "Creative Expression", "core_value", 0.05)
    strengthen(G, "Community Building", "core_value", 0.10)
    strengthen(G, "User Advocacy", "core_value", 0.06)
    strengthen(G, "Integrity", "core_value", 0.08)
    strengthen(G, "Teaching", "core_value", 0.10)
    strengthen(G, "Discipline", "core_value", 0.06)
    strengthen(G, "Jordan", "person", 0.05)
    strengthen(G, "Priya", "person", 0.08)
    strengthen(G, "Kai", "person", 0.12)
    add_value(G, "Gratitude", 0.52, evidence=1)
    add_person(G, "Alex", "friend", "romantic", 0.48)
    add_person(G, "Dad", "family", "neutral", 0.55)
    add_goal(G, "Earn enough to hire a contractor", 0.58)
    clear_states(G)
    add_state(G, "Overwhelmed by client deadlines", 0.75, "stressor")
    add_state(G, "Inspired by design collective feedback", 0.68, "task")
    add_state(G, "Craving a week off", 0.70, "craving")
    add_state(G, "Anxious about tax season", 0.58, "stressor")
    add_state(G, "Excited about mentoring opportunity", 0.62, "task")
    add_state(G, "Thinking about Alex a lot", 0.55, "craving")
    snapshots.append(("llama3", G.copy()))

    return snapshots


# ═════════════════════════════════════════════════════════════════════════════
# DEV — Software engineer at a fast-growing startup
# 5 sessions tracking career growth, burnout, side projects, and romance
# ═════════════════════════════════════════════════════════════════════════════

def dev_snapshots():
    snapshots = []

    # ── Session 1: New at the startup — excited but overwhelmed
    G = _fresh_graph()
    add_value(G, "Discipline", 0.70, evidence=2)
    add_value(G, "Intellectual Curiosity", 0.68, evidence=1)
    add_value(G, "Clean Code", 0.65, evidence=1)
    add_value(G, "Empathy", 0.45, evidence=1)
    add_goal(G, "Get promoted to staff engineer", 0.60)
    add_goal(G, "Learn the full system architecture", 0.72)
    add_state(G, "Onboarding information overload", 0.70, "stressor")
    add_state(G, "Excited about the tech stack", 0.65, "craving")
    add_person(G, "Sarah", "mentor", "supportive", 0.62)
    add_person(G, "Chen", "colleague", "collaborative", 0.55)
    G.graph["profile_id"] = "dev"
    snapshots.append(("llama3", G.copy()))

    # ── Session 2: Getting into the groove, discovering problems
    strengthen(G, "Discipline", "core_value", 0.08)
    strengthen(G, "Clean Code", "core_value", 0.12)
    strengthen(G, "Intellectual Curiosity", "core_value", 0.07)
    add_value(G, "Ownership Mentality", 0.62, evidence=1)
    add_value(G, "Systems Thinking", 0.55, evidence=1)
    strengthen(G, "Sarah", "person", 0.08)
    strengthen(G, "Chen", "person", 0.12)
    add_person(G, "Jake", "friend", "neutral", 0.50)
    add_person(G, "Aisha", "colleague", "collaborative", 0.48)
    add_goal(G, "Refactor the payment service", 0.65)
    add_goal(G, "Launch a side project SaaS", 0.45)
    clear_states(G)
    add_state(G, "Found a critical bug in prod", 0.82, "stressor")
    add_state(G, "Wrote a design doc everyone loved", 0.70, "task")
    add_state(G, "Want to try Rust for side project", 0.55, "craving")
    snapshots.append(("llama3", G.copy()))

    # ── Session 3: Sprint madness, team friction, side project starts
    strengthen(G, "Discipline", "core_value", 0.06)
    strengthen(G, "Ownership Mentality", "core_value", 0.12)
    strengthen(G, "Systems Thinking", "core_value", 0.10)
    add_value(G, "Continuous Learning", 0.60, evidence=1)
    add_value(G, "Pragmatism", 0.55, evidence=1)
    # Aisha relationship turns tense
    G["user"][make_node_id("person", "Aisha")]["tone"] = "tense"
    G["user"][make_node_id("person", "Aisha")]["strength"] = 0.42
    strengthen(G, "Chen", "person", 0.08)
    add_person(G, "Nina", "friend", "neutral", 0.50)
    add_goal(G, "Contribute to a major open source project", 0.48)
    add_goal(G, "Read 30 technical books this year", 0.42)
    clear_states(G)
    add_state(G, "Frustrated with sprint scope creep", 0.78, "stressor")
    add_state(G, "Aisha merged untested code — had to fix it", 0.68, "stressor")
    add_state(G, "Rust side project is actually working", 0.72, "craving")
    add_state(G, "Staying up too late coding", 0.55, "stressor")
    snapshots.append(("mistral:7b", G.copy()))

    # ── Session 4: Burnout warning signs, romance, growth
    strengthen(G, "Empathy", "core_value", 0.12)
    strengthen(G, "Continuous Learning", "core_value", 0.10)
    strengthen(G, "Pragmatism", "core_value", 0.08)
    add_value(G, "Open Source", 0.52, evidence=1)
    add_value(G, "Work-Life Balance", 0.48, evidence=1)
    add_value(G, "Vulnerability", 0.42, evidence=1)
    strengthen(G, "Sarah", "person", 0.10)
    strengthen(G, "Jake", "person", 0.08)
    # Nina becomes romantic interest
    G["user"][make_node_id("person", "Nina")]["tone"] = "romantic"
    G["user"][make_node_id("person", "Nina")]["strength"] = 0.68
    add_person(G, "CTO Dave", "mentor", "collaborative", 0.55)
    add_person(G, "Gym Buddy Mike", "friend", "supportive", 0.45)
    add_goal(G, "Improve public speaking skills", 0.40)
    add_goal(G, "Take a real vacation", 0.52)
    clear_states(G)
    add_state(G, "Tired from on-call rotation", 0.75, "stressor")
    add_state(G, "Sarah suggested I pace myself", 0.58, "task")
    add_state(G, "Date with Nina went really well", 0.72, "craving")
    add_state(G, "Thinking about what I actually want from career", 0.60, "task")
    add_state(G, "Need to start working out again", 0.50, "craving")
    snapshots.append(("llama3", G.copy()))

    # ── Session 5: Current state — finding balance
    strengthen(G, "Discipline", "core_value", 0.05)
    strengthen(G, "Intellectual Curiosity", "core_value", 0.06)
    strengthen(G, "Clean Code", "core_value", 0.05)
    strengthen(G, "Ownership Mentality", "core_value", 0.06)
    strengthen(G, "Continuous Learning", "core_value", 0.08)
    strengthen(G, "Open Source", "core_value", 0.10)
    strengthen(G, "Work-Life Balance", "core_value", 0.12)
    strengthen(G, "Vulnerability", "core_value", 0.10)
    strengthen(G, "Empathy", "core_value", 0.06)
    strengthen(G, "Chen", "person", 0.05)
    strengthen(G, "Nina", "person", 0.15)
    strengthen(G, "CTO Dave", "person", 0.10)
    strengthen(G, "Gym Buddy Mike", "person", 0.12)
    add_value(G, "Gratitude", 0.50, evidence=1)
    add_value(G, "Patience", 0.48, evidence=1)
    add_person(G, "Open Source Community", "colleague", "collaborative", 0.50)
    add_goal(G, "Give a lightning talk at work", 0.52)
    clear_states(G)
    add_state(G, "Debugging a gnarly race condition", 0.80, "task")
    add_state(G, "Frustrated with scope creep again", 0.65, "stressor")
    add_state(G, "Excited about Rust side project progress", 0.70, "craving")
    add_state(G, "Tired from on-call but coping better", 0.55, "stressor")
    add_state(G, "Nina made me dinner last night", 0.78, "craving")
    add_state(G, "First open source PR got merged", 0.82, "task")
    add_state(G, "Actually went to the gym twice this week", 0.60, "task")
    snapshots.append(("llama3", G.copy()))

    return snapshots


# ═════════════════════════════════════════════════════════════════════════════
# MIA — Pre-med student juggling science, family expectations, and identity
# 5 sessions tracking academic pressure, self-discovery, and finding her voice
# ═════════════════════════════════════════════════════════════════════════════

def mia_snapshots():
    snapshots = []

    # ── Session 1: Start of semester — pressure from all sides
    G = _fresh_graph()
    add_value(G, "Discipline", 0.68, evidence=2)
    add_value(G, "Empathy", 0.72, evidence=2)
    add_value(G, "Intellectual Curiosity", 0.60, evidence=1)
    add_value(G, "Family Loyalty", 0.75, evidence=2)
    add_goal(G, "Get into medical school", 0.82)
    add_goal(G, "Maintain a 3.9 GPA", 0.78)
    add_state(G, "Overwhelmed by organic chemistry", 0.75, "stressor")
    add_state(G, "Worried about MCAT prep", 0.68, "stressor")
    add_person(G, "Mom", "family", "supportive", 0.85)
    add_person(G, "Dad", "family", "tense", 0.62)
    add_person(G, "Roommate Bri", "friend", "supportive", 0.65)
    G.graph["profile_id"] = "mia"
    snapshots.append(("llama3", G.copy()))

    # ── Session 2: Lab work sparks something, dad tension grows
    strengthen(G, "Intellectual Curiosity", "core_value", 0.12)
    strengthen(G, "Empathy", "core_value", 0.08)
    add_value(G, "Scientific Rigor", 0.62, evidence=1)
    add_value(G, "Compassion", 0.58, evidence=1)
    strengthen(G, "Mom", "person", 0.05)
    add_person(G, "Dr. Reyes", "mentor", "supportive", 0.58)
    add_person(G, "Study Group", "colleague", "collaborative", 0.52)
    add_goal(G, "Publish undergraduate research paper", 0.50)
    add_goal(G, "Volunteer at free clinic", 0.55)
    clear_states(G)
    add_state(G, "Fascinated by neuroscience lab rotation", 0.72, "task")
    add_state(G, "Dad wants me to do surgery not research", 0.68, "stressor")
    add_state(G, "Pulled an all-nighter for biochem exam", 0.65, "stressor")
    add_state(G, "Bri convinced me to go to yoga", 0.45, "craving")
    snapshots.append(("llama3", G.copy()))

    # ── Session 3: Identity crisis — does she even want to be a doctor?
    strengthen(G, "Intellectual Curiosity", "core_value", 0.08)
    strengthen(G, "Scientific Rigor", "core_value", 0.10)
    strengthen(G, "Compassion", "core_value", 0.12)
    add_value(G, "Authenticity", 0.52, evidence=1)
    add_value(G, "Independence", 0.48, evidence=1)
    add_value(G, "Wellness", 0.45, evidence=1)
    strengthen(G, "Dr. Reyes", "person", 0.12)
    strengthen(G, "Roommate Bri", "person", 0.10)
    add_person(G, "Therapist", "mentor", "supportive", 0.55)
    add_person(G, "Ethan", "friend", "neutral", 0.45)
    add_goal(G, "Figure out if medicine is really for me", 0.65)
    add_goal(G, "Start journaling regularly", 0.42)
    clear_states(G)
    add_state(G, "Questioning if med school is my dream or dad's", 0.82, "stressor")
    add_state(G, "Loved shadowing at the free clinic", 0.75, "task")
    add_state(G, "Feeling guilty about doubting the plan", 0.68, "stressor")
    add_state(G, "Started yoga twice a week — helps a lot", 0.55, "craving")
    add_state(G, "Crying in the car after dinner with Dad", 0.72, "stressor")
    snapshots.append(("mistral:7b", G.copy()))

    # ── Session 4: Finding her own path — neuroscience research
    strengthen(G, "Authenticity", "core_value", 0.15)
    strengthen(G, "Independence", "core_value", 0.12)
    strengthen(G, "Discipline", "core_value", 0.06)
    strengthen(G, "Wellness", "core_value", 0.10)
    strengthen(G, "Compassion", "core_value", 0.08)
    add_value(G, "Courage", 0.52, evidence=1)
    add_value(G, "Patience", 0.48, evidence=1)
    strengthen(G, "Dr. Reyes", "person", 0.10)
    strengthen(G, "Therapist", "person", 0.10)
    # Ethan becomes closer
    G["user"][make_node_id("person", "Ethan")]["tone"] = "romantic"
    G["user"][make_node_id("person", "Ethan")]["strength"] = 0.62
    add_person(G, "Cousin Priya", "family", "supportive", 0.55)
    add_goal(G, "Apply to neuroscience PhD programs", 0.58)
    add_goal(G, "Have an honest conversation with Dad", 0.72)
    clear_states(G)
    add_state(G, "Dr. Reyes offered me a research position", 0.85, "task")
    add_state(G, "Told Mom about neuroscience interest — she listened", 0.65, "task")
    add_state(G, "Ethan and I went hiking — felt really seen", 0.70, "craving")
    add_state(G, "Dad hasn't returned my call in 3 days", 0.72, "stressor")
    add_state(G, "Journaling every night now", 0.55, "task")
    snapshots.append(("llama3", G.copy()))

    # ── Session 5: Current — standing in her truth
    strengthen(G, "Authenticity", "core_value", 0.08)
    strengthen(G, "Courage", "core_value", 0.12)
    strengthen(G, "Intellectual Curiosity", "core_value", 0.05)
    strengthen(G, "Empathy", "core_value", 0.06)
    strengthen(G, "Family Loyalty", "core_value", 0.05)
    strengthen(G, "Independence", "core_value", 0.08)
    strengthen(G, "Patience", "core_value", 0.10)
    strengthen(G, "Mom", "person", 0.08)
    strengthen(G, "Ethan", "person", 0.10)
    strengthen(G, "Roommate Bri", "person", 0.05)
    strengthen(G, "Cousin Priya", "person", 0.10)
    add_value(G, "Gratitude", 0.50, evidence=1)
    add_person(G, "Lab Partner Zoe", "colleague", "collaborative", 0.52)
    add_goal(G, "Write personal statement for PhD apps", 0.62)
    clear_states(G)
    add_state(G, "Had the conversation with Dad — it was hard but real", 0.88, "stressor")
    add_state(G, "Submitted first research abstract", 0.80, "task")
    add_state(G, "Ethan said he's proud of me", 0.72, "craving")
    add_state(G, "Mom is coming around to the idea", 0.60, "task")
    add_state(G, "Nervous but excited about the future", 0.75, "craving")
    add_state(G, "Finally sleeping 7 hours a night", 0.50, "task")
    snapshots.append(("llama3", G.copy()))

    return snapshots


# ═════════════════════════════════════════════════════════════════════════════
# RAY — Music producer & part-time barista trying to make it
# 5 sessions tracking creative hustle, money stress, and artistic identity
# ═════════════════════════════════════════════════════════════════════════════

def ray_snapshots():
    snapshots = []

    # ── Session 1: Grinding — beats by night, coffee by day
    G = _fresh_graph()
    add_value(G, "Creative Expression", 0.75, evidence=2)
    add_value(G, "Discipline", 0.50, evidence=1)
    add_value(G, "Intellectual Curiosity", 0.55, evidence=1)
    add_value(G, "Authenticity", 0.65, evidence=1)
    add_goal(G, "Get a song placed on a major playlist", 0.70)
    add_goal(G, "Quit the barista job within 2 years", 0.62)
    add_state(G, "Exhausted from double shifts", 0.72, "stressor")
    add_state(G, "Made a beat at 2am that actually slaps", 0.78, "craving")
    add_person(G, "T-Bone", "friend", "collaborative", 0.68)
    add_person(G, "Manager Kim", "colleague", "tense", 0.42)
    add_person(G, "Sister Jade", "family", "supportive", 0.75)
    G.graph["profile_id"] = "ray"
    snapshots.append(("llama3", G.copy()))

    # ── Session 2: First real collab, imposter syndrome hits
    strengthen(G, "Creative Expression", "core_value", 0.08)
    strengthen(G, "Authenticity", "core_value", 0.07)
    add_value(G, "Perseverance", 0.58, evidence=1)
    add_value(G, "Community", 0.52, evidence=1)
    strengthen(G, "T-Bone", "person", 0.10)
    add_person(G, "Vocalist Mira", "colleague", "collaborative", 0.55)
    add_person(G, "Sound Engineer Dex", "mentor", "neutral", 0.48)
    add_goal(G, "Finish the EP — 5 tracks", 0.65)
    add_goal(G, "Learn mixing and mastering properly", 0.55)
    clear_states(G)
    add_state(G, "Collab with Mira went amazing", 0.82, "task")
    add_state(G, "Imposter syndrome — am I good enough?", 0.75, "stressor")
    add_state(G, "Rent is due and I'm short $200", 0.70, "stressor")
    add_state(G, "Listening to J Dilla for inspiration", 0.55, "craving")
    snapshots.append(("llama3", G.copy()))

    # ── Session 3: Money crisis, almost quits, sister saves him
    strengthen(G, "Perseverance", "core_value", 0.15)
    strengthen(G, "Discipline", "core_value", 0.10)
    strengthen(G, "Community", "core_value", 0.10)
    add_value(G, "Gratitude", 0.50, evidence=1)
    add_value(G, "Humility", 0.48, evidence=1)
    strengthen(G, "Sister Jade", "person", 0.12)
    strengthen(G, "Vocalist Mira", "person", 0.08)
    add_person(G, "Landlord", "colleague", "tense", 0.35)
    add_person(G, "Open Mic Crowd", "friend", "supportive", 0.45)
    add_goal(G, "Build a home studio on a budget", 0.52)
    add_goal(G, "Play at least one live show a month", 0.48)
    clear_states(G)
    add_state(G, "Almost gave up music last Tuesday", 0.85, "stressor")
    add_state(G, "Jade lent me money and said keep going", 0.78, "craving")
    add_state(G, "Played open mic — strangers loved it", 0.80, "task")
    add_state(G, "Kim cut my hours at the café", 0.68, "stressor")
    add_state(G, "Writing lyrics about struggle — feels honest", 0.65, "task")
    snapshots.append(("mistral:7b", G.copy()))

    # ── Session 4: Momentum building — blog feature, new connections
    strengthen(G, "Creative Expression", "core_value", 0.06)
    strengthen(G, "Authenticity", "core_value", 0.10)
    strengthen(G, "Perseverance", "core_value", 0.08)
    strengthen(G, "Gratitude", "core_value", 0.10)
    add_value(G, "Empathy", 0.52, evidence=1)
    add_value(G, "Self-Belief", 0.55, evidence=1)
    strengthen(G, "T-Bone", "person", 0.08)
    strengthen(G, "Sound Engineer Dex", "person", 0.12)
    G["user"][make_node_id("person", "Sound Engineer Dex")]["tone"] = "supportive"
    add_person(G, "Music Blogger Tanya", "colleague", "supportive", 0.52)
    add_person(G, "DJ collective", "friend", "collaborative", 0.48)
    add_goal(G, "Get 1000 monthly listeners on Spotify", 0.58)
    add_goal(G, "Collab with an artist outside my genre", 0.45)
    clear_states(G)
    add_state(G, "Blog featured my track — 500 new plays!", 0.88, "task")
    add_state(G, "Dex is teaching me proper EQ technique", 0.65, "task")
    add_state(G, "Still broke but momentum feels real", 0.60, "stressor")
    add_state(G, "Dreaming about performing at SXSW", 0.72, "craving")
    add_state(G, "Mom called — she doesn't get the music thing", 0.55, "stressor")
    snapshots.append(("llama3", G.copy()))

    # ── Session 5: Current — EP almost done, identity solidifying
    strengthen(G, "Creative Expression", "core_value", 0.05)
    strengthen(G, "Discipline", "core_value", 0.08)
    strengthen(G, "Perseverance", "core_value", 0.06)
    strengthen(G, "Community", "core_value", 0.10)
    strengthen(G, "Self-Belief", "core_value", 0.12)
    strengthen(G, "Humility", "core_value", 0.08)
    strengthen(G, "Empathy", "core_value", 0.06)
    strengthen(G, "Sister Jade", "person", 0.08)
    strengthen(G, "Vocalist Mira", "person", 0.10)
    strengthen(G, "Music Blogger Tanya", "person", 0.10)
    strengthen(G, "DJ collective", "person", 0.10)
    add_value(G, "Storytelling", 0.55, evidence=1)
    add_person(G, "Fan from Open Mic", "friend", "supportive", 0.42)
    add_person(G, "Mom", "family", "neutral", 0.58)
    add_goal(G, "Release EP by end of summer", 0.78)
    clear_states(G)
    add_state(G, "Track 4 of 5 is mixed and mastered", 0.82, "task")
    add_state(G, "T-Bone wants to do a joint project next", 0.72, "craving")
    add_state(G, "Got a tip jar at the café — regulars are generous", 0.55, "task")
    add_state(G, "Jade is coming to my first real show", 0.80, "craving")
    add_state(G, "Still anxious about money but less paralyzed", 0.58, "stressor")
    add_state(G, "Wrote a song about Mom — might send it to her", 0.65, "task")
    add_state(G, "Feeling like an actual artist for the first time", 0.90, "craving")
    snapshots.append(("llama3", G.copy()))

    return snapshots


# ═════════════════════════════════════════════════════════════════════════════
# JUN — Data scientist at a biotech company navigating corporate politics
# 5 sessions tracking ambition, ethical dilemmas, immigration stress, mentoring
# ═════════════════════════════════════════════════════════════════════════════

def jun_snapshots():
    snapshots = []

    # ── Session 1: New role at BioGenix — high expectations
    G = _fresh_graph()
    add_value(G, "Discipline", 0.72, evidence=2)
    add_value(G, "Intellectual Curiosity", 0.78, evidence=2)
    add_value(G, "Integrity", 0.65, evidence=1)
    add_value(G, "Empathy", 0.55, evidence=1)
    add_goal(G, "Lead the genomics ML pipeline", 0.72)
    add_goal(G, "Get H-1B visa renewed", 0.85)
    add_state(G, "Excited about the genomics dataset", 0.70, "task")
    add_state(G, "Anxious about visa timeline", 0.78, "stressor")
    add_person(G, "Manager Lisa", "mentor", "collaborative", 0.62)
    add_person(G, "Wei", "colleague", "collaborative", 0.58)
    add_person(G, "Parents in Beijing", "family", "supportive", 0.80)
    G.graph["profile_id"] = "jun"
    snapshots.append(("llama3", G.copy()))

    # ── Session 2: First results, but corners being cut
    strengthen(G, "Intellectual Curiosity", "core_value", 0.08)
    strengthen(G, "Integrity", "core_value", 0.12)
    strengthen(G, "Discipline", "core_value", 0.06)
    add_value(G, "Scientific Rigor", 0.62, evidence=1)
    add_value(G, "Thoroughness", 0.55, evidence=1)
    strengthen(G, "Wei", "person", 0.10)
    add_person(G, "VP Brad", "colleague", "tense", 0.45)
    add_person(G, "Intern Lily", "colleague", "supportive", 0.48)
    add_goal(G, "Publish a paper on the genomics findings", 0.58)
    add_goal(G, "Mentor Lily through her first project", 0.50)
    clear_states(G)
    add_state(G, "Model is showing promising accuracy", 0.75, "task")
    add_state(G, "Brad wants to skip validation — I pushed back", 0.82, "stressor")
    add_state(G, "Visa lawyer said there might be a delay", 0.72, "stressor")
    add_state(G, "Lily is eager and reminds me of myself", 0.55, "task")
    snapshots.append(("llama3", G.copy()))

    # ── Session 3: Ethics confrontation, isolation, parents worried
    strengthen(G, "Integrity", "core_value", 0.12)
    strengthen(G, "Scientific Rigor", "core_value", 0.10)
    strengthen(G, "Empathy", "core_value", 0.10)
    add_value(G, "Courage", 0.52, evidence=1)
    add_value(G, "Cultural Identity", 0.58, evidence=1)
    add_value(G, "Patience", 0.48, evidence=1)
    strengthen(G, "Manager Lisa", "person", 0.08)
    strengthen(G, "Parents in Beijing", "person", 0.08)
    strengthen(G, "Intern Lily", "person", 0.10)
    add_person(G, "Immigration Lawyer", "colleague", "neutral", 0.42)
    add_person(G, "Friend Sanjay", "friend", "supportive", 0.55)
    add_goal(G, "Document all validation steps for compliance", 0.68)
    add_goal(G, "Learn Mandarin idioms to stay connected to roots", 0.40)
    clear_states(G)
    add_state(G, "Wrote a formal objection to Brad's shortcut plan", 0.88, "task")
    add_state(G, "Feeling isolated — am I the only one who cares?", 0.75, "stressor")
    add_state(G, "Mom asked when I'm coming home — felt guilty", 0.70, "stressor")
    add_state(G, "Sanjay invited me to a data science meetup", 0.55, "craving")
    add_state(G, "Lily brought me tea during late night — small kindness", 0.50, "craving")
    snapshots.append(("mistral:7b", G.copy()))

    # ── Session 4: Vindication, community, new perspective
    strengthen(G, "Integrity", "core_value", 0.08)
    strengthen(G, "Courage", "core_value", 0.12)
    strengthen(G, "Thoroughness", "core_value", 0.10)
    strengthen(G, "Cultural Identity", "core_value", 0.10)
    strengthen(G, "Patience", "core_value", 0.08)
    add_value(G, "Mentorship", 0.55, evidence=1)
    add_value(G, "Resilience", 0.52, evidence=1)
    strengthen(G, "Friend Sanjay", "person", 0.12)
    strengthen(G, "Intern Lily", "person", 0.08)
    strengthen(G, "Wei", "person", 0.08)
    add_person(G, "Data Science Meetup", "colleague", "collaborative", 0.50)
    add_person(G, "Grandma", "family", "supportive", 0.72)
    add_goal(G, "Give a talk at the data science meetup", 0.52)
    add_goal(G, "Visit family in Beijing this year", 0.65)
    clear_states(G)
    add_state(G, "Lisa backed my validation approach — Brad overruled", 0.80, "task")
    add_state(G, "Visa approved for another 3 years!", 0.90, "task")
    add_state(G, "Grandma is getting older — want to see her", 0.72, "craving")
    add_state(G, "Spoke at meetup — people actually listened", 0.75, "task")
    add_state(G, "Starting to feel like I belong here", 0.60, "craving")
    snapshots.append(("llama3", G.copy()))

    # ── Session 5: Current — leading with integrity, building roots
    strengthen(G, "Discipline", "core_value", 0.05)
    strengthen(G, "Intellectual Curiosity", "core_value", 0.06)
    strengthen(G, "Integrity", "core_value", 0.05)
    strengthen(G, "Mentorship", "core_value", 0.12)
    strengthen(G, "Resilience", "core_value", 0.10)
    strengthen(G, "Empathy", "core_value", 0.08)
    strengthen(G, "Cultural Identity", "core_value", 0.06)
    strengthen(G, "Manager Lisa", "person", 0.10)
    strengthen(G, "Intern Lily", "person", 0.06)
    strengthen(G, "Friend Sanjay", "person", 0.08)
    strengthen(G, "Data Science Meetup", "person", 0.10)
    strengthen(G, "Grandma", "person", 0.08)
    add_value(G, "Gratitude", 0.50, evidence=1)
    add_value(G, "Balance", 0.48, evidence=1)
    add_person(G, "New Hire Omar", "colleague", "collaborative", 0.45)
    add_goal(G, "Transition Lily from intern to full-time", 0.62)
    add_goal(G, "Start a dim sum night with coworkers", 0.42)
    clear_states(G)
    add_state(G, "Pipeline hit 96% accuracy — best in the company", 0.88, "task")
    add_state(G, "Booked flights to Beijing for summer", 0.82, "craving")
    add_state(G, "Brad is being reassigned — quiet relief", 0.65, "task")
    add_state(G, "Lily got a full-time offer — so proud", 0.78, "craving")
    add_state(G, "Teaching Omar the codebase — feels full circle", 0.60, "task")
    add_state(G, "Miss grandma's cooking", 0.55, "craving")
    add_state(G, "Finally feels like home and career can coexist", 0.70, "craving")
    snapshots.append(("llama3", G.copy()))

    return snapshots


# ═════════════════════════════════════════════════════════════════════════════
# AVA — High school teacher becoming a union organizer
# 5 sessions tracking passion for education, burnout, activism, and purpose
# ═════════════════════════════════════════════════════════════════════════════

def ava_snapshots():
    snapshots = []

    # ── Session 1: Third year teaching — loves the kids, hates the system
    G = _fresh_graph()
    add_value(G, "Empathy", 0.80, evidence=3)
    add_value(G, "Discipline", 0.60, evidence=1)
    add_value(G, "Intellectual Curiosity", 0.58, evidence=1)
    add_value(G, "Justice", 0.72, evidence=2)
    add_value(G, "Dedication", 0.68, evidence=2)
    add_goal(G, "Get my students to love reading", 0.78)
    add_goal(G, "Survive this school year without burning out", 0.65)
    add_state(G, "Spent $300 of my own money on classroom supplies", 0.75, "stressor")
    add_state(G, "Student Maya wrote her first real essay — so proud", 0.82, "task")
    add_person(G, "Principal Davis", "colleague", "tense", 0.48)
    add_person(G, "Co-teacher Roz", "colleague", "supportive", 0.72)
    add_person(G, "Partner Sam", "friend", "supportive", 0.80)
    G.graph["profile_id"] = "ava"
    snapshots.append(("llama3", G.copy()))

    # ── Session 2: Budget cuts announced, anger ignites
    strengthen(G, "Justice", "core_value", 0.12)
    strengthen(G, "Empathy", "core_value", 0.06)
    strengthen(G, "Dedication", "core_value", 0.08)
    add_value(G, "Advocacy", 0.62, evidence=1)
    add_value(G, "Courage", 0.55, evidence=1)
    strengthen(G, "Co-teacher Roz", "person", 0.08)
    strengthen(G, "Partner Sam", "person", 0.06)
    add_person(G, "Union Rep Darius", "colleague", "collaborative", 0.55)
    add_person(G, "Student Maya", "colleague", "supportive", 0.52)
    add_goal(G, "Fight the budget cuts at the board meeting", 0.72)
    add_goal(G, "Get more teachers to the union meeting", 0.58)
    clear_states(G)
    add_state(G, "They're cutting the reading program I built", 0.88, "stressor")
    add_state(G, "Roz and I are writing a petition", 0.72, "task")
    add_state(G, "Sam says I'm bringing work stress home", 0.62, "stressor")
    add_state(G, "Went to my first union meeting — felt powerful", 0.75, "craving")
    add_state(G, "Can't sleep — too angry", 0.65, "stressor")
    snapshots.append(("llama3", G.copy()))

    # ── Session 3: Organizing, exhaustion, relationship strain
    strengthen(G, "Advocacy", "core_value", 0.15)
    strengthen(G, "Justice", "core_value", 0.08)
    strengthen(G, "Courage", "core_value", 0.10)
    strengthen(G, "Discipline", "core_value", 0.08)
    add_value(G, "Community Power", 0.58, evidence=1)
    add_value(G, "Self-Care", 0.42, evidence=1)
    add_value(G, "Vulnerability", 0.45, evidence=1)
    strengthen(G, "Union Rep Darius", "person", 0.12)
    add_person(G, "Parent Committee", "colleague", "collaborative", 0.52)
    add_person(G, "Superintendent", "colleague", "tense", 0.38)
    add_person(G, "Therapist Dr. Kim", "mentor", "supportive", 0.55)
    add_goal(G, "Organize a teacher walkout if needed", 0.55)
    add_goal(G, "Learn to separate work from personal life", 0.50)
    clear_states(G)
    add_state(G, "Petition got 200 signatures — momentum!", 0.80, "task")
    add_state(G, "Sam and I had a big fight about boundaries", 0.78, "stressor")
    add_state(G, "Principal gave me a warning for speaking up", 0.72, "stressor")
    add_state(G, "Dr. Kim says I need to rest or I'll crash", 0.65, "task")
    add_state(G, "Maya's mom joined the parent committee because of me", 0.70, "craving")
    add_state(G, "Haven't exercised in 3 weeks", 0.50, "stressor")
    snapshots.append(("mistral:7b", G.copy()))

    # ── Session 4: Breakthrough — board meeting, reconnecting with Sam
    strengthen(G, "Empathy", "core_value", 0.08)
    strengthen(G, "Community Power", "core_value", 0.12)
    strengthen(G, "Self-Care", "core_value", 0.15)
    strengthen(G, "Vulnerability", "core_value", 0.10)
    strengthen(G, "Advocacy", "core_value", 0.06)
    add_value(G, "Gratitude", 0.52, evidence=1)
    add_value(G, "Persistence", 0.58, evidence=1)
    strengthen(G, "Partner Sam", "person", 0.10)
    strengthen(G, "Co-teacher Roz", "person", 0.08)
    strengthen(G, "Parent Committee", "person", 0.10)
    strengthen(G, "Therapist Dr. Kim", "person", 0.08)
    add_person(G, "Board Member Chen", "colleague", "neutral", 0.48)
    add_person(G, "Local Reporter", "colleague", "supportive", 0.45)
    add_goal(G, "Run for union chapter president", 0.55)
    add_goal(G, "Plan a real date night with Sam", 0.62)
    clear_states(G)
    add_state(G, "Spoke at the board meeting — standing ovation from parents", 0.90, "task")
    add_state(G, "Board voted to restore partial funding!", 0.85, "task")
    add_state(G, "Sam came to the meeting — cried together after", 0.80, "craving")
    add_state(G, "Took a whole Sunday off for the first time in months", 0.65, "craving")
    add_state(G, "Still exhausted but it's a different kind of tired", 0.55, "stressor")
    snapshots.append(("llama3", G.copy()))

    # ── Session 5: Current — becoming a leader, integrating all parts
    strengthen(G, "Justice", "core_value", 0.06)
    strengthen(G, "Advocacy", "core_value", 0.05)
    strengthen(G, "Empathy", "core_value", 0.05)
    strengthen(G, "Self-Care", "core_value", 0.08)
    strengthen(G, "Gratitude", "core_value", 0.10)
    strengthen(G, "Persistence", "core_value", 0.08)
    strengthen(G, "Dedication", "core_value", 0.06)
    strengthen(G, "Courage", "core_value", 0.05)
    strengthen(G, "Discipline", "core_value", 0.06)
    strengthen(G, "Partner Sam", "person", 0.08)
    strengthen(G, "Union Rep Darius", "person", 0.08)
    strengthen(G, "Student Maya", "person", 0.10)
    strengthen(G, "Local Reporter", "person", 0.10)
    strengthen(G, "Board Member Chen", "person", 0.08)
    add_value(G, "Hope", 0.55, evidence=1)
    add_person(G, "New Teacher Mentee", "colleague", "supportive", 0.48)
    add_goal(G, "Create a teacher wellness program", 0.58)
    add_goal(G, "Write an op-ed about education funding", 0.52)
    clear_states(G)
    add_state(G, "Won the union chapter election", 0.92, "task")
    add_state(G, "Maya got accepted to honors English", 0.85, "craving")
    add_state(G, "Sam and I are in a good place again", 0.72, "craving")
    add_state(G, "Local paper wants to interview me", 0.68, "task")
    add_state(G, "Planning summer curriculum with Roz", 0.60, "task")
    add_state(G, "Jogging again — 3 times this week", 0.55, "craving")
    add_state(G, "Scared and hopeful about next year", 0.70, "stressor")
    snapshots.append(("llama3", G.copy()))

    return snapshots


# ═════════════════════════════════════════════════════════════════════════════
# Main — generate everything
# ═════════════════════════════════════════════════════════════════════════════

def main():
    profiles = {
        "ted": ted_snapshots,
        "sal": sal_snapshots,
        "dev": dev_snapshots,
        "mia": mia_snapshots,
        "ray": ray_snapshots,
        "jun": jun_snapshots,
        "ava": ava_snapshots,
    }

    print("=" * 62)
    print("  Generating demo profiles for Serendipity")
    print("=" * 62)

    for name, builder in profiles.items():
        print(f"\n  ── Profile: {name} ──")

        # Wipe existing profile data for a clean start
        profile_dir = PROFILES_DIR / name
        if profile_dir.exists():
            shutil.rmtree(profile_dir)
            print(f"    Cleared existing data at {profile_dir}")

        snapshots = builder()
        total = len(snapshots)

        for i, (model, G) in enumerate(snapshots, 1):
            snap = save_snapshot(G, model=model, profile=name)
            n = G.number_of_nodes()
            e = G.number_of_edges()
            print(f"    Snapshot {i}/{total}: {n:>2} nodes, {e:>2} edges  ({model})  {snap.name}")
            time.sleep(1.1)  # timestamps must differ

        # Save the final snapshot as the live graph
        final_model, final_G = snapshots[-1]
        save_graph(final_G, model=final_model, profile=name)

        # Summary
        from src.graph_store import get_profile_summary
        summary = get_profile_summary(final_G)
        v = len(summary["core_values"])
        g = len(summary["long_term_goals"])
        s = len(summary["short_term_states"])
        r = len(summary["relationships"])
        print(f"    Live: {final_G.number_of_nodes()} nodes  "
              f"({v} values, {g} goals, {s} states, {r} people)")

    # Final summary
    print(f"\n  Profiles directory: {PROFILES_DIR}")
    print()

    for name in profiles:
        from src.graph_store import list_snapshots
        snaps = list_snapshots(profile=name)
        print(f"  {name}: {len(snaps)} snapshots")

    print(f"\n{'=' * 62}")
    print("  Done! Launch the app:")
    print("    .venv/bin/streamlit run app.py")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
