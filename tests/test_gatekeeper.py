"""
Tests for src/gatekeeper.py

Covers: apply_actions with each operation type, edge cases (strengthen on
non-existent node), short-term state clearing, relationship handling,
and config value usage.
"""

import pytest
import networkx as nx
from unittest.mock import patch

from src.gatekeeper import apply_actions, run_gatekeeper
from src.schemas import GatekeeperOutput, GatekeeperAction
from src.config import DEFAULT_CONFIDENCE, STRENGTHEN_INCREMENT, CONFIDENCE_CAP
from src.graph_store import _fresh_graph


# ─── apply_actions: core value / goal / state ────────────────────────────────

def test_add_new_core_value(fresh_graph):
    output = GatekeeperOutput(actions=[
        GatekeeperAction(operation="add", node_type="core_value", label="discipline", confidence=0.8),
    ])
    G = apply_actions(fresh_graph, output)
    assert "cv_discipline" in G.nodes
    assert G.has_edge("user", "cv_discipline")
    assert G["user"]["cv_discipline"]["weight"] == pytest.approx(0.8)


def test_add_does_not_duplicate_existing(sample_graph):
    # discipline already exists in sample_graph with weight 0.8
    output = GatekeeperOutput(actions=[
        GatekeeperAction(operation="add", node_type="core_value", label="discipline", confidence=0.5),
    ])
    G = apply_actions(sample_graph, output)
    # Weight should remain 0.8, not be overwritten by the add
    assert G["user"]["cv_discipline"]["weight"] == pytest.approx(0.8)


def test_strengthen_existing_node(sample_graph):
    output = GatekeeperOutput(actions=[
        GatekeeperAction(operation="strengthen", node_type="core_value", label="discipline", confidence=0.8),
    ])
    G = apply_actions(sample_graph, output)
    new_weight = G["user"]["cv_discipline"]["weight"]
    assert new_weight == pytest.approx(0.8 + STRENGTHEN_INCREMENT, abs=0.001)


def test_strengthen_capped_at_confidence_cap(sample_graph):
    # Set weight close to cap
    sample_graph["user"]["cv_discipline"]["weight"] = 0.95
    output = GatekeeperOutput(actions=[
        GatekeeperAction(operation="strengthen", node_type="core_value", label="discipline", confidence=0.9),
    ])
    G = apply_actions(sample_graph, output)
    assert G["user"]["cv_discipline"]["weight"] <= CONFIDENCE_CAP


def test_strengthen_non_existent_node_treated_as_add(fresh_graph, caplog):
    import logging
    output = GatekeeperOutput(actions=[
        GatekeeperAction(operation="strengthen", node_type="core_value", label="new_value", confidence=0.7),
    ])
    with caplog.at_level(logging.WARNING, logger="src.gatekeeper"):
        G = apply_actions(fresh_graph, output)
    assert "cv_new_value" in G.nodes
    assert "treat" in caplog.text.lower() or "add" in caplog.text.lower()


def test_update_metadata_on_existing_node(sample_graph):
    output = GatekeeperOutput(actions=[
        GatekeeperAction(
            operation="update",
            node_type="core_value",
            label="discipline",
            confidence=0.9,
            metadata={"category": "work_ethic"},
        ),
    ])
    G = apply_actions(sample_graph, output)
    assert G.nodes["cv_discipline"].get("category") == "work_ethic"


def test_remove_existing_node(sample_graph):
    output = GatekeeperOutput(actions=[
        GatekeeperAction(operation="remove", node_type="core_value", label="discipline"),
    ])
    G = apply_actions(sample_graph, output)
    assert "cv_discipline" not in G.nodes


def test_remove_nonexistent_node_is_safe(fresh_graph):
    output = GatekeeperOutput(actions=[
        GatekeeperAction(operation="remove", node_type="core_value", label="nonexistent"),
    ])
    # Should not raise
    G = apply_actions(fresh_graph, output)
    assert "user" in G.nodes


def test_short_term_states_cleared_before_new_ones(sample_graph):
    # Add a short-term state to sample_graph
    sample_graph.add_node("sts_stressed", node_type="short_term_state", label="stressed")
    sample_graph.add_edge("user", "sts_stressed", relation="experiencing", intensity=0.7)

    output = GatekeeperOutput(actions=[
        GatekeeperAction(operation="add", node_type="short_term_state", label="excited about defense", confidence=0.8),
    ])
    G = apply_actions(sample_graph, output)

    # Old STS should be gone
    assert "sts_stressed" not in G.nodes
    # New STS should be present
    assert "sts_excited_about_defense" in G.nodes


def test_default_confidence_used_when_none(fresh_graph):
    output = GatekeeperOutput(actions=[
        GatekeeperAction(operation="add", node_type="core_value", label="honesty", confidence=None),
    ])
    G = apply_actions(fresh_graph, output)
    edge = G["user"]["cv_honesty"]
    assert edge["weight"] == pytest.approx(DEFAULT_CONFIDENCE)


# ─── apply_actions: relationships ────────────────────────────────────────────

def test_add_relationship(fresh_graph):
    output = GatekeeperOutput(actions=[
        GatekeeperAction(
            operation="add",
            node_type="relationship",
            label="Alice",
            confidence=0.75,
            metadata={"relationship_type": "friend", "tone": "supportive"},
        ),
    ])
    G = apply_actions(fresh_graph, output)
    assert "person_alice" in G.nodes
    assert G["user"]["person_alice"]["relationship_type"] == "friend"
    assert G["user"]["person_alice"]["tone"] == "supportive"


def test_update_existing_relationship(sample_graph):
    # Alice is already a friend in sample_graph — update to mentor
    output = GatekeeperOutput(actions=[
        GatekeeperAction(
            operation="update",
            node_type="relationship",
            label="Alice",
            confidence=0.9,
            metadata={"relationship_type": "mentor", "tone": "collaborative"},
        ),
    ])
    G = apply_actions(sample_graph, output)
    assert G["user"]["person_alice"]["relationship_type"] == "mentor"
    assert G["user"]["person_alice"]["tone"] == "collaborative"


def test_strengthen_relationship(sample_graph):
    original = sample_graph["user"]["person_alice"]["strength"]
    output = GatekeeperOutput(actions=[
        GatekeeperAction(operation="strengthen", node_type="relationship", label="Alice", confidence=0.7),
    ])
    G = apply_actions(sample_graph, output)
    assert G["user"]["person_alice"]["strength"] == pytest.approx(original + STRENGTHEN_INCREMENT, abs=0.001)


def test_remove_relationship(sample_graph):
    output = GatekeeperOutput(actions=[
        GatekeeperAction(operation="remove", node_type="relationship", label="Alice"),
    ])
    G = apply_actions(sample_graph, output)
    assert "person_alice" not in G.nodes


# ─── run_gatekeeper (integration with mocked LLM) ────────────────────────────

def test_run_gatekeeper_uses_gatekeeper_prompt(fresh_graph, mock_llm_gatekeeper_response):
    with patch("src.gatekeeper.call_llm", return_value=(mock_llm_gatekeeper_response, "{}")) as mock_call, \
         patch("src.gatekeeper.get_gatekeeper_prompt", return_value="test prompt"):
        G, raw = run_gatekeeper(fresh_graph, __import__("src.schemas", fromlist=["ExtractionOutput"]).ExtractionOutput(), "llama3")
    mock_call.assert_called_once()
    assert mock_call.call_args[1]["system_prompt"] == "test prompt"


@pytest.fixture
def sample_graph():
    """Local sample graph for gatekeeper tests with discipline + Alice pre-populated."""
    from src.graph_store import _fresh_graph
    G = _fresh_graph()
    G.add_node("cv_discipline", node_type="core_value", label="discipline", evidence_count=2)
    G.add_edge("user", "cv_discipline", relation="holds_value", weight=0.8, intensity=0.8)
    G.add_node("person_alice", node_type="person", name="Alice")
    G.add_edge("user", "person_alice", relation="knows", relationship_type="friend", tone="supportive", strength=0.7)
    return G
