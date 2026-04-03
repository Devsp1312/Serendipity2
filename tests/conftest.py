"""
Shared pytest fixtures for the Serendipity test suite.

Fixtures are designed to be composable and dependency-free:
  - fresh_graph / sample_graph: pre-built NetworkX graphs
  - mock_llm_*: raw dicts matching what Ollama would return
  - sample_extraction_output: validated ExtractionOutput from the mock dict
  - tmp_data_dir: isolated temporary data directory for file I/O tests
"""

import pytest
import networkx as nx

from src.graph_store import _fresh_graph
from src.schemas import (
    ExtractionOutput,
    GatekeeperOutput,
    GatekeeperAction,
    RelationshipEntry,
    validate_extraction,
    validate_gatekeeper,
)


# ─── Graph fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def fresh_graph() -> nx.DiGraph:
    """Returns a fresh graph with only the user anchor node."""
    return _fresh_graph()


@pytest.fixture
def sample_graph() -> nx.DiGraph:
    """
    Returns a graph with pre-populated nodes:
      - core_value:    discipline (weight 0.8, evidence 2)
      - person:        Alice (friend, supportive, strength 0.7)
    """
    G = _fresh_graph()
    G.add_node("cv_discipline", node_type="core_value", label="discipline", evidence_count=2)
    G.add_edge("user", "cv_discipline", relation="holds_value", weight=0.8, intensity=0.8)

    G.add_node("person_alice", node_type="person", name="Alice")
    G.add_edge(
        "user", "person_alice",
        relation="knows",
        relationship_type="friend",
        tone="supportive",
        strength=0.7,
    )
    return G


# ─── LLM response fixtures ────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_extraction_response() -> dict:
    """Simulates a clean extraction LLM response dict."""
    return {
        "metadata": {"main_subject_id": "SPEAKER_00", "reasoning": "Most consistent speaker."},
        "core_values":       [{"item": "discipline",      "confidence": 0.9},
                              {"item": "academic_rigor",  "confidence": 0.8}],
        "long_term_goals":   [{"item": "finish PhD",      "confidence": 0.85}],
        "short_term_values": [{"item": "stressed about deadline", "confidence": 0.75}],
        "interests":         [],
        "relationships": [
            {
                "speaker_id": "SPEAKER_01",
                "name": "Alice",
                "relationship_type": "friend",
                "interaction_type": "two_way_conversation",
                "confidence": 0.8,
            }
        ],
    }


@pytest.fixture
def mock_llm_gatekeeper_response() -> dict:
    """Simulates a clean gatekeeper LLM response dict."""
    return {
        "actions": [
            {
                "operation": "strengthen",
                "node_type": "core_value",
                "label": "discipline",
                "confidence": 0.9,
                "metadata": {},
            },
            {
                "operation": "add",
                "node_type": "short_term_state",
                "label": "stressed about deadline",
                "confidence": 0.8,
                "metadata": {},
            },
        ]
    }


@pytest.fixture
def sample_extraction_output(mock_llm_extraction_response) -> ExtractionOutput:
    """Returns a validated ExtractionOutput built from the mock extraction response."""
    return validate_extraction(mock_llm_extraction_response)


@pytest.fixture
def sample_gatekeeper_output(mock_llm_gatekeeper_response) -> GatekeeperOutput:
    """Returns a validated GatekeeperOutput built from the mock gatekeeper response."""
    return validate_gatekeeper(mock_llm_gatekeeper_response)


# ─── File system fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def tmp_data_dir(tmp_path):
    """
    Creates a temporary data directory structure.
    Tests that touch graph files should use this to avoid corrupting real data.

    Yields the root tmp directory. Subdirs:
      - tmp_path/snapshots/
    """
    (tmp_path / "snapshots").mkdir()
    return tmp_path
