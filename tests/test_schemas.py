"""
Tests for src/schemas.py

Covers: Pydantic validation, enum coercion with warning logging,
and custom LLMSchemaError. Updated for v3 schema (speaker_id required,
interaction_type replaces tone, ExtractionItem replaces bare strings).
"""

import logging
import pytest

from src.schemas import (
    ExtractionItem,
    ExtractionMetadata,
    RelationshipEntry,
    ExtractionOutput,
    GatekeeperAction,
    GatekeeperOutput,
    LLMSchemaError,
    validate_extraction,
    validate_gatekeeper,
    VALID_RELATIONSHIP_TYPES,
    VALID_INTERACTION_TYPES,
    VALID_OPERATIONS,
    VALID_NODE_TYPES,
)


# ─── RelationshipEntry coercion ───────────────────────────────────────────────

def test_valid_relationship_type_passes():
    entry = RelationshipEntry(speaker_id="SPEAKER_01", name="Alice", relationship_type="friend", confidence=0.8)
    assert entry.relationship_type == "friend"


def test_invalid_relationship_type_coerced_to_unknown(caplog):
    with caplog.at_level(logging.WARNING, logger="src.schemas"):
        entry = RelationshipEntry(speaker_id="SPEAKER_01", name="Alice", relationship_type="helper", confidence=0.5)
    assert entry.relationship_type == "unknown"
    assert "helper" in caplog.text


def test_invalid_interaction_type_coerced_to_two_way(caplog):
    with caplog.at_level(logging.WARNING, logger="src.schemas"):
        entry = RelationshipEntry(speaker_id="SPEAKER_01", name="Bob", relationship_type="friend", interaction_type="live_stream", confidence=0.5)
    assert entry.interaction_type == "two_way_conversation"
    assert "live_stream" in caplog.text


def test_valid_interaction_type_passes():
    entry = RelationshipEntry(speaker_id="SPEAKER_01", name="Bob", relationship_type="mentor", interaction_type="one_way_media", confidence=0.7)
    assert entry.interaction_type == "one_way_media"


def test_all_valid_relationship_types_pass():
    for rt in VALID_RELATIONSHIP_TYPES:
        entry = RelationshipEntry(speaker_id="SPEAKER_01", name="X", relationship_type=rt, confidence=0.5)
        assert entry.relationship_type == rt


def test_all_valid_interaction_types_pass():
    for it in VALID_INTERACTION_TYPES:
        entry = RelationshipEntry(speaker_id="SPEAKER_01", name="X", interaction_type=it, confidence=0.5)
        assert entry.interaction_type == it


def test_name_is_optional():
    entry = RelationshipEntry(speaker_id="SPEAKER_02", confidence=0.5)
    assert entry.name is None


def test_confidence_out_of_range_raises():
    with pytest.raises(Exception):  # Pydantic ValidationError
        RelationshipEntry(speaker_id="SPEAKER_01", name="X", relationship_type="friend", confidence=1.5)


# ─── GatekeeperAction coercion ────────────────────────────────────────────────

def test_invalid_operation_coerced_to_add(caplog):
    with caplog.at_level(logging.WARNING, logger="src.schemas"):
        action = GatekeeperAction(operation="insert", node_type="core_value", label="discipline")
    assert action.operation == "add"
    assert "insert" in caplog.text


def test_unknown_node_type_passes_through_as_custom(caplog):
    """Non-standard node types pass through as custom categories (not coerced)."""
    with caplog.at_level(logging.DEBUG, logger="src.schemas"):
        action = GatekeeperAction(operation="add", node_type="trait", label="discipline")
    assert action.node_type == "trait"


def test_valid_operations_pass():
    for op in VALID_OPERATIONS:
        action = GatekeeperAction(operation=op, node_type="core_value", label="test")
        assert action.operation == op


def test_valid_node_types_pass():
    for nt in VALID_NODE_TYPES:
        action = GatekeeperAction(operation="add", node_type=nt, label="test")
        assert action.node_type == nt


# ─── validate_extraction ─────────────────────────────────────────────────────

def test_validate_extraction_valid_input():
    raw = {
        "metadata": {"main_subject_id": "SPEAKER_00", "reasoning": "Most turns."},
        "core_values":      [{"item": "discipline", "confidence": 0.9}],
        "long_term_goals":  [{"item": "finish PhD", "confidence": 0.8}],
        "short_term_values": [{"item": "stressed about deadline", "confidence": 0.7}],
        "interests":        [{"item": "running", "confidence": 0.6}],
        "relationships": [
            {"speaker_id": "SPEAKER_01", "name": "Alice", "relationship_type": "friend",
             "interaction_type": "two_way_conversation", "confidence": 0.8}
        ],
    }
    result = validate_extraction(raw)
    assert isinstance(result, ExtractionOutput)
    assert result.metadata.main_subject_id == "SPEAKER_00"
    assert result.core_values[0].item == "discipline"
    assert result.short_term_values[0].item == "stressed about deadline"
    assert result.interests[0].item == "running"
    assert result.relationships[0].speaker_id == "SPEAKER_01"


def test_validate_extraction_missing_fields_defaults_to_empty():
    """All list fields are optional and default to empty."""
    result = validate_extraction({})
    assert result.core_values == []
    assert result.long_term_goals == []
    assert result.short_term_values == []
    assert result.interests == []
    assert result.relationships == []


def test_validate_extraction_raises_llm_schema_error_on_bad_type():
    """If the structure is fundamentally wrong (e.g. list instead of dict), raise LLMSchemaError."""
    with pytest.raises(LLMSchemaError):
        validate_extraction({"core_values": "not_a_list"})


# ─── validate_gatekeeper ─────────────────────────────────────────────────────

def test_validate_gatekeeper_valid_input():
    raw = {
        "actions": [
            {"operation": "add", "node_type": "core_value", "label": "discipline", "confidence": 0.8, "metadata": {}}
        ]
    }
    result = validate_gatekeeper(raw)
    assert isinstance(result, GatekeeperOutput)
    assert len(result.actions) == 1


def test_validate_gatekeeper_empty_actions():
    result = validate_gatekeeper({"actions": []})
    assert result.actions == []


def test_validate_gatekeeper_raises_on_invalid_structure():
    with pytest.raises(LLMSchemaError):
        validate_gatekeeper({"actions": "not_a_list"})


# ─── LLMSchemaError ──────────────────────────────────────────────────────────

def test_llm_schema_error_carries_raw_response():
    err = LLMSchemaError("bad json", raw_response='{"broken": }')
    assert err.raw_response == '{"broken": }'
    assert "bad json" in str(err)
