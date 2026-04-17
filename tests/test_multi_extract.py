"""
Tests for src/multi_extract.py — the multi-pass focused extraction pipeline.

Covers: pre-pass validation, focused extraction, dedup, identity smart update,
and the full pipeline orchestrator. All LLM calls are mocked.
"""

import pytest
import networkx as nx
from unittest.mock import patch, MagicMock

from src.pipeline.extract import (
    run_pre_pass,
    run_focused_extraction,
    run_dedup,
    run_identity_extraction,
    run_multi_pass_pipeline,
    _get_existing_identity,
)
from src.core.schemas import (
    PrePassOutput,
    PassDefinition,
    FocusedExtractionOutput,
    DedupOutput,
    IdentityOutput,
    IdentityInfo,
    ExtractionItem,
    validate_pre_pass,
    validate_focused_extraction,
    validate_dedup,
    validate_identity,
)
from src.storage.graph import _fresh_graph
from src.core.config import CORE_EXTRACTION_CATEGORIES, MAX_EXTRA_PASSES


# ─── Schema validation tests ────────────────────────────────────────────────

class TestPrePassSchema:
    def test_valid_pre_pass_output(self):
        raw = {
            "main_subject_id": "SPEAKER_00",
            "reasoning": "Most consistent speaker",
            "passes": [
                {"category": "short_term_state", "focus_prompt": "Look for stressors", "priority": 1},
                {"category": "long_term_goal", "focus_prompt": "Look for goals", "priority": 2},
                {"category": "core_value", "focus_prompt": "Look for values", "priority": 3},
            ],
        }
        result = validate_pre_pass(raw)
        assert result.main_subject_id == "SPEAKER_00"
        assert len(result.passes) == 3

    def test_defaults_on_empty(self):
        raw = {}
        result = validate_pre_pass(raw)
        assert result.main_subject_id == "SPEAKER_00"
        assert result.passes == []


class TestFocusedExtractionSchema:
    def test_valid_output(self):
        raw = {
            "category": "core_value",
            "items": [{"item": "Discipline", "confidence": 0.9}],
            "relationships": [],
        }
        result = validate_focused_extraction(raw)
        assert result.category == "core_value"
        assert len(result.items) == 1

    def test_empty_items(self):
        raw = {"category": "short_term_state", "items": [], "relationships": []}
        result = validate_focused_extraction(raw)
        assert result.items == []


class TestDedupSchema:
    def test_valid_dedup_output(self):
        raw = {
            "core_values": [{"item": "Discipline", "confidence": 0.9}],
            "long_term_goals": [],
            "short_term_values": [{"item": "Stressed about deadline", "confidence": 0.7}],
            "interests": [],
            "relationships": [],
            "cross_category_insights": ["Stress connects to discipline value"],
            "custom_categories": {},
        }
        result = validate_dedup(raw)
        assert len(result.core_values) == 1
        assert len(result.cross_category_insights) == 1

    def test_with_custom_categories(self):
        raw = {
            "core_values": [],
            "long_term_goals": [],
            "short_term_values": [],
            "interests": [],
            "relationships": [],
            "cross_category_insights": [],
            "custom_categories": {
                "communication_style": [{"item": "Direct and assertive", "confidence": 0.8}],
            },
        }
        result = validate_dedup(raw)
        assert "communication_style" in result.custom_categories
        assert len(result.custom_categories["communication_style"]) == 1


class TestIdentitySchema:
    def test_valid_identity(self):
        raw = {
            "identity": {
                "name": "Ted",
                "name_confidence": 0.95,
                "age": "28",
                "age_confidence": 0.6,
                "occupation": "PhD Student",
                "occupation_confidence": 0.85,
                "location": None,
                "location_confidence": 0.0,
                "additional": {},
                "additional_confidence": {},
            }
        }
        result = validate_identity(raw)
        assert result.identity.name == "Ted"
        assert result.identity.occupation == "PhD Student"

    def test_empty_identity(self):
        raw = {"identity": {}}
        result = validate_identity(raw)
        assert result.identity.name is None


# ─── Pre-pass logic tests ───────────────────────────────────────────────────

class TestRunPrePass:
    @patch("src.pipeline.extract.call_llm")
    def test_enforces_core_categories(self, mock_llm):
        """If the LLM returns only 1 core category, the other 2 are auto-added."""
        mock_llm.return_value = (
            {
                "main_subject_id": "SPEAKER_00",
                "reasoning": "test",
                "passes": [
                    {"category": "core_value", "focus_prompt": "values", "priority": 1},
                ],
            },
            "{}",
        )
        result = run_pre_pass("transcript text", "llama3")
        categories = {p.category for p in result.passes}
        for core in CORE_EXTRACTION_CATEGORIES:
            assert core in categories

    @patch("src.pipeline.extract.call_llm")
    def test_caps_custom_passes(self, mock_llm):
        """Custom passes beyond MAX_EXTRA_PASSES are trimmed."""
        mock_llm.return_value = (
            {
                "main_subject_id": "SPEAKER_00",
                "reasoning": "test",
                "passes": [
                    {"category": "short_term_state", "focus_prompt": "s", "priority": 1},
                    {"category": "long_term_goal", "focus_prompt": "l", "priority": 2},
                    {"category": "core_value", "focus_prompt": "c", "priority": 3},
                    {"category": "custom_a", "focus_prompt": "a", "priority": 4},
                    {"category": "custom_b", "focus_prompt": "b", "priority": 5},
                    {"category": "custom_c", "focus_prompt": "c", "priority": 6},
                ],
            },
            "{}",
        )
        result = run_pre_pass("transcript text", "llama3")
        custom = [p for p in result.passes if p.category not in CORE_EXTRACTION_CATEGORIES]
        assert len(custom) <= MAX_EXTRA_PASSES


# ─── Identity smart update tests ────────────────────────────────────────────

class TestIdentitySmartUpdate:
    def test_get_existing_identity_from_fresh_graph(self):
        G = _fresh_graph()
        identity = _get_existing_identity(G)
        assert identity["name"] is None
        assert identity["name_confidence"] == 0.0

    def test_get_existing_identity_with_data(self):
        G = _fresh_graph()
        G.nodes["user"]["identity_name"] = "Ted"
        G.nodes["user"]["identity_name_confidence"] = 0.9
        identity = _get_existing_identity(G)
        assert identity["name"] == "Ted"
        assert identity["name_confidence"] == 0.9

    @patch("src.pipeline.extract.call_llm")
    def test_smart_update_keeps_higher_confidence(self, mock_llm):
        """If existing confidence is higher, the old value is kept."""
        G = _fresh_graph()
        G.nodes["user"]["identity_name"] = "Ted"
        G.nodes["user"]["identity_name_confidence"] = 0.95

        mock_llm.return_value = (
            {
                "identity": {
                    "name": "Theodore",
                    "name_confidence": 0.5,
                    "age": "28",
                    "age_confidence": 0.7,
                }
            },
            "{}",
        )

        result, _ = run_identity_extraction("transcript", G, "SPEAKER_00", "llama3")
        # Name should stay as "Ted" (0.95 > 0.5)
        assert result.identity.name == "Ted"
        assert result.identity.name_confidence == 0.95
        # Age should be new (no existing data)
        assert result.identity.age == "28"

    @patch("src.pipeline.extract.call_llm")
    def test_smart_update_replaces_lower_confidence(self, mock_llm):
        """If new confidence is higher, the new value wins."""
        G = _fresh_graph()
        G.nodes["user"]["identity_name"] = "T"
        G.nodes["user"]["identity_name_confidence"] = 0.3

        mock_llm.return_value = (
            {
                "identity": {
                    "name": "Ted",
                    "name_confidence": 0.9,
                }
            },
            "{}",
        )

        result, _ = run_identity_extraction("transcript", G, "SPEAKER_00", "llama3")
        assert result.identity.name == "Ted"
        assert result.identity.name_confidence == 0.9


# ─── Gatekeeper with DedupOutput ────────────────────────────────────────────

class TestGatekeeperWithDedupOutput:
    def test_run_gatekeeper_with_dedup_output(self):
        from src.pipeline.gatekeeper import apply_actions
        from src.core.schemas import GatekeeperOutput, GatekeeperAction

        G = _fresh_graph()
        output = GatekeeperOutput(actions=[
            GatekeeperAction(operation="add", node_type="core_value", label="Discipline", confidence=0.8),
        ])
        G = apply_actions(G, output)
        assert "cv_discipline" in G.nodes

    def test_custom_category_action(self):
        from src.pipeline.gatekeeper import apply_actions, _apply_custom_action

        G = _fresh_graph()
        from src.core.schemas import GatekeeperOutput, GatekeeperAction
        output = GatekeeperOutput(actions=[
            GatekeeperAction(
                operation="add",
                node_type="communication_style",
                label="Direct and assertive",
                confidence=0.8,
                metadata={},
            ),
        ])
        G = apply_actions(G, output)
        # Custom node should exist with the custom type prefix
        custom_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "communication_style"]
        assert len(custom_nodes) == 1

    def test_conditional_short_term_wipe(self):
        from src.pipeline.gatekeeper import apply_actions
        from src.core.schemas import GatekeeperOutput, GatekeeperAction

        G = _fresh_graph()
        G.add_node("sts_old_state", node_type="short_term_state", label="old state")
        G.add_edge("user", "sts_old_state", relation="experiencing", intensity=0.5)

        # wipe_short_term=False should preserve old state
        output = GatekeeperOutput(actions=[
            GatekeeperAction(operation="add", node_type="core_value", label="honesty", confidence=0.7),
        ])
        G = apply_actions(G, output, wipe_short_term=False)
        assert "sts_old_state" in G.nodes

    def test_identity_applied_to_user_node(self):
        from src.pipeline.gatekeeper import apply_identity

        G = _fresh_graph()
        identity_output = IdentityOutput(
            identity=IdentityInfo(
                name="Ted",
                name_confidence=0.9,
                age="28",
                age_confidence=0.7,
                occupation="PhD Student",
                occupation_confidence=0.85,
            )
        )
        apply_identity(G, identity_output)
        assert G.nodes["user"]["identity_name"] == "Ted"
        assert G.nodes["user"]["identity_age"] == "28"
        assert G.nodes["user"]["identity_occupation"] == "PhD Student"


# ─── Full pipeline orchestration ────────────────────────────────────────────

class TestMultiPassPipeline:
    @patch("src.pipeline.extract.call_llm")
    def test_full_pipeline_returns_expected_types(self, mock_llm):
        """Smoke test: the full pipeline returns (DedupOutput, IdentityOutput, list[str])."""
        # Mock responses for: pre_pass, 3 extractions, dedup, identity
        mock_llm.side_effect = [
            # Pre-pass
            (
                {
                    "main_subject_id": "SPEAKER_00",
                    "reasoning": "test",
                    "passes": [
                        {"category": "short_term_state", "focus_prompt": "s", "priority": 1},
                        {"category": "long_term_goal", "focus_prompt": "l", "priority": 2},
                        {"category": "core_value", "focus_prompt": "c", "priority": 3},
                    ],
                },
                "{}",
            ),
            # Extraction: short_term_state
            ({"category": "short_term_state", "items": [{"item": "Stressed", "confidence": 0.7}], "relationships": []}, "{}"),
            # Extraction: long_term_goal
            ({"category": "long_term_goal", "items": [{"item": "Finish PhD", "confidence": 0.8}], "relationships": []}, "{}"),
            # Extraction: core_value
            ({"category": "core_value", "items": [{"item": "Discipline", "confidence": 0.9}], "relationships": []}, "{}"),
            # Dedup
            (
                {
                    "core_values": [{"item": "Discipline", "confidence": 0.9}],
                    "long_term_goals": [{"item": "Finish PhD", "confidence": 0.8}],
                    "short_term_values": [{"item": "Stressed", "confidence": 0.7}],
                    "interests": [],
                    "relationships": [],
                    "cross_category_insights": ["Stress connects to PhD goal"],
                    "custom_categories": {},
                },
                "{}",
            ),
            # Identity
            (
                {
                    "identity": {
                        "name": "Ted",
                        "name_confidence": 0.9,
                    }
                },
                "{}",
            ),
        ]

        G = _fresh_graph()
        dedup, identity, logs = run_multi_pass_pipeline("transcript text", "llama3", G)

        assert isinstance(dedup, DedupOutput)
        assert isinstance(identity, IdentityOutput)
        assert isinstance(logs, list)
        assert len(logs) >= 4  # pre-pass + 3 extractions + dedup + identity

        assert len(dedup.core_values) == 1
        assert dedup.core_values[0].item == "Discipline"
        assert identity.identity.name == "Ted"
