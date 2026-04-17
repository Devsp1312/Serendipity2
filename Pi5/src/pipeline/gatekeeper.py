"""
Pipeline Step 5 — Gatekeeper / Knowledge Graph Editor.

Input:  current NetworkX graph  +  DedupOutput (new extraction results)
Output: updated NetworkX graph  (saved to SQLite by the caller)

The gatekeeper is a second LLM call that acts as a smart diff editor.
Instead of blindly dumping every extracted item into the graph, it sees:
  - the EXISTING profile (what we already know)
  - the NEW insights (what just came from the transcript)
…and emits a targeted list of graph edits:

  add       → new insight not yet in the graph  → create node + edge
  strengthen → same insight seen again           → boost edge confidence (+0.1)
  update    → something changed (e.g. tone)      → overwrite node metadata
  remove    → new evidence contradicts old node  → delete the node

Why a separate LLM call instead of just adding everything?
  Without a gatekeeper, the graph would accumulate every noisy extraction and
  grow unbounded. The gatekeeper acts like a fact-checker: it only commits
  things that are consistent with the existing profile, resolves conflicts, and
  prunes contradictions — keeping the graph clean and trustworthy.

Short-term states are wiped before applying new ones (transient by design):
  values and goals accumulate over time, but today's emotional state replaces
  yesterday's — they should not pile up.

Accepts both ExtractionOutput (legacy single-pass) and DedupOutput (multi-pass).
"""

import json
from typing import Union

import networkx as nx

from src.core.config import DEFAULT_CONFIDENCE, STRENGTHEN_INCREMENT, CONFIDENCE_CAP, RELATION_FOR_TYPE, NODE_TYPE_PREFIX
from src.storage.graph import get_profile_summary, make_node_id
from src.core.llm_client import call_llm, get_gatekeeper_prompt
from src.core.logger import get_logger
from src.core.schemas import (
    ExtractionOutput,
    DedupOutput,
    IdentityOutput,
    GatekeeperOutput,
    validate_gatekeeper,
)

logger = get_logger(__name__)


def _build_new_insights_from_extraction(extraction: ExtractionOutput) -> dict:
    """Build gatekeeper input from the original single-pass ExtractionOutput."""
    return {
        "core_values":       [item.model_dump() for item in extraction.core_values],
        "long_term_goals":   [item.model_dump() for item in extraction.long_term_goals],
        "short_term_values": [item.model_dump() for item in extraction.short_term_values],
        "interests":         [item.model_dump() for item in extraction.interests],
        "relationships":     [r.model_dump() for r in extraction.relationships],
    }


def _build_new_insights_from_dedup(dedup: DedupOutput) -> dict:
    """Build gatekeeper input from the multi-pass DedupOutput."""
    insights = {
        "core_values":       [item.model_dump() for item in dedup.core_values],
        "long_term_goals":   [item.model_dump() for item in dedup.long_term_goals],
        "short_term_values": [item.model_dump() for item in dedup.short_term_values],
        "interests":         [item.model_dump() for item in dedup.interests],
        "relationships":     [r.model_dump() for r in dedup.relationships],
    }
    # Include cross-category insights as context for the gatekeeper
    if dedup.cross_category_insights:
        insights["cross_category_insights"] = dedup.cross_category_insights
    # Include custom categories
    for cat_name, items in dedup.custom_categories.items():
        insights[cat_name] = [item.model_dump() for item in items]
    return insights


def run_gatekeeper(
    G: nx.DiGraph,
    extraction: Union[ExtractionOutput, DedupOutput],
    model: str,
) -> tuple[nx.DiGraph, str]:
    """
    Runs the gatekeeper LLM and applies its edit actions to the graph.
    Accepts either ExtractionOutput (single-pass) or DedupOutput (multi-pass).
    Returns (updated_graph, raw_llm_json_string).
    """
    existing_profile = get_profile_summary(G)

    if isinstance(extraction, DedupOutput):
        new_insights = _build_new_insights_from_dedup(extraction)
        has_short_term = bool(extraction.short_term_values)
    else:
        new_insights = _build_new_insights_from_extraction(extraction)
        has_short_term = bool(extraction.short_term_values)

    user_prompt = (
        f"Existing profile:\n{json.dumps(existing_profile, indent=2)}\n\n"
        f"New insights from today:\n{json.dumps(new_insights, indent=2)}\n\n"
        "Generate the update actions."
    )

    logger.info("Gatekeeper phase started  model=%s  graph_nodes=%d", model, G.number_of_nodes())
    raw_dict, raw_str = call_llm(
        system_prompt=get_gatekeeper_prompt(),
        user_prompt=user_prompt,
        model=model,
    )

    gatekeeper_output = validate_gatekeeper(raw_dict)
    logger.info("Gatekeeper returned %d action(s)", len(gatekeeper_output.actions))
    return apply_actions(G, gatekeeper_output, wipe_short_term=has_short_term), raw_str


def apply_identity(G: nx.DiGraph, identity_output: IdentityOutput) -> nx.DiGraph:
    """
    Stores identity info as attributes on the 'user' node.
    Called after the gatekeeper — identity is managed separately from graph delta actions.
    """
    identity = identity_output.identity
    user = G.nodes["user"]

    for field in ("name", "age", "occupation", "location"):
        value = getattr(identity, field)
        conf = getattr(identity, f"{field}_confidence")
        if value is not None:
            user[f"identity_{field}"] = value
            user[f"identity_{field}_confidence"] = conf
            logger.debug("Identity set: %s=%r (confidence=%.2f)", field, value, conf)

    if identity.additional:
        user["identity_additional"] = identity.additional
        user["identity_additional_confidence"] = identity.additional_confidence

    return G


def apply_actions(G: nx.DiGraph, gatekeeper: GatekeeperOutput,
                  wipe_short_term: bool = True) -> nx.DiGraph:
    """
    Applies gatekeeper actions to the graph.

    Short-term states are wiped first ONLY if the current run extracted short-term data.
    Unlike values/goals (which accumulate), states represent the user's *current*
    snapshot and go stale immediately.
    """
    if wipe_short_term:
        sts_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "short_term_state"]
        if sts_nodes:
            logger.debug("Clearing %d short-term state node(s)", len(sts_nodes))
        G.remove_nodes_from(sts_nodes)

    logger.info("Applying %d gatekeeper action(s)", len(gatekeeper.actions))
    for action in gatekeeper.actions:
        if not action.label:
            logger.warning("Skipping gatekeeper action with missing label: op=%s type=%s metadata=%s",
                           action.operation, action.node_type, action.metadata)
            continue

        confidence = action.confidence if action.confidence is not None else DEFAULT_CONFIDENCE
        logger.debug("Action: op=%s  type=%s  label=%r  confidence=%.2f",
                     action.operation, action.node_type, action.label, confidence)

        # "relationship" is the LLM-facing label; graph stores them as node_type="person"
        if action.node_type == "relationship":
            _apply_relationship_action(G, action.operation, action.label, confidence, action.metadata)
        elif action.node_type in NODE_TYPE_PREFIX or action.node_type in ("core_value", "long_term_goal", "short_term_state"):
            _apply_value_action(G, action.operation, action.label, action.node_type, confidence, action.metadata)
        else:
            # Custom category from model-invented passes
            _apply_custom_action(G, action.operation, action.label, action.node_type, confidence, action.metadata)

    return G


def _apply_value_action(G: nx.DiGraph, op: str, label: str, node_type: str,
                        confidence: float, metadata: dict) -> None:
    """
    Handles add/strengthen/update/remove for core_value, long_term_goal, short_term_state.
    Node IDs are deterministic slugs (e.g. "cv_academic_rigor") — same label always
    maps to the same ID so we never create duplicates.
    """
    node_id  = make_node_id(node_type, label)
    relation = RELATION_FOR_TYPE.get(node_type, "related_to")

    if op == "add":
        if node_id not in G:
            G.add_node(node_id, node_type=node_type, label=label, evidence_count=1, **metadata)
        if not G.has_edge("user", node_id):
            G.add_edge("user", node_id, relation=relation, weight=confidence, intensity=confidence)

    elif op == "strengthen":
        if node_id in G:
            G.nodes[node_id]["evidence_count"] = G.nodes[node_id].get("evidence_count", 1) + 1
            if G.has_edge("user", node_id):
                current = G["user"][node_id].get("weight", confidence)
                # Cap at 1.0 — unbounded confidence would break the visualizer's radius math
                G["user"][node_id]["weight"] = min(CONFIDENCE_CAP, round(current + STRENGTHEN_INCREMENT, 3))
        else:
            # LLM said "strengthen" but node doesn't exist — treat as add to avoid losing the insight
            logger.warning("strengthen on non-existent node %r (type=%s) — treating as add", node_id, node_type)
            G.add_node(node_id, node_type=node_type, label=label, evidence_count=1, **metadata)
            G.add_edge("user", node_id, relation=relation, weight=confidence, intensity=confidence)

    elif op == "update":
        if node_id in G:
            G.nodes[node_id].update(metadata)
            if G.has_edge("user", node_id) and confidence:
                G["user"][node_id]["weight"] = confidence

    elif op == "remove":
        if node_id in G:
            G.remove_node(node_id)


def _apply_relationship_action(G: nx.DiGraph, op: str, label: str,
                                confidence: float, metadata: dict) -> None:
    """
    Handles add/update/strengthen/remove for person nodes (relationships).
    Person edges carry extra metadata: relationship_type and tone.
    """
    node_id  = make_node_id("person", label)
    rel_type = metadata.get("relationship_type", "unknown")
    tone     = metadata.get("tone", "neutral")

    if op in ("add", "update"):
        if node_id not in G:
            G.add_node(node_id, node_type="person", name=label)
        if not G.has_edge("user", node_id):
            G.add_edge("user", node_id, relation="knows",
                       relationship_type=rel_type, tone=tone, strength=confidence)
        else:
            G["user"][node_id].update({"relationship_type": rel_type, "tone": tone, "strength": confidence})

    elif op == "strengthen":
        if G.has_edge("user", node_id):
            current = G["user"][node_id].get("strength", confidence)
            G["user"][node_id]["strength"] = min(CONFIDENCE_CAP, round(current + STRENGTHEN_INCREMENT, 3))

    elif op == "remove":
        if node_id in G:
            G.remove_node(node_id)


def _apply_custom_action(G: nx.DiGraph, op: str, label: str, node_type: str,
                         confidence: float, metadata: dict) -> None:
    """
    Handles add/strengthen/update/remove for model-invented custom categories.
    Custom nodes use prefix "custom_" and edge relation "has_trait".
    """
    node_id = make_node_id(node_type, label)
    relation = "has_trait"

    if op == "add":
        if node_id not in G:
            G.add_node(node_id, node_type=node_type, label=label, evidence_count=1, **metadata)
        if not G.has_edge("user", node_id):
            G.add_edge("user", node_id, relation=relation, weight=confidence)

    elif op == "strengthen":
        if node_id in G:
            G.nodes[node_id]["evidence_count"] = G.nodes[node_id].get("evidence_count", 1) + 1
            if G.has_edge("user", node_id):
                current = G["user"][node_id].get("weight", confidence)
                G["user"][node_id]["weight"] = min(CONFIDENCE_CAP, round(current + STRENGTHEN_INCREMENT, 3))
        else:
            logger.warning("strengthen on non-existent custom node %r (type=%s) — treating as add", node_id, node_type)
            G.add_node(node_id, node_type=node_type, label=label, evidence_count=1, **metadata)
            G.add_edge("user", node_id, relation=relation, weight=confidence)

    elif op == "update":
        if node_id in G:
            G.nodes[node_id].update(metadata)
            if G.has_edge("user", node_id) and confidence:
                G["user"][node_id]["weight"] = confidence

    elif op == "remove":
        if node_id in G:
            G.remove_node(node_id)
