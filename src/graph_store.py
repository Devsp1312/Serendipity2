"""
Knowledge graph persistence, querying, snapshotting, diffing, and rollback.

The graph is a NetworkX DiGraph. "user" is the anchor node at the centre.
Every other node radiates outward from user via a typed edge:

    user --[holds_value]-->  core_value node
    user --[pursues_goal]--> long_term_goal node
    user --[experiencing]--> short_term_state node
    user --[knows]-->        person node

Confidence is stored on the edge, not the node:
  - core_value / long_term_goal → "weight"
  - short_term_state            → "intensity"
  - person                      → "strength"

The graph serializes to JSON via NetworkX's node_link_data format.
"""

import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import networkx as nx
from networkx.readwrite import json_graph

from src.config import DEFAULT_PROFILE, PROFILE_PRESETS, PROFILES_DIR, NODE_TYPE_PREFIX
from src.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "normalize_profile_id", "get_graph_path", "get_snapshots_dir", "list_profiles",
    "load_graph", "save_graph", "save_snapshot", "list_snapshots",
    "rollback_to_snapshot", "make_node_id", "get_profile_summary",
    "diff_graphs", "NodeDiff", "GraphDiff", "_fresh_graph",
]


# ─── Path helpers ─────────────────────────────────────────────────────────────

def normalize_profile_id(profile: str | None) -> str:
    """Converts a profile name to a safe filesystem slug (e.g. 'Ted Smith' → 'ted_smith')."""
    raw     = (profile or DEFAULT_PROFILE).strip().lower()
    cleaned = re.sub(r"[^a-z0-9_-]", "_", raw)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or DEFAULT_PROFILE


def get_graph_path(profile: str | None = None) -> Path:
    return PROFILES_DIR / normalize_profile_id(profile) / "profile_graph.json"


def get_snapshots_dir(profile: str | None = None) -> Path:
    return PROFILES_DIR / normalize_profile_id(profile) / "snapshots"


def list_profiles() -> list[str]:
    """Returns all known profile IDs (presets + any on-disk folders)."""
    profiles = {normalize_profile_id(DEFAULT_PROFILE)}
    profiles.update(normalize_profile_id(p) for p in PROFILE_PRESETS)
    if PROFILES_DIR.exists():
        profiles.update(normalize_profile_id(c.name) for c in PROFILES_DIR.iterdir() if c.is_dir())
    return sorted(profiles)


# ─── Load / Save ──────────────────────────────────────────────────────────────

def load_graph(path: Optional[Path] = None, profile: str | None = None) -> nx.DiGraph:
    """
    Loads the profile graph from JSON. Returns a fresh graph (just the 'user' node)
    if the file doesn't exist or is corrupt — so the app never crashes on first run.
    """
    resolved_path = path or get_graph_path(profile)
    logger.debug("Loading graph from %s", resolved_path)
    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # multigraph=False is critical: without it NetworkX builds a MultiDiGraph
        # with different edge access patterns that break get_profile_summary() and diff_graphs()
        G = json_graph.node_link_graph(data, directed=True, multigraph=False)
        logger.debug("Graph loaded: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
        return G
    except FileNotFoundError:
        logger.debug("Graph file not found at %s — starting fresh", resolved_path)
        return _fresh_graph()
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Graph file corrupt at %s — starting fresh: %s", resolved_path, e)
        return _fresh_graph()


def save_graph(G: nx.DiGraph, path: Optional[Path] = None,
               model: str = "", profile: str | None = None) -> None:
    """Serializes the graph to JSON. Stamps it with the current timestamp and model name."""
    resolved_path = path or get_graph_path(profile)
    G.graph["last_updated"] = _now()
    if model:
        G.graph["last_model"] = model
    if profile is not None:
        G.graph["profile_id"] = normalize_profile_id(profile)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved_path, "w", encoding="utf-8") as f:
        json.dump(json_graph.node_link_data(G), f, indent=2)
    logger.info("Graph saved to %s (%d nodes, %d edges)", resolved_path,
                G.number_of_nodes(), G.number_of_edges())


def save_snapshot(G: nx.DiGraph, model: str = "", profile: str | None = None) -> Path:
    """Saves a timestamped copy of the graph. Filename: profile_20260326_120000_llama3.json"""
    snapshots_dir = get_snapshots_dir(profile)
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    ts         = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_model = re.sub(r"[^\w\-.]", "_", model) if model else "unknown"
    snapshot_path = snapshots_dir / f"profile_{ts}_{safe_model}.json"
    save_graph(G, path=snapshot_path, model=model, profile=profile)
    logger.info("Snapshot saved: %s", snapshot_path.name)
    return snapshot_path


def list_snapshots(profile: str | None = None) -> list[dict]:
    """
    Returns metadata for all snapshots, newest first.
    Each entry: {path, filename, timestamp, model, node_count}.
    """
    snapshots_dir = get_snapshots_dir(profile)
    if not snapshots_dir.exists():
        return []
    snapshots = []
    for p in sorted(snapshots_dir.glob("profile_*.json"), reverse=True):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            graph_meta = data.get("graph", {})
            snapshots.append({
                "path":       p,
                "filename":   p.name,
                "timestamp":  graph_meta.get("last_updated", "unknown"),
                "model":      graph_meta.get("last_model", "unknown"),
                "node_count": len(data.get("nodes", [])),
            })
        except Exception:
            continue
    return snapshots


def rollback_to_snapshot(snapshot_path: Path, profile: str | None = None) -> nx.DiGraph:
    """
    Restores the live graph to a previous snapshot.
    Always saves a safety backup of the current graph first so the rollback is reversible.

    Raises FileNotFoundError if snapshot_path doesn't exist.
    Raises ValueError if the snapshot is missing the 'user' anchor node.
    """
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    # Safety snapshot — save current state before overwriting
    current = load_graph(profile=profile)
    safety_path = save_snapshot(current, model=current.graph.get("last_model", "pre-rollback"), profile=profile)
    logger.info("Pre-rollback safety snapshot saved to %s", safety_path.name)

    restored = load_graph(snapshot_path)
    if "user" not in restored.nodes:
        raise ValueError(f"Snapshot {snapshot_path.name} is missing the 'user' anchor node.")

    save_graph(restored, model=restored.graph.get("last_model", ""), profile=profile)
    logger.info("Rolled back to snapshot %s (%d nodes)", snapshot_path.name, restored.number_of_nodes())
    return restored


# ─── Graph helpers ────────────────────────────────────────────────────────────

def _fresh_graph() -> nx.DiGraph:
    """Creates an empty graph with just the 'user' anchor node."""
    G = nx.DiGraph()
    G.graph["created"]      = _now()
    G.graph["last_updated"] = _now()
    G.add_node("user", node_type="person", name="User")
    return G


def make_node_id(node_type: str, label: str) -> str:
    """
    Creates a deterministic, stable ID from a node type + label.
    Same concept always maps to the same ID — prevents duplicates from casing differences.
    Example: ("core_value", "Academic Rigor") → "cv_academic_rigor"
    """
    prefix     = NODE_TYPE_PREFIX.get(node_type, node_type[:3])
    normalized = unicodedata.normalize("NFKD", label)
    ascii_label = normalized.encode("ascii", "ignore").decode()
    slug = re.sub(r"\s+", "_", re.sub(r"[^\w\s]", "", ascii_label).lower().strip())
    return f"{prefix}_{slug}"


def get_profile_summary(G: nx.DiGraph) -> dict:
    """
    Walks all edges out from 'user' and returns a clean summary dict.
    This is what we pass to the gatekeeper LLM as context — no raw NetworkX internals.
    Includes identity info from the user node and any custom category nodes.
    """
    summary: dict = {
        "identity": {},
        "core_values": [],
        "long_term_goals": [],
        "short_term_states": [],
        "relationships": [],
        "custom_categories": {},
    }

    # Pull identity info from the user node
    user_data = G.nodes.get("user", {})
    for field in ("name", "age", "occupation", "location"):
        value = user_data.get(f"identity_{field}")
        if value is not None:
            summary["identity"][field] = value
            summary["identity"][f"{field}_confidence"] = user_data.get(f"identity_{field}_confidence", 0.0)
    if user_data.get("identity_additional"):
        summary["identity"]["additional"] = user_data["identity_additional"]

    known_types = {"core_value", "long_term_goal", "short_term_state", "person"}

    for _, target, edge_data in G.out_edges("user", data=True):
        node_data = G.nodes[target]
        node_type = node_data.get("node_type", "")
        relation  = edge_data.get("relation", "")

        if node_type == "core_value":
            summary["core_values"].append({
                "label": node_data.get("label", target),
                "confidence": edge_data.get("weight", 0.5),
                "evidence_count": node_data.get("evidence_count", 1),
            })
        elif node_type == "long_term_goal":
            summary["long_term_goals"].append({
                "label": node_data.get("label", target),
                "confidence": edge_data.get("weight", 0.5),
            })
        elif node_type == "short_term_state":
            summary["short_term_states"].append({
                "label":     node_data.get("label", target),
                "category":  node_data.get("category", "stressor"),
                "intensity": edge_data.get("intensity", 0.5),
            })
        elif node_type == "person" and relation == "knows":
            summary["relationships"].append({
                "name":              node_data.get("name", target),
                "relationship_type": edge_data.get("relationship_type", "unknown"),
                "tone":              edge_data.get("tone", "neutral"),
                "strength":          edge_data.get("strength", 0.5),
            })
        elif node_type not in known_types and node_type:
            # Custom category node
            if node_type not in summary["custom_categories"]:
                summary["custom_categories"][node_type] = []
            summary["custom_categories"][node_type].append({
                "label": node_data.get("label", target),
                "confidence": edge_data.get("weight", 0.5),
                "evidence_count": node_data.get("evidence_count", 1),
            })

    # Remove empty sections to keep the summary clean for the LLM
    if not summary["identity"]:
        del summary["identity"]
    if not summary["custom_categories"]:
        del summary["custom_categories"]

    return summary


# ─── Diff (used by the History tab) ──────────────────────────────────────────

@dataclass
class NodeDiff:
    """The state of one node as seen in a diff between two snapshots."""
    node_id:        str
    label:          str
    node_type:      str
    old_confidence: Optional[float] = None
    new_confidence: Optional[float] = None

    @property
    def delta(self) -> Optional[float]:
        """Confidence change between the two snapshots."""
        if self.old_confidence is not None and self.new_confidence is not None:
            return round(self.new_confidence - self.old_confidence, 3)
        return None


@dataclass
class GraphDiff:
    """Result of comparing two graph snapshots."""
    added:        list[NodeDiff] = field(default_factory=list)
    removed:      list[NodeDiff] = field(default_factory=list)
    strengthened: list[NodeDiff] = field(default_factory=list)
    unchanged:    list[NodeDiff] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.removed or self.strengthened)


def diff_graphs(G_old: nx.DiGraph, G_new: nx.DiGraph) -> GraphDiff:
    """
    Compares two graph snapshots via set arithmetic on node IDs.
    Complexity: O(n) — fine for personal knowledge graphs (< 1000 nodes).
    """
    diff = GraphDiff()

    def confidence(G: nx.DiGraph, node_id: str) -> Optional[float]:
        """Pull confidence from the user→node edge (field name varies by type)."""
        if "user" not in G or node_id not in G:
            return None
        e = G.edges.get(("user", node_id), {})
        # person edges use "strength", others use "weight" or "intensity"
        return e.get("weight") or e.get("intensity") or e.get("strength")

    def label(G: nx.DiGraph, node_id: str) -> str:
        n = G.nodes.get(node_id, {})
        return n.get("label") or n.get("name") or node_id

    def ntype(G: nx.DiGraph, node_id: str) -> str:
        return G.nodes.get(node_id, {}).get("node_type", "unknown")

    old_ids = {n for n in G_old.nodes if n != "user"}
    new_ids = {n for n in G_new.nodes if n != "user"}

    for node_id in new_ids - old_ids:
        diff.added.append(NodeDiff(node_id, label(G_new, node_id), ntype(G_new, node_id),
                                   new_confidence=confidence(G_new, node_id)))

    for node_id in old_ids - new_ids:
        diff.removed.append(NodeDiff(node_id, label(G_old, node_id), ntype(G_old, node_id),
                                     old_confidence=confidence(G_old, node_id)))

    for node_id in old_ids & new_ids:
        old_c = confidence(G_old, node_id)
        new_c = confidence(G_new, node_id)
        nd = NodeDiff(node_id, label(G_new, node_id), ntype(G_new, node_id), old_c, new_c)
        (diff.strengthened if (old_c and new_c and new_c > old_c) else diff.unchanged).append(nd)

    return diff


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
