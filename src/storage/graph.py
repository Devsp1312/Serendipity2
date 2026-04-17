"""
Storage layer — Knowledge Graph persistence, snapshots, rollback, and diff.

Input:  NetworkX DiGraph (from the gatekeeper, fully updated)
Output: profile.db  (SQLite file per profile in data/profiles/<name>/)

Knowledge graph persistence, querying, snapshotting, diffing, and rollback.

Storage backend: SQLite (one `profile.db` per profile directory).

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

All edge attributes are preserved exactly as-is in the `attrs_json` column,
so the in-memory DiGraph is byte-for-byte identical whether loaded from the
new SQLite backend or from the old JSON files.

Backward compatibility
─────────────────────
• load_graph(path=some_file.json) / save_graph(G, path=some_file.json)
  still read/write plain JSON — old test helpers and scripts are unaffected.
• On first load of a profile that has profile_graph.json but no profile.db,
  the JSON file (and any snapshot JSONs) are automatically migrated to SQLite.

Snapshot virtual paths
──────────────────────
save_snapshot() writes a row into the snapshots table and returns a "virtual"
Path like:
    data/profiles/spongebob/snapshots/snap_7_20260412_160315_llama3_2_3b.db

This path does not exist on disk. load_graph() and rollback_to_snapshot()
recognise it and resolve the row from the DB — callers in app.py need no
changes.
"""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional

import networkx as nx
from networkx.readwrite import json_graph

from src.core.config import DEFAULT_PROFILE, PROFILE_PRESETS, PROFILES_DIR, NODE_TYPE_PREFIX
from src.core.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "normalize_profile_id", "get_graph_path", "get_snapshots_dir", "list_profiles",
    "load_graph", "save_graph", "save_snapshot", "list_snapshots",
    "rollback_to_snapshot", "make_node_id", "get_profile_summary",
    "diff_graphs", "NodeDiff", "GraphDiff", "_fresh_graph",
]

# ─── SQLite schema ─────────────────────────────────────────────────────────────

_SCHEMA = """
-- WAL (Write-Ahead Log) mode: allows concurrent reads while a write is in progress.
-- Without WAL, any write would lock the file and block the Streamlit UI from
-- reading the graph at the same time. WAL is the right choice for local apps.
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS nodes (
    id             TEXT PRIMARY KEY,
    node_type      TEXT NOT NULL DEFAULT '',
    label          TEXT NOT NULL DEFAULT '',
    evidence_count INTEGER NOT NULL DEFAULT 1,
    attrs_json     TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS edges (
    source     TEXT NOT NULL,
    target     TEXT NOT NULL,
    relation   TEXT NOT NULL DEFAULT '',
    weight     REAL NOT NULL DEFAULT 0.5,
    attrs_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (source, target)
);

CREATE TABLE IF NOT EXISTS snapshots (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp  TEXT    NOT NULL,
    model      TEXT    NOT NULL DEFAULT '',
    node_count INTEGER NOT NULL DEFAULT 0,
    graph_json TEXT    NOT NULL
);
"""


# ─── Path helpers ──────────────────────────────────────────────────────────────

def normalize_profile_id(profile: str | None) -> str:
    """Converts a profile name to a safe filesystem slug (e.g. 'Ted Smith' → 'ted_smith')."""
    raw     = (profile or DEFAULT_PROFILE).strip().lower()
    cleaned = re.sub(r"[^a-z0-9_-]", "_", raw)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or DEFAULT_PROFILE


def get_db_path(profile: str | None = None) -> Path:
    """Returns the SQLite DB path for a profile."""
    return PROFILES_DIR / normalize_profile_id(profile) / "profile.db"


def get_graph_path(profile: str | None = None) -> Path:
    """Returns the storage path for a profile (now a .db file)."""
    return get_db_path(profile)


def get_snapshots_dir(profile: str | None = None) -> Path:
    """Returns the snapshots directory (kept for backward compat; new snapshots live in the DB)."""
    return PROFILES_DIR / normalize_profile_id(profile) / "snapshots"


def list_profiles() -> list[str]:
    """Returns all known profile IDs (presets + any on-disk folders)."""
    profiles = {normalize_profile_id(DEFAULT_PROFILE)}
    profiles.update(normalize_profile_id(p) for p in PROFILE_PRESETS)
    if PROFILES_DIR.exists():
        profiles.update(normalize_profile_id(c.name) for c in PROFILES_DIR.iterdir() if c.is_dir())
    return sorted(profiles)


# ─── Internal SQLite helpers ───────────────────────────────────────────────────

@contextmanager
def _connect(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    """Opens a WAL-mode SQLite connection with foreign keys and row_factory."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


def _ensure_schema(db_path: Path) -> None:
    """Creates tables if they don't exist yet."""
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)
        conn.commit()


def _is_virtual_snapshot(path: Path) -> bool:
    """True if path is a virtual snapshot ref (snap_<id>_…_.db)."""
    return path.suffix == ".db" and path.stem.startswith("snap_")


def _virtual_to_db_and_id(path: Path) -> tuple[Path, int]:
    """Extract (profile_db_path, snapshot_id) from a virtual snapshot path."""
    snap_id = int(path.stem.split("_")[1])
    db_path = path.parent.parent / "profile.db"
    return db_path, snap_id


def _snapshot_virtual_path(db_path: Path, snap_id: int, ts: str, model: str) -> Path:
    """Build the virtual Path that callers pass back to load_graph / rollback."""
    safe_model = re.sub(r"[^\w\-]", "_", model) if model else "unknown"
    safe_ts    = ts.replace(":", "").replace("-", "").replace("T", "_").replace("Z", "")
    filename   = f"snap_{snap_id}_{safe_ts}_{safe_model}.db"
    return db_path.parent / "snapshots" / filename


# ─── Graph ↔ SQLite serialisation ─────────────────────────────────────────────

def _graph_to_db(G: nx.DiGraph, db_path: Path) -> None:
    """Replace the live graph in the DB with G (atomic transaction)."""
    _ensure_schema(db_path)
    with _connect(db_path) as conn:
        with conn:
            # ── meta ──
            conn.execute("DELETE FROM meta")
            for k, v in G.graph.items():
                conn.execute("INSERT INTO meta VALUES (?, ?)", (k, str(v)))

            # ── nodes ──
            conn.execute("DELETE FROM edges")   # edges first (FK)
            conn.execute("DELETE FROM nodes")
            for node_id, attrs in G.nodes(data=True):
                label          = attrs.get("label") or attrs.get("name", "")
                node_type      = attrs.get("node_type", "")
                evidence_count = attrs.get("evidence_count", 1)
                conn.execute(
                    "INSERT INTO nodes VALUES (?, ?, ?, ?, ?)",
                    (node_id, node_type, label, evidence_count, json.dumps(attrs)),
                )

            # ── edges ──
            for source, target, attrs in G.edges(data=True):
                # Unified weight for the indexed column (field name varies by type)
                weight = (
                    attrs.get("weight")
                    or attrs.get("intensity")
                    or attrs.get("strength")
                    or 0.5
                )
                conn.execute(
                    "INSERT INTO edges VALUES (?, ?, ?, ?, ?)",
                    (source, target, attrs.get("relation", ""), float(weight), json.dumps(attrs)),
                )


def _db_to_graph(db_path: Path) -> nx.DiGraph:
    """Load the live graph from SQLite. Returns a fresh graph if DB is empty/missing."""
    if not db_path.exists():
        return _fresh_graph()

    _ensure_schema(db_path)
    G = nx.DiGraph()

    try:
        with _connect(db_path) as conn:
            # meta
            for row in conn.execute("SELECT key, value FROM meta"):
                G.graph[row["key"]] = row["value"]

            # nodes — restore ALL original attrs from attrs_json
            for row in conn.execute("SELECT attrs_json FROM nodes"):
                attrs = json.loads(row["attrs_json"])
                node_id = attrs.get("id") or next(
                    k for k, v in attrs.items() if k not in ("node_type", "label",
                                                              "evidence_count")
                )
                # node id is not stored inside attrs, so fetch it separately
                break
            for row in conn.execute("SELECT id, attrs_json FROM nodes"):
                attrs = json.loads(row["attrs_json"])
                G.add_node(row["id"], **attrs)

            # edges — restore ALL original attrs from attrs_json
            for row in conn.execute("SELECT source, target, attrs_json FROM edges"):
                attrs = json.loads(row["attrs_json"])
                G.add_edge(row["source"], row["target"], **attrs)

    except Exception as e:
        logger.warning("Failed to read DB %s — starting fresh: %s", db_path, e)
        return _fresh_graph()

    if "user" not in G.nodes:
        G.add_node("user", node_type="person", name="User")
    if not G.graph.get("created"):
        G.graph["created"] = _now()

    logger.debug("Graph loaded from SQLite: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


def _insert_snapshot(db_path: Path, G: nx.DiGraph, model: str = "") -> tuple[int, str]:
    """Insert a snapshot row and return (snapshot_id, timestamp)."""
    _ensure_schema(db_path)
    ts = G.graph.get("last_updated") or _now()
    graph_json = json.dumps(json_graph.node_link_data(G))
    with _connect(db_path) as conn:
        with conn:
            cur = conn.execute(
                "INSERT INTO snapshots (timestamp, model, node_count, graph_json) VALUES (?, ?, ?, ?)",
                (ts, model, G.number_of_nodes(), graph_json),
            )
            return cur.lastrowid, ts


def _load_snapshot_by_id(db_path: Path, snap_id: int) -> nx.DiGraph:
    """Load a snapshot row from the DB and return it as a DiGraph."""
    if not db_path.exists():
        raise FileNotFoundError(f"Profile DB not found: {db_path}")
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT graph_json FROM snapshots WHERE id = ?", (snap_id,)
        ).fetchone()
    if row is None:
        raise FileNotFoundError(f"Snapshot #{snap_id} not found in {db_path}")
    data = json.loads(row["graph_json"])
    return json_graph.node_link_graph(data, directed=True, multigraph=False)


# ─── JSON helpers (backward compat) ───────────────────────────────────────────

def _load_from_json(path: Path) -> nx.DiGraph:
    """Load a NetworkX graph from a legacy JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        G = json_graph.node_link_graph(data, directed=True, multigraph=False)
        logger.debug("Graph loaded from JSON: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
        return G
    except FileNotFoundError:
        logger.debug("JSON file not found at %s — starting fresh", path)
        return _fresh_graph()
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning("Graph file unreadable at %s — starting fresh: %s", path, e)
        return _fresh_graph()


def _save_to_json(G: nx.DiGraph, path: Path, model: str = "", profile: str | None = None) -> None:
    """Serialize the graph to a legacy JSON file."""
    G.graph["last_updated"] = _now()
    if model:
        G.graph["last_model"] = model
    if profile is not None:
        G.graph["profile_id"] = normalize_profile_id(profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_graph.node_link_data(G), f, indent=2)


# ─── Auto-migration: JSON → SQLite ────────────────────────────────────────────

def _migrate_if_needed(db_path: Path, profile: str | None = None) -> None:
    """
    If no DB exists but profile_graph.json does, migrate it (and any snapshot
    JSON files) into the SQLite DB automatically. Runs once per profile.
    """
    if db_path.exists():
        return

    pid       = normalize_profile_id(profile)
    json_path = db_path.parent / "profile_graph.json"
    if not json_path.exists():
        return  # Fresh profile — no migration needed

    logger.info("Migrating profile '%s' from JSON → SQLite …", pid)

    # Migrate live graph
    G = _load_from_json(json_path)
    _graph_to_db(G, db_path)

    # Migrate snapshot JSON files
    snaps_dir = db_path.parent / "snapshots"
    migrated  = 0
    if snaps_dir.exists():
        for snap_file in sorted(snaps_dir.glob("profile_*.json")):
            try:
                G_snap = _load_from_json(snap_file)
                _insert_snapshot(db_path, G_snap, G_snap.graph.get("last_model", ""))
                migrated += 1
            except Exception as exc:
                logger.warning("Could not migrate snapshot %s: %s", snap_file.name, exc)

    logger.info(
        "Migration complete for '%s': live graph + %d snapshot(s) imported.",
        pid, migrated,
    )


# ─── Public API ───────────────────────────────────────────────────────────────

def load_graph(path: Optional[Path] = None, profile: str | None = None) -> nx.DiGraph:
    """
    Loads the profile graph.

    Resolution order:
    1. Virtual snapshot path  (snap_<id>_…_.db)  → loads row from DB
    2. Explicit .db path                          → loads live graph from DB
    3. Explicit .json path                        → loads legacy JSON file
    4. Default (profile= kwarg or default)        → loads from profile.db (with auto-migration)
    """
    if path is not None:
        if _is_virtual_snapshot(path):
            db_path, snap_id = _virtual_to_db_and_id(path)
            try:
                return _load_snapshot_by_id(db_path, snap_id)
            except FileNotFoundError:
                logger.warning("Virtual snapshot %s not found — fresh graph", path)
                return _fresh_graph()

        if path.suffix == ".db":
            return _db_to_graph(path)

        # .json path (backward compat — test helpers, old snapshot files)
        return _load_from_json(path)

    # No explicit path: use the profile's DB (with migration from JSON if needed)
    db_path = get_db_path(profile)
    _migrate_if_needed(db_path, profile)
    return _db_to_graph(db_path)


def save_graph(G: nx.DiGraph, path: Optional[Path] = None,
               model: str = "", profile: str | None = None) -> None:
    """
    Serializes the graph to storage.

    If an explicit .json path is given, writes a legacy JSON file (backward
    compat for scripts and test helpers).  Otherwise writes to the profile's
    SQLite DB.
    """
    G.graph["last_updated"] = _now()
    if model:
        G.graph["last_model"] = model
    if profile is not None:
        G.graph["profile_id"] = normalize_profile_id(profile)

    if path is not None and path.suffix == ".json":
        _save_to_json(G, path, model=model, profile=profile)
        logger.info("Graph saved to JSON %s (%d nodes, %d edges)",
                    path, G.number_of_nodes(), G.number_of_edges())
        return

    db_path = (path if path is not None and path.suffix == ".db" else None) or get_db_path(profile)
    _graph_to_db(G, db_path)
    logger.info("Graph saved to SQLite %s (%d nodes, %d edges)",
                db_path, G.number_of_nodes(), G.number_of_edges())


def save_snapshot(G: nx.DiGraph, model: str = "", profile: str | None = None) -> Path:
    """
    Saves a snapshot of the graph as a row in the snapshots table.
    Returns a virtual Path that can be passed back to load_graph() or
    rollback_to_snapshot() — no real file is written.
    """
    G.graph["last_updated"] = _now()
    if model:
        G.graph["last_model"] = model

    db_path = get_db_path(profile)
    snap_id, ts = _insert_snapshot(db_path, G, model)
    virtual_path = _snapshot_virtual_path(db_path, snap_id, ts, model)
    logger.info("Snapshot #%d saved (model=%s, nodes=%d)", snap_id, model, G.number_of_nodes())
    return virtual_path


def list_snapshots(profile: str | None = None) -> list[dict]:
    """
    Returns metadata for all snapshots, newest first.
    Each entry: {path, filename, timestamp, model, node_count}.

    'path' is a virtual Path suitable for passing to load_graph() or
    rollback_to_snapshot().
    """
    db_path = get_db_path(profile)
    if not db_path.exists():
        return []

    _ensure_schema(db_path)
    snapshots = []
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                "SELECT id, timestamp, model, node_count FROM snapshots ORDER BY id DESC"
            ).fetchall()
        for row in rows:
            snap_id   = row["id"]
            ts        = row["timestamp"]
            model     = row["model"]
            node_count = row["node_count"]
            vpath     = _snapshot_virtual_path(db_path, snap_id, ts, model)
            snapshots.append({
                "path":       vpath,
                "filename":   vpath.name,
                "timestamp":  ts,
                "model":      model,
                "node_count": node_count,
            })
    except Exception as exc:
        logger.warning("Could not list snapshots for %s: %s", db_path, exc)

    return snapshots


def rollback_to_snapshot(snapshot_path: Path, profile: str | None = None) -> nx.DiGraph:
    """
    Restores the live graph to a previous snapshot.
    Always saves a safety snapshot of the current state first.

    Accepts both virtual .db snapshot paths (new) and real .json paths (legacy).
    Raises FileNotFoundError if the snapshot doesn't exist.
    Raises ValueError if the snapshot is missing the 'user' anchor node.
    """
    # Validate the snapshot exists before touching anything
    if _is_virtual_snapshot(snapshot_path):
        db_path, snap_id = _virtual_to_db_and_id(snapshot_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Profile DB not found: {db_path}")
        # Probe for the row
        with _connect(db_path) as conn:
            row = conn.execute(
                "SELECT id FROM snapshots WHERE id = ?", (snap_id,)
            ).fetchone()
        if row is None:
            raise FileNotFoundError(f"Snapshot #{snap_id} not found in {db_path}")
    elif snapshot_path.suffix == ".json":
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")
    else:
        raise FileNotFoundError(f"Unrecognised snapshot reference: {snapshot_path}")

    # Safety snapshot of current state
    current      = load_graph(profile=profile)
    safety_path  = save_snapshot(current, model=current.graph.get("last_model", "pre-rollback"),
                                 profile=profile)
    logger.info("Pre-rollback safety snapshot saved (%s)", safety_path.name)

    # Load and validate the target snapshot
    restored = load_graph(snapshot_path)
    if "user" not in restored.nodes:
        raise ValueError(
            f"Snapshot {snapshot_path.name} is missing the 'user' anchor node."
        )

    save_graph(restored, model=restored.graph.get("last_model", ""), profile=profile)
    logger.info("Rolled back to %s (%d nodes)", snapshot_path.name, restored.number_of_nodes())
    return restored


# ─── Graph helpers ─────────────────────────────────────────────────────────────

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
    prefix      = NODE_TYPE_PREFIX.get(node_type, node_type[:3])
    normalized  = unicodedata.normalize("NFKD", label)
    ascii_label = normalized.encode("ascii", "ignore").decode()
    slug = re.sub(r"\s+", "_", re.sub(r"[^\w\s]", "", ascii_label).lower().strip())
    return f"{prefix}_{slug}"


def get_profile_summary(G: nx.DiGraph) -> dict:
    """
    Walks all edges out from 'user' and returns a clean summary dict.
    This is what we pass to the gatekeeper LLM as context.
    """
    summary: dict = {
        "identity": {},
        "core_values": [],
        "long_term_goals": [],
        "short_term_states": [],
        "relationships": [],
        "custom_categories": {},
    }

    user_data = G.nodes.get("user", {})
    for f in ("name", "age", "occupation", "location"):
        value = user_data.get(f"identity_{f}")
        if value is not None:
            summary["identity"][f]              = value
            summary["identity"][f"{f}_confidence"] = user_data.get(f"identity_{f}_confidence", 0.0)
    if user_data.get("identity_additional"):
        summary["identity"]["additional"] = user_data["identity_additional"]

    known_types = {"core_value", "long_term_goal", "short_term_state", "person"}

    for _, target, edge_data in G.out_edges("user", data=True):
        node_data = G.nodes[target]
        node_type = node_data.get("node_type", "")
        relation  = edge_data.get("relation", "")

        if node_type == "core_value":
            summary["core_values"].append({
                "label":          node_data.get("label", target),
                "confidence":     edge_data.get("weight", 0.5),
                "evidence_count": node_data.get("evidence_count", 1),
            })
        elif node_type == "long_term_goal":
            summary["long_term_goals"].append({
                "label":      node_data.get("label", target),
                "confidence": edge_data.get("weight", 0.5),
            })
        elif node_type == "short_term_state":
            summary["short_term_states"].append({
                "label":    node_data.get("label", target),
                "category": node_data.get("category", "stressor"),
                "intensity": edge_data.get("intensity", edge_data.get("weight", 0.5)),
            })
        elif node_type == "person" and relation == "knows":
            summary["relationships"].append({
                "name":              node_data.get("name", target),
                "relationship_type": edge_data.get("relationship_type", "unknown"),
                "tone":              edge_data.get("tone", "neutral"),
                "strength":          edge_data.get("strength", edge_data.get("weight", 0.5)),
            })
        elif node_type not in known_types and node_type:
            if node_type not in summary["custom_categories"]:
                summary["custom_categories"][node_type] = []
            summary["custom_categories"][node_type].append({
                "label":          node_data.get("label", target),
                "confidence":     edge_data.get("weight", 0.5),
                "evidence_count": node_data.get("evidence_count", 1),
            })

    if not summary["identity"]:
        del summary["identity"]
    if not summary["custom_categories"]:
        del summary["custom_categories"]

    return summary


# ─── Diff ─────────────────────────────────────────────────────────────────────

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
    O(n) — fine for personal knowledge graphs (< 1 000 nodes).
    """
    diff = GraphDiff()

    def confidence(G: nx.DiGraph, node_id: str) -> Optional[float]:
        if "user" not in G or node_id not in G:
            return None
        e = G.edges.get(("user", node_id), {})
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
        nd    = NodeDiff(node_id, label(G_new, node_id), ntype(G_new, node_id), old_c, new_c)
        (diff.strengthened if (old_c and new_c and new_c > old_c) else diff.unchanged).append(nd)

    return diff


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
