"""
Tests for src/graph_store.py  (SQLite backend)

Covers: load/save roundtrip, snapshot creation, listing, diffing,
rollback, make_node_id, get_profile_summary, and error handling.
"""

import json
import pytest
import networkx as nx
from pathlib import Path

from src.storage.graph import (
    _fresh_graph,
    get_graph_path,
    get_snapshots_dir,
    load_graph,
    normalize_profile_id,
    save_graph,
    save_snapshot,
    list_snapshots,
    rollback_to_snapshot,
    make_node_id,
    get_profile_summary,
    diff_graphs,
    NodeDiff,
    GraphDiff,
)


# ─── load_graph ────────────────────────────────────────────────────────────────

def test_load_graph_fresh_when_file_missing(tmp_data_dir):
    missing = tmp_data_dir / "nonexistent.json"
    G = load_graph(missing)
    assert "user" in G.nodes
    assert G.number_of_nodes() == 1


def test_load_graph_fresh_when_corrupt(tmp_data_dir):
    corrupt = tmp_data_dir / "corrupt.json"
    corrupt.write_text("{not valid json", encoding="utf-8")
    G = load_graph(corrupt)
    assert "user" in G.nodes


def test_load_graph_fresh_logs_warning_for_corrupt(tmp_data_dir, caplog):
    import logging
    corrupt = tmp_data_dir / "corrupt.json"
    corrupt.write_text("{not valid json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="src.storage.graph"):
        load_graph(corrupt)
    assert "corrupt" in caplog.text.lower() or "unreadable" in caplog.text.lower()


# ─── save_graph / load_graph roundtrip ────────────────────────────────────────

def test_save_and_load_roundtrip_json(tmp_data_dir, sample_graph):
    """Explicit .json path still writes/reads JSON (backward compat)."""
    path = tmp_data_dir / "graph.json"
    save_graph(sample_graph, path=path, model="test_model")
    loaded = load_graph(path)

    assert loaded.number_of_nodes() == sample_graph.number_of_nodes()
    assert loaded.number_of_edges() == sample_graph.number_of_edges()
    assert loaded.graph["last_model"] == "test_model"
    assert "user" in loaded.nodes
    assert "cv_discipline" in loaded.nodes


def test_save_and_load_roundtrip_sqlite(tmp_data_dir, monkeypatch, sample_graph):
    """profile= kwarg uses SQLite; graph survives a full round-trip."""
    import src.storage.graph as gs
    monkeypatch.setattr(gs, "PROFILES_DIR", tmp_data_dir / "profiles")

    save_graph(sample_graph, profile="test", model="test_model")
    loaded = load_graph(profile="test")

    assert loaded.number_of_nodes() == sample_graph.number_of_nodes()
    assert loaded.number_of_edges() == sample_graph.number_of_edges()
    assert loaded.graph["last_model"] == "test_model"
    assert "user" in loaded.nodes
    assert "cv_discipline" in loaded.nodes


def test_save_graph_creates_parent_dir(tmp_path):
    nested_path = tmp_path / "deep" / "nested" / "graph.json"
    G = _fresh_graph()
    save_graph(G, path=nested_path)
    assert nested_path.exists()


def test_profile_graph_paths_are_normalized(tmp_data_dir, monkeypatch):
    import src.storage.graph as gs
    monkeypatch.setattr(gs, "PROFILES_DIR", tmp_data_dir / "profiles")

    assert normalize_profile_id(" Ted ") == "ted"
    assert get_graph_path("TeD") == (tmp_data_dir / "profiles" / "ted" / "profile.db")
    assert get_snapshots_dir("sal") == (tmp_data_dir / "profiles" / "sal" / "snapshots")


def test_save_and_load_isolated_by_profile(tmp_data_dir, monkeypatch):
    import src.storage.graph as gs
    monkeypatch.setattr(gs, "PROFILES_DIR", tmp_data_dir / "profiles")

    G_ted = _fresh_graph()
    G_ted.add_node("cv_focus", node_type="core_value", label="focus")
    G_ted.add_edge("user", "cv_focus", relation="holds_value", weight=0.8)

    G_sal = _fresh_graph()
    G_sal.add_node("cv_kindness", node_type="core_value", label="kindness")
    G_sal.add_edge("user", "cv_kindness", relation="holds_value", weight=0.9)

    save_graph(G_ted, profile="ted")
    save_graph(G_sal, profile="sal")

    loaded_ted = load_graph(profile="ted")
    loaded_sal = load_graph(profile="sal")

    assert "cv_focus" in loaded_ted.nodes
    assert "cv_kindness" not in loaded_ted.nodes
    assert "cv_kindness" in loaded_sal.nodes
    assert "cv_focus" not in loaded_sal.nodes


# ─── save_snapshot ─────────────────────────────────────────────────────────────

def test_save_snapshot_returns_loadable_path(tmp_data_dir, monkeypatch, sample_graph):
    import src.storage.graph as gs
    monkeypatch.setattr(gs, "PROFILES_DIR", tmp_data_dir / "profiles")

    # Save a live graph first so the DB exists
    save_graph(sample_graph, profile="test")
    snap_path = save_snapshot(sample_graph, model="llama3", profile="test")

    # Virtual path ends in .db and starts with "snap_"
    assert snap_path.suffix == ".db"
    assert snap_path.stem.startswith("snap_")
    # Model name should appear somewhere in the filename
    assert "llama3" in snap_path.name

    # Can round-trip through load_graph
    loaded = load_graph(snap_path)
    assert "cv_discipline" in loaded.nodes


def test_save_snapshot_sanitises_model_name(tmp_data_dir, monkeypatch, fresh_graph):
    import src.storage.graph as gs
    monkeypatch.setattr(gs, "PROFILES_DIR", tmp_data_dir / "profiles")

    save_graph(fresh_graph, profile="test")
    snap_path = save_snapshot(fresh_graph, model="mistral:7b", profile="test")
    # Colons must be sanitised
    assert ":" not in snap_path.name


# ─── list_snapshots ────────────────────────────────────────────────────────────

def test_list_snapshots_empty_when_no_db(tmp_data_dir, monkeypatch):
    import src.storage.graph as gs
    monkeypatch.setattr(gs, "PROFILES_DIR", tmp_data_dir / "profiles")
    result = list_snapshots(profile="test")
    assert result == []


def test_list_snapshots_sorted_newest_first(tmp_data_dir, monkeypatch, fresh_graph):
    import src.storage.graph as gs
    monkeypatch.setattr(gs, "PROFILES_DIR", tmp_data_dir / "profiles")

    save_graph(fresh_graph, profile="test")
    save_snapshot(fresh_graph, model="m1", profile="test")
    save_snapshot(fresh_graph, model="m2", profile="test")

    snapshots = list_snapshots(profile="test")
    assert len(snapshots) == 2
    # Newest first — higher snapshot id means newer
    assert snapshots[0]["filename"] >= snapshots[1]["filename"]


def test_list_snapshots_isolated_by_profile(tmp_data_dir, monkeypatch, fresh_graph):
    import src.storage.graph as gs
    monkeypatch.setattr(gs, "PROFILES_DIR", tmp_data_dir / "profiles")

    save_graph(fresh_graph, profile="ted")
    save_graph(fresh_graph, profile="sal")
    save_snapshot(fresh_graph, model="m1", profile="ted")
    save_snapshot(fresh_graph, model="m2", profile="sal")

    ted_snaps = list_snapshots(profile="ted")
    sal_snaps = list_snapshots(profile="sal")

    assert len(ted_snaps) == 1
    assert len(sal_snaps) == 1
    assert "m1" in ted_snaps[0]["filename"]
    assert "m2" in sal_snaps[0]["filename"]


def test_list_snapshots_contains_required_keys(tmp_data_dir, monkeypatch, fresh_graph):
    import src.storage.graph as gs
    monkeypatch.setattr(gs, "PROFILES_DIR", tmp_data_dir / "profiles")

    save_graph(fresh_graph, profile="test")
    save_snapshot(fresh_graph, model="llama3", profile="test")
    snaps = list_snapshots(profile="test")
    assert len(snaps) == 1
    for key in ("path", "filename", "timestamp", "model", "node_count"):
        assert key in snaps[0], f"Missing key: {key}"


# ─── rollback_to_snapshot ──────────────────────────────────────────────────────

def test_rollback_restores_graph(tmp_data_dir, monkeypatch, sample_graph, fresh_graph):
    import src.storage.graph as gs
    monkeypatch.setattr(gs, "PROFILES_DIR", tmp_data_dir / "profiles")

    save_graph(sample_graph, profile="test")
    snapshot_path = save_snapshot(sample_graph, model="test", profile="test")
    save_graph(fresh_graph, profile="test")

    restored = rollback_to_snapshot(snapshot_path, profile="test")

    assert "cv_discipline" in restored.nodes
    assert restored.number_of_nodes() == sample_graph.number_of_nodes()


def test_rollback_creates_safety_snapshot(tmp_data_dir, monkeypatch, sample_graph, fresh_graph):
    import src.storage.graph as gs
    monkeypatch.setattr(gs, "PROFILES_DIR", tmp_data_dir / "profiles")

    save_graph(sample_graph, profile="test")
    snapshot_path = save_snapshot(sample_graph, model="test", profile="test")
    save_graph(fresh_graph, profile="test")

    before_count = len(list_snapshots(profile="test"))
    rollback_to_snapshot(snapshot_path, profile="test")
    after_count = len(list_snapshots(profile="test"))

    # A new safety snapshot should have been created
    assert after_count == before_count + 1


def test_rollback_raises_for_missing_virtual_snapshot(tmp_data_dir, monkeypatch, fresh_graph):
    import src.storage.graph as gs
    monkeypatch.setattr(gs, "PROFILES_DIR", tmp_data_dir / "profiles")
    save_graph(fresh_graph, profile="test")

    fake_path = get_graph_path("test").parent / "snapshots" / "snap_9999_20260101_000000_none.db"
    with pytest.raises(FileNotFoundError):
        rollback_to_snapshot(fake_path, profile="test")


def test_rollback_raises_for_missing_json_file():
    with pytest.raises(FileNotFoundError):
        rollback_to_snapshot(Path("/nonexistent/snapshot.json"))


def test_rollback_raises_for_invalid_graph(tmp_data_dir, monkeypatch):
    import src.storage.graph as gs
    monkeypatch.setattr(gs, "PROFILES_DIR", tmp_data_dir / "profiles")

    # Create a JSON snapshot file without the "user" node
    bad_graph = nx.DiGraph()
    bad_graph.add_node("something_else")
    bad_path = tmp_data_dir / "bad_snap.json"
    save_graph(bad_graph, path=bad_path)

    with pytest.raises(ValueError, match="user"):
        rollback_to_snapshot(bad_path, profile="test")


# ─── migration: JSON → SQLite ──────────────────────────────────────────────────

def test_auto_migration_from_json(tmp_data_dir, monkeypatch, sample_graph):
    """If profile_graph.json exists but no .db, load_graph migrates automatically."""
    import src.storage.graph as gs
    monkeypatch.setattr(gs, "PROFILES_DIR", tmp_data_dir / "profiles")

    profile_dir = tmp_data_dir / "profiles" / "migtest"
    profile_dir.mkdir(parents=True)

    # Write legacy JSON
    from networkx.readwrite import json_graph as jg
    import json as _json
    sample_graph.graph["last_updated"] = "2026-01-01T00:00:00Z"
    (profile_dir / "profile_graph.json").write_text(
        _json.dumps(jg.node_link_data(sample_graph)), encoding="utf-8"
    )

    # load_graph should migrate and return the full graph
    loaded = load_graph(profile="migtest")
    assert "cv_discipline" in loaded.nodes
    # DB should now exist
    assert (profile_dir / "profile.db").exists()
    # Second load reads from DB
    loaded2 = load_graph(profile="migtest")
    assert "cv_discipline" in loaded2.nodes


# ─── make_node_id ──────────────────────────────────────────────────────────────

def test_make_node_id_basic():
    assert make_node_id("core_value", "discipline") == "cv_discipline"
    assert make_node_id("long_term_goal", "Finish PhD") == "ltg_finish_phd"
    assert make_node_id("person", "Alice") == "person_alice"


def test_make_node_id_is_deterministic():
    a = make_node_id("core_value", "Academic Rigor")
    b = make_node_id("core_value", "Academic Rigor")
    assert a == b


def test_make_node_id_case_insensitive():
    assert make_node_id("core_value", "Discipline") == make_node_id("core_value", "discipline")


def test_make_node_id_handles_spaces():
    nid = make_node_id("long_term_goal", "finish my phd thesis")
    assert " " not in nid
    assert "_" in nid


def test_make_node_id_handles_unicode():
    nid = make_node_id("core_value", "café culture")
    assert isinstance(nid, str)
    assert nid.startswith("cv_")


# ─── get_profile_summary ───────────────────────────────────────────────────────

def test_get_profile_summary_empty_graph(fresh_graph):
    summary = get_profile_summary(fresh_graph)
    assert summary["core_values"]       == []
    assert summary["long_term_goals"]   == []
    assert summary["short_term_states"] == []
    assert summary["relationships"]     == []


def test_get_profile_summary_populated(sample_graph):
    summary = get_profile_summary(sample_graph)
    assert len(summary["core_values"]) == 1
    assert summary["core_values"][0]["label"] == "discipline"
    assert len(summary["relationships"]) == 1
    assert summary["relationships"][0]["name"] == "Alice"


# ─── diff_graphs ───────────────────────────────────────────────────────────────

def test_diff_detects_added_nodes(fresh_graph, sample_graph):
    diff = diff_graphs(fresh_graph, sample_graph)
    added_ids = [nd.node_id for nd in diff.added]
    assert "cv_discipline" in added_ids
    assert "person_alice" in added_ids


def test_diff_detects_removed_nodes(sample_graph, fresh_graph):
    diff = diff_graphs(sample_graph, fresh_graph)
    removed_ids = [nd.node_id for nd in diff.removed]
    assert "cv_discipline" in removed_ids


def test_diff_detects_strengthened_nodes(sample_graph):
    G_new = sample_graph.copy()
    G_new["user"]["cv_discipline"]["weight"] = 0.95  # was 0.8
    diff = diff_graphs(sample_graph, G_new)
    strengthened_ids = [nd.node_id for nd in diff.strengthened]
    assert "cv_discipline" in strengthened_ids


def test_diff_has_changes_false_when_identical(sample_graph):
    diff = diff_graphs(sample_graph, sample_graph)
    assert diff.has_changes is False


def test_diff_has_changes_true_when_nodes_added(fresh_graph, sample_graph):
    diff = diff_graphs(fresh_graph, sample_graph)
    assert diff.has_changes is True
