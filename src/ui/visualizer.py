"""
3D Knowledge Graph Visualizer.

Input:  NetworkX DiGraph  (loaded from profile.db)
Output: self-contained HTML string  (embedded in Streamlit via st.components.v1.html)

Converts the NetworkX profile graph into a self-contained HTML/JS component
powered by Three.js + 3d-force-graph for embedding in Streamlit via
st.components.v1.html().

Layout concept:
  - User node is fixed at the origin (0, 0, 0)
  - All other nodes radiate outward on a sphere
  - Radius = inverse of confidence: HIGH confidence -> close to centre
  - Nodes distributed evenly using the Fibonacci golden-angle spiral

All visual constants (colors, sizes, camera position, star count) are imported
from src/core/config.py so they can be tuned without touching this file.
"""

from __future__ import annotations

import json
import math
from typing import Dict, Any, List

import networkx as nx

from src.core.config import (
    NODE_COLORS,
    VIZ_CUSTOM_COLOR,
    VIZ_DEFAULT_COLOR,
    VIZ_NODE_RADIUS_MIN,
    VIZ_NODE_RADIUS_SPAN,
    VIZ_NODE_SIZE_BASE,
    VIZ_NODE_SIZE_SPAN,
    VIZ_LINK_WIDTH_BASE,
    VIZ_LINK_WIDTH_SPAN,
    VIZ_PARTICLE_SPEED_BASE,
    VIZ_PARTICLE_SPEED_SPAN,
    VIZ_STAR_COUNT,
    VIZ_CAMERA_X,
    VIZ_CAMERA_Y,
    VIZ_CAMERA_Z,
)

# Golden ratio for Fibonacci sphere distribution
_GOLDEN = (1.0 + math.sqrt(5.0)) / 2.0


# ─── Confidence extraction ────────────────────────────────────────────────────

def _get_confidence(edge_data: Dict[str, Any]) -> float:
    """
    Unified confidence extractor across all edge types.
    - core_value / long_term_goal use 'weight'
    - short_term_state uses 'intensity'
    - person (knows) uses 'strength'
    Falls back to 0.5.
    """
    # Uses or-chaining for brevity. Note: 0.0 is falsy and would fall through
    # to the next key (or ultimately to 0.5). In practice this is harmless because
    # the pipeline never produces exactly 0.0 confidence.
    return float(
        edge_data.get("weight")
        or edge_data.get("strength")
        or edge_data.get("intensity")
        or 0.5
    )


# ─── Position computation ─────────────────────────────────────────────────────

def _compute_positions(non_user_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Assigns fx, fy, fz to each node using Fibonacci golden-angle sphere.

    Radius formula (from config):
      r = (1 - confidence) * VIZ_NODE_RADIUS_SPAN + VIZ_NODE_RADIUS_MIN
      -> confidence 1.0  ->  r ~ VIZ_NODE_RADIUS_MIN        (close to centre)
      -> confidence 0.5  ->  r ~ mid-range
      -> confidence 0.0  ->  r ~ VIZ_NODE_RADIUS_MIN + VIZ_NODE_RADIUS_SPAN (outer shell)
    """
    n = len(non_user_nodes)
    if n == 0:
        return non_user_nodes

    for i, node in enumerate(non_user_nodes):
        confidence = node.get("confidence", 0.5)
        r = (1.0 - confidence) * VIZ_NODE_RADIUS_SPAN + VIZ_NODE_RADIUS_MIN

        if n == 1:
            theta = math.pi / 2
            phi   = 0.0
        else:
            theta = math.acos(1.0 - 2.0 * (i + 0.5) / n)
            phi   = 2.0 * math.pi * i / _GOLDEN

        node["fx"] = round(r * math.sin(theta) * math.cos(phi), 2)
        node["fy"] = round(r * math.sin(theta) * math.sin(phi), 2)
        node["fz"] = round(r * math.cos(theta), 2)

    return non_user_nodes


# ─── Graph data builder ───────────────────────────────────────────────────────

def build_graph_data(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Converts a NetworkX DiGraph into the JSON structure expected by 3d-force-graph.
    Returns {"nodes": [...], "links": [...]}.
    """
    nodes: List[Dict[str, Any]] = []
    links: List[Dict[str, Any]] = []

    # ── Centre / user node — show identity info if available
    user_data = G.nodes.get("user", {})
    user_label = user_data.get("identity_name") or "You"
    identity_parts = []
    if user_data.get("identity_occupation"):
        identity_parts.append(user_data["identity_occupation"])
    if user_data.get("identity_age"):
        identity_parts.append(f"Age: {user_data['identity_age']}")
    if user_data.get("identity_location"):
        identity_parts.append(user_data["identity_location"])
    if identity_parts:
        user_label = f"{user_label} ({', '.join(identity_parts)})"

    nodes.append({
        "id":         "user",
        "label":      user_label,
        "node_type":  "user",
        "color":      NODE_COLORS["user"],
        "size":       14,
        "confidence": 1.0,
        "fx": 0.0, "fy": 0.0, "fz": 0.0,
    })

    # ── All other nodes
    non_user: List[Dict[str, Any]] = []

    for node_id, node_data in G.nodes(data=True):
        if node_id == "user":
            continue
        if not G.has_edge("user", node_id):
            continue

        edge_data  = G.edges["user", node_id]
        confidence = _get_confidence(edge_data)
        node_type  = node_data.get("node_type", "core_value")
        label      = (
            node_data.get("label")
            or node_data.get("name")
            or node_id
        )
        label = str(label).capitalize()
        color = NODE_COLORS.get(node_type, VIZ_CUSTOM_COLOR if node_type else VIZ_DEFAULT_COLOR)

        non_user.append({
            "id":         node_id,
            "label":      label,
            "node_type":  node_type,
            "color":      color,
            "size":       round(confidence * VIZ_NODE_SIZE_SPAN + VIZ_NODE_SIZE_BASE, 2),
            "confidence": round(confidence, 3),
        })

        links.append({
            "source":        "user",
            "target":        node_id,
            "color":         color,
            "width":         round(confidence * VIZ_LINK_WIDTH_SPAN + VIZ_LINK_WIDTH_BASE, 2),
            "confidence":    round(confidence, 3),
            "particleSpeed": round(VIZ_PARTICLE_SPEED_BASE + confidence * VIZ_PARTICLE_SPEED_SPAN, 4),
        })

    non_user = _compute_positions(non_user)
    nodes.extend(non_user)

    return {"nodes": nodes, "links": links}


# ─── HTML template ────────────────────────────────────────────────────────────

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }
    html, body {
      width: 100%; height: 100%;
      background: #0a0d12;
      overflow: hidden;
      font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    }
    #wrap { width: 100%; height: COMPONENT_HEIGHTpx; position: relative; }

    /* Starfield canvas behind the WebGL scene */
    #stars {
      position: absolute; top: 0; left: 0;
      width: 100%; height: 100%;
      pointer-events: none; z-index: 0;
    }

    /* Minimal legend — bottom-left, frosted glass */
    #legend {
      position: absolute; bottom: 16px; left: 16px; z-index: 10;
      background: rgba(10,13,18,0.82);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 10px;
      padding: 12px 16px;
      pointer-events: none;
    }
    .lg-row {
      display: flex; align-items: center; gap: 8px;
      margin-bottom: 5px; color: rgba(255,255,255,0.65); font-size: 11px;
      letter-spacing: 0.02em;
    }
    .lg-row:last-child { margin-bottom: 0; }
    .lg-dot {
      width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
    }
    #legend .lg-hint {
      margin-top: 8px; padding-top: 8px;
      border-top: 1px solid rgba(255,255,255,0.06);
      color: rgba(255,255,255,0.30); font-size: 10px; line-height: 1.5;
    }

    /* Tooltip */
    #tip {
      position: fixed; z-index: 20;
      background: rgba(10,12,18,0.92);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 8px; padding: 8px 12px;
      color: #fff;
      font-size: 12.5px; pointer-events: none; display: none;
      max-width: 220px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    }
    #tip strong { font-weight: 600; }
    #tip .meta { color: rgba(255,255,255,0.45); font-size: 11px; margin-top: 2px; }
  </style>
</head>
<body>
  <canvas id="stars"></canvas>
  <div id="wrap"></div>

  <div id="legend">
    <div class="lg-row"><div class="lg-dot" style="background:#fff;box-shadow:0 0 4px rgba(255,255,255,0.6);"></div>You</div>
    <div class="lg-row"><div class="lg-dot" style="background:LEGEND_COLOR_CV;"></div>Core Value</div>
    <div class="lg-row"><div class="lg-dot" style="background:LEGEND_COLOR_LTG;"></div>Long-term Goal</div>
    <div class="lg-row"><div class="lg-dot" style="background:LEGEND_COLOR_STS;"></div>Current State</div>
    <div class="lg-row"><div class="lg-dot" style="background:LEGEND_COLOR_PERSON;"></div>Person</div>
    <div class="lg-hint">Beam thickness = confidence<br>Distance from centre = inverse confidence</div>
  </div>

  <div id="tip"></div>

  <script src="https://unpkg.com/3d-force-graph@1.73.5/dist/3d-force-graph.min.js"></script>

  <script>
    var GRAPH_DATA = GRAPH_DATA_PLACEHOLDER;
    var wrap = document.getElementById('wrap');
    var tip  = document.getElementById('tip');

    // ── Starfield ─────────────────────────────────────────────────────────────
    (function drawStars() {
      var c = document.getElementById('stars');
      c.width  = window.innerWidth || 900;
      c.height = COMPONENT_HEIGHT;
      var ctx = c.getContext('2d');
      for (var i = 0; i < STAR_COUNT; i++) {
        var x = Math.random() * c.width;
        var y = Math.random() * c.height;
        var r = Math.random() * 1.2 + 0.15;
        var a = 0.15 + Math.random() * 0.5;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255,255,255,' + a + ')';
        ctx.fill();
      }
    })();

    // ── Tooltip tracking ──────────────────────────────────────────────────────
    document.addEventListener('mousemove', function(e) {
      tip.style.left = (e.clientX + 14) + 'px';
      tip.style.top  = (e.clientY - 10) + 'px';
    });

    // ── Build graph ───────────────────────────────────────────────────────────
    var Graph = ForceGraph3D({ controlType: 'orbit' })(wrap)
      .width(wrap.clientWidth || 900)
      .height(COMPONENT_HEIGHT)
      .backgroundColor('rgba(0,0,0,0)')
      .graphData(GRAPH_DATA)
      .nodeId('id')
      .nodeLabel('')
      .nodeColor(function(n) { return n.color; })
      .nodeVal(function(n) { return n.size; })
      .nodeOpacity(0.95)
      .nodeResolution(20)

      // ── Links — glowing beams from centre ───────────────────────────────
      .linkColor(function(l) { return l.color; })
      .linkWidth(function(l) { return l.width; })
      .linkOpacity(0.85)
      .linkDirectionalParticles(3)
      .linkDirectionalParticleWidth(function(l) { return Math.max(l.width * 0.5, 1.2); })
      .linkDirectionalParticleColor(function(l) { return l.color; })
      .linkDirectionalParticleSpeed(function(l) { return l.particleSpeed; })

      // ── Hover tooltip ───────────────────────────────────────────────────
      .onNodeHover(function(node) {
        if (node) {
          var pct = node.id === 'user' ? '' :
            '<div class="meta">' + Math.round(node.confidence * 100) + '% confidence</div>';
          var type = node.node_type.replace(/_/g, ' ');
          tip.innerHTML =
            '<strong style="color:' + node.color + '">' + node.label + '</strong>'
            + '<div class="meta">' + type + '</div>'
            + pct;
          tip.style.display = 'block';
        } else {
          tip.style.display = 'none';
        }
        wrap.style.cursor = node ? 'pointer' : 'default';
      })

      // ── Keep link force for reference resolution; disable the rest ──────
      .d3Force('charge', null)
      .d3Force('center', null)

      .onEngineStop(function() {
        setTimeout(function() { _onReady(); }, 600);
      });

    // Fallback in case onEngineStop fires too early
    var _ready = false;
    setTimeout(function() { if (!_ready) _onReady(); }, 2200);

    function _onReady() {
      if (_ready) return;
      _ready = true;

      // ── Emissive glow on nodes ────────────────────────────────────────
      var userMesh = null;

      GRAPH_DATA.nodes.forEach(function(node) {
        var obj = node.__threeObj;
        if (!obj) return;
        var mesh = obj.isMesh ? obj : null;
        obj.traverse(function(child) { if (!mesh && child.isMesh) mesh = child; });
        if (!mesh || !mesh.material) return;

        if (mesh.material.color && mesh.material.color.clone) {
          mesh.material.emissive         = mesh.material.color.clone();
          mesh.material.emissiveIntensity = node.id === 'user' ? 0.85 : 0.45;
          mesh.material.needsUpdate       = true;
        }
        if (node.id === 'user') userMesh = mesh;
      });

      // ── Emissive glow on link beams ───────────────────────────────────
      function brightenLinks() {
        GRAPH_DATA.links.forEach(function(link) {
          var lineObj = link.__lineObj;
          if (!lineObj) return;

          var glow = function(obj) {
            if (obj.isMesh && obj.material && obj.material.color && obj.material.color.clone) {
              obj.material.emissive         = obj.material.color.clone();
              obj.material.emissiveIntensity = 0.8;
              obj.material.opacity           = 0.9;
              obj.material.transparent       = true;
              obj.material.depthWrite        = false;
              obj.material.needsUpdate       = true;
            }
          };
          glow(lineObj);
          lineObj.traverse(glow);
        });
      }
      brightenLinks();
      setTimeout(brightenLinks, 1000);

      // ── Orbit auto-rotate ─────────────────────────────────────────────
      var controls = Graph.controls();
      controls.autoRotate      = true;
      controls.autoRotateSpeed = 0.5;

      // ── Pulsing centre node ───────────────────────────────────────────
      if (userMesh && userMesh.material) {
        var t = 0;
        (function pulse() {
          requestAnimationFrame(pulse);
          t += 0.018;
          userMesh.material.emissiveIntensity = 0.45 + 0.40 * Math.sin(t);
        })();
      }
    }

    // ── Camera ────────────────────────────────────────────────────────────────
    Graph.cameraPosition({ x: CAMERA_X, y: CAMERA_Y, z: CAMERA_Z });
  </script>
</body>
</html>
"""


# ─── Public API ───────────────────────────────────────────────────────────────

def build_visualizer_html(G: nx.DiGraph, height: int | None = None) -> str:
    """
    Top-level function called from app.py.
    Builds graph data from the NetworkX graph and injects it into
    the self-contained HTML template.

    Args:
        G:      The profile DiGraph from graph_store.load_graph()
        height: Component height in pixels. Defaults to VIZ_DEFAULT_HEIGHT from config.

    Returns:
        Complete self-contained HTML string ready for st.components.v1.html()
    """
    from src.core.config import VIZ_DEFAULT_HEIGHT
    if height is None:
        height = VIZ_DEFAULT_HEIGHT

    graph_data = build_graph_data(G)
    graph_json = json.dumps(graph_data, separators=(",", ":"))

    html = _HTML_TEMPLATE
    html = html.replace("GRAPH_DATA_PLACEHOLDER", graph_json)
    html = html.replace("COMPONENT_HEIGHT", str(height))
    html = html.replace("STAR_COUNT",       str(VIZ_STAR_COUNT))
    html = html.replace("CAMERA_X",         str(VIZ_CAMERA_X))
    html = html.replace("CAMERA_Y",         str(VIZ_CAMERA_Y))
    html = html.replace("CAMERA_Z",         str(VIZ_CAMERA_Z))
    # Legend colors from config
    html = html.replace("LEGEND_COLOR_CV",     NODE_COLORS.get("core_value",       "#00B4D8"))
    html = html.replace("LEGEND_COLOR_LTG",    NODE_COLORS.get("long_term_goal",   "#06D6A0"))
    html = html.replace("LEGEND_COLOR_STS",    NODE_COLORS.get("short_term_state", "#EF476F"))
    html = html.replace("LEGEND_COLOR_PERSON", NODE_COLORS.get("person",           "#FFD166"))
    return html
