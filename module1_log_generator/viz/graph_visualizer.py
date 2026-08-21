from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import networkx as nx

from module1_log_generator.core.service_graph import ServiceGraph
from module1_log_generator.models.schema import ServiceGraphConfig

# Generic tokens that don't make useful subsystem names on their own.
_STOPWORD_TOKENS = {
    "service", "svc", "gateway", "api", "backend", "server",
    "db", "database", "store", "worker", "engine",
}


def _classify_role(graph: ServiceGraph, name: str) -> str:

    has_dependents = len(graph.get_dependents(name)) > 0
    has_dependencies = len(graph.get_dependencies(name)) > 0

    if not has_dependents and not has_dependencies:
        return "isolated"
    if not has_dependents:
        return "entry_point"
    if not has_dependencies:
        return "data_store"
    return "internal"


def _detect_subsystems(undirected: nx.Graph) -> list[set[str]]:

    if undirected.number_of_nodes() <= 1:
        return [set(undirected.nodes())] if undirected.number_of_nodes() else []

    if undirected.number_of_edges() == 0:
        # No edges at all -> every node is its own subsystem.
        return [{n} for n in undirected.nodes()]

    try:
        communities = nx.community.greedy_modularity_communities(undirected)
        result = [set(c) for c in communities]
        if result:
            return result
    except Exception:
        pass

    return [set(c) for c in nx.connected_components(undirected)]


def _subsystem_label(members: set[str], graph: ServiceGraph) -> str:

    token_counts: dict[str, int] = {}
    for name in members:
        tokens = {t for t in name.replace("_", "-").split("-") if t and t not in _STOPWORD_TOKENS}
        for t in tokens:
            token_counts[t] = token_counts.get(t, 0) + 1

    if token_counts:
        best_token, best_count = max(token_counts.items(), key=lambda kv: (kv[1], -len(kv[0])))
        if best_count >= max(2, math.ceil(len(members) / 2)):
            return best_token.capitalize()

    hub = max(
        members,
        key=lambda n: len(graph.get_dependencies(n)) + len(graph.get_dependents(n)),
    )
    return f"{hub} group"


def _ring_layout(
    clusters_meta: list[dict[str, Any]],
    cell_radius: dict[str, float],
    quotient: "nx.Graph",
    margin: float = 130.0,
) -> dict[str, tuple[float, float]]:

    if len(clusters_meta) <= 1:
        return {clusters_meta[0]["id"]: (0.0, 0.0)} if clusters_meta else {}

    ids = [c["id"] for c in clusters_meta]
    weight = {}
    for u, v, d in quotient.edges(data=True):
        weight[frozenset((u, v))] = d.get("weight", 1)

    remaining = set(ids)
    start = max(ids, key=lambda cid: cell_radius[cid])
    order = [start]
    remaining.discard(start)
    while remaining:
        last = order[-1]
        nxt = max(
            remaining,
            key=lambda cid: (weight.get(frozenset((last, cid)), 0), cell_radius[cid]),
        )
        order.append(nxt)
        remaining.discard(nxt)

    n = len(order)
    required_r = 0.0
    theta = 2 * math.pi / n
    chord_factor = 2 * math.sin(theta / 2)
    for i in range(n):
        a, b = order[i], order[(i + 1) % n]
        needed = (cell_radius[a] + cell_radius[b] + margin) / chord_factor
        required_r = max(required_r, needed)

    macro_pos = {}
    for i, cid in enumerate(order):
        angle = theta * i - math.pi / 2
        macro_pos[cid] = (required_r * math.cos(angle), required_r * math.sin(angle))
    return macro_pos


def build_graph_data(config: ServiceGraphConfig, graph: ServiceGraph) -> dict[str, Any]:

    service_names = graph.all_services()
    n = max(len(service_names), 1)
    seed = config.simulation_seed if config.simulation_seed is not None else 42
    undirected = graph._graph.to_undirected()

    subsystem_members = sorted(_detect_subsystems(undirected), key=len, reverse=True)
    cluster_of: dict[str, str] = {}
    clusters_meta: list[dict[str, Any]] = []
    for idx, members in enumerate(subsystem_members):
        cid = f"c{idx}"
        for name in members:
            cluster_of[name] = cid
        clusters_meta.append(
            {
                "id": cid,
                "label": _subsystem_label(members, graph),
                "members": sorted(members),
                "size": len(members),
                "color": f"hsl({(idx * 137.508) % 360:.0f}, 68%, 60%)",
            }
        )

    cell_radius = {
        c["id"]: 92.0 + 48.0 * math.sqrt(c["size"]) for c in clusters_meta
    }

    quotient = nx.Graph()
    quotient.add_nodes_from(c["id"] for c in clusters_meta)
    for u, v in graph._graph.edges():
        cu, cv = cluster_of[u], cluster_of[v]
        if cu != cv:
            if quotient.has_edge(cu, cv):
                quotient[cu][cv]["weight"] += 1
            else:
                quotient.add_edge(cu, cv, weight=1)

    macro_pos = _ring_layout(clusters_meta, cell_radius, quotient)

    node_pos: dict[str, tuple[float, float]] = {}
    for c in clusters_meta:
        members = c["members"]
        cx, cy = macro_pos[c["id"]]
        inner_radius = cell_radius[c["id"]] - 40.0

        if len(members) == 1:
            node_pos[members[0]] = (cx, cy)
            continue

        sub_g = undirected.subgraph(members)
        local = nx.spring_layout(
            sub_g, seed=seed, k=1.9 / math.sqrt(len(members)), iterations=300
        )
        lx = [p[0] for p in local.values()]
        ly = [p[1] for p in local.values()]
        centroid_x, centroid_y = sum(lx) / len(lx), sum(ly) / len(ly)
        max_r = max(math.hypot(p[0] - centroid_x, p[1] - centroid_y) for p in local.values()) or 1.0
        scale = (inner_radius * 0.85) / max_r

        for name, (px, py) in local.items():
            node_pos[name] = (
                cx + (px - centroid_x) * scale,
                cy + (py - centroid_y) * scale,
            )

    instances = {name: graph.service_config(name).instances for name in service_names}
    max_instances = max(instances.values()) if instances else 1

    nodes = []
    for name in service_names:
        cfg = graph.service_config(name)
        x, y = node_pos.get(name, (0.0, 0.0))
        cid = cluster_of.get(name, "c0")
        nodes.append(
            {
                "id": name,
                "x": round(x, 1),
                "y": round(y, 1),
                "instances": cfg.instances,
                "radius": round(14 + 16 * math.sqrt(cfg.instances / max_instances), 1),
                "role": _classify_role(graph, name),
                "cluster": cid,
                "log_volume_per_sec": cfg.log_volume_per_sec,
                "normal_latency_ms": cfg.normal_latency_ms,
                "normal_error_rate": cfg.normal_error_rate,
                "peak_hours": cfg.peak_hours,
                "resource_limits": {
                    "cpu_cores": cfg.resource_limits.cpu_cores,
                    "memory_mb": cfg.resource_limits.memory_mb,
                },
                "dependencies": graph.get_dependencies(name),
                "dependents": graph.get_dependents(name),
                "special_call_rate": cfg.special_call_rate,
            }
        )

    call_volumes = [graph.edge_call_volume(u, v) for u, v in graph._graph.edges()]
    max_call_volume = max(call_volumes) if call_volumes else 1

    edges = []
    for u, v in graph._graph.edges():
        cfg = graph.service_config(u)
        call_volume = graph.edge_call_volume(u, v)
        edges.append(
            {
                "source": u,
                "target": v,
                "call_volume": call_volume,
                "width": round(1.3 + 4.2 * (math.log1p(call_volume) / math.log1p(max_call_volume or 1)), 2),
                "call_rate": cfg.special_call_rate.get(v, 1.0),
                "intra": cluster_of.get(u) == cluster_of.get(v),
            }
        )

    clusters_out = [
        {
            "id": c["id"],
            "label": c["label"],
            "size": c["size"],
            "color": c["color"],
            "cx": round(macro_pos[c["id"]][0], 1),
            "cy": round(macro_pos[c["id"]][1], 1),
            "radius": round(cell_radius[c["id"]], 1),
        }
        for c in clusters_meta
    ]

    if nodes:
        xs = [nd["x"] - nd["radius"] for nd in nodes] + [c["cx"] - c["radius"] for c in clusters_out]
        xs_max = [nd["x"] + nd["radius"] for nd in nodes] + [c["cx"] + c["radius"] for c in clusters_out]
        ys = [nd["y"] - nd["radius"] for nd in nodes] + [c["cy"] - c["radius"] for c in clusters_out]
        ys_max = [nd["y"] + nd["radius"] for nd in nodes] + [c["cy"] + c["radius"] for c in clusters_out]
        pad = 60.0
        canvas_w = max(xs_max) - min(xs) + 2 * pad
        canvas_h = max(ys_max) - min(ys) + 2 * pad
        offset_x = -min(xs) + pad
        offset_y = -min(ys) + pad
        for nd in nodes:
            nd["x"] = round(nd["x"] + offset_x, 1)
            nd["y"] = round(nd["y"] + offset_y, 1)
        for c in clusters_out:
            c["cx"] = round(c["cx"] + offset_x, 1)
            c["cy"] = round(c["cy"] + offset_y, 1)
    else:
        canvas_w, canvas_h = 1000.0, 640.0

    cycles = graph.detect_cycles()

    return {
        "meta": {
            "service_count": len(service_names),
            "edge_count": len(edges),
            "total_instances": sum(instances.values()),
            "subsystem_count": len(clusters_out),
            "allow_cycles": config.allow_cycles,
            "cycles": cycles,
            "simulation_seed": config.simulation_seed,
            "canvas_width": round(canvas_w, 1),
            "canvas_height": round(canvas_h, 1),
        },
        "nodes": nodes,
        "edges": edges,
        "clusters": clusters_out,
    }


def render_service_graph_html(
    config: ServiceGraphConfig,
    graph: ServiceGraph,
    output_path: str | Path,
    title: str = "LogSentinel AI — Service Dependency Graph",
) -> Path:
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = build_graph_data(config, graph)
    html = _TEMPLATE.replace("__TITLE__", title).replace(
        "__GRAPH_DATA__", json.dumps(data)
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>__TITLE__</title>
<style>
  :root {
    --bg-0: #0a0e1a;
    --bg-1: #0f1526;
    --bg-2: #141b2f;
    --panel: rgba(20, 27, 47, 0.88);
    --border: rgba(148, 163, 184, 0.14);
    --text-0: #e8ecf6;
    --text-1: #9aa4bd;
    --text-2: #6b7590;
    --accent: #6ea8ff;
    --entry: #22d3ee;
    --store: #f59e0b;
    --internal: #8b7cf6;
    --isolated: #64748b;
    --edge: #3a4361;
    --edge-inter: #4a5578;
    --edge-hi: #6ea8ff;
    --danger: #f87171;
    --radius: 14px;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0; height: 100%;
    background:
      radial-gradient(1200px 600px at 15% -10%, #16223f 0%, transparent 60%),
      radial-gradient(1000px 700px at 110% 10%, #1a1440 0%, transparent 55%),
      var(--bg-0);
    color: var(--text-0);
    font-family: -apple-system, "Segoe UI", Inter, Roboto, Arial, sans-serif;
    overflow: hidden;
  }
  #app { display: flex; flex-direction: column; height: 100%; }

  header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 22px; border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, rgba(15,21,38,0.9), rgba(15,21,38,0.55));
    backdrop-filter: blur(6px);
    z-index: 5;
  }
  header h1 {
    font-size: 15px; font-weight: 650; margin: 0; letter-spacing: .2px;
    display: flex; align-items: center; gap: 10px;
  }
  header h1 .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--accent); box-shadow: 0 0 10px var(--accent);
  }
  .stats { display: flex; gap: 8px; flex-wrap: wrap; }
  .chip {
    font-size: 11.5px; color: var(--text-1);
    background: rgba(148,163,184,0.08); border: 1px solid var(--border);
    padding: 5px 10px; border-radius: 999px; white-space: nowrap;
  }
  .chip b { color: var(--text-0); font-weight: 650; }
  .chip.warn { color: #fca5a5; border-color: rgba(248,113,113,0.35); background: rgba(248,113,113,0.08); }

  main { position: relative; flex: 1; min-height: 0; }
  svg { width: 100%; height: 100%; display: block; cursor: grab; }
  svg.panning { cursor: grabbing; }

  .cluster-blob { transition: opacity .15s; }
  .cluster-blob.dim { opacity: 0.18; }
  .cluster-label {
    fill: var(--text-1); font-size: 12px; font-weight: 700;
    text-anchor: middle; pointer-events: none; letter-spacing: .3px;
  }
  .cluster-count {
    fill: var(--text-2); font-size: 10px; text-anchor: middle; pointer-events: none;
  }

  .edge { stroke: var(--edge); fill: none; transition: stroke .15s, opacity .15s; opacity: 0.65; }
  .edge.inter { stroke: var(--edge-inter); stroke-dasharray: 4 4; opacity: 0.4; }
  .edge-arrow { fill: var(--edge); transition: fill .15s, opacity .15s; opacity: 0.65; }
  .edge-arrow.inter { fill: var(--edge-inter); opacity: 0.4; }
  .dim { opacity: 0.08 !important; }
  .edge.hi { stroke: var(--edge-hi); opacity: 1 !important; stroke-dasharray: none; }
  .edge-arrow.hi { fill: var(--edge-hi); opacity: 1 !important; }
  .edge.flow {
    stroke-dasharray: 5 5;
    animation: flow 0.7s linear infinite;
  }
  @keyframes flow { to { stroke-dashoffset: -20; } }

  .node-group { cursor: pointer; }
  .node-circle {
    stroke-width: 2px; transition: filter .15s, stroke .15s, r .15s;
  }
  .node-group.hi .node-circle { filter: drop-shadow(0 0 10px currentColor); }
  .node-group.selected .node-circle {
    stroke: #fff; stroke-width: 2.5px;
    animation: pulse 1.6s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { stroke-opacity: 1; }
    50% { stroke-opacity: .35; }
  }
  .node-label {
    fill: var(--text-0); font-size: 11px; font-weight: 600;
    text-anchor: middle; pointer-events: none; paint-order: stroke;
    stroke: var(--bg-0); stroke-width: 3px;
  }
  .node-sub {
    fill: var(--text-2); font-size: 9px; text-anchor: middle;
    pointer-events: none; paint-order: stroke; stroke: var(--bg-0); stroke-width: 3px;
  }
  .badge-bg { fill: var(--bg-1); stroke: var(--border); stroke-width: 1px; }
  .badge-text { fill: var(--text-0); font-size: 9px; font-weight: 700; text-anchor: middle; }

  aside.panel {
    position: absolute; top: 16px; right: 16px; width: 300px; max-height: calc(100% - 32px);
    background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 16px; backdrop-filter: blur(10px); overflow-y: auto;
    box-shadow: 0 20px 50px rgba(0,0,0,0.45);
  }
  aside.panel h2 { font-size: 13px; margin: 0 0 10px; color: var(--text-0); }
  aside.panel .empty { color: var(--text-2); font-size: 12.5px; line-height: 1.5; }
  .role-pill {
    display: inline-block; font-size: 10.5px; font-weight: 700; padding: 3px 9px;
    border-radius: 999px; margin-bottom: 10px; letter-spacing: .3px; text-transform: uppercase;
  }
  .kv { display: grid; grid-template-columns: 1fr auto; gap: 4px 10px; font-size: 12px; margin: 3px 0; }
  .kv .k { color: var(--text-1); }
  .kv .v { color: var(--text-0); font-weight: 600; text-align: right; }
  .section-title {
    font-size: 10.5px; text-transform: uppercase; letter-spacing: .6px;
    color: var(--text-2); margin: 14px 0 6px;
  }
  .tag-list { display: flex; flex-wrap: wrap; gap: 6px; }
  .tag {
    font-size: 11px; background: rgba(148,163,184,0.1); border: 1px solid var(--border);
    padding: 3px 8px; border-radius: 8px; color: var(--text-0); cursor: pointer;
  }
  .tag:hover { border-color: var(--accent); color: var(--accent); }

  .toolbar {
    position: absolute; top: 16px; left: 16px; display: flex; flex-direction: column; gap: 10px;
    max-height: calc(100% - 32px);
  }
  .search-box {
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 8px 12px; backdrop-filter: blur(10px); width: 230px;
  }
  .search-box input {
    width: 100%; background: transparent; border: none; outline: none;
    color: var(--text-0); font-size: 12.5px;
  }
  .search-box input::placeholder { color: var(--text-2); }
  .btn-row { display: flex; gap: 8px; }
  .btn {
    background: var(--panel); border: 1px solid var(--border); color: var(--text-1);
    font-size: 11.5px; padding: 7px 10px; border-radius: 8px; cursor: pointer;
    backdrop-filter: blur(10px);
  }
  .btn:hover { color: var(--text-0); border-color: var(--accent); }

  .subsystem-nav {
    background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 10px; backdrop-filter: blur(10px); width: 230px; overflow-y: auto; flex: 1;
  }
  .subsystem-nav .title {
    font-size: 10.5px; text-transform: uppercase; letter-spacing: .6px; color: var(--text-2);
    padding: 2px 4px 8px;
  }
  .subsystem-item {
    display: flex; align-items: center; gap: 8px; padding: 6px 6px; border-radius: 8px;
    cursor: pointer; font-size: 12px;
  }
  .subsystem-item:hover { background: rgba(148,163,184,0.08); }
  .subsystem-item .swatch { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
  .subsystem-item .name { color: var(--text-0); flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .subsystem-item .count { color: var(--text-2); font-size: 10.5px; }

  .legend {
    position: absolute; left: 16px; bottom: 16px; background: var(--panel);
    border: 1px solid var(--border); border-radius: var(--radius); padding: 12px 14px;
    backdrop-filter: blur(10px); font-size: 11px; color: var(--text-1);
  }
  .legend .row { display: flex; align-items: center; gap: 8px; margin: 5px 0; }
  .legend .swatch { width: 10px; height: 10px; border-radius: 50%; }
  .legend .line { width: 22px; height: 0; border-top: 3px solid var(--edge); }
  .legend .line.dashed { border-top: 2px dashed var(--edge-inter); }

  .cycle-banner {
    position: absolute; top: 16px; left: 50%; transform: translateX(-50%);
    background: rgba(248,113,113,0.12); border: 1px solid rgba(248,113,113,0.4);
    color: #fca5a5; font-size: 12px; padding: 8px 14px; border-radius: 10px;
    backdrop-filter: blur(10px);
  }
</style>
</head>
<body>
<div id="app">
  <header>
    <h1><span class="dot"></span>__TITLE__</h1>
    <div class="stats" id="stats"></div>
  </header>
  <main>
    <svg id="svg">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" class="edge-arrow"></path>
        </marker>
        <marker id="arrow-inter" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" class="edge-arrow inter"></path>
        </marker>
        <marker id="arrow-hi" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" class="edge-arrow hi"></path>
        </marker>
      </defs>
      <g id="viewport">
        <g id="cluster-layer"></g>
        <g id="edge-layer"></g>
        <g id="node-layer"></g>
      </g>
    </svg>

    <div class="toolbar">
      <div class="search-box"><input id="search" placeholder="Search a service…" autocomplete="off" /></div>
      <div class="btn-row">
        <button class="btn" id="btn-settle">↻ Resettle</button>
        <button class="btn" id="btn-reset">⤢ Fit view</button>
      </div>
      <div class="subsystem-nav" id="subsystem-nav"></div>
    </div>

    <aside class="panel" id="detail">
      <h2>Service detail</h2>
      <div class="empty">Click a node to inspect its configuration, or hover to trace its dependency chain. Click a subsystem on the left to focus it.</div>
    </aside>

    <div class="legend">
      <div class="row"><span class="swatch" style="background:var(--entry)"></span>Entry point (no callers)</div>
      <div class="row"><span class="swatch" style="background:var(--store)"></span>Data store (no dependencies)</div>
      <div class="row"><span class="swatch" style="background:var(--internal)"></span>Internal service</div>
      <div class="row"><span class="line"></span>Within-subsystem call</div>
      <div class="row"><span class="line dashed"></span>Cross-subsystem call</div>
      <div class="row">Node size ∝ &radic;instances</div>
    </div>
  </main>
</div>

<script>
const DATA = __GRAPH_DATA__;
const NS = "http://www.w3.org/2000/svg";
const ROLE_COLOR = { entry_point: "var(--entry)", data_store: "var(--store)", internal: "var(--internal)", isolated: "var(--isolated)" };
const ROLE_LABEL = { entry_point: "Entry point", data_store: "Data store", internal: "Internal service", isolated: "Isolated" };

//  stats
const statsEl = document.getElementById("stats");
const m = DATA.meta;
let statsHtml = `
  <span class="chip"><b>${m.service_count}</b>&nbsp;services</span>
  <span class="chip"><b>${m.subsystem_count}</b>&nbsp;subsystems</span>
  <span class="chip"><b>${m.edge_count}</b>&nbsp;dependencies</span>
  <span class="chip"><b>${m.total_instances}</b>&nbsp;total instances</span>
`;
if (m.simulation_seed !== null) statsHtml += `<span class="chip">seed <b>${m.simulation_seed}</b></span>`;
if (m.cycles && m.cycles.length) statsHtml += `<span class="chip warn">⚠ ${m.cycles.length} cycle(s) allowed</span>`;
statsEl.innerHTML = statsHtml;

if (m.cycles && m.cycles.length) {
  const banner = document.createElement("div");
  banner.className = "cycle-banner";
  banner.textContent = `⚠ Graph contains ${m.cycles.length} cycle(s) — allow_cycles is set, but double-check this is intentional.`;
  document.querySelector("main").appendChild(banner);
}

//  state
const nodes = DATA.nodes.map(n => ({...n, vx: 0, vy: 0}));
const edges = DATA.edges;
const clusters = DATA.clusters;
const nodeById = Object.fromEntries(nodes.map(n => [n.id, n]));
const clusterById = Object.fromEntries(clusters.map(c => [c.id, c]));
const neighborMap = {};
nodes.forEach(n => neighborMap[n.id] = new Set());
edges.forEach(e => { neighborMap[e.source].add(e.target); neighborMap[e.target].add(e.source); });

let selected = null;
let hovered = null;
let focusedCluster = null;

//  render
const svg = document.getElementById("svg");
const viewport = document.getElementById("viewport");
const clusterLayer = document.getElementById("cluster-layer");
const edgeLayer = document.getElementById("edge-layer");
const nodeLayer = document.getElementById("node-layer");

function edgePath(e) {
  const s = nodeById[e.source], t = nodeById[e.target];
  const dx = t.x - s.x, dy = t.y - s.y;
  const dist = Math.max(Math.hypot(dx, dy), 0.001);
  const ux = dx / dist, uy = dy / dist;
  const x1 = s.x + ux * (s.radius + 2), y1 = s.y + uy * (s.radius + 2);
  const x2 = t.x - ux * (t.radius + 8), y2 = t.y - uy * (t.radius + 8);
  const bow = e.intra ? 18 : 34;
  const mx = (x1 + x2) / 2 + (uy) * bow, my = (y1 + y2) / 2 - (ux) * bow;
  return `M${x1},${y1} Q${mx},${my} ${x2},${y2}`;
}

const clusterEls = clusters.map(c => {
  const g = document.createElementNS(NS, "g");
  g.classList.add("cluster-blob");
  g.dataset.id = c.id;

  const circle = document.createElementNS(NS, "circle");
  circle.setAttribute("cx", c.cx);
  circle.setAttribute("cy", c.cy);
  circle.setAttribute("r", c.radius);
  circle.setAttribute("fill", c.color);
  circle.setAttribute("fill-opacity", "0.06");
  circle.setAttribute("stroke", c.color);
  circle.setAttribute("stroke-opacity", "0.35");
  circle.setAttribute("stroke-width", "1.5");
  circle.setAttribute("stroke-dasharray", "3 5");
  g.appendChild(circle);

  const label = document.createElementNS(NS, "text");
  label.classList.add("cluster-label");
  label.setAttribute("x", c.cx);
  label.setAttribute("y", c.cy - c.radius + 20);
  label.setAttribute("fill", c.color);
  label.textContent = c.label;
  g.appendChild(label);

  const count = document.createElementNS(NS, "text");
  count.classList.add("cluster-count");
  count.setAttribute("x", c.cx);
  count.setAttribute("y", c.cy - c.radius + 34);
  count.textContent = `${c.size} service${c.size === 1 ? "" : "s"}`;
  g.appendChild(count);

  clusterLayer.appendChild(g);
  return g;
});

const edgeEls = edges.map(e => {
  const path = document.createElementNS(NS, "path");
  path.classList.add("edge");
  if (!e.intra) path.classList.add("inter");
  path.setAttribute("stroke-width", e.width);
  path.setAttribute("marker-end", e.intra ? "url(#arrow)" : "url(#arrow-inter)");
  path.setAttribute("d", edgePath(e));
  edgeLayer.appendChild(path);
  return path;
});

const nodeEls = nodes.map(n => {
  const g = document.createElementNS(NS, "g");
  g.classList.add("node-group");
  g.dataset.id = n.id;
  g.dataset.cluster = n.cluster;

  const circle = document.createElementNS(NS, "circle");
  circle.classList.add("node-circle");
  circle.setAttribute("r", n.radius);
  circle.setAttribute("fill", ROLE_COLOR[n.role]);
  circle.setAttribute("fill-opacity", "0.28");
  circle.setAttribute("stroke", ROLE_COLOR[n.role]);
  circle.style.color = ROLE_COLOR[n.role];
  g.appendChild(circle);

  const label = document.createElementNS(NS, "text");
  label.classList.add("node-label");
  label.setAttribute("y", n.radius + 14);
  label.textContent = n.id;
  g.appendChild(label);

  const sub = document.createElementNS(NS, "text");
  sub.classList.add("node-sub");
  sub.setAttribute("y", n.radius + 25);
  sub.textContent = `${n.log_volume_per_sec} log/s`;
  g.appendChild(sub);

  const badgeBg = document.createElementNS(NS, "circle");
  badgeBg.classList.add("badge-bg");
  badgeBg.setAttribute("r", 8);
  badgeBg.setAttribute("cx", n.radius * 0.72);
  badgeBg.setAttribute("cy", -n.radius * 0.72);
  g.appendChild(badgeBg);

  const badgeText = document.createElementNS(NS, "text");
  badgeText.classList.add("badge-text");
  badgeText.setAttribute("x", n.radius * 0.72);
  badgeText.setAttribute("y", -n.radius * 0.72 + 3);
  badgeText.textContent = n.instances;
  g.appendChild(badgeText);

  nodeLayer.appendChild(g);
  return g;
});

function layout() {
  nodes.forEach((n, i) => {
    nodeEls[i].setAttribute("transform", `translate(${n.x},${n.y})`);
  });
  edges.forEach((e, i) => edgeEls[i].setAttribute("d", edgePath(e)));
}
layout();

//  subsystem nav
const navEl = document.getElementById("subsystem-nav");
navEl.innerHTML = `<div class="title">Subsystems</div>` + clusters.map(c => `
  <div class="subsystem-item" data-cluster="${c.id}">
    <span class="swatch" style="background:${c.color}"></span>
    <span class="name">${c.label}</span>
    <span class="count">${c.size}</span>
  </div>
`).join("");
navEl.querySelectorAll(".subsystem-item").forEach(item => {
  item.addEventListener("click", () => focusCluster(item.dataset.cluster));
});

//  focus / highlight
function clearFocus() {
  nodeEls.forEach(el => el.classList.remove("hi", "dim", "selected"));
  edgeEls.forEach(el => {
    el.classList.remove("hi", "dim", "flow");
    const e = edges[edgeEls.indexOf(el)];
  });
  edges.forEach((e, i) => {
    edgeEls[i].setAttribute("marker-end", e.intra ? "url(#arrow)" : "url(#arrow-inter)");
  });
  clusterEls.forEach(el => el.classList.remove("dim"));
}

function applyFocus(id) {
  clearFocus();
  const focusSet = new Set([id, ...neighborMap[id]]);
  nodeEls.forEach(el => {
    if (focusSet.has(el.dataset.id)) el.classList.add("hi");
    else el.classList.add("dim");
    if (el.dataset.id === selected) el.classList.add("selected");
  });
  edgeEls.forEach((el, i) => {
    const e = edges[i];
    if (e.source === id || e.target === id) {
      el.classList.add("hi", "flow");
      el.setAttribute("marker-end", "url(#arrow-hi)");
    } else {
      el.classList.add("dim");
    }
  });
}

function fmtLatency(range) { return `${range[0]}–${range[1]} ms`; }
function fmtPct(x) { return (x * 100).toFixed(2) + "%"; }

function showDetail(n) {
  const panel = document.getElementById("detail");
  const cluster = clusterById[n.cluster];
  const deps = n.dependencies.length
    ? n.dependencies.map(d => `<span class="tag" data-goto="${d}">${d}</span>`).join("")
    : `<span class="empty">none</span>`;
  const dependents = n.dependents.length
    ? n.dependents.map(d => `<span class="tag" data-goto="${d}">${d}</span>`).join("")
    : `<span class="empty">none</span>`;
  const peak = n.peak_hours ? `${n.peak_hours[0]}:00 – ${n.peak_hours[1]}:00 (3× traffic)` : "none configured";
  const overrides = Object.keys(n.special_call_rate || {}).length
    ? Object.entries(n.special_call_rate).map(([k, v]) => `<div class="kv"><span class="k">→ ${k}</span><span class="v">${fmtPct(v)}</span></div>`).join("")
    : `<div class="empty">none</div>`;

  panel.innerHTML = `
    <h2>${n.id}</h2>
    <span class="role-pill" style="color:${ROLE_COLOR[n.role]};border:1px solid ${ROLE_COLOR[n.role]};background:${ROLE_COLOR[n.role]}22">${ROLE_LABEL[n.role]}</span>
    <div class="kv"><span class="k">Subsystem</span><span class="v" style="color:${cluster ? cluster.color : 'inherit'}">${cluster ? cluster.label : '—'}</span></div>
    <div class="kv"><span class="k">Instances</span><span class="v">${n.instances}</span></div>
    <div class="kv"><span class="k">Log volume</span><span class="v">${n.log_volume_per_sec}/s</span></div>
    <div class="kv"><span class="k">Normal latency</span><span class="v">${fmtLatency(n.normal_latency_ms)}</span></div>
    <div class="kv"><span class="k">Normal error rate</span><span class="v">${fmtPct(n.normal_error_rate)}</span></div>
    <div class="kv"><span class="k">CPU cores</span><span class="v">${n.resource_limits.cpu_cores}</span></div>
    <div class="kv"><span class="k">Memory limit</span><span class="v">${n.resource_limits.memory_mb} MB</span></div>
    <div class="kv"><span class="k">Peak hours</span><span class="v">${peak}</span></div>

    <div class="section-title">Calls (dependencies)</div>
    <div class="tag-list">${deps}</div>

    <div class="section-title">Called by (dependents)</div>
    <div class="tag-list">${dependents}</div>

    <div class="section-title">Special call rate overrides</div>
    ${overrides}
  `;
  panel.querySelectorAll("[data-goto]").forEach(t => {
    t.addEventListener("click", () => selectNode(t.dataset.goto));
  });
}

function selectNode(id) {
  selected = id;
  focusedCluster = null;
  applyFocus(id);
  showDetail(nodeById[id]);
}

nodeEls.forEach(g => {
  g.addEventListener("mouseenter", () => { hovered = g.dataset.id; if (!selected) applyFocus(hovered); });
  g.addEventListener("mouseleave", () => { hovered = null; if (!selected) clearFocus(); });
  g.addEventListener("click", (ev) => { ev.stopPropagation(); selectNode(g.dataset.id); });
});
svg.addEventListener("click", () => {
  selected = null;
  focusedCluster = null;
  clearFocus();
  document.getElementById("detail").innerHTML = `<h2>Service detail</h2><div class="empty">Click a node to inspect its configuration, or hover to trace its dependency chain. Click a subsystem on the left to focus it.</div>`;
});

//  search
document.getElementById("search").addEventListener("input", (ev) => {
  const q = ev.target.value.trim().toLowerCase();
  if (!q) { if (!selected) clearFocus(); return; }
  const matches = nodes.filter(n => n.id.toLowerCase().includes(q)).map(n => n.id);
  nodeEls.forEach(el => {
    el.classList.toggle("dim", !matches.includes(el.dataset.id));
    el.classList.toggle("hi", matches.includes(el.dataset.id));
  });
  edgeEls.forEach(el => el.classList.add("dim"));
  clusterEls.forEach(el => el.classList.add("dim"));
});

//  subsystem focus + camera fit
function bboxFit(items, marginFactor = 1.12) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  items.forEach(([x, y, r]) => {
    minX = Math.min(minX, x - r); maxX = Math.max(maxX, x + r);
    minY = Math.min(minY, y - r); maxY = Math.max(maxY, y + r);
  });
  const rect = svg.getBoundingClientRect();
  const contentW = (maxX - minX) * marginFactor, contentH = (maxY - minY) * marginFactor;
  const scale = Math.max(0.15, Math.min(2.2, Math.min(rect.width / contentW, rect.height / contentH)));
  viewScale = scale;
  viewX = rect.width / 2 - ((minX + maxX) / 2) * scale;
  viewY = rect.height / 2 - ((minY + maxY) / 2) * scale;
  applyView();
}

function fitToView() {
  const items = [
    ...nodes.map(n => [n.x, n.y, n.radius + 30]),
    ...clusters.map(c => [c.cx, c.cy, c.radius]),
  ];
  if (items.length) bboxFit(items, 1.06);
}

function focusCluster(clusterId) {
  selected = null;
  focusedCluster = clusterId;
  const members = nodes.filter(n => n.cluster === clusterId).map(n => n.id);
  const memberSet = new Set(members);

  clearFocus();
  nodeEls.forEach(el => el.classList.toggle("dim", !memberSet.has(el.dataset.id)));
  nodeEls.forEach(el => el.classList.toggle("hi", memberSet.has(el.dataset.id)));
  edgeEls.forEach((el, i) => {
    const e = edges[i];
    el.classList.toggle("dim", !(memberSet.has(e.source) && memberSet.has(e.target)));
  });
  clusterEls.forEach(el => el.classList.toggle("dim", el.dataset.id !== clusterId));

  const c = clusterById[clusterId];
  bboxFit([[c.cx, c.cy, c.radius]], 1.25);

  const panel = document.getElementById("detail");
  panel.innerHTML = `
    <h2>${c.label}</h2>
    <span class="role-pill" style="color:${c.color};border:1px solid ${c.color};background:${c.color}22">Subsystem · ${c.size} service${c.size === 1 ? "" : "s"}</span>
    <div class="section-title">Members</div>
    <div class="tag-list">${members.map(id => `<span class="tag" data-goto="${id}">${id}</span>`).join("")}</div>
  `;
  panel.querySelectorAll("[data-goto]").forEach(t => {
    t.addEventListener("click", (ev) => { ev.stopPropagation(); selectNode(t.dataset.goto); });
  });
}

//  force settle
function settle(iterations = 220, fitAfter = false) {
  let alpha = 1.0;
  let tick = 0;
  function step() {
    alpha *= 0.985;
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        let dx = a.x - b.x, dy = a.y - b.y;
        let dist2 = dx * dx + dy * dy || 0.01;
        let dist = Math.sqrt(dist2);
        const force = (2200 / dist2) * alpha;
        dx /= dist; dy /= dist;
        a.vx += dx * force; a.vy += dy * force;
        b.vx -= dx * force; b.vy -= dy * force;
      }
    }
    edges.forEach(e => {
      
      if (!e.intra) return;
      const a = nodeById[e.source], b = nodeById[e.target];
      let dx = b.x - a.x, dy = b.y - a.y;
      const dist = Math.hypot(dx, dy) || 0.01;
      const force = (dist - 170) * 0.024 * alpha;
      dx /= dist; dy /= dist;
      a.vx += dx * force; a.vy += dy * force;
      b.vx -= dx * force; b.vy -= dy * force;
    });
    
    nodes.forEach(n => {
      const c = clusterById[n.cluster];
      n.vx += (c.cx - n.x) * 0.006 * alpha;
      n.vy += (c.cy - n.y) * 0.006 * alpha;
      n.x += n.vx; n.y += n.vy;
      n.vx *= 0.75; n.vy *= 0.75;

     
      const dx = n.x - c.cx, dy = n.y - c.cy;
      const dist = Math.hypot(dx, dy);
      const maxDist = c.radius - n.radius - 16;
      if (dist > maxDist && dist > 0.01) {
        const k = maxDist / dist;
        n.x = c.cx + dx * k;
        n.y = c.cy + dy * k;
        n.vx *= 0.2; n.vy *= 0.2;
      }
    });
    layout();
    tick++;
    if (tick < iterations && alpha > 0.02) {
      requestAnimationFrame(step);
    } else if (fitAfter) {
      fitToView();
    }
  }
  requestAnimationFrame(step);
}

document.getElementById("btn-settle").addEventListener("click", () => settle(200, false));
document.getElementById("btn-reset").addEventListener("click", () => { focusedCluster = null; clearFocus(); fitToView(); });

//  drag
let dragNode = null;
nodeEls.forEach((g, i) => {
  g.addEventListener("mousedown", (ev) => {
    ev.stopPropagation();
    dragNode = nodes[i];
  });
});
svg.addEventListener("mousemove", (ev) => {
  if (!dragNode) return;
  const pt = toViewportPoint(ev);
  dragNode.x = pt.x; dragNode.y = pt.y;
  layout();
});
window.addEventListener("mouseup", () => { dragNode = null; });

//  pan & zoom
let viewX = 0, viewY = 0, viewScale = 1;
let panning = false, panStart = null;

function applyView() {
  viewport.setAttribute("transform", `translate(${viewX},${viewY}) scale(${viewScale})`);
}
function toViewportPoint(ev) {
  const rect = svg.getBoundingClientRect();
  return { x: (ev.clientX - rect.left - viewX) / viewScale, y: (ev.clientY - rect.top - viewY) / viewScale };
}
svg.addEventListener("mousedown", (ev) => {
  if (dragNode) return;
  panning = true;
  panStart = { x: ev.clientX - viewX, y: ev.clientY - viewY };
  svg.classList.add("panning");
});
svg.addEventListener("mousemove", (ev) => {
  if (!panning || dragNode) return;
  viewX = ev.clientX - panStart.x;
  viewY = ev.clientY - panStart.y;
  applyView();
});
window.addEventListener("mouseup", () => { panning = false; svg.classList.remove("panning"); });
svg.addEventListener("wheel", (ev) => {
  ev.preventDefault();
  const rect = svg.getBoundingClientRect();
  const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
  const prevScale = viewScale;
  viewScale = Math.max(0.12, Math.min(2.6, viewScale * (ev.deltaY > 0 ? 0.9 : 1.1)));
  viewX = mx - (mx - viewX) * (viewScale / prevScale);
  viewY = my - (my - viewY) * (viewScale / prevScale);
  applyView();
}, { passive: false });

//  boot
fitToView();
settle(220, true);
</script>
</body>
</html>
"""
