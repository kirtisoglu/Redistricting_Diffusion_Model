"""Corridor analysis on the ε=0.2 fiber (193,128 partitions, 970k edges).

Goals:
  1. Identify communities (Louvain — Q=0.64 expected from prior greedy-mod run).
  2. Quantify "corridors" — narrow inter-community throats:
        a) inter-community edge counts per (a, b) pair,
        b) high-betweenness inter-community edges (corridor edges),
        c) 2-cut candidates: very few inter-community edges between big pairs.
  3. Two-level community-aware layout for visualisation.
  4. Save: layout, corridor edges highlighted, communities coloured.

This complements the ε=0 fiber bridge analysis (`fiber_islands_necks.py`):
  ε=0 fiber: 4006 partitions, 13,416 edges, 16 BRIDGES, modularity 0.65.
  ε=0.2 fiber: 193,128 partitions, 969,760 edges, 0 BRIDGES, modularity ~0.64.

The ε=0.2 fiber has no bridges (the bridge structure is "thickened" into
multi-vertex corridors).  This script characterises those corridors.
"""

import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


HERE = Path(__file__).parent

# CLI: corridor_analysis_eps02.py [fiber_prefix] [n_fiber]
# Default: fiber_5x5_eps02, n=193128
# 4x4 example: fiber_4x4_eps025, n=1953
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "fiber_5x5_eps02"
N_FIBER = int(sys.argv[2]) if len(sys.argv) > 2 else 193128
EDGES = HERE / f"{PREFIX}_edges.txt"


def load_fiber():
    print(f"Loading fiber edges from {EDGES} ...")
    t0 = time.time()
    F = nx.read_edgelist(str(EDGES), nodetype=int)
    F.add_nodes_from(range(N_FIBER))
    print(f"  {F.number_of_nodes()} nodes, {F.number_of_edges()} edges "
          f"in {time.time() - t0:.1f}s")
    return F


def detect_communities(F):
    print("Running Louvain ...")
    t0 = time.time()
    communities = list(
        nx.community.louvain_communities(F, resolution=1.0, seed=42))
    Q = nx.community.modularity(F, communities)
    print(f"  {len(communities)} communities, Q={Q:.4f} in "
          f"{time.time() - t0:.1f}s")
    sizes = sorted([len(c) for c in communities], reverse=True)
    print(f"  sizes (top 20): {sizes[:20]}")
    return communities, Q


def build_meta_graph(F, node_comm, communities):
    """Community-collapsed graph: nodes = communities, edges = inter-comm
    edges weighted by raw count."""
    meta = nx.Graph()
    for ci in range(len(communities)):
        meta.add_node(ci, size=len(communities[ci]))
    counts = Counter()
    for u, v in F.edges():
        cu, cv = node_comm[u], node_comm[v]
        if cu != cv:
            key = (min(cu, cv), max(cu, cv))
            counts[key] += 1
    for (a, b), w in counts.items():
        meta.add_edge(a, b, weight=w)
    return meta, counts


def corridor_throat_table(meta_counts, communities):
    """For each pair of communities, report inter-community edge count.
    Sort ascending — small counts = narrow throat = corridor candidate."""
    rows = []
    for (a, b), w in meta_counts.items():
        sa, sb = len(communities[a]), len(communities[b])
        max_possible = sa * sb  # if completely connected (uninteresting; just for normalising)
        rel = w / min(sa, sb)   # edges per node in smaller community
        rows.append((a, b, w, sa, sb, rel))
    rows.sort(key=lambda r: r[2])  # narrowest throat first
    return rows


def edge_betweenness_sampled(F, node_comm, k=100, seed=42):
    """Sampled edge betweenness via NetworkX's Brandes algorithm with
    k random source-pivots.  O(k·|E|) instead of full O(|V|·|E|)."""
    print(f"Computing edge betweenness (sampled, k={k}) ...")
    t0 = time.time()
    eb = nx.edge_betweenness_centrality(F, k=k, seed=seed)
    eb = {(min(u, v), max(u, v)): w for (u, v), w in eb.items()}
    print(f"  done in {time.time() - t0:.1f}s")
    return eb


def corridor_edges(eb, node_comm, top_frac=0.001):
    """Top-betweenness edges that cross community boundaries (corridor
    structure).  Returns sorted list of (edge, score, is_inter)."""
    items = sorted(eb.items(), key=lambda kv: -kv[1])
    n_top = max(1, int(len(items) * top_frac))
    top = items[:n_top]
    inter_top = [(e, s) for e, s in top if node_comm[e[0]] != node_comm[e[1]]]
    return top, inter_top


def two_level_layout(F, communities, meta, node_comm, seed=42):
    """Place each community's centroid via spring on meta-graph, then
    place its nodes around the centroid via local spring."""
    print("Computing two-level layout ...")
    t0 = time.time()
    # 1. Meta-graph layout
    meta_pos = nx.spring_layout(meta, seed=seed, weight="weight",
                                 k=3.0, iterations=200)
    # Scale meta-graph
    coords = np.array(list(meta_pos.values()))
    spread = max(coords.ptp(axis=0).max(), 1.0)
    meta_pos = {ci: (p[0] / spread * 10, p[1] / spread * 10)
                for ci, p in meta_pos.items()}

    # 2. Within each community: spring layout, scaled by community size
    pos = {}
    for ci, c in enumerate(communities):
        sub = F.subgraph(c)
        radius = 0.5 + 0.5 * np.log10(max(len(c), 2))
        try:
            sub_pos = nx.spring_layout(
                sub, seed=seed + ci,
                k=2.0 / np.sqrt(len(c)),
                iterations=30,
                scale=radius)
        except Exception:
            sub_pos = {n: (0, 0) for n in c}
        cx, cy = meta_pos[ci]
        for n, (x, y) in sub_pos.items():
            pos[n] = (x + cx, y + cy)
    print(f"  layout in {time.time()-t0:.1f}s")
    return pos, meta_pos


def visualise(F, communities, node_comm, pos, meta_pos, eb_top,
              out_path):
    print(f"Rendering plot to {out_path} ...")
    t0 = time.time()
    fig, ax = plt.subplots(figsize=(22, 22))

    intra = [(u, v) for u, v in F.edges() if node_comm[u] == node_comm[v]]
    inter = [(u, v) for u, v in F.edges() if node_comm[u] != node_comm[v]]
    print(f"  intra={len(intra)}, inter={len(inter)}")

    # Light intra-community edges
    nx.draw_networkx_edges(F, pos=pos, edgelist=intra,
                           edge_color="lightgray",
                           width=0.05, alpha=0.20, ax=ax)
    # Mid-tone inter-community edges (the corridor backbone)
    nx.draw_networkx_edges(F, pos=pos, edgelist=inter,
                           edge_color="steelblue",
                           width=0.25, alpha=0.55, ax=ax)
    # Top-betweenness edges (corridor highlights)
    if eb_top:
        keys = [e for e, _ in eb_top]
        nx.draw_networkx_edges(F, pos=pos, edgelist=keys,
                               edge_color="crimson",
                               width=1.0, alpha=0.9, ax=ax)
    # Nodes coloured by community
    cmap = plt.cm.tab20
    colors = [cmap(node_comm[n] % 20) for n in F.nodes()]
    nx.draw_networkx_nodes(F, pos=pos, node_color=colors,
                           node_size=1.5, linewidths=0, ax=ax)

    # Community labels at centroids
    for ci, (cx, cy) in meta_pos.items():
        ax.text(cx, cy, f"C{ci}\n|n|={len(communities[ci])}",
                fontsize=8, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", alpha=0.85,
                          edgecolor="black"))

    ax.set_title(
        f"ε=0.2 fiber: {F.number_of_nodes()} partitions, "
        f"{F.number_of_edges()} edges, {len(communities)} communities. "
        f"Crimson edges = top {len(eb_top)} by sampled betweenness "
        f"(corridor highlights).",
        fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  rendered in {time.time()-t0:.1f}s")


def main():
    F = load_fiber()
    communities, Q = detect_communities(F)
    node_comm = {n: ci for ci, c in enumerate(communities) for n in c}
    meta, meta_counts = build_meta_graph(F, node_comm, communities)

    # ── 1. Corridor throat table ──────────────────────────────────────────────
    rows = corridor_throat_table(meta_counts, communities)
    print("\nNarrowest inter-community throats (top 12 — corridor candidates):")
    print(f"  {'pair':<10} {'edges':>6}  {'|A|':>6}  {'|B|':>6}  "
          f"{'edges/min(|A|,|B|)':>20}")
    for a, b, w, sa, sb, rel in rows[:12]:
        print(f"  ({a:>2},{b:>2})    {w:>6}  {sa:>6}  {sb:>6}  {rel:>20.4f}")

    # ── 2. Sampled edge betweenness (corridor highlight) ──────────────────────
    # Use small k for compute reasons (193k nodes is very large)
    eb = edge_betweenness_sampled(F, node_comm, k=100)
    eb_top, eb_inter_top = corridor_edges(eb, node_comm, top_frac=0.001)
    print(f"\nTop 0.1% by sampled edge-betweenness: {len(eb_top)} edges, "
          f"{len(eb_inter_top)} are inter-community (corridor edges).")

    # ── 3. Layout + visualise ─────────────────────────────────────────────────
    pos, meta_pos = two_level_layout(F, communities, meta, node_comm)
    out_path = HERE / f"{PREFIX}_corridors.png"
    visualise(F, communities, node_comm, pos, meta_pos, eb_top, out_path)

    # ── 4. Save community labels for downstream chain step-trace test ─────────
    comms_path = HERE / f"{PREFIX}_communities.npz"
    np.savez(comms_path,
             node_comm=np.array([node_comm[i] for i in range(N_FIBER)]),
             n_communities=len(communities),
             modularity=Q)
    print(f"\nCommunity labels saved to {comms_path}")
    print(f"\nOutputs:")
    print(f"  Plot: {out_path}")
    print(f"  Community labels: {comms_path.name}")


if __name__ == "__main__":
    main()
