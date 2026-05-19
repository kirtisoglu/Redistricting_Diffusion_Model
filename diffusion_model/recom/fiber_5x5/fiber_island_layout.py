"""Better fiber visualization + per-island radii.

Strategy:
  1. Reuse the 10 communities (islands) found by greedy modularity.
  2. For each island, compute the graph-theoretic radius and diameter
     of the induced subgraph (largest connected component if it is not
     connected).
  3. Layout the fiber with a two-level scheme:
       * place each island's centroid on a 'meta' spring layout of the
         island-level supergraph;
       * within each island, run an internal spring layout and translate
         it onto the centroid.
     This packs island nodes tightly together, with thin necks between.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def main():
    here = Path(__file__).parent
    F = nx.read_edgelist(here / "fiber_5x5_edges.txt", nodetype=int)
    F.add_nodes_from(range(4006))
    print(f"Loaded fiber: {F.number_of_nodes()} nodes, {F.number_of_edges()} edges")

    print("\nDetecting communities (greedy modularity)...")
    communities = list(nx.community.greedy_modularity_communities(F))
    Q = nx.community.modularity(F, communities)
    print(f"  {len(communities)} islands, modularity Q = {Q:.4f}")

    node_to_comm_pre = {}
    for ci, c in enumerate(communities):
        for n in c:
            node_to_comm_pre[n] = ci
    external_count = [0] * len(communities)
    for u, v in F.edges():
        cu, cv = node_to_comm_pre[u], node_to_comm_pre[v]
        if cu != cv:
            external_count[cu] += 1
            external_count[cv] += 1

    print("\n--- island radii, diameters, and edge counts ---")
    print(f"  {'id':>2}  {'size':>4}  {'connected':>9}  "
          f"{'radius':>6}  {'diameter':>8}  "
          f"{'internal':>8}  {'external':>8}  {'center(s)':<25}")
    island_info = []
    for ci, c in enumerate(communities):
        sub = F.subgraph(c).copy()
        if nx.is_connected(sub):
            connected = "yes"
            sub_main = sub
        else:
            comps = list(nx.connected_components(sub))
            sub_main = sub.subgraph(max(comps, key=len)).copy()
            connected = f"no ({len(comps)})"
        if sub_main.number_of_nodes() == 1:
            radius = diameter = 0
            centers = list(sub_main.nodes())
        else:
            ecc = nx.eccentricity(sub_main)
            radius = min(ecc.values())
            diameter = max(ecc.values())
            centers = [n for n, e in ecc.items() if e == radius]
        island_info.append({
            "id": ci, "nodes": list(c),
            "size": len(c), "connected": connected,
            "radius": radius, "diameter": diameter,
            "centers": centers,
            "internal_edges": sub.number_of_edges(),
            "external_edges": external_count[ci],
        })
        center_str = (",".join(map(str, centers[:3]))
                      + ("..." if len(centers) > 3 else ""))
        print(f"  {ci:>2}  {len(c):>4}  {connected:>9}  "
              f"{radius:>6}  {diameter:>8}  "
              f"{sub.number_of_edges():>8}  {external_count[ci]:>8}  "
              f"{center_str:<25}")

    print("\nBuilding two-level community layout...")
    t0 = time.time()
    rng = np.random.default_rng(42)

    node_to_comm = {}
    for ci, c in enumerate(communities):
        for n in c:
            node_to_comm[n] = ci

    super_g = nx.Graph()
    for ci, c in enumerate(communities):
        super_g.add_node(ci, weight=len(c))
    inter_edges = {}
    for u, v in F.edges():
        cu, cv = node_to_comm[u], node_to_comm[v]
        if cu != cv:
            key = (min(cu, cv), max(cu, cv))
            inter_edges[key] = inter_edges.get(key, 0) + 1
    for (a, b), w in inter_edges.items():
        super_g.add_edge(a, b, weight=w)
    super_pos = nx.spring_layout(super_g, seed=42, k=2.0, iterations=200,
                                 weight="weight")

    pos = {}
    for ci, c in enumerate(communities):
        sub = F.subgraph(c)
        if sub.number_of_nodes() == 1:
            n0 = next(iter(c))
            pos[n0] = super_pos[ci]
            continue
        sub_pos = nx.spring_layout(
            sub, seed=42 + ci,
            k=1.5 / (len(c) ** 0.5),
            iterations=100,
            scale=0.18 * (len(c) ** 0.5) / 30,
        )
        cx, cy = super_pos[ci]
        for n, (x, y) in sub_pos.items():
            pos[n] = (x + cx, y + cy)

    print(f"  layout computed in {time.time() - t0:.1f}s")

    print("\nDrawing...")
    cmap = plt.cm.tab20
    colors = [cmap(node_to_comm[n] % 20) for n in F.nodes()]

    intra_edges = [(u, v) for u, v in F.edges()
                   if node_to_comm[u] == node_to_comm[v]]
    crossing_edges = [(u, v) for u, v in F.edges()
                      if node_to_comm[u] != node_to_comm[v]]
    bridges = set(map(frozenset, nx.bridges(F)))
    bridge_edges = [(u, v) for u, v in F.edges() if frozenset((u, v)) in bridges]

    fig, ax = plt.subplots(figsize=(20, 20))
    nx.draw_networkx_edges(F, pos=pos, edgelist=intra_edges,
                           edge_color="lightgray", width=0.15, alpha=0.4, ax=ax)
    nx.draw_networkx_edges(F, pos=pos, edgelist=crossing_edges,
                           edge_color="dimgray", width=0.5, alpha=0.7, ax=ax)
    if bridge_edges:
        nx.draw_networkx_edges(F, pos=pos, edgelist=bridge_edges,
                               edge_color="red", width=1.4, alpha=0.95, ax=ax)
    nx.draw_networkx_nodes(F, pos=pos, node_size=12,
                           node_color=colors, linewidths=0, ax=ax)

    for ci, info in enumerate(island_info):
        cx, cy = super_pos[ci]
        ax.text(cx, cy, f"I{ci}\nn={info['size']}\nr={info['radius']}",
                fontsize=10, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor="black", alpha=0.85))

    ax.set_title(
        f"Fiber arranged by island. "
        f"{len(communities)} islands, Q={Q:.3f}. "
        f"Red = bridges, dark gray = inter-island, light = intra-island.",
        fontsize=14,
    )
    ax.set_axis_off()
    plt.tight_layout()
    out = here / "fiber_5x5_island_layout.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  plot saved to {out}")


if __name__ == "__main__":
    main()
