"""Test the 'islands and necks' hypothesis on the 5x5 fiber graph.

Runs:
  1. Bridges + articulation points (literal necks).
  2. Edge betweenness centrality (necks carry many shortest paths).
  3. Louvain community detection + modularity (do islands exist?).
  4. k-core decomposition (cores = islands; shells = necks).
  5. Conductance of the discovered communities.
  6. A community-colored layout, with bridges highlighted.
"""

import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    here = Path(__file__).parent
    F = nx.read_edgelist(here / "fiber_5x5_edges.txt", nodetype=int)
    F.add_nodes_from(range(4006))
    print(f"Loaded fiber: {F.number_of_nodes()} nodes, {F.number_of_edges()} edges")

    section("1. Bridges and articulation points")
    bridges = list(nx.bridges(F))
    aps = list(nx.articulation_points(F))
    print(f"Bridges (edges whose removal disconnects F): {len(bridges)}")
    print(f"Articulation points (cut vertices): {len(aps)}")
    if bridges:
        print(f"  example bridges: {bridges[:10]}")
    if aps:
        print(f"  example articulation points: {aps[:10]}")

    section("2. Edge betweenness centrality (top 15)")
    t0 = time.time()
    eb = nx.edge_betweenness_centrality(F, normalized=True)
    print(f"  computed in {time.time() - t0:.1f}s")
    top_eb = sorted(eb.items(), key=lambda x: x[1], reverse=True)[:15]
    print("  rank  edge                   betweenness")
    for r, (e, v) in enumerate(top_eb, 1):
        print(f"  {r:>3}   {str(e):<22} {v:.4f}")
    print(f"  median edge betweenness: "
          f"{sorted(eb.values())[len(eb) // 2]:.5f}")
    print(f"  ratio top/median: "
          f"{top_eb[0][1] / sorted(eb.values())[len(eb) // 2]:.0f}x")

    section("3. Communities (greedy modularity)")
    t0 = time.time()
    communities = list(nx.community.greedy_modularity_communities(F))
    Q = nx.community.modularity(F, communities)
    print(f"  found {len(communities)} communities in {time.time() - t0:.1f}s")
    print(f"  modularity Q = {Q:.4f}  (>0.3 = clear structure, "
          f">0.5 = strong)")
    sizes = sorted([len(c) for c in communities], reverse=True)
    print(f"  community sizes (top 15): {sizes[:15]}")
    print(f"  community sizes (smallest 5): {sizes[-5:]}")

    section("4. k-core decomposition")
    core = nx.core_number(F)
    core_dist = Counter(core.values())
    print("  core    nodes")
    for k in sorted(core_dist):
        print(f"  {k:>4}    {core_dist[k]:>5}")
    print(f"  max core (densest island level): {max(core_dist)}")
    print(f"  fraction of nodes in lowest 2 cores: "
          f"{sum(core_dist[k] for k in sorted(core_dist)[:2]) / 4006:.1%}")

    section("5. Conductance of largest community vs rest")
    largest = max(communities, key=len)
    rest = set(F.nodes()) - set(largest)
    cond = nx.conductance(F, largest, rest)
    print(f"  size of largest community: {len(largest)}")
    print(f"  conductance(largest vs rest) = {cond:.4f}  "
          f"(low = strong neck/bottleneck)")

    print("\n--- per-community conductance (top 10 by size) ---")
    print("  size    conductance   internal-edge-frac")
    for c in sorted(communities, key=len, reverse=True)[:10]:
        nodes = set(c)
        if len(nodes) == F.number_of_nodes():
            continue
        cd = nx.conductance(F, nodes, set(F.nodes()) - nodes)
        internal = F.subgraph(nodes).number_of_edges()
        outgoing = sum(1 for u, v in F.edges()
                       if (u in nodes) != (v in nodes))
        frac_internal = internal / (internal + outgoing) if internal + outgoing else 0
        print(f"  {len(nodes):>4}    {cd:>6.4f}        {frac_internal:>6.4f}")

    section("6. Visualization with community colors + bridges")
    print("Computing layout...")
    t0 = time.time()
    pos = nx.spring_layout(F, seed=42, iterations=80,
                           k=2.5 / (F.number_of_nodes() ** 0.5))
    print(f"  layout in {time.time() - t0:.1f}s")

    node_to_comm = {}
    for ci, c in enumerate(communities):
        for n in c:
            node_to_comm[n] = ci
    cmap = plt.cm.tab20
    colors = [cmap(node_to_comm[n] % 20) for n in F.nodes()]

    fig, ax = plt.subplots(figsize=(20, 20))
    nx.draw_networkx_edges(F, pos=pos, edge_color="lightgray",
                           width=0.15, alpha=0.4, ax=ax)
    if bridges:
        nx.draw_networkx_edges(F, pos=pos, edgelist=bridges,
                               edge_color="red", width=1.0, alpha=0.9, ax=ax)
    nx.draw_networkx_nodes(F, pos=pos, node_size=10,
                           node_color=colors, linewidths=0, ax=ax)
    ax.set_title(
        f"Fiber communities ({len(communities)}, Q={Q:.3f}). "
        f"Red edges = bridges ({len(bridges)}).",
        fontsize=14,
    )
    ax.set_axis_off()
    plt.tight_layout()
    out = here / "fiber_5x5_communities.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  plot saved to {out}")


if __name__ == "__main__":
    main()
