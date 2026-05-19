"""Build the fiber graph of the 5x5 grid under equal-population 5-partitions.

Nodes of the fiber graph: each connected 5-partition of the 5x5 grid where
every district has exactly 5 nodes. (4006 of them.) Each partition is
assigned an integer id 0..4005 in the order produced by enumeration.

Edges of the fiber graph: two partitions are adjacent iff one is obtained
from the other by a single population-preserving flip — i.e. a swap of
exactly two nodes u in district a and v in district b across the (a,b)
boundary, such that the two affected districts remain connected.
"""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def connected_subsets_containing(graph, seed, size):
    """All connected subsets of `size` nodes in `graph` containing `seed`."""
    results = set()

    def dfs(current, frontier):
        if len(current) == size:
            results.add(frozenset(current))
            return
        for u in list(frontier):
            new_current = current | {u}
            new_frontier = (frontier - {u}) | (set(graph.neighbors(u)) - new_current)
            dfs(new_current, new_frontier)

    dfs(frozenset({seed}), frozenset(graph.neighbors(seed)))
    return results


def enumerate_partitions(graph, k, size):
    """All unordered partitions of `graph` into k connected parts of `size`."""
    partitions = []

    def helper(remaining, current_parts):
        if len(remaining) == 0:
            partitions.append(frozenset(current_parts))
            return
        sub = graph.subgraph(remaining)
        seed = min(remaining)
        for piece in connected_subsets_containing(sub, seed, size):
            helper(remaining - piece, current_parts + [piece])

    helper(frozenset(graph.nodes()), [])
    return partitions


def fiber_neighbors(P, graph):
    """All partitions reachable from P by a single boundary swap (flip-pair)."""
    neighbors = set()
    parts = list(P)
    for i in range(len(parts)):
        A = parts[i]
        for j in range(i + 1, len(parts)):
            B = parts[j]
            boundary_A = [u for u in A if any(w in B for w in graph.neighbors(u))]
            boundary_B = [v for v in B if any(w in A for w in graph.neighbors(v))]
            if not boundary_A or not boundary_B:
                continue
            for u in boundary_A:
                for v in boundary_B:
                    A_new = (A - {u}) | {v}
                    B_new = (B - {v}) | {u}
                    if not nx.is_connected(graph.subgraph(A_new)):
                        continue
                    if not nx.is_connected(graph.subgraph(B_new)):
                        continue
                    new_P = (P - {A, B}) | {frozenset(A_new), frozenset(B_new)}
                    neighbors.add(new_P)
    return neighbors


def main():
    out_dir = Path(__file__).parent
    G = nx.grid_graph([5, 5])

    print("Enumerating equal-population connected 5-partitions...")
    t0 = time.time()
    partitions = enumerate_partitions(G, 5, 5)
    print(f"  {len(partitions)} partitions in {time.time() - t0:.2f}s")

    P_to_idx = {P: i for i, P in enumerate(partitions)}

    print("Building fiber graph (single-swap adjacency)...")
    t0 = time.time()
    F = nx.Graph()
    F.add_nodes_from(range(len(partitions)))
    for i, P in enumerate(partitions):
        if i % 500 == 0 and i > 0:
            print(f"  processed {i}/{len(partitions)}  edges so far: {F.number_of_edges()}")
        for nb in fiber_neighbors(P, G):
            j = P_to_idx[nb]
            if j > i:
                F.add_edge(i, j)
    print(f"  {F.number_of_nodes()} nodes, {F.number_of_edges()} edges in {time.time() - t0:.2f}s")

    components = list(nx.connected_components(F))
    print(f"  Connected components: {len(components)} "
          f"(largest = {max(len(c) for c in components)})")
    degrees = [d for _, d in F.degree()]
    print(f"  Degree: min={min(degrees)}, mean={sum(degrees) / len(degrees):.2f}, "
          f"max={max(degrees)}")

    nx.write_edgelist(F, out_dir / "fiber_5x5_edges.txt", data=False)
    print(f"  Edge list saved to {out_dir / 'fiber_5x5_edges.txt'}")

    id_map = {
        str(i): [sorted(list(part)) for part in sorted(P, key=lambda s: min(s))]
        for i, P in enumerate(partitions)
    }
    with open(out_dir / "fiber_5x5_partitions.json", "w") as f:
        json.dump(id_map, f, indent=2, default=str)
    print(f"  Partition id->parts map saved to "
          f"{out_dir / 'fiber_5x5_partitions.json'}")

    print("Computing layout (spring) ...")
    t0 = time.time()
    pos = nx.spring_layout(F, seed=42, iterations=80,
                           k=2.5 / (len(partitions) ** 0.5))
    print(f"  Layout in {time.time() - t0:.2f}s")

    fig, ax = plt.subplots(figsize=(40, 40))
    nx.draw_networkx_edges(F, pos=pos, edge_color="lightgray",
                           width=0.1, alpha=0.4, ax=ax)
    nx.draw_networkx_nodes(F, pos=pos, node_size=55,
                           node_color="white", edgecolors="steelblue",
                           linewidths=0.4, ax=ax)
    nx.draw_networkx_labels(F, pos=pos, labels={i: str(i) for i in F.nodes()},
                            font_size=2.2, font_color="black", ax=ax)
    ax.set_title(
        f"Fiber: equal-population connected 5-partitions of the 5x5 grid\n"
        f"{F.number_of_nodes()} partitions, {F.number_of_edges()} flip-swap edges, "
        f"{len(components)} component(s)",
        fontsize=14,
    )
    ax.set_axis_off()
    plt.tight_layout()
    out_path_png = out_dir / "fiber_5x5.png"
    plt.savefig(out_path_png, dpi=300, bbox_inches="tight")
    out_path_pdf = out_dir / "fiber_5x5.pdf"
    plt.savefig(out_path_pdf, bbox_inches="tight")
    print(f"  Plots saved to {out_path_png} and {out_path_pdf}")


if __name__ == "__main__":
    main()
