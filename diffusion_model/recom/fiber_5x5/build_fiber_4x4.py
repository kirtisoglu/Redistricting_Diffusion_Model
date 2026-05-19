"""Enumerate the ε=0.25 single-flip fiber on the 4×4 grid (k=4 districts).

Each node has population 50 → ideal district pop 200.
ε = 0.25 → district pops in [150, 250] → district sizes in {3, 4, 5}.
Allowed multisets summing to 16 with each size ∈ {3,4,5}:
  (4,4,4,4)   — balanced layer L0
  (3,4,4,5)   — one-off layer L1
  (3,3,5,5)   — two-off layer L2

Edge: P ~ P' iff P' is obtained from P by moving one node u from
district a to district b across the (a,b) boundary, with both new
districts (a−{u}, b∪{u}) connected and of size in {3,4,5}.

Outputs:
  fiber_4x4_eps025_edges.txt        — edge list
  fiber_4x4_eps025_partitions.json  — id → {layer, parts}
  fiber_4x4_eps025_degree.png       — degree distribution
"""

import json
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


ALLOWED_SIZES = (3, 4, 5)
K = 4
GRID_DIM = 4


def connected_subsets_containing(graph, seed, size):
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


def enumerate_partitions(graph):
    partitions = []
    min_s, max_s = min(ALLOWED_SIZES), max(ALLOWED_SIZES)

    def helper(remaining, current_parts):
        if len(current_parts) == K:
            if not remaining:
                partitions.append(frozenset(current_parts))
            return
        sub = graph.subgraph(remaining)
        seed = min(remaining)
        parts_left = K - len(current_parts) - 1
        for size in ALLOWED_SIZES:
            if size > len(remaining):
                continue
            after = len(remaining) - size
            if parts_left == 0 and after != 0:
                continue
            if parts_left > 0:
                if after < parts_left * min_s or after > parts_left * max_s:
                    continue
            for piece in connected_subsets_containing(sub, seed, size):
                helper(remaining - piece, current_parts + [piece])

    helper(frozenset(graph.nodes()), [])
    return partitions


def fiber_neighbors_flip(P, graph):
    neighbors = set()
    parts = list(P)
    for A in parts:
        if len(A) - 1 not in ALLOWED_SIZES:
            continue
        for B in parts:
            if A is B:
                continue
            if len(B) + 1 not in ALLOWED_SIZES:
                continue
            for u in A:
                if not any(w in B for w in graph.neighbors(u)):
                    continue
                A_new = A - {u}
                if not nx.is_connected(graph.subgraph(A_new)):
                    continue
                B_new = B | {u}
                new_P = (P - {A, B}) | {frozenset(A_new), frozenset(B_new)}
                neighbors.add(new_P)
    return neighbors


def layer_of(P):
    return tuple(sorted(len(part) for part in P))


def main():
    out_dir = Path(__file__).parent
    G = nx.grid_graph([GRID_DIM, GRID_DIM])

    print(f"Enumerating ε=0.25 connected k={K} partitions of "
          f"{GRID_DIM}×{GRID_DIM} grid …")
    t0 = time.time()
    partitions = enumerate_partitions(G)
    print(f"  {len(partitions)} partitions in {time.time() - t0:.2f}s")

    layers = [layer_of(P) for P in partitions]
    layer_counts = Counter(layers)
    for L, c in sorted(layer_counts.items()):
        print(f"  layer {L}: {c}")

    P_to_idx = {P: i for i, P in enumerate(partitions)}

    print(f"\nBuilding single-flip fiber graph...")
    t0 = time.time()
    F = nx.Graph()
    F.add_nodes_from(range(len(partitions)))
    for i, P in enumerate(partitions):
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(partitions)}  edges: {F.number_of_edges()}")
        for nb in fiber_neighbors_flip(P, G):
            j = P_to_idx[nb]
            if j > i:
                F.add_edge(i, j)
    print(f"  {F.number_of_nodes()} nodes, {F.number_of_edges()} edges "
          f"in {time.time() - t0:.1f}s")

    components = list(nx.connected_components(F))
    print(f"  Components: {len(components)}, "
          f"largest: {max(len(c) for c in components)}")
    bridges = list(nx.bridges(F))
    aps = list(nx.articulation_points(F))
    print(f"  Bridges: {len(bridges)}  Articulation points: {len(aps)}")

    nx.write_edgelist(F, out_dir / "fiber_4x4_eps025_edges.txt", data=False)
    print(f"  Edge list saved to {out_dir / 'fiber_4x4_eps025_edges.txt'}")

    pmap = {}
    for i, P in enumerate(partitions):
        pmap[str(i)] = {
            "layer": "-".join(map(str, layers[i])),
            "parts": [sorted([str(x) for x in part])
                      for part in sorted(P, key=lambda s: min(s))],
        }
    with open(out_dir / "fiber_4x4_eps025_partitions.json", "w") as f:
        json.dump(pmap, f)
    print(f"  Partitions saved to "
          f"{out_dir / 'fiber_4x4_eps025_partitions.json'}")

    # Degree distribution
    degs = [d for _, d in F.degree()]
    deg_count = Counter(degs)
    plt.figure(figsize=(10, 5))
    ks = sorted(deg_count)
    plt.bar(ks, [deg_count[k] for k in ks], color="steelblue", edgecolor="black")
    plt.xlabel("degree")
    plt.ylabel("partitions")
    plt.title(f"4×4 ε=0.25 fiber: degree distribution\n"
              f"{F.number_of_nodes()} partitions, "
              f"{F.number_of_edges()} edges, mean deg {np.mean(degs):.2f}")
    plt.tight_layout()
    plt.savefig(out_dir / "fiber_4x4_eps025_degree.png", dpi=150)
    print(f"  Degree plot: {out_dir / 'fiber_4x4_eps025_degree.png'}")


if __name__ == "__main__":
    main()
