"""Build the ε=0.2 single-flip fiber graph on the 5x5 grid and verify
predictions about its layered structure.

Allowed district sizes: {4, 5, 6}  (deviation ≤ 20% from ideal 250).
Allowed multisets summing to 25 with 5 districts:
    L0 = (5,5,5,5,5)
    L1 = (4,5,5,5,6)
    L2 = (4,4,5,6,6)

Edge: P ~ P' iff P' is obtained from P by moving one node u from
district a to district b across the (a,b) boundary, with both new
districts (a−{u}, b∪{u}) connected and of size in {4,5,6}.
"""

import json
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


ALLOWED_SIZES = (4, 5, 6)
K = 5


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
    sizes = tuple(sorted(len(part) for part in P))
    if sizes == (5, 5, 5, 5, 5):
        return 0
    if sizes == (4, 5, 5, 5, 6):
        return 1
    if sizes == (4, 4, 5, 6, 6):
        return 2
    return -1


def main():
    here = Path(__file__).parent
    G = nx.grid_graph([5, 5])

    print("Enumerating ε=0.2 connected partitions...")
    t0 = time.time()
    partitions = enumerate_partitions(G)
    print(f"  {len(partitions)} partitions in {time.time() - t0:.2f}s")

    layers = [layer_of(P) for P in partitions]
    layer_counts = Counter(layers)
    for L in [0, 1, 2]:
        print(f"  L{L}: {layer_counts[L]}")

    P_to_idx = {P: i for i, P in enumerate(partitions)}

    print("\nBuilding single-flip fiber graph...")
    t0 = time.time()
    F = nx.Graph()
    F.add_nodes_from(range(len(partitions)))
    for i, P in enumerate(partitions):
        if i % 5000 == 0 and i > 0:
            print(f"  {i}/{len(partitions)}  edges so far: {F.number_of_edges()}")
        for nb in fiber_neighbors_flip(P, G):
            j = P_to_idx[nb]
            if j > i:
                F.add_edge(i, j)
    print(f"  {F.number_of_nodes()} nodes, {F.number_of_edges()} edges in "
          f"{time.time() - t0:.1f}s")

    print("\n=== predictions vs reality ===")

    print("\n[A] L0 induced subgraph:")
    L0_nodes = [i for i, L in enumerate(layers) if L == 0]
    L0_sub = F.subgraph(L0_nodes)
    print(f"  L0 partitions: {len(L0_nodes)}")
    print(f"  edges within L0: {L0_sub.number_of_edges()}  "
          f"(prediction: 0)")

    print("\n[B] edges by layer-pair:")
    layer_edges = Counter()
    for u, v in F.edges():
        a, b = layers[u], layers[v]
        layer_edges[tuple(sorted([a, b]))] += 1
    total = F.number_of_edges()
    for k in sorted(layer_edges):
        c = layer_edges[k]
        print(f"  L{k[0]}-L{k[1]}: {c:>7}  ({100 * c / total:5.1f}%)")
    forbidden = layer_edges.get((0, 0), 0) + layer_edges.get((0, 2), 0) + layer_edges.get((2, 2), 0)
    # Note: L2-L2 IS allowed by single flip, only L0-L0 and L0-L2 forbidden
    print(f"  forbidden (L0-L0, L0-L2): "
          f"{layer_edges.get((0, 0), 0) + layer_edges.get((0, 2), 0)}")

    print("\n[C] connectivity:")
    components = list(nx.connected_components(F))
    print(f"  components: {len(components)}, largest: "
          f"{max(len(c) for c in components)}")

    print("\n[D] bridges and articulation points:")
    bridges = list(nx.bridges(F))
    aps = list(nx.articulation_points(F))
    print(f"  bridges: {len(bridges)}  (ε=0 fiber: 16)")
    print(f"  articulation points: {len(aps)}  (ε=0 fiber: 16)")

    print("\n[E] modularity (greedy):")
    t0 = time.time()
    communities = list(nx.community.greedy_modularity_communities(F))
    Q = nx.community.modularity(F, communities)
    print(f"  {len(communities)} communities, Q={Q:.4f} in "
          f"{time.time() - t0:.1f}s  (ε=0 fiber: Q=0.654)")

    print("\n[F] degree statistics:")
    degs = [d for _, d in F.degree()]
    print(f"  mean: {np.mean(degs):.2f}, median: {np.median(degs)}, "
          f"min: {min(degs)}, max: {max(degs)}  (ε=0: mean 6.70, max 13)")
    for L in [0, 1, 2]:
        L_nodes = [i for i, l in enumerate(layers) if l == L]
        if L_nodes:
            L_degs = [F.degree(n) for n in L_nodes]
            print(f"  L{L}: mean deg = {np.mean(L_degs):.2f}, "
                  f"min = {min(L_degs)}, max = {max(L_degs)}")

    print("\n[G] L0-to-L0 shortest path in the new fiber (sample):")
    sample = L0_nodes[:8]
    for s in sample:
        dists = nx.single_source_shortest_path_length(F, s, cutoff=3)
        l0d = [dists[t] for t in L0_nodes if t != s and t in dists]
        if l0d:
            print(f"  from id={s}: min L0-L0 dist = {min(l0d)} "
                  f"(must be ≥2 since L0 is independent), "
                  f"L0 reachable in ≤3 hops: {len(l0d)}")

    print("\nSaving outputs...")
    nx.write_edgelist(F, here / "fiber_5x5_eps02_edges.txt", data=False)
    pmap = {
        str(i): {
            "layer": layers[i],
            "parts": [sorted([str(x) for x in part])
                      for part in sorted(P, key=lambda s: min(s))],
        }
        for i, P in enumerate(partitions)
    }
    with open(here / "fiber_5x5_eps02_partitions.json", "w") as f:
        json.dump(pmap, f)
    print(f"  edges -> {here / 'fiber_5x5_eps02_edges.txt'}")
    print(f"  partitions -> {here / 'fiber_5x5_eps02_partitions.json'}")

    print("\nDegree distribution plot...")
    plt.figure(figsize=(10, 5))
    deg_count = Counter(degs)
    ks = sorted(deg_count)
    plt.bar(ks, [deg_count[k] for k in ks], color="steelblue", edgecolor="black")
    plt.xlabel("degree (single-flip neighbors)")
    plt.ylabel("partitions")
    plt.title(f"ε=0.2 single-flip fiber: degree distribution\n"
              f"{F.number_of_nodes()} partitions, {F.number_of_edges()} edges")
    plt.tight_layout()
    plt.savefig(here / "fiber_5x5_eps02_degree.png", dpi=150)
    print(f"  -> {here / 'fiber_5x5_eps02_degree.png'}")


if __name__ == "__main__":
    main()
