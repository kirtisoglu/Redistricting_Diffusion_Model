"""
Waste fraction scaling experiment.

For grid sizes 6, 10, 15, 20, 25, 30:
  - Build NxN grid, partition into k=4 quadrants
  - Run MEW cycle basis steps
  - Measure: % of steps that don't change boundary edges (within-fiber waste)
  - Measure: % of interior edges (theoretical prediction)
"""

import numpy as np
import networkx as nx
import random
import time

random.seed(42)
np.random.seed(42)


def wilson_random_spanning_tree(G):
    """Sample a uniformly random spanning tree via Wilson's algorithm."""
    nodes = list(G.nodes())
    in_tree = {nodes[0]}
    tree_edges = []

    for start in nodes[1:]:
        if start in in_tree:
            continue
        path = [start]
        current = start
        visited = {start: 0}
        while current not in in_tree:
            neighbors = list(G.neighbors(current))
            current = random.choice(neighbors)
            if current in visited:
                idx = visited[current]
                for node in path[idx+1:]:
                    del visited[node]
                path = path[:idx+1]
            else:
                visited[current] = len(path)
                path.append(current)
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            tree_edges.append((min(u, v), max(u, v)))
            in_tree.add(path[i])

    return set(tree_edges)


def cycle_basis_step(T_edges, G, all_edges_set):
    """
    One MEW cycle basis step.
    Returns new tree edges and whether the boundary changed.
    """
    tree_edges_set = set(T_edges)
    non_tree_edges = list(all_edges_set - tree_edges_set)

    if not non_tree_edges:
        return T_edges

    e_plus = random.choice(non_tree_edges)
    u, v = e_plus

    # Build tree graph for path finding
    T = nx.Graph()
    T.add_edges_from(tree_edges_set)

    try:
        path = nx.shortest_path(T, u, v)
    except nx.NetworkXNoPath:
        return T_edges

    cycle_tree_edges = []
    for i in range(len(path) - 1):
        a, b = path[i], path[i+1]
        cycle_tree_edges.append((min(a, b), max(a, b)))

    if not cycle_tree_edges:
        return T_edges

    e_minus = random.choice(cycle_tree_edges)
    new_edges = (tree_edges_set | {e_plus}) - {e_minus}
    return new_edges


def make_quadrant_assignment(N):
    """Partition NxN grid into 4 quadrants."""
    half = N // 2
    assignment = {}
    for node in range(N * N):
        row = node // N
        col = node % N
        if row < half and col < half:
            assignment[node] = 0
        elif row < half and col >= half:
            assignment[node] = 1
        elif row >= half and col < half:
            assignment[node] = 2
        else:
            assignment[node] = 3
    return assignment


def count_boundary_edges_in_tree(T_edges, assignment):
    """Count how many tree edges cross district boundaries."""
    count = 0
    for u, v in T_edges:
        if assignment[u] != assignment[v]:
            count += 1
    return count


def get_boundary_signature(T_edges, assignment):
    """Get the set of boundary edges in the tree (for detecting changes)."""
    return frozenset((u, v) for u, v in T_edges if assignment[u] != assignment[v])


def run_experiment(N, num_steps):
    """Run MEW on NxN grid and measure waste fraction."""
    G = nx.grid_2d_graph(N, N)
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()
    m = G.number_of_edges()

    assignment = make_quadrant_assignment(N)

    # Precompute all edges as canonical pairs
    all_edges_set = set((min(u, v), max(u, v)) for u, v in G.edges())

    # Count interior vs boundary edges
    n_boundary_total = sum(1 for u, v in G.edges() if assignment[u] != assignment[v])
    n_interior_total = m - n_boundary_total
    interior_fraction = n_interior_total / m

    # Kirchhoff: spanning tree counts
    L = nx.laplacian_matrix(G).toarray()
    L_red = L[1:, 1:]
    log_det = np.linalg.slogdet(L_red.astype(float))
    log_t_G = log_det[1]  # log of number of spanning trees

    # District spanning tree counts
    districts = {}
    for node, d in assignment.items():
        districts.setdefault(d, []).append(node)

    log_t_districts = 0
    for d in sorted(districts):
        subg = G.subgraph(districts[d])
        L_sub = nx.laplacian_matrix(subg).toarray()
        L_sub_red = L_sub[1:, 1:]
        ld = np.linalg.slogdet(L_sub_red.astype(float))
        log_t_districts += ld[1]

    # Sample initial tree
    T_current = wilson_random_spanning_tree(G)
    last_sig = get_boundary_signature(T_current, assignment)

    within_fiber_count = 0
    cross_boundary_count = 0

    t0 = time.time()
    for step in range(num_steps):
        T_new = cycle_basis_step(T_current, G, all_edges_set)
        new_sig = get_boundary_signature(T_new, assignment)

        if new_sig == last_sig:
            within_fiber_count += 1
        else:
            cross_boundary_count += 1

        last_sig = new_sig
        T_current = T_new

    t1 = time.time()

    waste_fraction = within_fiber_count / num_steps
    return {
        'N': N,
        'n': n,
        'm': m,
        'interior_edges': n_interior_total,
        'boundary_edges': n_boundary_total,
        'interior_fraction': interior_fraction,
        'num_steps': num_steps,
        'within_fiber': within_fiber_count,
        'cross_boundary': cross_boundary_count,
        'waste_fraction': waste_fraction,
        'log_t_G': log_t_G,
        'log_t_districts': log_t_districts,
        'time': t1 - t0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RUN EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 75)
print("WASTE FRACTION SCALING EXPERIMENT")
print("MEW cycle basis steps on NxN grids with 4-quadrant partition")
print("=" * 75)
print()

grid_sizes = [6, 10, 15, 20, 25, 30]
num_steps_per_grid = 10000
results = []

for N in grid_sizes:
    print(f"Running N={N} ({N}x{N} = {N*N} nodes)...", end=" ", flush=True)
    res = run_experiment(N, num_steps_per_grid)
    results.append(res)
    print(f"done in {res['time']:.1f}s — waste = {res['waste_fraction']*100:.1f}%")

print()
print("=" * 75)
print(f"{'Grid':>6} {'Nodes':>6} {'Edges':>6} {'Int.Edges':>9} {'Bdy.Edges':>9} "
      f"{'Int.Frac':>8} {'Waste%':>8} {'Useful%':>8} {'Speedup':>8}")
print("-" * 75)

for r in results:
    speedup = 1.0 / (1.0 - r['waste_fraction']) if r['waste_fraction'] < 1 else float('inf')
    print(f"{r['N']:>4}x{r['N']:<2} {r['n']:>5} {r['m']:>6} "
          f"{r['interior_edges']:>9} {r['boundary_edges']:>9} "
          f"{r['interior_fraction']*100:>7.1f}% "
          f"{r['waste_fraction']*100:>7.1f}% "
          f"{(1-r['waste_fraction'])*100:>7.1f}% "
          f"{speedup:>7.1f}x")

print()
print("Interior fraction: % of graph edges that are within a single district")
print("Waste%: % of MEW steps that don't change the boundary (within-fiber)")
print("Useful%: % of MEW steps that change the boundary (cross-boundary)")
print("Speedup: factor by which two-level chain avoids wasted steps (1/(1-waste))")

print()
print("=" * 75)
print("LOG SPANNING TREE COUNTS")
print("-" * 75)
print(f"{'Grid':>6} {'log t(G)':>14} {'log ∏t(G[Di])':>14} {'log ratio':>14} {'ratio':>14}")
print("-" * 75)
for r in results:
    log_ratio = r['log_t_G'] - r['log_t_districts']
    print(f"{r['N']:>4}x{r['N']:<2} {r['log_t_G']:>14.2f} {r['log_t_districts']:>14.2f} "
          f"{log_ratio:>14.2f} {np.exp(log_ratio):>14.2e}")

print()
print("The ratio t(G)/∏t(G[Di]) measures how many boundary configurations")
print("exist across ALL partitions per within-district fiber.")
