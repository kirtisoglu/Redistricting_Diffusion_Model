"""
Picard group experiment on a 6×6 grid with 4 districts.

We compute:
1. The Picard group Pic⁰(G) = Div⁰(G) / im(Δ) via Smith normal form of the Laplacian
2. The number of spanning trees |T(G)| = |Pic⁰(G)|
3. For a fixed partition ξ, the fiber F(ξ) = {trees compatible with ξ}
4. The subgroup structure: ∏ Pic⁰(G[Dᵢ]) and whether F(ξ) aligns with cosets
5. How many distinct partitions the cycle basis step actually visits
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from itertools import product as iterproduct
import random
import time

random.seed(42)
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. BUILD 6×6 GRID
# ═══════════════════════════════════════════════════════════════════════════════
N = 6
G = nx.grid_2d_graph(N, N)
G = nx.convert_node_labels_to_integers(G)
n = G.number_of_nodes()   # 36
m = G.number_of_edges()   # 60
g = m - n + 1             # genus = 25

print(f"6×6 grid: {n} nodes, {m} edges, genus g = {g}")
print(f"Genus = dimension of cycle space = {g}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# 2. COMPUTE PICARD GROUP VIA SMITH NORMAL FORM
# ═══════════════════════════════════════════════════════════════════════════════
L = nx.laplacian_matrix(G).toarray()

# Reduced Laplacian: delete row and column of one vertex (say vertex 0)
L_red = L[1:, 1:]

# Number of spanning trees = det(L_red) (Kirchhoff's theorem)
det_L = round(np.linalg.det(L_red.astype(float)))
print(f"|T(G)| = det(L_red) = {det_L:,}")
print(f"  (this is also |Pic⁰(G)|)")
print()

# Smith normal form to find group structure
# Pic⁰(G) ≅ Z/d₁ × Z/d₂ × ... × Z/dₖ where d₁|d₂|...|dₖ
# We use integer SVD via sympy for exact computation
try:
    from sympy import Matrix as SympyMatrix
    L_red_sympy = SympyMatrix(L_red.tolist())

    print("Computing Smith Normal Form of reduced Laplacian...")
    t0 = time.time()
    smith = L_red_sympy.smith_normal_form()
    t1 = time.time()

    # Extract diagonal (invariant factors)
    diag = [int(smith[i, i]) for i in range(smith.rows)]
    invariant_factors = [d for d in diag if d > 1]

    print(f"  Time: {t1-t0:.2f}s")
    print(f"  Invariant factors (d > 1): {invariant_factors}")
    print(f"  Pic⁰(G) ≅ {'×'.join(f'Z/{d}' for d in invariant_factors)}")
    print(f"  |Pic⁰(G)| = {np.prod(invariant_factors)} (should equal {det_L})")
    has_smith = True
except Exception as e:
    print(f"  Smith normal form computation failed: {e}")
    print(f"  Falling back to determinant only.")
    has_smith = False

print()

# ═══════════════════════════════════════════════════════════════════════════════
# 3. CREATE A BALANCED 4-DISTRICT PARTITION OF THE 6×6 GRID
# ═══════════════════════════════════════════════════════════════════════════════
# Simple partition: 4 quadrants of 3×3 = 9 nodes each
# D0: rows 0-2, cols 0-2   D1: rows 0-2, cols 3-5
# D2: rows 3-5, cols 0-2   D3: rows 3-5, cols 3-5
grid_pos = {}
for node in range(n):
    row = node // N
    col = node % N
    grid_pos[node] = (col, row)

assignment = {}
for node in range(n):
    row = node // N
    col = node % N
    if row < 3 and col < 3:
        assignment[node] = 0
    elif row < 3 and col >= 3:
        assignment[node] = 1
    elif row >= 3 and col < 3:
        assignment[node] = 2
    else:
        assignment[node] = 3

# Verify
districts = defaultdict(list)
for node, d in assignment.items():
    districts[d].append(node)

print("Partition ξ (4 quadrants of 3×3):")
for d in sorted(districts):
    nodes = sorted(districts[d])
    print(f"  District {d}: {len(nodes)} nodes — {nodes}")

# Check connectivity of each district
for d in sorted(districts):
    subg = G.subgraph(districts[d])
    print(f"  District {d} connected: {nx.is_connected(subg)}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# 4. COUNT SPANNING TREES PER DISTRICT AND THE DEGENERACY FACTOR
# ═══════════════════════════════════════════════════════════════════════════════
print("Spanning trees per district subgraph:")
district_tree_counts = {}
for d in sorted(districts):
    subg = G.subgraph(districts[d])
    L_sub = nx.laplacian_matrix(subg).toarray()
    # Remove one row/col for reduced Laplacian
    nodes_list = sorted(districts[d])
    L_sub_red = L_sub[1:, 1:]
    count = round(np.linalg.det(L_sub_red.astype(float)))
    district_tree_counts[d] = count
    print(f"  |T(G[D_{d}])| = {count}")

# Boundary edges between districts
boundary_edges = {}
for u, v in G.edges():
    du, dv = assignment[u], assignment[v]
    if du != dv:
        key = (min(du, dv), max(du, dv))
        boundary_edges[key] = boundary_edges.get(key, 0) + 1

print(f"\nBoundary edges between districts:")
for (d1, d2), count in sorted(boundary_edges.items()):
    print(f"  |∂(D_{d1}, D_{d2})| = {count}")

# Weighted spanning tree count of quotient graph Q via Kirchhoff
# Q has k nodes (districts), edge (i,j) with weight |∂(Dᵢ,Dⱼ)|
Q = nx.Graph()
Q.add_nodes_from(range(4))
quotient_edge_weights = {}
for (d1, d2), count in boundary_edges.items():
    Q.add_edge(d1, d2, weight=count)
    quotient_edge_weights[(d1, d2)] = count

# Weighted Laplacian of Q
L_Q = np.zeros((4, 4))
for (d1, d2), w in quotient_edge_weights.items():
    L_Q[d1, d2] -= w
    L_Q[d2, d1] -= w
    L_Q[d1, d1] += w
    L_Q[d2, d2] += w

L_Q_red = L_Q[1:, 1:]
t_w_Q = round(np.linalg.det(L_Q_red))  # weighted spanning tree count of Q

print(f"\nQuotient graph Q (districts as nodes, weighted by boundary edge counts):")
print(f"  Weighted Laplacian of Q:")
for row in L_Q.astype(int):
    print(f"    {row}")
print(f"  t_w(Q) = det(L_Q_red) = {t_w_Q}")
print(f"    (= sum over spanning trees T of Q: ∏_{{(i,j)∈T}} |∂(Dᵢ,Dⱼ)|)")

# Degeneracy factor τ(ξ) = ∏ t(G[Dᵢ]) × t_w(Q)
product_district_trees = 1
for d, count in district_tree_counts.items():
    product_district_trees *= count
tau = product_district_trees * t_w_Q

print(f"\nτ(ξ) = ∏ t(G[Dᵢ]) × t_w(Q) = {product_district_trees:,} × {t_w_Q} = {tau:,}")
print(f"Fraction of all trees compatible with ξ: τ(ξ) / |T(G)| = {tau/det_L:.6e}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# 5. PICARD GROUPS OF DISTRICT SUBGRAPHS
# ═══════════════════════════════════════════════════════════════════════════════
print("Picard groups of district subgraphs:")
for d in sorted(districts):
    subg = G.subgraph(districts[d])
    n_sub = subg.number_of_nodes()
    m_sub = subg.number_of_edges()
    g_sub = m_sub - n_sub + 1
    print(f"  G[D_{d}]: {n_sub} nodes, {m_sub} edges, genus = {g_sub}")
    print(f"    |Pic⁰(G[D_{d}])| = {district_tree_counts[d]}")

product_size = 1
for d in district_tree_counts:
    product_size *= district_tree_counts[d]

print(f"\n|∏ Pic⁰(G[Dᵢ])| = {product_size:,}")
print(f"|Pic⁰(G)| = {det_L:,}")
print(f"Ratio |Pic⁰(G)| / |∏ Pic⁰(G[Dᵢ])| = {det_L / product_size:.1f}")
print(f"  (NOT a group-theoretic index — ∏ Pic⁰(G[Dᵢ]) does not embed in Pic⁰(G))")
print(f"  True boundary configurations for THIS partition: t_w(Q) = {t_w_Q}")
print(f"  Within-district fiber per boundary config: ∏ t(G[Dᵢ]) = {product_size:,}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# 6. ENUMERATE ALL SPANNING TREES (feasible for 6×6 grid)
#    AND CLASSIFY BY PARTITION
# ═══════════════════════════════════════════════════════════════════════════════
print("Enumerating all spanning trees of 6×6 grid...")
print("(This may take a moment...)")
t0 = time.time()

# Use Kirchhoff's matrix-tree theorem approach with actual enumeration
# For a 36-node graph, direct enumeration is too slow.
# Instead, we'll sample trees and track partition statistics.

# First, let's count how many trees Wilson's algorithm generates
# that give distinct partitions (with 3 marked edges for 4 districts)

def wilson_random_spanning_tree(G):
    """Sample a uniformly random spanning tree via Wilson's algorithm."""
    nodes = list(G.nodes())
    n = len(nodes)
    in_tree = {nodes[0]}
    next_node = {nodes[0]: None}
    parent = {nodes[0]: None}
    tree_edges = []

    for start in nodes[1:]:
        if start in in_tree:
            continue
        # Random walk from start until hitting tree
        path = [start]
        current = start
        visited = {start: 0}
        while current not in in_tree:
            neighbors = list(G.neighbors(current))
            current = random.choice(neighbors)
            if current in visited:
                # Loop erase
                idx = visited[current]
                for node in path[idx+1:]:
                    del visited[node]
                path = path[:idx+1]
            else:
                visited[current] = len(path)
                path.append(current)

        # Add path to tree
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            tree_edges.append((u, v) if u < v else (v, u))
            in_tree.add(path[i])

    return set(tree_edges)


def tree_to_partition(tree_edges, G, assignment, k=4):
    """
    Given a spanning tree and k-1=3 marked edges to remove,
    check if removing boundary edges gives the target partition.
    Return the partition signature if we remove all boundary edges from tree.
    """
    # Find which tree edges are boundary edges (cross districts)
    boundary_tree_edges = []
    for u, v in tree_edges:
        if assignment[u] != assignment[v]:
            boundary_tree_edges.append((u, v))
    return len(boundary_tree_edges), tuple(sorted(boundary_tree_edges))


def partition_from_tree(tree_edges, G, k=4):
    """
    For a spanning tree, find ALL ways to choose k-1 edges to remove
    that give k connected components. Return the set of resulting partitions.
    """
    # Build tree as a graph
    T = nx.Graph()
    T.add_nodes_from(G.nodes())
    T.add_edges_from(tree_edges)

    # The boundary edges in the tree determine the partition
    # (if we think of it as MEW: marked edges = boundary edges)
    # But actually, ANY k-1 edges can be removed.
    # For simplicity, let's just look at what partition the tree
    # is compatible with when we remove its boundary edges.

    # Count boundary edges
    partitions = set()
    edges_list = list(tree_edges)
    n_edges = len(edges_list)

    # For k=4, we need to remove 3 edges
    # Only feasible for small trees
    if n_edges > 40:
        return partitions

    from itertools import combinations
    for combo in combinations(range(n_edges), k-1):
        remaining = [edges_list[i] for i in range(n_edges) if i not in combo]
        F = nx.Graph()
        F.add_nodes_from(G.nodes())
        F.add_edges_from(remaining)
        components = list(nx.connected_components(F))
        if len(components) == k:
            # Encode partition as frozenset of frozensets
            partition = frozenset(frozenset(c) for c in components)
            partitions.add(partition)

    return partitions


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SAMPLE SPANNING TREES AND ANALYZE STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════
NUM_SAMPLES = 5000

print(f"\nSampling {NUM_SAMPLES} spanning trees via Wilson's algorithm...")
t0 = time.time()

tree_set = set()  # track unique trees
boundary_edge_counts = Counter()  # how many boundary edges per tree
trees_per_partition_type = Counter()  # keyed by number of boundary edges in tree

sampled_trees = []
for i in range(NUM_SAMPLES):
    T_edges = wilson_random_spanning_tree(G)
    tree_key = frozenset(T_edges)
    tree_set.add(tree_key)

    n_boundary, boundary_tuple = tree_to_partition(T_edges, G, assignment, k=4)
    boundary_edge_counts[n_boundary] += 1
    sampled_trees.append((T_edges, n_boundary))

t1 = time.time()
print(f"  Time: {t1-t0:.2f}s")
print(f"  Unique trees sampled: {len(tree_set)} / {NUM_SAMPLES}")
print(f"  Total spanning trees: {det_L:,}")
print(f"  Coverage: {len(tree_set)/det_L*100:.4f}%")
print()

print("Distribution of boundary edges in random spanning trees:")
for k_val in sorted(boundary_edge_counts):
    count = boundary_edge_counts[k_val]
    print(f"  {k_val} boundary edges: {count} trees ({count/NUM_SAMPLES*100:.1f}%)")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# 8. SIMULATE MEW CYCLE BASIS STEP AND TRACK REDUNDANCY
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SIMULATING MEW CYCLE BASIS STEP — TRACKING REDUNDANCY")
print("=" * 70)


def cycle_basis_step(T_edges, G):
    """
    One MEW cycle basis step:
    1. Pick random non-tree edge e+
    2. Find fundamental cycle C in T ∪ {e+}
    3. Pick random edge e- from C (not e+)
    4. Return T' = (T ∪ {e+}) \ {e-}
    """
    T = nx.Graph()
    T.add_nodes_from(G.nodes())
    T.add_edges_from(T_edges)

    all_edges = set((min(u,v), max(u,v)) for u, v in G.edges())
    tree_edges_set = set((min(u,v), max(u,v)) for u, v in T_edges)
    non_tree_edges = list(all_edges - tree_edges_set)

    if not non_tree_edges:
        return T_edges

    # Pick random non-tree edge
    e_plus = random.choice(non_tree_edges)
    u, v = e_plus

    # Find fundamental cycle: path from u to v in T, plus edge (u,v)
    try:
        path = nx.shortest_path(T, u, v)
    except nx.NetworkXNoPath:
        return T_edges

    # Cycle edges (in T)
    cycle_tree_edges = []
    for i in range(len(path) - 1):
        a, b = path[i], path[i+1]
        cycle_tree_edges.append((min(a,b), max(a,b)))

    if not cycle_tree_edges:
        return T_edges

    # Pick random edge to remove
    e_minus = random.choice(cycle_tree_edges)

    # New tree
    new_edges = (tree_edges_set | {e_plus}) - {e_minus}
    return new_edges


# Start from a random tree
T_current = wilson_random_spanning_tree(G)

NUM_STEPS = 10000
visited_trees = set()
visited_trees.add(frozenset(T_current))

revisit_count = 0
new_tree_count = 0
partition_changes = 0
last_boundary = tree_to_partition(T_current, G, assignment, k=4)[0]

# Track when we see new trees
new_tree_timeline = []

for step in range(NUM_STEPS):
    T_new = cycle_basis_step(T_current, G)
    key = frozenset(T_new)

    if key in visited_trees:
        revisit_count += 1
    else:
        new_tree_count += 1
        visited_trees.add(key)

    new_boundary = tree_to_partition(T_new, G, assignment, k=4)[0]
    if new_boundary != last_boundary:
        partition_changes += 1
    last_boundary = new_boundary

    T_current = T_new

    if (step + 1) % 1000 == 0:
        new_tree_timeline.append((step + 1, len(visited_trees)))

print(f"\nAfter {NUM_STEPS} cycle basis steps:")
print(f"  Unique trees visited:  {len(visited_trees)}")
print(f"  Revisits:              {revisit_count} ({revisit_count/NUM_STEPS*100:.1f}%)")
print(f"  New trees discovered:  {new_tree_count} ({new_tree_count/NUM_STEPS*100:.1f}%)")
print(f"  Boundary edge changes: {partition_changes} ({partition_changes/NUM_STEPS*100:.1f}%)")
print(f"  Total spanning trees:  {det_L:,}")
print(f"  Coverage:              {len(visited_trees)/det_L*100:.4f}%")
print()

print("Discovery rate over time:")
for step, count in new_tree_timeline:
    print(f"  Step {step:>6}: {count:>6} unique trees ({count/det_L*100:.4f}%)")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# 9. ANALYZE THE FIBER: FOR A FIXED PARTITION, HOW MANY TREES?
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("FIBER ANALYSIS: Fixing partition ξ, sampling compatible trees")
print("=" * 70)


def sample_compatible_tree(G, districts):
    """
    Sample a spanning tree compatible with partition ξ:
    1. Sample spanning tree of each G[Dᵢ] via Wilson's
    2. Pick one boundary edge per adjacent district pair
    3. Combine
    """
    tree_edges = set()

    # Per-district spanning trees
    for d in sorted(districts):
        subg = G.subgraph(districts[d])
        sub_tree = wilson_random_spanning_tree(subg)
        tree_edges |= sub_tree

    # Boundary edges: need to connect the 4 districts into one tree
    # The quotient graph has 4 nodes (districts)
    # Need 3 edges in the quotient = 3 boundary edges
    # Build quotient graph
    quotient_edges = defaultdict(list)
    for u, v in G.edges():
        du, dv = assignment[u], assignment[v]
        if du != dv:
            key = (min(du, dv), max(du, dv))
            quotient_edges[key].append((min(u,v), max(u,v)))

    # Sample spanning tree of quotient graph (4 nodes)
    q_nodes = list(range(4))
    q_graph = nx.Graph()
    q_graph.add_nodes_from(q_nodes)
    for (d1, d2) in quotient_edges:
        q_graph.add_edge(d1, d2)

    q_tree = wilson_random_spanning_tree(q_graph)

    # For each quotient tree edge, pick a random boundary edge
    for d1, d2 in q_tree:
        key = (min(d1, d2), max(d1, d2))
        edge = random.choice(quotient_edges[key])
        tree_edges.add(edge)

    return tree_edges


print(f"\nSampling {NUM_SAMPLES} trees compatible with partition ξ (4 quadrants)...")
t0 = time.time()

fiber_trees = set()
for i in range(NUM_SAMPLES):
    T = sample_compatible_tree(G, districts)
    fiber_trees.add(frozenset(T))

t1 = time.time()
print(f"  Time: {t1-t0:.2f}s")
print(f"  Unique compatible trees found: {len(fiber_trees)}")
print(f"  Theoretical fiber size τ(ξ):   {tau:,}")
print(f"  Coverage of fiber:             {len(fiber_trees)/tau*100:.2f}%")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# 10. THE KEY COMPARISON: QUOTIENT SIZE
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("QUOTIENT ANALYSIS — THE PUNCHLINE")
print("=" * 70)
print()
print(f"Full state space:       |T(G)| = {det_L:,} spanning trees")
print(f"Fiber over ξ:           τ(ξ)   = {tau:,} trees for THIS partition")
print(f"Fiber fraction:         τ(ξ)/|T(G)| = {tau/det_L:.6e}")
print()
print(f"DECOMPOSITION OF THE FIBER over ξ:")
print(f"  τ(ξ) = ∏ t(G[Dᵢ]) × t_w(Q) = {product_size:,} × {t_w_Q} = {tau:,}")
print(f"  Within-district factor:  ∏ t(G[Dᵢ]) = {product_size:,}  (192⁴ — independent per district)")
print(f"  Cross-boundary factor:   t_w(Q)      = {t_w_Q}  (boundary configs for THIS partition)")
print()
print(f"KEY OBSTACLE: ∏ Pic⁰(G[Dᵢ]) does NOT embed in Pic⁰(G)")
print(f"  Chip-firing v in G[Dᵢ] ≠ chip-firing v in G (boundary edges change the diagonal)")
print(f"  The ratio |Pic⁰(G)| / |∏ Pic⁰(G[Dᵢ])| = {det_L/product_size:.1f} is NOT a group index")
print()
print(f"WHAT IS COMPUTABLE: Spanning tree decomposition for fixed ξ")
print(f"  Distinct boundary configs: t_w(Q) = {t_w_Q}")
print(f"  Trees per boundary config: ∏ t(G[Dᵢ]) = {product_size:,}")
print(f"  → A chain on boundary configs has state space {t_w_Q}, not {det_L:,}")
print()

# How many cycle basis steps actually change the partition?
print(f"From MEW simulation:")
print(f"  {partition_changes}/{NUM_STEPS} steps changed boundary edge count")
print(f"  = {partition_changes/NUM_STEPS*100:.1f}% of cycle basis steps are 'useful'")
print(f"  = {(NUM_STEPS-partition_changes)/NUM_STEPS*100:.1f}% are within-fiber shuffling")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# 11. VISUALIZE
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: The partition
ax = axes[0]
colors_map = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange'}
node_colors = [colors_map[assignment[node]] for node in range(n)]
pos = grid_pos
nx.draw(G, pos=pos, node_color=node_colors, node_size=200,
        node_shape='s', edge_color='gray', alpha=0.7,
        with_labels=True, font_size=6, ax=ax)
ax.set_title("Partition ξ (4 quadrants of 3×3)", fontsize=12)

# Plot 2: Distribution of boundary edges in random trees
ax = axes[1]
x = sorted(boundary_edge_counts.keys())
y = [boundary_edge_counts[k_val] for k_val in x]
ax.bar(x, y, color='steelblue', alpha=0.8)
ax.set_xlabel("Number of boundary edges in tree")
ax.set_ylabel("Count (out of 5000 samples)")
ax.set_title("Boundary edges in random spanning trees", fontsize=12)
ax.axvline(x=3, color='red', linestyle='--', label='k-1=3 (needed for ξ)')
ax.legend()

# Plot 3: Tree discovery rate
ax = axes[2]
steps = [s for s, c in new_tree_timeline]
counts = [c for s, c in new_tree_timeline]
ax.plot(steps, counts, 'o-', color='darkgreen', linewidth=2)
ax.set_xlabel("MEW cycle basis steps")
ax.set_ylabel("Unique trees discovered")
ax.set_title("Tree discovery rate (cycle basis walk)", fontsize=12)
ax.axhline(y=det_L, color='red', linestyle='--', alpha=0.5,
           label=f'|T(G)| = {det_L:,}')
ax.legend()

plt.tight_layout()
plt.savefig("/Users/kirtisoglu/GitHub/Redistricting_Diffusion_Model/plots/picard_experiment.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Plot saved to plots/picard_experiment.png")
