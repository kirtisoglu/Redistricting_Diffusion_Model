"""
Acceptance rate scaling: merge-and-resplit proposals.

Compare three approaches on NxN grids with k=4 quadrant districts:
  1. Vanilla MEW: cycle basis step + fixed M (proper mechanics with rejection)
  2. Merge-resplit + random cut: Wilson's on merged district pair, random edge cut
  3. Merge-resplit + QP cut: Wilson's on merged district pair, QP-guided edge cut

The hypothesis: merge-resplit produces genuinely new tree structures where
balanced cuts exist, and QP finds them reliably.
"""

import numpy as np
import networkx as nx
import random
import time
from collections import defaultdict

random.seed(42)
np.random.seed(42)


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def wilson_random_spanning_tree(G):
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


def get_components(T_edges, removed_edges, all_nodes):
    """Get connected components of T with some edges removed."""
    F = nx.Graph()
    F.add_nodes_from(all_nodes)
    F.add_edges_from(set(T_edges) - set(removed_edges))
    return list(nx.connected_components(F))


def check_balance(components, populations, pbar, epsilon):
    """Check if all components are within population tolerance."""
    if len(components) != 4:
        return False
    for comp in components:
        pop = sum(populations[v] for v in comp)
        if abs(pop - pbar) > epsilon * pbar:
            return False
    return True


def subtree_pops_from_root(T_edges, root, all_nodes, populations):
    """BFS + bottom-up pass to compute subtree populations."""
    T = nx.Graph()
    T.add_nodes_from(all_nodes)
    T.add_edges_from(T_edges)

    parent = {root: None}
    children = defaultdict(list)
    visited = {root}
    queue = [root]
    order = [root]
    while queue:
        v = queue.pop(0)
        for w in T.neighbors(v):
            if w not in visited:
                visited.add(w)
                parent[w] = v
                children[v].append(w)
                queue.append(w)
                order.append(w)

    sigma = {}
    for v in reversed(order):
        sigma[v] = populations[v]
        for c in children[v]:
            sigma[v] += sigma[c]

    return sigma, parent


# ═══════════════════════════════════════════════════════════════════════════════
# 1. VANILLA MEW (proper mechanics with rejection)
# ═══════════════════════════════════════════════════════════════════════════════

def sample_compatible_tree(G, districts, assignment):
    """Sample a spanning tree compatible with partition."""
    tree_edges = set()
    for d in sorted(districts):
        subg = G.subgraph(districts[d])
        sub_tree = wilson_random_spanning_tree(subg)
        tree_edges |= sub_tree

    quotient_edges = defaultdict(list)
    for u, v in G.edges():
        du, dv = assignment[u], assignment[v]
        if du != dv:
            key = (min(du, dv), max(du, dv))
            quotient_edges[key].append((min(u, v), max(u, v)))

    q_graph = nx.Graph()
    q_graph.add_nodes_from(range(4))
    for (d1, d2) in quotient_edges:
        q_graph.add_edge(d1, d2)
    q_tree = wilson_random_spanning_tree(q_graph)

    marked = set()
    for d1, d2 in q_tree:
        key = (min(d1, d2), max(d1, d2))
        edge = random.choice(quotient_edges[key])
        tree_edges.add(edge)
        marked.add(edge)

    return tree_edges, marked


def mew_step(T_edges, M, all_edges_set, all_nodes, populations, pbar, epsilon):
    """
    One full MEW step with rejection.
    Returns (T_new, M, accepted).
    """
    tree_edges_set = set(T_edges)
    non_tree_edges = list(all_edges_set - tree_edges_set)
    if not non_tree_edges:
        return T_edges, M, False

    e_plus = random.choice(non_tree_edges)
    u, v = e_plus

    T = nx.Graph()
    T.add_nodes_from(all_nodes)
    T.add_edges_from(tree_edges_set)

    try:
        path = nx.shortest_path(T, u, v)
    except nx.NetworkXNoPath:
        return T_edges, M, False

    cycle_tree_edges = []
    for i in range(len(path) - 1):
        a, b = path[i], path[i+1]
        cycle_tree_edges.append((min(a, b), max(a, b)))

    eligible = [e for e in cycle_tree_edges if e not in M]
    if not eligible:
        return T_edges, M, False

    e_minus = random.choice(eligible)
    T_new = (tree_edges_set | {e_plus}) - {e_minus}

    # Check if partition changed
    comps = get_components(T_new, M, all_nodes)
    if check_balance(comps, populations, pbar, epsilon):
        return T_new, M, True
    else:
        # Reject: stay at current state
        return T_edges, M, False


def run_mew(G, districts, assignment, all_edges_set, all_nodes,
            populations, pbar, epsilon, num_steps):
    """Run vanilla MEW and count accepted steps."""
    T, M = sample_compatible_tree(G, districts, assignment)
    accepted = 0
    for _ in range(num_steps):
        T, M, acc = mew_step(T, M, all_edges_set, all_nodes,
                             populations, pbar, epsilon)
        if acc:
            accepted += 1
    return accepted


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MERGE-RESPLIT + RANDOM CUT
# ═══════════════════════════════════════════════════════════════════════════════

def merge_resplit_random(G, current_districts, all_nodes, populations,
                         pbar, epsilon):
    """
    Merge two adjacent districts, resample tree via Wilson's,
    pick a random edge to cut.
    Returns (new_districts, feasible).
    """
    # Find adjacent district pairs
    adj_pairs = []
    for u, v in G.edges():
        du = current_districts.get(u)
        dv = current_districts.get(v)
        if du is not None and dv is not None and du != dv:
            pair = (min(du, dv), max(du, dv))
            if pair not in adj_pairs:
                adj_pairs.append(pair)

    if not adj_pairs:
        return current_districts, False

    # Pick random adjacent pair
    di, dj = random.choice(adj_pairs)

    # Get nodes in each district
    nodes_i = [v for v, d in current_districts.items() if d == di]
    nodes_j = [v for v, d in current_districts.items() if d == dj]
    merged_nodes = set(nodes_i + nodes_j)

    # Sample spanning tree of merged subgraph
    subg = G.subgraph(merged_nodes)
    if not nx.is_connected(subg):
        return current_districts, False

    T_merged = wilson_random_spanning_tree(subg)

    # Pick a random edge to cut
    edges_list = list(T_merged)
    cut_edge = random.choice(edges_list)

    # Check resulting components
    T_graph = nx.Graph()
    T_graph.add_nodes_from(merged_nodes)
    T_graph.add_edges_from(T_merged - {cut_edge})
    comps = list(nx.connected_components(T_graph))

    if len(comps) != 2:
        return current_districts, False

    # Check balance of the two new districts
    for comp in comps:
        pop = sum(populations[v] for v in comp)
        if abs(pop - pbar) > epsilon * pbar:
            return current_districts, False

    # Build new assignment
    new_districts = dict(current_districts)
    for v in comps[0]:
        new_districts[v] = di
    for v in comps[1]:
        new_districts[v] = dj

    return new_districts, True


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MERGE-RESPLIT + QP CUT
# ═══════════════════════════════════════════════════════════════════════════════

def merge_resplit_qp(G, current_districts, all_nodes, populations,
                     pbar, epsilon):
    """
    Merge two adjacent districts, resample tree via Wilson's,
    use QP scoring to find the best balanced cut.
    Returns (new_districts, feasible).
    """
    adj_pairs = []
    for u, v in G.edges():
        du = current_districts.get(u)
        dv = current_districts.get(v)
        if du is not None and dv is not None and du != dv:
            pair = (min(du, dv), max(du, dv))
            if pair not in adj_pairs:
                adj_pairs.append(pair)

    if not adj_pairs:
        return current_districts, False

    di, dj = random.choice(adj_pairs)
    nodes_i = [v for v, d in current_districts.items() if d == di]
    nodes_j = [v for v, d in current_districts.items() if d == dj]
    merged_nodes = set(nodes_i + nodes_j)

    subg = G.subgraph(merged_nodes)
    if not nx.is_connected(subg):
        return current_districts, False

    T_merged = wilson_random_spanning_tree(subg)

    # QP scoring: find the edge whose removal gives the most balanced split
    root = random.choice(list(merged_nodes))
    sigma, parent = subtree_pops_from_root(T_merged, root, merged_nodes, populations)

    # Score each edge by |sigma_v - pbar| (lower = more balanced cut)
    best_edge = None
    best_dev = float('inf')
    for v in merged_nodes:
        if v == root or v not in parent:
            continue
        p = parent[v]
        edge = (min(v, p), max(v, p))
        dev = abs(sigma[v] - pbar)
        if dev < best_dev:
            best_dev = dev
            best_edge = edge

    if best_edge is None:
        return current_districts, False

    # Cut the best edge
    T_graph = nx.Graph()
    T_graph.add_nodes_from(merged_nodes)
    T_graph.add_edges_from(T_merged - {best_edge})
    comps = list(nx.connected_components(T_graph))

    if len(comps) != 2:
        return current_districts, False

    for comp in comps:
        pop = sum(populations[v] for v in comp)
        if abs(pop - pbar) > epsilon * pbar:
            return current_districts, False

    new_districts = dict(current_districts)
    for v in comps[0]:
        new_districts[v] = di
    for v in comps[1]:
        new_districts[v] = dj

    return new_districts, True


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MERGE-RESPLIT + EXHAUSTIVE SEARCH (count ALL balanced cuts)
# ═══════════════════════════════════════════════════════════════════════════════

def count_balanced_cuts(G, current_districts, populations, pbar, epsilon,
                        num_samples=200):
    """
    For each of num_samples merge-resplit proposals, count how many edges
    in the resampled tree give a balanced cut. This tells us how "easy" it
    is to find a balanced cut on a resampled tree.
    """
    adj_pairs = []
    for u, v in G.edges():
        du = current_districts.get(u)
        dv = current_districts.get(v)
        if du is not None and dv is not None and du != dv:
            pair = (min(du, dv), max(du, dv))
            if pair not in adj_pairs:
                adj_pairs.append(pair)

    if not adj_pairs:
        return []

    results = []
    for _ in range(num_samples):
        di, dj = random.choice(adj_pairs)
        nodes_i = [v for v, d in current_districts.items() if d == di]
        nodes_j = [v for v, d in current_districts.items() if d == dj]
        merged_nodes = set(nodes_i + nodes_j)

        subg = G.subgraph(merged_nodes)
        if not nx.is_connected(subg):
            continue

        T_merged = wilson_random_spanning_tree(subg)
        n_merged = len(merged_nodes)

        # Check every edge
        root = random.choice(list(merged_nodes))
        sigma, parent = subtree_pops_from_root(T_merged, root, merged_nodes,
                                                populations)

        total_edges = 0
        balanced_edges = 0
        for v in merged_nodes:
            if v == root or v not in parent:
                continue
            total_edges += 1
            if abs(sigma[v] - pbar) <= epsilon * pbar:
                balanced_edges += 1

        results.append((total_edges, balanced_edges))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# RUN EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_experiment(N, num_steps, epsilon=0.10):
    G = nx.grid_2d_graph(N, N)
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    k = 4
    pbar = n / k

    populations = {v: 1.0 for v in range(n)}

    half = N // 2
    districts = {0: [], 1: [], 2: [], 3: []}
    assignment = {}
    for node in range(n):
        row = node // N
        col = node % N
        if row < half and col < half:
            d = 0
        elif row < half and col >= half:
            d = 1
        elif row >= half and col < half:
            d = 2
        else:
            d = 3
        districts[d].append(node)
        assignment[node] = d

    all_edges_set = set((min(u, v), max(u, v)) for u, v in G.edges())
    all_nodes = set(range(n))

    # --- 1. Vanilla MEW ---
    t0 = time.time()
    mew_accepted = run_mew(G, districts, assignment, all_edges_set,
                           all_nodes, populations, pbar, epsilon, num_steps)
    t_mew = time.time() - t0

    # --- 2. Merge-resplit + random cut ---
    t0 = time.time()
    mr_random_accepted = 0
    current_d = dict(assignment)
    for _ in range(num_steps):
        new_d, feasible = merge_resplit_random(G, current_d, all_nodes,
                                               populations, pbar, epsilon)
        if feasible:
            mr_random_accepted += 1
            current_d = new_d

    t_random = time.time() - t0

    # --- 3. Merge-resplit + QP cut ---
    t0 = time.time()
    mr_qp_accepted = 0
    current_d = dict(assignment)
    for _ in range(num_steps):
        new_d, feasible = merge_resplit_qp(G, current_d, all_nodes,
                                           populations, pbar, epsilon)
        if feasible:
            mr_qp_accepted += 1
            current_d = new_d

    t_qp = time.time() - t0

    # --- 4. Count balanced cuts available ---
    t0 = time.time()
    cut_data = count_balanced_cuts(G, assignment, populations, pbar, epsilon,
                                   num_samples=500)
    t_count = time.time() - t0

    avg_total = np.mean([t for t, b in cut_data]) if cut_data else 0
    avg_balanced = np.mean([b for t, b in cut_data]) if cut_data else 0
    frac_balanced = avg_balanced / avg_total if avg_total > 0 else 0

    return {
        'N': N, 'n': n,
        'mew_accepted': mew_accepted,
        'mew_rate': mew_accepted / num_steps,
        'mr_random_accepted': mr_random_accepted,
        'mr_random_rate': mr_random_accepted / num_steps,
        'mr_qp_accepted': mr_qp_accepted,
        'mr_qp_rate': mr_qp_accepted / num_steps,
        'avg_tree_edges': avg_total,
        'avg_balanced_cuts': avg_balanced,
        'frac_balanced_cuts': frac_balanced,
        't_mew': t_mew,
        't_random': t_random,
        't_qp': t_qp,
    }


print("=" * 90)
print("ACCEPTANCE RATE SCALING: MEW vs MERGE-RESPLIT (random cut vs QP cut)")
print("Uniform populations, epsilon = 0.10, k = 4 quadrant districts")
print("=" * 90)
print()

grid_sizes = [6, 10, 14, 20, 24, 30]
num_steps = 5000
results = []

for N in grid_sizes:
    print(f"Running N={N} ({N}x{N} = {N*N} nodes)...", flush=True)
    res = run_full_experiment(N, num_steps, epsilon=0.10)
    results.append(res)
    print(f"  MEW:           {res['mew_rate']*100:>6.2f}% accepted  ({res['t_mew']:.1f}s)")
    print(f"  MR+random cut: {res['mr_random_rate']*100:>6.2f}% accepted  ({res['t_random']:.1f}s)")
    print(f"  MR+QP cut:     {res['mr_qp_rate']*100:>6.2f}% accepted  ({res['t_qp']:.1f}s)")
    print(f"  Balanced cuts available: {res['avg_balanced_cuts']:.1f} / {res['avg_tree_edges']:.0f} "
          f"edges ({res['frac_balanced_cuts']*100:.1f}%)")
    print()

print("=" * 90)
print("SUMMARY TABLE")
print("-" * 90)
print(f"{'Grid':>7} {'n':>5} "
      f"{'MEW':>8} {'MR+rand':>8} {'MR+QP':>8} "
      f"{'Bal.cuts':>9} {'QP/MEW':>8} {'QP/rand':>8}")
print(f"{'':>7} {'':>5} "
      f"{'acc%':>8} {'acc%':>8} {'acc%':>8} "
      f"{'avail%':>9} {'speedup':>8} {'speedup':>8}")
print("-" * 90)

for r in results:
    qp_vs_mew = r['mr_qp_rate'] / r['mew_rate'] if r['mew_rate'] > 0 else float('inf')
    qp_vs_rand = r['mr_qp_rate'] / r['mr_random_rate'] if r['mr_random_rate'] > 0 else float('inf')
    print(f"{r['N']:>4}x{r['N']:<3} {r['n']:>4} "
          f"{r['mew_rate']*100:>7.2f}% "
          f"{r['mr_random_rate']*100:>7.2f}% "
          f"{r['mr_qp_rate']*100:>7.2f}% "
          f"{r['frac_balanced_cuts']*100:>8.1f}% "
          f"{qp_vs_mew:>7.1f}x "
          f"{qp_vs_rand:>7.1f}x")

print()
print("=" * 90)
print("INTERPRETATION")
print("-" * 90)
print("MEW acc%:      vanilla MEW (cycle basis + fixed M, with rejection)")
print("MR+rand acc%:  merge-resplit with random edge cut")
print("MR+QP acc%:    merge-resplit with QP-guided best-balance cut")
print("Bal.cuts:      % of edges in resampled merged tree giving balanced split")
print("QP/MEW:        acceptance rate improvement of MR+QP over vanilla MEW")
print("QP/rand:       acceptance rate improvement of QP cut over random cut")
