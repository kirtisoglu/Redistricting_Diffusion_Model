"""
Full MEW comparison: vanilla MEW vs boundary cycle + QP marked edge.

Vanilla MEW:
  1. Cycle step: e+ from ALL non-tree edges, e- from C \ M
  2. Marked edge step: slide one marked edge to random tree neighbor
  3. Accept/reject on population balance

Our algorithm:
  1. Boundary cycle step: e+ from BOUNDARY non-tree edges only
  2. QP marked edge step: select k-1 edges minimizing |sigma_e - pbar|
  3. Accept/reject on population balance
"""

import numpy as np
import networkx as nx
import random
import time
from collections import defaultdict

random.seed(42)
np.random.seed(42)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def wilson_ust(G):
    """Uniform spanning tree via Wilson's algorithm."""
    nodes = list(G.nodes())
    in_tree = {nodes[0]}
    edges = set()
    for start in nodes[1:]:
        if start in in_tree:
            continue
        path = [start]
        current = start
        visited = {start: 0}
        while current not in in_tree:
            nbrs = list(G.neighbors(current))
            current = random.choice(nbrs)
            if current in visited:
                idx = visited[current]
                for nd in path[idx + 1:]:
                    del visited[nd]
                path = path[:idx + 1]
            else:
                visited[current] = len(path)
                path.append(current)
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edges.add((min(u, v), max(u, v)))
            in_tree.add(path[i])
    return edges


def sample_compatible(G, assignment, k):
    """Sample (T, M) compatible with partition."""
    districts = defaultdict(list)
    for node, d in assignment.items():
        districts[d].append(node)

    tree = set()
    for d in range(k):
        sub = G.subgraph(districts[d])
        tree |= wilson_ust(sub)

    cross = defaultdict(list)
    for u, v in G.edges():
        du, dv = assignment[u], assignment[v]
        if du != dv:
            key = (min(du, dv), max(du, dv))
            cross[key].append((min(u, v), max(u, v)))

    Q = nx.Graph()
    Q.add_nodes_from(range(k))
    for (d1, d2) in cross:
        Q.add_edge(d1, d2)
    qtree = wilson_ust(Q)

    marked = set()
    for d1, d2 in qtree:
        key = (min(d1, d2), max(d1, d2))
        e = random.choice(cross[key])
        tree.add(e)
        marked.add(e)

    return tree, marked


def bfs_path(adj, u, v):
    """BFS shortest path."""
    parent = {u: None}
    queue = [u]
    i = 0
    while i < len(queue):
        nd = queue[i]
        i += 1
        if nd == v:
            break
        for nbr in adj[nd]:
            if nbr not in parent:
                parent[nbr] = nd
                queue.append(nbr)
    path = []
    c = v
    while c is not None:
        path.append(c)
        c = parent.get(c)
    path.reverse()
    return path


def make_adj(edges, nodes):
    adj = {n: [] for n in nodes}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def get_partition(T, M, nodes):
    """Assignment dict from T \\ M."""
    forest = set(T) - set(M)
    adj = make_adj(forest, nodes)
    asn = {}
    did = 0
    visited = set()
    for s in sorted(nodes):
        if s in visited:
            continue
        stack = [s]
        visited.add(s)
        while stack:
            nd = stack.pop()
            asn[nd] = did
            for nbr in adj[nd]:
                if nbr not in visited:
                    visited.add(nbr)
                    stack.append(nbr)
        did += 1
    return asn


def same_partition(a1, a2, nodes):
    """Check if two assignments are the same partition (up to relabeling)."""
    fwd = {}
    for n in nodes:
        d1, d2 = a1[n], a2[n]
        if d1 in fwd:
            if fwd[d1] != d2:
                return False
        else:
            fwd[d1] = d2
    return True


def pop_balanced(asn, pops, pbar, eps, k):
    dp = defaultdict(float)
    ds = set()
    for n, d in asn.items():
        dp[d] += pops[n]
        ds.add(d)
    if len(ds) != k:
        return False
    return all(abs(p - pbar) / pbar <= eps for p in dp.values())


# ═══════════════════════════════════════════════════════════════════════════════
# MARKED EDGE STEPS
# ═══════════════════════════════════════════════════════════════════════════════

def random_marked_edge_step(T, M, nodes):
    """
    MEW marked edge step: pick a marked edge m, pick endpoint u,
    pick random tree neighbor v of u, slide m to (u, v).
    """
    if not M:
        return M
    m = random.choice(list(M))
    m_nodes = list(m)
    u = random.choice(m_nodes)

    adj = make_adj(T, nodes)
    nbrs = adj[u]
    if not nbrs:
        return M
    v = random.choice(nbrs)

    M_new = set(M)
    M_new.discard(m)
    M_new.add((min(u, v), max(u, v)))
    return M_new


def qp_marked_edge_step(T, M, nodes, pops, pbar, k):
    """
    QP-guided marked edge selection: pick the k-1 tree edges
    whose removal gives the most balanced components.

    For each tree edge, compute subtree population on one side.
    Select k-1 edges with smallest |sigma_e - pbar|.
    """
    adj = make_adj(T, nodes)

    # Root the tree and compute subtree populations
    root = min(nodes)
    parent = {root: None}
    children = defaultdict(list)
    order = [root]
    visited = {root}
    queue = [root]
    while queue:
        nd = queue.pop(0)
        for nbr in adj[nd]:
            if nbr not in visited:
                visited.add(nbr)
                parent[nbr] = nd
                children[nd].append(nbr)
                queue.append(nbr)
                order.append(nbr)

    # Bottom-up subtree populations
    sigma = {}
    for nd in reversed(order):
        sigma[nd] = pops[nd]
        for c in children[nd]:
            sigma[nd] += sigma[c]

    # Score each tree edge by |sigma_child - pbar|
    edge_scores = []
    for nd in order:
        if nd == root:
            continue
        p = parent[nd]
        edge = (min(nd, p), max(nd, p))
        dev = abs(sigma[nd] - pbar)
        edge_scores.append((dev, edge))

    # Select k-1 edges with smallest deviation
    edge_scores.sort(key=lambda x: x[0])
    M_new = set()
    for _, edge in edge_scores[:k - 1]:
        M_new.add(edge)

    return M_new


# ═══════════════════════════════════════════════════════════════════════════════
# FULL STEPS
# ═══════════════════════════════════════════════════════════════════════════════

def vanilla_mew_step(T, M, all_e, nodes, pops, pbar, epsilon, k):
    """
    Full vanilla MEW step:
      1. Cycle step: e+ from all non-tree edges
      2. Marked edge step: random slide
      3. Balance check
    """
    non_tree = list(all_e - T)
    if not non_tree:
        return T, M, False

    e1 = random.choice(non_tree)
    adj = make_adj(T, nodes)
    path = bfs_path(adj, e1[0], e1[1])
    cycle = [(min(path[i], path[i + 1]), max(path[i], path[i + 1]))
             for i in range(len(path) - 1)]
    eligible = [e for e in cycle if e not in M]

    if not eligible:
        return T, M, False

    e2 = random.choice(eligible)
    T_new = (T | {e1}) - {e2}

    # Marked edge step
    M_new = random_marked_edge_step(T_new, M, nodes)

    # Check partition
    new_asn = get_partition(T_new, M_new, nodes)
    if not pop_balanced(new_asn, pops, pbar, epsilon, k):
        return T, M, False

    return T_new, M_new, True


def boundary_qp_step(T, M, all_e, nodes, asn, pops, pbar, epsilon, k):
    """
    Our algorithm:
      1. Boundary cycle step: e+ from boundary non-tree edges
      2. QP marked edge step: select k-1 best-balance edges
      3. Balance check
    """
    non_tree = [e for e in (all_e - T) if asn[e[0]] != asn[e[1]]]
    if not non_tree:
        return T, M, False

    e1 = random.choice(non_tree)
    adj = make_adj(T, nodes)
    path = bfs_path(adj, e1[0], e1[1])
    cycle = [(min(path[i], path[i + 1]), max(path[i], path[i + 1]))
             for i in range(len(path) - 1)]
    eligible = [e for e in cycle if e not in M]

    if not eligible:
        return T, M, False

    e2 = random.choice(eligible)
    T_new = (T | {e1}) - {e2}

    # QP marked edge step
    M_new = qp_marked_edge_step(T_new, M, nodes, pops, pbar, k)

    # Check partition
    new_asn = get_partition(T_new, M_new, nodes)
    if not pop_balanced(new_asn, pops, pbar, epsilon, k):
        return T, M, False

    return T_new, M_new, True


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════

def run(N, num_steps, epsilon=0.10):
    G = nx.grid_2d_graph(N, N)
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()
    k = 4
    pbar = n / k
    pops = {v: 1.0 for v in range(n)}

    half = N // 2
    init_asn = {}
    for nd in range(n):
        r, c = nd // N, nd % N
        init_asn[nd] = (0 if r < half else 2) + (0 if c < half else 1)

    all_e = set((min(u, v), max(u, v)) for u, v in G.edges())
    nodes = set(range(n))

    results = {}

    # --- Vanilla MEW (cycle + random marked edge) ---
    T, M = sample_compatible(G, init_asn, k)
    asn = get_partition(T, M, nodes)

    v_changed = 0
    v_accepted = 0

    t0 = time.time()
    for _ in range(num_steps):
        old_asn = asn
        T_new, M_new, acc = vanilla_mew_step(T, M, all_e, nodes, pops, pbar, epsilon, k)
        if acc:
            v_accepted += 1
            T, M = T_new, M_new
            asn = get_partition(T, M, nodes)
            if not same_partition(old_asn, asn, nodes):
                v_changed += 1
        # If rejected, (T, M, asn) stay the same

    t_vanilla = time.time() - t0

    results['vanilla'] = {
        'accepted': v_accepted,
        'changed': v_changed,
        'accept_rate': v_accepted / num_steps,
        'change_rate': v_changed / num_steps,
        'useful_rate': v_changed / num_steps,  # changed = useful for vanilla
        'time': t_vanilla,
    }

    # --- Boundary cycle + QP marked edge ---
    T, M = sample_compatible(G, init_asn, k)
    asn = get_partition(T, M, nodes)

    b_changed = 0
    b_accepted = 0

    t0 = time.time()
    for _ in range(num_steps):
        old_asn = asn
        T_new, M_new, acc = boundary_qp_step(T, M, all_e, nodes, asn, pops, pbar, epsilon, k)
        if acc:
            b_accepted += 1
            T, M = T_new, M_new
            asn = get_partition(T, M, nodes)
            if not same_partition(old_asn, asn, nodes):
                b_changed += 1

    t_boundary = time.time() - t0

    results['boundary_qp'] = {
        'accepted': b_accepted,
        'changed': b_changed,
        'accept_rate': b_accepted / num_steps,
        'change_rate': b_changed / num_steps,
        'useful_rate': b_changed / num_steps,
        'time': t_boundary,
    }

    return N, n, results


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 85)
print("FULL ALGORITHM: Vanilla MEW vs Boundary Cycle + QP Marked Edge")
print("NxN grids, k=4 quadrants, uniform pop, epsilon=0.10")
print("=" * 85)
print()

grid_sizes = [6, 10, 15, 20, 25, 30]
num_steps = 5000
all_results = []

for N in grid_sizes:
    print(f"Running N={N} ({N}x{N} = {N * N} nodes)...", flush=True)
    N_, n_, res = run(N, num_steps, epsilon=0.10)
    all_results.append((N_, n_, res))
    v, b = res['vanilla'], res['boundary_qp']
    print(f"  Vanilla MEW:      accept={v['accept_rate'] * 100:>6.2f}%  "
          f"changed={v['change_rate'] * 100:>6.2f}%  ({v['time']:.1f}s)")
    print(f"  Boundary + QP:    accept={b['accept_rate'] * 100:>6.2f}%  "
          f"changed={b['change_rate'] * 100:>6.2f}%  ({b['time']:.1f}s)")

print()
print("=" * 85)
print("SUMMARY TABLE")
print("-" * 85)
print(f"{'Grid':>7} {'n':>5} | "
      f"{'Accept':>8} {'Changed':>8} | "
      f"{'Accept':>8} {'Changed':>8} | "
      f"{'Speedup':>8}")
print(f"{'':>7} {'':>5} | "
      f"{'-- Vanilla MEW --':>17} | "
      f"{'-- Boundary+QP --':>17} | "
      f"{'':>8}")
print("-" * 85)

for N, n, res in all_results:
    v, b = res['vanilla'], res['boundary_qp']
    if v['change_rate'] > 0:
        speedup = b['change_rate'] / v['change_rate']
    elif b['change_rate'] > 0:
        speedup = float('inf')
    else:
        speedup = 1.0

    print(f"{N:>4}x{N:<3} {n:>4} | "
          f"{v['accept_rate'] * 100:>7.2f}% {v['change_rate'] * 100:>7.2f}% | "
          f"{b['accept_rate'] * 100:>7.2f}% {b['change_rate'] * 100:>7.2f}% | "
          f"{speedup:>7.1f}x")

print()
print("=" * 85)
print("INTERPRETATION")
print("-" * 85)
print("Accept:  fraction of steps where the proposal was accepted")
print("Changed: fraction of steps where the partition actually changed")
print("         (accepted steps may not change partition if cycle is within-fiber)")
print("Speedup: ratio of changed rates (boundary+QP / vanilla)")
print()
print("Vanilla MEW: cycle step picks any non-tree edge + random marked edge walk")
print("Boundary+QP: cycle step picks boundary edges + QP selects best marked edges")
