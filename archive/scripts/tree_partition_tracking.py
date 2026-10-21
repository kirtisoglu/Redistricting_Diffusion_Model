"""
Track quotient trees and partitions through MEW steps on 6x6 grid.

Classify each step:
  A) Quotient tree changed AND partition changed  (boundary jump)
  B) Quotient tree same   AND partition changed   (local adjustment via M)
  C) Quotient tree changed AND partition same      (boundary move, same partition)
  D) Quotient tree same   AND partition same       (no change)
"""

import numpy as np
import networkx as nx
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

random.seed(42)
np.random.seed(42)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def wilson_ust(G):
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


def make_adj(edges, nodes):
    adj = {n: [] for n in nodes}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def bfs_path(adj, u, v):
    parent = {u: None}
    queue = [u]
    i = 0
    while i < len(queue):
        nd = queue[i]; i += 1
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


def get_partition(T, M, nodes):
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


def partition_key(asn, nodes):
    groups = defaultdict(set)
    for n in sorted(nodes):
        groups[asn[n]].add(n)
    return frozenset(frozenset(s) for s in groups.values())


def quotient_tree_key(T, asn):
    return frozenset(e for e in T if asn[e[0]] != asn[e[1]])


def full_mew_step(T, M, all_e, nodes):
    non_tree = list(all_e - T)
    if not non_tree:
        return T, M
    e1 = random.choice(non_tree)
    adj = make_adj(T, nodes)
    path = bfs_path(adj, e1[0], e1[1])
    cycle = [(min(path[i], path[i + 1]), max(path[i], path[i + 1]))
             for i in range(len(path) - 1)]
    eligible = [e for e in cycle if e not in M]
    if not eligible:
        return T, M
    e2 = random.choice(eligible)
    T_new = (T | {e1}) - {e2}

    M_new = set(M)
    if M_new:
        m = random.choice(list(M_new))
        m_tuple = tuple(sorted(m))
        u = random.choice(list(m_tuple))
        adj_new = make_adj(T_new, nodes)
        nbrs = adj_new[u]
        if nbrs:
            v = random.choice(nbrs)
            M_new.discard(m)
            M_new.add((min(u, v), max(u, v)))

    return T_new, M_new


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

N = 6
G = nx.grid_2d_graph(N, N)
G = nx.convert_node_labels_to_integers(G)
n = G.number_of_nodes()
k = 4

init_asn = {}
half = N // 2
for nd in range(n):
    r, c = nd // N, nd % N
    init_asn[nd] = (0 if r < half else 2) + (0 if c < half else 1)

all_e = set((min(u, v), max(u, v)) for u, v in G.edges())
nodes = set(range(n))

T, M = sample_compatible(G, init_asn, k)
asn = get_partition(T, M, nodes)
prev_qt = quotient_tree_key(T, asn)
prev_pk = partition_key(asn, nodes)

num_steps = 50000

# Counters for the four cases
case_A = 0  # QT changed, partition changed  (boundary jump)
case_B = 0  # QT same,    partition changed  (local adjustment)
case_C = 0  # QT changed, partition same     (boundary move, same partition)
case_D = 0  # QT same,    partition same     (no change)

# Track unique partitions discovered by each case
partitions_from_A = set()
partitions_from_B = set()

# Track new partitions (first time seen)
all_seen_partitions = {prev_pk}
new_from_A = 0
new_from_B = 0
new_from_D = 0  # should be 0

for step in range(num_steps):
    T, M = full_mew_step(T, M, all_e, nodes)
    asn = get_partition(T, M, nodes)

    qt = quotient_tree_key(T, asn)
    pk = partition_key(asn, nodes)

    qt_changed = (qt != prev_qt)
    pk_changed = (pk != prev_pk)
    is_new = pk not in all_seen_partitions

    if qt_changed and pk_changed:
        case_A += 1
        partitions_from_A.add(pk)
        if is_new:
            new_from_A += 1
    elif not qt_changed and pk_changed:
        case_B += 1
        partitions_from_B.add(pk)
        if is_new:
            new_from_B += 1
    elif qt_changed and not pk_changed:
        case_C += 1
    else:
        case_D += 1

    all_seen_partitions.add(pk)
    prev_qt = qt
    prev_pk = pk


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

total = case_A + case_B + case_C + case_D
total_partitions = len(all_seen_partitions)

print("=" * 75)
print(f"6x6 grid, k=4, {num_steps} MEW steps (cycle + marked edge)")
print("=" * 75)
print()
print("STEP CLASSIFICATION")
print("-" * 75)
print(f"{'Case':>5}  {'QT':>10}  {'Partition':>10}  {'Count':>8}  {'%':>8}  {'Description'}")
print("-" * 75)
print(f"{'A':>5}  {'changed':>10}  {'changed':>10}  {case_A:>8}  {case_A/total*100:>7.1f}%  boundary jump")
print(f"{'B':>5}  {'same':>10}  {'changed':>10}  {case_B:>8}  {case_B/total*100:>7.1f}%  local adjustment (M)")
print(f"{'C':>5}  {'changed':>10}  {'same':>10}  {case_C:>8}  {case_C/total*100:>7.1f}%  boundary move, same partition")
print(f"{'D':>5}  {'same':>10}  {'same':>10}  {case_D:>8}  {case_D/total*100:>7.1f}%  no change")
print("-" * 75)
print(f"{'Total':>5}  {'':>10}  {'':>10}  {total:>8}  {'100.0':>7}%")
print()

print("PARTITION DISCOVERY")
print("-" * 75)
print(f"Total distinct partitions discovered: {total_partitions}")
print(f"  New partitions from A (boundary jump):      {new_from_A} ({new_from_A/max(total_partitions,1)*100:.1f}%)")
print(f"  New partitions from B (local adjustment):   {new_from_B} ({new_from_B/max(total_partitions,1)*100:.1f}%)")
print(f"  Initial partition:                           1")
print()
print(f"Unique partitions ever visited via A: {len(partitions_from_A)}")
print(f"Unique partitions ever visited via B: {len(partitions_from_B)}")
print(f"Overlap (visited via both A and B):   {len(partitions_from_A & partitions_from_B)}")
print()

# Partition changes that are "useful" vs "wasted"
total_changes = case_A + case_B
print("AMONG STEPS THAT CHANGE THE PARTITION:")
print("-" * 75)
if total_changes > 0:
    print(f"  Via boundary jump (A):     {case_A:>8}  ({case_A/total_changes*100:.1f}% of changes)")
    print(f"  Via local adjustment (B):  {case_B:>8}  ({case_B/total_changes*100:.1f}% of changes)")
print()

total_no_change = case_C + case_D
print("AMONG STEPS THAT DO NOT CHANGE THE PARTITION:")
print("-" * 75)
if total_no_change > 0:
    print(f"  QT changed but same partition (C):  {case_C:>8}  ({case_C/total_no_change*100:.1f}% of non-changes)")
    print(f"  Nothing changed (D):                {case_D:>8}  ({case_D/total_no_change*100:.1f}% of non-changes)")

print()
print("=" * 75)

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Step classification pie chart
labels = [
    f'A: boundary jump\n({case_A/total*100:.1f}%)',
    f'B: local adjust\n({case_B/total*100:.1f}%)',
    f'C: QT change,\nsame part. ({case_C/total*100:.1f}%)',
    f'D: no change\n({case_D/total*100:.1f}%)',
]
sizes = [case_A, case_B, case_C, case_D]
colors = ['#e74c3c', '#f39c12', '#3498db', '#bdc3c7']
axes[0].pie(sizes, labels=labels, colors=colors, startangle=90,
            textprops={'fontsize': 9})
axes[0].set_title('Step classification (50,000 MEW steps)', fontsize=11)

# Plot 2: Partition discovery source
labels2 = [
    f'Boundary jump (A)\n{new_from_A} ({new_from_A/max(total_partitions,1)*100:.1f}%)',
    f'Local adjustment (B)\n{new_from_B} ({new_from_B/max(total_partitions,1)*100:.1f}%)',
    f'Initial\n1',
]
sizes2 = [new_from_A, new_from_B, 1]
colors2 = ['#e74c3c', '#f39c12', '#2ecc71']
axes[1].pie(sizes2, labels=labels2, colors=colors2, startangle=90,
            textprops={'fontsize': 9})
axes[1].set_title('How new partitions were discovered', fontsize=11)

plt.tight_layout()
plt.savefig('plots/mew/step_classification.png', dpi=150, bbox_inches='tight')
print("Plot saved to plots/mew/step_classification.png")
