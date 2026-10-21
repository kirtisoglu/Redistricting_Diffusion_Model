"""
mew_model.py — Marked Edge Walk (MEW) for redistricting
========================================================

State space:  X = {(T, M) : T spanning tree of G, M ⊆ T}
Forest F = T \\ M gives a balanced k-partition (each component is a district).

Two-step transition (Section 2 of the paper):
  1. Cycle basis step — swap a non-tree edge into the tree
  2. Marked edge step — slide one marked edge along the tree

Metropolis–Hastings correction uses:
  • Transition ratio  P(x|x')/P(x'|x)  from Eq. 5
  • Target  p(x) ∝ exp(J(ξ(x))) / τ(ξ(x))
  • Degeneracy factor  τ(ξ) = Π t(ξ_i) · Π |∂(ξ_i, ξ_j)|   (Eq. 8)

Reference
---------
McWhorter & DeFord, "The Marked Edge Walk: A Novel MCMC Algorithm
for Sampling of Graph Partitions" (arXiv:2510.17714, 2025).
"""

import random
import numpy as np
import networkx as nx
from collections import deque


# ═══════════════════════════════════════════════════════════════════════════════
#  MEWState
# ═══════════════════════════════════════════════════════════════════════════════

class MEWState:
    """
    Lifted state  x = (T, M)  for the Marked Edge Walk.

    Parameters
    ----------
    graph : nx.Graph
        Underlying graph G (read-only throughout the walk).
    tree  : nx.Graph
        Spanning tree T of G.
    marked : set of frozenset
        Marked edges M ⊆ E(T).  |M| = k − 1 for a k-partition.
    """

    __slots__ = ("graph", "tree", "marked")

    def __init__(self, graph, tree, marked):
        self.graph  = graph
        self.tree   = tree
        self.marked = marked          # set of frozenset pairs

    # ── partition extraction ────────────────────────────────────────────────
    def get_assignment(self):
        """Return dict {node → district_id} from the forest T \\ M."""
        forest = self.tree.copy()
        for e in self.marked:
            forest.remove_edge(*e)
        assignment = {}
        for i, comp in enumerate(nx.connected_components(forest)):
            for node in comp:
                assignment[node] = i
        return assignment

    def copy(self):
        return MEWState(self.graph, self.tree.copy(), set(self.marked))


# ═══════════════════════════════════════════════════════════════════════════════
#  Initialisation
# ═══════════════════════════════════════════════════════════════════════════════

def initialize_from_partition(graph, assignment):
    """
    Build a valid MEW state (T, M) from a partition.

    1. For each district, sample a random spanning tree of its induced subgraph.
    2. Pick a spanning tree of the *quotient* graph (districts as nodes).
    3. For each quotient-tree edge, pick one cross-district edge at random
       and add it to T.  These k−1 edges become the marked set M.

    Returns
    -------
    MEWState
    """
    # group nodes by district
    districts = {}
    for node, d in assignment.items():
        districts.setdefault(d, []).append(node)

    tree = nx.Graph()

    # Step 1: spanning tree per district
    for d_id, nodes in districts.items():
        sub = graph.subgraph(nodes)
        if sub.number_of_nodes() == 1:
            tree.add_node(nodes[0])
        else:
            st = nx.random_spanning_tree(sub)
            tree.add_edges_from(st.edges())

    # Step 2: build quotient graph and find cross-district edges
    cross_edges = {}                          # (min_d, max_d) → [edges]
    quotient    = nx.Graph()
    for u, v in graph.edges():
        d1, d2 = assignment[u], assignment[v]
        if d1 != d2:
            key = (min(d1, d2), max(d1, d2))
            quotient.add_edge(key[0], key[1])
            cross_edges.setdefault(key, []).append((u, v))

    # spanning tree of quotient → which district-pairs to connect
    q_tree = nx.random_spanning_tree(quotient)

    marked = set()
    for d1, d2 in q_tree.edges():
        key  = (min(d1, d2), max(d1, d2))
        edge = random.choice(cross_edges[key])
        tree.add_edge(*edge)
        marked.add(frozenset(edge))

    return MEWState(graph, tree, marked)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _find_path(tree, u, v):
    """BFS path from *u* to *v* in tree (list of nodes)."""
    parent = {u: None}
    queue  = deque([u])
    while queue:
        node = queue.popleft()
        if node == v:
            break
        for nbr in tree.neighbors(node):
            if nbr not in parent:
                parent[nbr] = node
                queue.append(nbr)
    path, cur = [], v
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return path


def _path_edges(path):
    """Convert a node-path to a set of frozenset edges."""
    return {frozenset((path[i], path[i + 1])) for i in range(len(path) - 1)}


# ═══════════════════════════════════════════════════════════════════════════════
#  Degeneracy factor  τ(ξ)   (Eq. 8)
# ═══════════════════════════════════════════════════════════════════════════════

def _log_spanning_tree_count(graph, nodes):
    """
    log t(G[nodes])  via Kirchhoff's matrix-tree theorem.

    t(G) = any cofactor of the Laplacian = det L[1:,1:].
    Uses np.linalg.slogdet for numerical stability.
    """
    n = len(nodes)
    if n <= 1:
        return 0.0
    sub = graph.subgraph(nodes)
    L   = nx.laplacian_matrix(sub).toarray().astype(np.float64)
    sign, logdet = np.linalg.slogdet(L[1:, 1:])
    if sign <= 0:
        return -np.inf                # disconnected subgraph
    return logdet


def compute_log_degeneracy(graph, assignment):
    """
    log τ(ξ) = Σ_i log t(ξ_i)  +  Σ_{i~j} log |∂(ξ_i, ξ_j)|

    Returns −∞ if any district is disconnected.
    """
    districts = {}
    for node, d in assignment.items():
        districts.setdefault(d, []).append(node)

    log_tau = 0.0

    # product of spanning tree counts
    for nodes in districts.values():
        lt = _log_spanning_tree_count(graph, nodes)
        if lt == -np.inf:
            return -np.inf
        log_tau += lt

    # product of border-edge counts
    border = {}
    for u, v in graph.edges():
        d1, d2 = assignment[u], assignment[v]
        if d1 != d2:
            key = (min(d1, d2), max(d1, d2))
            border[key] = border.get(key, 0) + 1

    for cnt in border.values():
        log_tau += np.log(cnt)

    return log_tau


# ═══════════════════════════════════════════════════════════════════════════════
#  MEW proposal  (one full transition: cycle step + marked-edge step)
# ═══════════════════════════════════════════════════════════════════════════════

def mew_proposal(state):
    """
    Propose  x' = (T', M')  from  x = (T, M).

    Returns
    -------
    new_state : MEWState  (or the same state on failure)
    log_trans_ratio : float   log P(x|x') / P(x'|x)  (Eq. 5)
    valid : bool   True if the proposal could be constructed
    """
    graph = state.graph
    T     = state.tree
    M     = state.marked

    # ── 1. Cycle basis step ─────────────────────────────────────────────────
    # Choose e+ uniformly from E(G) \ E(T)
    non_tree = [(u, v) for u, v in graph.edges() if not T.has_edge(u, v)]
    if not non_tree:
        return state, 0.0, False

    e_plus = random.choice(non_tree)
    e_plus_fs = frozenset(e_plus)

    # Unique cycle C in T ∪ {e+}:  tree-path(e+[0], e+[1]) ∪ {e+}
    path        = _find_path(T, e_plus[0], e_plus[1])
    cycle_edges = _path_edges(path)               # tree edges in C

    # Choose e− from C \ M  (marked edges cannot be removed)
    removable = [e for e in cycle_edges if e not in M]
    if not removable:
        return state, 0.0, False

    e_minus = random.choice(removable)

    # T' = (T ∪ {e+}) \ {e−}
    T_new = T.copy()
    T_new.add_edge(*e_plus)
    T_new.remove_edge(*e_minus)

    # ── 2. Marked edge step ─────────────────────────────────────────────────
    M_new = set(M)

    if not M_new:
        return MEWState(graph, T_new, M_new), 0.0, True

    # choose m ∈ M uniformly
    m = random.choice(list(M_new))
    m_nodes = list(m)

    # choose endpoint u ∈ m uniformly
    u = random.choice(m_nodes)
    v_other = m_nodes[0] if m_nodes[1] == u else m_nodes[1]   # other endpoint

    # choose neighbour v ∈ N_{T'}(u) uniformly
    nbrs = list(T_new.neighbors(u))
    if not nbrs:
        return state, 0.0, False
    v = random.choice(nbrs)

    m_prime = frozenset((u, v))

    # update M
    M_new.discard(m)
    M_new.add(m_prime)

    # ── 3. Transition ratio  log P(x|x') / P(x'|x)   (Eq. 5) ──────────────

    # indicator: if m' = e+  →  reverse tree-step impossible → reject
    if m_prime == e_plus_fs:
        return state, 0.0, False

    # |C \ M|  (forward)
    len_C_M = len(removable)

    # |C' \ M'|  (reverse cycle: C' in T' ∪ {e−})
    e_minus_tuple = tuple(e_minus)
    path_rev        = _find_path(T_new, e_minus_tuple[0], e_minus_tuple[1])
    cycle_edges_rev = _path_edges(path_rev)
    removable_rev   = [e for e in cycle_edges_rev if e not in M_new]
    len_C_M_rev     = len(removable_rev)

    if len_C_M_rev == 0:
        return state, 0.0, False

    # tree degrees
    deg_T_u     = T.degree(u)
    deg_T_new_u = T_new.degree(u)
    if deg_T_u == 0 or deg_T_new_u == 0:
        return state, 0.0, False

    # log ratio  (Eq. 5)
    log_ratio = (np.log(len_C_M) - np.log(len_C_M_rev)
                 + np.log(deg_T_new_u) - np.log(deg_T_u))

    # extra correction when m = m'  (marked edge didn't move)  — Eq. 4
    if m == m_prime:
        deg_T_v     = T.degree(v_other)
        deg_T_new_v = T_new.degree(v_other)
        if deg_T_v == 0 or deg_T_new_v == 0:
            return state, 0.0, False
        log_ratio += (np.log(deg_T_u + deg_T_v)
                      - np.log(deg_T_new_u + deg_T_new_v)
                      + np.log(deg_T_new_v) - np.log(deg_T_v))

    return MEWState(graph, T_new, M_new), log_ratio, True


# ═══════════════════════════════════════════════════════════════════════════════
#  Full MH step
# ═══════════════════════════════════════════════════════════════════════════════

def mew_step(state, energy_fn, pop_target, epsilon):
    """
    One MEW step with Metropolis–Hastings acceptance.

    Target distribution:  p(x) ∝ exp(J(ξ(x))) / τ(ξ(x))

    Parameters
    ----------
    state      : MEWState
    energy_fn  : callable(graph, assignment) → float   (= J(ξ))
    pop_target : float   ideal total population per district
    epsilon    : float   fractional population tolerance

    Returns
    -------
    (new_state, accepted : bool)
    """
    # propose
    new_state, log_trans, valid = mew_proposal(state)
    if not valid:
        return state, False

    new_asn = new_state.get_assignment()

    # ── population balance check ────────────────────────────────────────────
    old_asn   = state.get_assignment()
    n_dist    = len(set(old_asn.values()))
    dist_ids  = set(new_asn.values())

    if len(dist_ids) != n_dist:
        return state, False

    pops = {}
    for node, d in new_asn.items():
        pops[d] = pops.get(d, 0) + state.graph.nodes[node]["population"]
    for pop in pops.values():
        if abs(pop - pop_target) / pop_target > epsilon:
            return state, False

    # ── connectivity check ──────────────────────────────────────────────────
    # Forest T'\M' components are connected by construction (they are
    # subtrees), so no extra check is needed.

    # ── MH acceptance ───────────────────────────────────────────────────────
    J_old = energy_fn(state.graph, old_asn)
    J_new = energy_fn(state.graph, new_asn)

    log_tau_old = compute_log_degeneracy(state.graph, old_asn)
    log_tau_new = compute_log_degeneracy(state.graph, new_asn)

    # log α = (J_new − J_old) + (log τ_old − log τ_new) + log_trans
    log_alpha = (J_new - J_old) + (log_tau_old - log_tau_new) + log_trans

    if np.log(random.random()) < log_alpha:
        return new_state, True
    else:
        return state, False


def mew_wilson_step(state, energy_fn, pop_target, epsilon):
    """
    Plain MEW step with Wilson tree resampling and NO degeneracy τ.

    Same proposal as mew_step, but:
      1. Resample T uniformly given ξ before proposing
      2. Drop τ from MH ratio

    Target: p(T,M) ∝ exp(J(ξ)), inducing π(ξ) ∝ τ(ξ)·exp(J(ξ)).
    """
    from mew.qp_proposal import resample_tree

    # Resample T given current partition (Gibbs step)
    old_asn = state.get_assignment()
    k = len(set(old_asn.values()))
    T_new, M_new = resample_tree(state.graph, old_asn, k)
    state = MEWState(state.graph, T_new, M_new)

    # Standard MEW proposal on fresh tree
    new_state, log_trans, valid = mew_proposal(state)
    if not valid:
        return state, False

    new_asn = new_state.get_assignment()
    n_dist = len(set(old_asn.values()))

    if len(set(new_asn.values())) != n_dist:
        return state, False

    pops = {}
    for node, d in new_asn.items():
        pops[d] = pops.get(d, 0) + state.graph.nodes[node]["population"]
    for pop in pops.values():
        if abs(pop - pop_target) / pop_target > epsilon:
            return state, False

    # MH acceptance — no τ
    J_old = energy_fn(state.graph, old_asn)
    J_new = energy_fn(state.graph, new_asn)

    log_alpha = (J_new - J_old) + log_trans
    if np.log(random.random()) < log_alpha:
        return new_state, True
    return state, False
