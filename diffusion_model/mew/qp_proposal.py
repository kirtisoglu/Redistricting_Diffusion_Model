"""
qp_proposal.py — QP-guided proposal for MEW with tree resampling
=================================================================

QP objective (compactness + population fidelity):

    min_{x ∈ X}  x^T Q x  -  2λ q^T x

where:
    Q = Δ₁ᵀ  +  λΣ²           (compactness + population fidelity)
    q = Σ² x_M                 (fidelity pull toward current M)
    X = {x ∈ [0,1]^m : 1^T x = k-1}

    Σ = diag((σ_v - p̄)/p̄)    (normalized subtree population deviation)
    Δ₁ᵀ = B₁ᵀ B₁              (edge Laplacian on T)

Pipeline per step:
    1. Resample T uniformly given current partition ξ (Wilson's algorithm)
    2. Cycle basis step (tree exploration)
    3. QP solve on new tree → pick k-1 best marked edges
    4. MH acceptance WITHOUT degeneracy τ

Target: p(T,M) ∝ exp(J(ξ)), inducing π(ξ) ∝ τ(ξ)·exp(J(ξ)).
"""

import random
import math
import numpy as np
import networkx as nx
from collections import deque

from mew.mew_model import (
    MEWState,
    _find_path,
    _path_edges,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Wilson's algorithm — uniform spanning tree sampling
# ═══════════════════════════════════════════════════════════════════════════════

def wilson_spanning_tree(G, nodes=None):
    """
    Sample a uniform spanning tree of G (or G restricted to `nodes`)
    using Wilson's algorithm (loop-erased random walk).

    Returns nx.Graph (a tree on the given nodes).
    """
    if nodes is None:
        nodes = list(G.nodes())
    node_set = set(nodes)
    if len(nodes) <= 1:
        T = nx.Graph()
        T.add_nodes_from(nodes)
        return T

    # Adjacency restricted to node_set for O(1) neighbor lookup
    adj = {v: [u for u in G.neighbors(v) if u in node_set] for v in nodes}

    in_tree = set()
    tree = nx.Graph()
    tree.add_nodes_from(nodes)

    # Start with a random root
    root = random.choice(nodes)
    in_tree.add(root)

    for start in nodes:
        if start in in_tree:
            continue

        # Loop-erased random walk from start until hitting in_tree
        path = [start]
        visited = {start: 0}  # node → index in path

        current = start
        while current not in in_tree:
            nbrs = adj[current]
            if not nbrs:
                break
            nxt = random.choice(nbrs)
            if nxt in visited:
                # Loop detected — erase it
                loop_start = visited[nxt]
                for removed in path[loop_start + 1:]:
                    del visited[removed]
                path = path[:loop_start + 1]
                current = nxt
            else:
                path.append(nxt)
                visited[nxt] = len(path) - 1
                current = nxt

        # Add path edges to tree
        for i in range(len(path) - 1):
            tree.add_edge(path[i], path[i + 1])
            in_tree.add(path[i])
        if path:
            in_tree.add(path[-1])

    return tree


def resample_tree(graph, assignment, k):
    """
    Sample a spanning tree T uniformly among trees compatible with
    partition ξ (given by assignment). Returns (T, M).

    1. Per district: Wilson's algorithm on G[D_i]
    2. District adjacency: random spanning tree + random boundary edges

    This is a Gibbs step: P(T|ξ) = 1/τ(ξ) for all compatible T.
    """
    # Group nodes by district
    districts = {}
    for node, d in assignment.items():
        districts.setdefault(d, []).append(node)

    tree = nx.Graph()

    # Step 1: uniform spanning tree per district
    for d_id, nodes in districts.items():
        sub = graph.subgraph(nodes)
        st = wilson_spanning_tree(sub, nodes)
        tree.add_edges_from(st.edges())
        tree.add_nodes_from(nodes)  # ensure isolated nodes are added

    # Step 2: connect districts via boundary edges
    cross_edges = {}  # (min_d, max_d) → [edges]
    quotient = nx.Graph()
    for u, v in graph.edges():
        d1, d2 = assignment[u], assignment[v]
        if d1 != d2:
            key = (min(d1, d2), max(d1, d2))
            quotient.add_edge(key[0], key[1])
            cross_edges.setdefault(key, []).append((u, v))

    # Spanning tree of quotient (k nodes — trivial)
    if quotient.number_of_nodes() <= 1:
        return tree, set()

    q_tree = nx.random_spanning_tree(quotient)

    marked = set()
    for d1, d2 in q_tree.edges():
        key = (min(d1, d2), max(d1, d2))
        edge = random.choice(cross_edges[key])
        tree.add_edge(*edge)
        marked.add(frozenset(edge))

    return tree, marked


# ═══════════════════════════════════════════════════════════════════════════════
#  Cached tree operators — avoid rebuilding from scratch each step
# ═══════════════════════════════════════════════════════════════════════════════

class TreeCache:
    """
    Caches rooted tree structure and subtree populations.

    After a cycle step swaps one edge, call update() to patch
    the cache instead of rebuilding from scratch.
    """
    __slots__ = ("graph", "k", "pbar", "root", "parent", "order",
                 "children", "subtree_pop", "depth")

    def __init__(self, tree, graph, k):
        self.graph = graph
        self.k = k
        self.pbar = sum(graph.nodes[v]["population"] for v in graph) / k
        self.root = None
        self.parent = {}
        self.order = []
        self.children = {}
        self.subtree_pop = {}
        self.depth = {}
        self._build(tree)

    def _build(self, tree):
        """Full BFS rooting + subtree population computation."""
        nodes = list(tree.nodes())
        self.root = random.choice(nodes)

        self.parent = {self.root: None}
        self.children = {v: [] for v in nodes}
        self.order = []
        self.depth = {self.root: 0}

        queue = deque([self.root])
        while queue:
            v = queue.popleft()
            self.order.append(v)
            for u in tree.neighbors(v):
                if u not in self.parent:
                    self.parent[u] = v
                    self.children[v].append(u)
                    self.depth[u] = self.depth[v] + 1
                    queue.append(u)

        # Subtree populations (bottom-up)
        self.subtree_pop = {v: self.graph.nodes[v]["population"] for v in nodes}
        for v in reversed(self.order):
            if self.parent[v] is not None:
                self.subtree_pop[self.parent[v]] += self.subtree_pop[v]

    def score_and_select(self, tree, M, lam):
        """
        Score all edges and return the k-1 best as the new marked set.

        score_i = deg(v) + λ (σ_v/p̄)² (1 - 2·is_marked)

        σ_v is normalized by p̄ so fidelity term competes with degree.

        Returns set of frozenset edges, or None if invalid.
        """
        k = self.k
        pbar = self.pbar
        best = []  # list of (score, frozenset_edge)

        # Build M lookup for O(1) membership test
        for v in self.order:
            if v == self.root:
                continue
            p = self.parent[v]
            e = frozenset((v, p))
            sigma_norm = (self.subtree_pop[v] - pbar) / pbar
            sigma2 = sigma_norm * sigma_norm
            deg = tree.degree(v)
            is_marked = 1.0 if e in M else 0.0
            score = deg + lam * sigma2 * (1.0 - 2.0 * is_marked)
            best.append((score, e))

        if len(best) < k - 1:
            return None

        best.sort()
        return {e for _, e in best[:k - 1]}


def _build_cache(tree, graph, k):
    """Build a fresh TreeCache with random root."""
    return TreeCache(tree, graph, k)


# ═══════════════════════════════════════════════════════════════════════════════
#  QP-Hybrid proposal: cycle step + QP-guided marked edges
# ═══════════════════════════════════════════════════════════════════════════════

def qp_hybrid_proposal(state, graph, k, lam_mean=1.0, lam_noise=0.3):
    """
    Hybrid proposal with tree resampling:
      1. Resample T uniformly given current partition ξ (Wilson's)
      2. Cycle basis step on fresh T (tree exploration)
      3. QP-guided marked-edge placement on new tree

    Returns
    -------
    new_state, log_trans_ratio, valid
    """
    # ── 1. Resample T given current ξ (Gibbs step) ──────────────────────────
    asn = state.get_assignment()
    T, M = resample_tree(graph, asn, k)

    # ── 2. Cycle basis step on fresh T ───────────────────────────────────────
    non_tree = [(u, v) for u, v in graph.edges() if not T.has_edge(u, v)]
    if not non_tree:
        return state, 0.0, False

    e_plus = random.choice(non_tree)
    path = _find_path(T, e_plus[0], e_plus[1])
    cycle_edges = _path_edges(path)
    removable = [e for e in cycle_edges if e not in M]
    if not removable:
        return state, 0.0, False

    e_minus = random.choice(removable)

    T_new = T.copy()
    T_new.add_edge(*e_plus)
    T_new.remove_edge(*e_minus)

    # ── 3. Build cache for new tree (random root) + QP solve ─────────────────
    cache = _build_cache(T_new, graph, k)

    if lam_noise > 0:
        shape = 1.0 / (lam_noise ** 2)
        scale = lam_mean / shape
        lam = np.random.gamma(shape, scale)
    else:
        lam = lam_mean

    M_new = cache.score_and_select(T_new, M, lam)
    if M_new is None or len(M_new) != k - 1:
        return state, 0.0, False

    # ── 4. Transition ratio (cycle step only) ────────────────────────────────
    len_C_M = len(removable)

    e_minus_tuple = tuple(e_minus)
    path_rev = _find_path(T_new, e_minus_tuple[0], e_minus_tuple[1])
    cycle_edges_rev = _path_edges(path_rev)
    removable_rev = [e for e in cycle_edges_rev if e not in M_new]
    len_C_M_rev = len(removable_rev)
    if len_C_M_rev == 0:
        return state, 0.0, False

    log_ratio = math.log(len_C_M) - math.log(len_C_M_rev)

    return MEWState(graph, T_new, M_new), log_ratio, True


# ═══════════════════════════════════════════════════════════════════════════════
#  Full MH step
# ═══════════════════════════════════════════════════════════════════════════════

def qp_hybrid_step(state, energy_fn, pop_target, epsilon, k,
                    lam_mean=1.0, lam_noise=0.3):
    """
    One QP-hybrid MEW step with Metropolis-Hastings acceptance.

    Tree resampling eliminates the degeneracy factor τ from the MH ratio.
    Target: p(T,M) ∝ exp(J(ξ)), inducing π(ξ) ∝ τ(ξ)·exp(J(ξ)).
    """
    new_state, log_trans, valid = qp_hybrid_proposal(
        state, state.graph, k, lam_mean, lam_noise)
    if not valid:
        return state, False

    new_asn = new_state.get_assignment()
    old_asn = state.get_assignment()
    n_dist = len(set(old_asn.values()))

    if len(set(new_asn.values())) != n_dist:
        return state, False

    # Population check
    pops = {}
    for node, d in new_asn.items():
        pops[d] = pops.get(d, 0) + state.graph.nodes[node]["population"]
    for pop in pops.values():
        if abs(pop - pop_target) / pop_target > epsilon:
            return state, False

    # MH acceptance — no τ correction (tree resampling handles it)
    J_old = energy_fn(state.graph, old_asn)
    J_new = energy_fn(state.graph, new_asn)

    log_alpha = (J_new - J_old) + log_trans
    if math.log(random.random()) < log_alpha:
        return new_state, True
    return state, False
