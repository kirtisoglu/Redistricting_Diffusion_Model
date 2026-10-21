"""
mew_step.py — Vanilla and boundary-restricted MEW proposal steps.

Implements the two-stage MEW transition:
    1. Cycle basis step: swap a non-tree edge e+ into T, remove a tree edge e-.
    2. Marked edge step: slide one marked edge along T to a neighbor.

Two variants of the cycle basis step are provided:
    - vanilla_cycle_step: e+ chosen uniformly from all non-tree edges (original MEW).
    - boundary_cycle_step: e+ chosen uniformly from boundary non-tree edges only.

The marked edge step and MH acceptance are shared between both variants.

Reference: McWhorter & DeFord, arXiv:2510.17714 (2025), Section 2.
"""

import random
import numpy as np
import networkx as nx
from collections import deque
from state import MEWState


# ---------------------------------------------------------------------------
#  Tree path utilities
# ---------------------------------------------------------------------------

def _find_path(tree, u, v):
    """BFS path from u to v in the tree. Returns list of nodes."""
    parent = {u: None}
    queue = deque([u])
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


# ---------------------------------------------------------------------------
#  Degeneracy factor τ(ξ)
# ---------------------------------------------------------------------------

def _log_spanning_tree_count(graph, nodes):
    """log t(G[nodes]) via Kirchhoff's matrix-tree theorem."""
    n = len(nodes)
    if n <= 1:
        return 0.0
    sub = graph.subgraph(nodes)
    L = nx.laplacian_matrix(sub).toarray().astype(np.float64)
    sign, logdet = np.linalg.slogdet(L[1:, 1:])
    if sign <= 0:
        return -np.inf
    return logdet


def compute_log_degeneracy(graph, assignment):
    """
    log τ(ξ) = Σ_i log t(G[ξ_i]) + Σ_{i~j} log |∂(ξ_i, ξ_j)|.

    Returns -inf if any district is disconnected.
    """
    districts = {}
    for node, d in assignment.items():
        districts.setdefault(d, []).append(node)

    log_tau = 0.0
    for nodes in districts.values():
        lt = _log_spanning_tree_count(graph, nodes)
        if lt == -np.inf:
            return -np.inf
        log_tau += lt

    border = {}
    for u, v in graph.edges():
        d1, d2 = assignment[u], assignment[v]
        if d1 != d2:
            key = (min(d1, d2), max(d1, d2))
            border[key] = border.get(key, 0) + 1
    for cnt in border.values():
        log_tau += np.log(cnt)

    return log_tau


# ---------------------------------------------------------------------------
#  Cycle basis step variants
# ---------------------------------------------------------------------------

def _cycle_step(state, e_plus):
    """
    Given a chosen non-tree edge e+, perform the cycle basis swap.

    Finds the fundamental cycle C = tree_path(e+) ∪ {e+}, picks e- uniformly
    from C \\ M, and returns (T', cycle_edges, removable, e_minus) or None.
    """
    T, M = state.tree, state.marked

    path = _find_path(T, e_plus[0], e_plus[1])
    cycle_edges = _path_edges(path)
    removable = [e for e in cycle_edges if e not in M]

    if not removable:
        return None

    e_minus = random.choice(removable)

    T_new = T.copy()
    T_new.add_edge(*e_plus)
    T_new.remove_edge(*e_minus)

    return T_new, cycle_edges, removable, e_minus


def vanilla_cycle_step(state):
    """
    Original MEW cycle step: e+ chosen uniformly from ALL non-tree edges.

    Returns
    -------
    (T_new, e_plus, cycle_edges, removable, e_minus) or None if no valid swap.
    """
    graph, T = state.graph, state.tree
    non_tree = [(u, v) for u, v in graph.edges() if not T.has_edge(u, v)]
    if not non_tree:
        return None

    e_plus = random.choice(non_tree)
    result = _cycle_step(state, e_plus)
    if result is None:
        return None

    T_new, cycle_edges, removable, e_minus = result
    return T_new, e_plus, cycle_edges, removable, e_minus


def boundary_cycle_step(state):
    """
    Boundary-restricted cycle step: e+ chosen uniformly from boundary
    non-tree edges only (edges whose endpoints are in different districts).

    Every successful proposal changes the partition boundary.

    Returns
    -------
    (T_new, e_plus, cycle_edges, removable, e_minus) or None if no valid swap.
    """
    boundary_edges = state.boundary_non_tree_edges()
    if not boundary_edges:
        return None

    e_plus = random.choice(boundary_edges)
    result = _cycle_step(state, e_plus)
    if result is None:
        return None

    T_new, cycle_edges, removable, e_minus = result
    return T_new, e_plus, cycle_edges, removable, e_minus


# ---------------------------------------------------------------------------
#  Marked edge step (shared by both variants)
# ---------------------------------------------------------------------------

def _marked_edge_step(T_new, M, e_plus_fs):
    """
    Slide one marked edge along the tree.

    Pick m ∈ M uniformly, pick an endpoint u of m, pick a neighbor v of u
    in T_new, set m' = (u, v). Update M accordingly.

    Returns
    -------
    (M_new, m, m_prime, u, v_other) or None if invalid.
    """
    if not M:
        return set(), None, None, None, None

    M_new = set(M)
    m = random.choice(list(M_new))
    m_nodes = list(m)

    u = random.choice(m_nodes)
    v_other = m_nodes[0] if m_nodes[1] == u else m_nodes[1]

    nbrs = list(T_new.neighbors(u))
    if not nbrs:
        return None
    v = random.choice(nbrs)

    m_prime = frozenset((u, v))

    # If m' equals the edge we just added, the reverse step is impossible
    if m_prime == e_plus_fs:
        return None

    M_new.discard(m)
    M_new.add(m_prime)

    return M_new, m, m_prime, u, v_other


# ---------------------------------------------------------------------------
#  Transition ratio (Eq. 5 of the paper)
# ---------------------------------------------------------------------------

def _compute_log_transition_ratio(
    T, T_new, M, M_new, removable, e_minus,
    m, m_prime, u, v_other
):
    """
    log P(x|x') / P(x'|x) from Eq. 5, with Eq. 4 correction when m = m'.
    """
    # |C \\ M| forward
    len_C_M = len(removable)

    # |C' \\ M'| reverse: cycle in T' ∪ {e-}
    e_minus_tuple = tuple(e_minus)
    path_rev = _find_path(T_new, e_minus_tuple[0], e_minus_tuple[1])
    cycle_edges_rev = _path_edges(path_rev)
    removable_rev = [e for e in cycle_edges_rev if e not in M_new]
    len_C_M_rev = len(removable_rev)

    if len_C_M_rev == 0:
        return None

    # Tree degrees at the sliding endpoint u
    deg_T_u = T.degree(u)
    deg_T_new_u = T_new.degree(u)
    if deg_T_u == 0 or deg_T_new_u == 0:
        return None

    log_ratio = (
        np.log(len_C_M) - np.log(len_C_M_rev)
        + np.log(deg_T_new_u) - np.log(deg_T_u)
    )

    # Eq. 4 correction when the marked edge didn't move
    if m == m_prime:
        deg_T_v = T.degree(v_other)
        deg_T_new_v = T_new.degree(v_other)
        if deg_T_v == 0 or deg_T_new_v == 0:
            return None
        log_ratio += (
            np.log(deg_T_u + deg_T_v)
            - np.log(deg_T_new_u + deg_T_new_v)
            + np.log(deg_T_new_v) - np.log(deg_T_v)
        )

    return log_ratio


# ---------------------------------------------------------------------------
#  Full proposal: cycle step + marked edge step + transition ratio
# ---------------------------------------------------------------------------

def _full_proposal(state, cycle_step_fn):
    """
    Combine a cycle step function with the marked edge step.

    Parameters
    ----------
    state : MEWState
    cycle_step_fn : callable(state) → (T_new, e_plus, cycle_edges, removable, e_minus) or None

    Returns
    -------
    (new_state, log_trans_ratio, valid)
    """
    cycle_result = cycle_step_fn(state)
    if cycle_result is None:
        return state, 0.0, False

    T_new, e_plus, cycle_edges, removable, e_minus = cycle_result
    e_plus_fs = frozenset(e_plus)

    # Marked edge step
    me_result = _marked_edge_step(T_new, state.marked, e_plus_fs)
    if me_result is None:
        return state, 0.0, False

    M_new, m, m_prime, u, v_other = me_result

    # Transition ratio
    if m is None:
        # No marked edges (k=1) — ratio is just from cycle step
        new_state = MEWState(state.graph, T_new, M_new)
        return new_state, 0.0, True

    log_trans = _compute_log_transition_ratio(
        state.tree, T_new, state.marked, M_new,
        removable, e_minus, m, m_prime, u, v_other
    )
    if log_trans is None:
        return state, 0.0, False

    new_state = MEWState(state.graph, T_new, M_new)
    return new_state, log_trans, True


# ---------------------------------------------------------------------------
#  Public API: one full MH step
# ---------------------------------------------------------------------------

def mew_step(state, energy_fn, pop_target, epsilon, boundary_only=False):
    """
    One MEW step with Metropolis-Hastings acceptance.

    Target: p(x) ∝ exp(J(ξ(x))) / τ(ξ(x))

    Parameters
    ----------
    state : MEWState
    energy_fn : callable(graph, assignment) → float
        Energy function J(ξ). Use lambda g, a: 0.0 for uniform target.
    pop_target : float
        Ideal population per district.
    epsilon : float
        Fractional population tolerance.
    boundary_only : bool
        If True, use boundary_cycle_step instead of vanilla_cycle_step.

    Returns
    -------
    (new_state, accepted, partition_changed)
        partition_changed is True if the new partition differs from the old.
    """
    cycle_fn = boundary_cycle_step if boundary_only else vanilla_cycle_step
    new_state, log_trans, valid = _full_proposal(state, cycle_fn)

    old_asn = state.get_partition()

    if not valid:
        return state, False, False

    new_asn = new_state.get_partition()

    # Cardinality check
    if len(set(new_asn.values())) != len(set(old_asn.values())):
        return state, False, False

    # Population balance check
    pops = {}
    for node, d in new_asn.items():
        pops[d] = pops.get(d, 0) + state.graph.nodes[node].get("population", 1)
    for pop in pops.values():
        if abs(pop - pop_target) / pop_target > epsilon:
            return state, False, False

    # MH acceptance
    J_old = energy_fn(state.graph, old_asn)
    J_new = energy_fn(state.graph, new_asn)
    log_tau_old = compute_log_degeneracy(state.graph, old_asn)
    log_tau_new = compute_log_degeneracy(state.graph, new_asn)

    log_alpha = (J_new - J_old) + (log_tau_old - log_tau_new) + log_trans

    if np.log(random.random()) < log_alpha:
        partition_changed = (set(new_asn.items()) != set(old_asn.items()))
        return new_state, True, partition_changed
    else:
        return state, False, False
