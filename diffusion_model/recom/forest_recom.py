"""The forest-lifted tree move (after Autry, Carter, Herschlag, Hunter and
Mattingly, "Metropolized forest recombination", 2023), the diffusion
local move lifted to the same state space, and their mixture. All three
are reversible for an arbitrary kernel-weighted compactness target, with
or without spanning-tree weighting.

WHY THIS FILE EXISTS.  The partition-level tree move of hybrid.py draws
its cut edge uniformly among ALL tree edges, which is what makes its
density closed-form, and that choice costs it its acceptance (under two
percent on Iowa). The paper's hybrid beats that move by 3x to 7x in
effective samples per second, but the move is not what practice runs,
so the comparison does not settle whether the local kernel adds value.
Lifting the state to a spanning forest recovers balanced-edge selection
(GerryChain's recom) with an exact density, and that is the baseline the
kill test has to be run against.

STATE SPACE.  A labeled spanning forest F = (asn, trees), where asn maps
each node to a district and trees[d] is a spanning tree of G[V_d], stored
as a list of frozenset edges. The fiber over a plan P has
tau(P) = prod_d tau(V_d) forests. The lifted target is

    pi~(F) = pi(P(F)) / tau(P(F)),

so the marginal of pi~ on plans is pi. Two targets on plans are supported:

    Target:             pi_w(P) ∝ exp(-lam * w(boundary(P)))
    TreeWeightedTarget: pi_w(P) * prod_d tau(V_d)

the second being the tree-weighted tempered measure that the paper's
calibration (Section 5.1) identifies as close to what tempered ReCom
samples, and the natural home ground of any tree-based move.

THE FOREST MOVE.  Pick an adjacent pair {i,j} uniformly from bp(P). Draw
a uniform spanning tree T of H = G[V_i ∪ V_j] by Wilson's algorithm. Let
B(T) be the set of edges of T whose removal leaves both components inside
the global balance window. If B(T) is empty, draw again, up to
max_tree_draws times, then reject. Otherwise pick e ∈ B(T) uniformly, let
{A, B} be the components, assign the labels i, j by a fair coin. A forest
F' = (.., T_A, T_B, ..) is reached from exactly those draws (T^, e^) with
T^ = T_A ∪ T_B ∪ f for a crossing edge f ∈ ∂(A,B) and e^ = f, so

    q(F -> F') = c_H / (2 |bp(P)| tau(H)) * S(T_A, T_B),
    S(T_A, T_B) = sum_{f in ∂(A,B)} 1 / |B(T_A ∪ T_B ∪ f)|,

where c_H = (1 - (1 - p_H)^M) / p_H accounts for the retry, p_H being the
probability that a uniform spanning tree of H has a balanced edge. The
retry is exact because p_H depends on H and the window only, never on
the endpoint, and the reverse move shares H, so c_H and tau(H) both
cancel. (This is NOT true of the local move's Bernoulli rounding, whose
feasibility event depends on the QP optimum at the endpoint; that is why
Chain E must be single-shot.) The Hastings ratio is

    [pi(P')/pi(P)] * [tau(V_i) tau(V_j) / (tau(A) tau(B))]
                   * [S(T_i, T_j) / S(T_A, T_B)] * [|bp(P)| / |bp(P')|].

The tree-count ratio is the lifted target's, not the proposal's. Under
TreeWeightedTarget it cancels against the same factor in pi(P')/pi(P),
which is why the forest move accepts far more often there: a tree
proposal is biased toward splits with many spanning trees, and targeting
plain pi_w makes the ratio undo that bias at every step (on Iowa that
term averages e^-6; see TODO_Q1.md). Each S is a sum over the crossing
edges of an O(n) subtree-population pass, and the two determinants of
the current districts are cached, so a step costs two determinants plus
O(|∂| n).

THE LIFTED LOCAL MOVE.  Run Chain E (hybrid.local_step) on the plan,
against whichever target is given. If it accepts, redraw the trees of the
two changed districts uniformly by Wilson; otherwise keep the forest.
With K_E pi-reversible on plans, K~(F, F') = K_E(P, P') / (tau(A) tau(B))
and pi~(F) K~(F, F') is symmetric because tau(P) = tau(V_i) tau(V_j) *
(the other districts), so the lifted move is pi~-reversible.

THE HYBRID.  A local step with probability p_local, else a forest step.

VERIFICATION.  db_check_forest_2x3() assembles all three kernels
analytically on the lifted 2x3 fiber for both targets and checks
pi~ K = (pi~ K)^T entrywise, and checks that the closed-form S reproduces
the enumerated proposal mass. forest_validate_5x5.py tests the chains
against enumerated targets on 4,006 plans.
"""

import math
import random
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

_DIFF = Path(__file__).resolve().parent.parent
if str(_DIFF) not in sys.path:
    sys.path.insert(0, str(_DIFF))

from helpers.metrics import border_pairs_set
from recom.hybrid import Target, wilson_ust, log_tau, ess_fft  # noqa: F401

MAX_TREE_DRAWS = 20


# ─── targets ─────────────────────────────────────────────────────────────────

class TreeWeightedTarget(Target):
    """pi(P) ∝ prod_d tau(V_d) * exp(-lam * w(boundary(P))). The log tau
    of each district is cached by its node set."""

    def __init__(self, graph, kernel_fn, lam):
        super().__init__(graph, kernel_fn, lam)
        self._lt = {}

    def log_tau_district(self, nodes):
        key = frozenset(nodes)
        v = self._lt.get(key)
        if v is None:
            if len(self._lt) > 200_000:
                self._lt.clear()
            v = log_tau(self.graph, list(nodes))
            self._lt[key] = v
        return v

    def log_pi(self, asn):
        parts = {}
        for v, d in asn.items():
            parts.setdefault(d, []).append(v)
        return (-self.lam * self.wcut(asn)
                + sum(self.log_tau_district(p) for p in parts.values()))


def make_target(kind, graph, kernel_fn, lam):
    if kind == "pi":
        return Target(graph, kernel_fn, lam)
    if kind == "tree":
        return TreeWeightedTarget(graph, kernel_fn, lam)
    raise ValueError(f"target kind must be 'pi' or 'tree', got {kind!r}")


# ─── tree utilities ──────────────────────────────────────────────────────────

def _tree_adj(edges, nodes):
    adj = {v: [] for v in nodes}
    for e in edges:
        a, b = tuple(e)
        adj[a].append(b)
        adj[b].append(a)
    return adj


def balanced_edges(edges, nodes, pop, lo, hi):
    """Edges of the tree (edges on `nodes`) whose removal leaves both
    components with population in [lo, hi]. O(n)."""
    adj = _tree_adj(edges, nodes)
    root = nodes[0]
    parent = {root: None}
    order = [root]
    stack = [root]
    while stack:
        v = stack.pop()
        for u in adj[v]:
            if u not in parent:
                parent[u] = v
                order.append(u)
                stack.append(u)
    sub = {v: pop[v] for v in nodes}
    for v in reversed(order):
        if parent[v] is not None:
            sub[parent[v]] += sub[v]
    total = sub[root]
    out = []
    for v in order[1:]:
        s = sub[v]
        if lo <= s <= hi and lo <= total - s <= hi:
            out.append(frozenset((v, parent[v])))
    return out


def _split(edges, nodes, e):
    """Components of the tree after removing edge e: (A, B) as sets."""
    adj = _tree_adj([f for f in edges if f != e], nodes)
    a0 = next(iter(e))
    A = set()
    stack = [a0]
    while stack:
        v = stack.pop()
        if v in A:
            continue
        A.add(v)
        stack.extend(adj[v])
    return A, set(nodes) - A


def crossing_edges(graph, A, B):
    B = set(B)
    return [frozenset((u, w)) for u in A for w in graph.neighbors(u) if w in B]


def proposal_sum(graph, T_A, T_B, A, B, nodes, pop, lo, hi):
    """S(T_A, T_B) = sum over crossing edges f of 1/|B(T_A ∪ T_B ∪ f)|."""
    base = list(T_A) + list(T_B)
    s = 0.0
    for f in crossing_edges(graph, A, B):
        nb = len(balanced_edges(base + [f], nodes, pop, lo, hi))
        s += 1.0 / nb          # nb >= 1: f itself is balanced
    return s


# ─── forest state ────────────────────────────────────────────────────────────

def initial_forest(graph, asn):
    """Uniform spanning tree of each district (Wilson), as the lifted
    initial state."""
    trees = {}
    for d in sorted(set(asn.values())):
        nodes = [v for v in asn if asn[v] == d]
        ns = set(nodes)
        adj = {v: [u for u in graph.neighbors(v) if u in ns] for v in nodes}
        trees[d] = [frozenset(e) for e in wilson_ust(nodes, adj)]
    return trees


class TauCache:
    """log tau(V_d) for the current districts, recomputed only when a
    district's node set changes."""

    def __init__(self, graph):
        self.graph = graph
        self.store = {}

    def get(self, d, nodes):
        key = frozenset(nodes)
        hit = self.store.get(d)
        if hit is not None and hit[0] == key:
            return hit[1]
        val = log_tau(self.graph, list(nodes))
        self.store[d] = (key, val)
        return val


# ─── the moves ───────────────────────────────────────────────────────────────

def forest_log_alpha(tgt, asn, new, d1, d2, Vi, Vj, A, B, T_i, T_j, T_A, T_B,
                     nodes, lo, hi, lt_i, lt_j, lt_A, lt_B):
    """log of the Hastings ratio of the forest move. Shared by the chain
    and the analytic check so the check validates the implementation."""
    graph = tgt.graph
    S_fwd = proposal_sum(graph, T_A, T_B, A, B, nodes, tgt.pop, lo, hi)
    S_rev = proposal_sum(graph, T_i, T_j, Vi, Vj, nodes, tgt.pop, lo, hi)
    bp_old = len(border_pairs_set(asn, graph))
    bp_new = len(border_pairs_set(new, graph))
    return ((tgt.log_pi(new) - tgt.log_pi(asn))
            + (lt_i + lt_j - lt_A - lt_B)
            + math.log(S_rev) - math.log(S_fwd)
            + math.log(bp_old) - math.log(bp_new))


def forest_step(tgt, asn, trees, ideal, eps, cache, max_tree_draws=MAX_TREE_DRAWS):
    graph = tgt.graph
    bp = sorted(border_pairs_set(asn, graph))
    d1, d2 = random.choice(bp)
    Vi = [v for v in asn if asn[v] == d1]
    Vj = [v for v in asn if asn[v] == d2]
    nodes = Vi + Vj
    nodeset = set(nodes)
    adj = {v: [u for u in graph.neighbors(v) if u in nodeset] for v in nodes}
    rec = dict(kind="forest", feasible=False, accepted=False, log_alpha=np.nan, draws=0,
               log_tau_ratio=np.nan)

    lo, hi = (1 - eps) * ideal, (1 + eps) * ideal
    bal = []
    for _ in range(max_tree_draws):
        T = [frozenset(e) for e in wilson_ust(nodes, adj)]
        rec["draws"] += 1
        bal = balanced_edges(T, nodes, tgt.pop, lo, hi)
        if bal:
            break
    if not bal:
        return asn, trees, rec
    rec["feasible"] = True
    e = random.choice(bal)
    A, B = _split(T, nodes, e)
    if random.random() < 0.5:          # fair coin: bijection on labeled forests
        A, B = B, A
    T_A = [f for f in T if f != e and next(iter(f)) in A]
    T_B = [f for f in T if f != e and next(iter(f)) in B]

    new = dict(asn)
    for v in A:
        new[v] = d1
    for v in B:
        new[v] = d2

    lt_i = cache.get(d1, Vi)
    lt_j = cache.get(d2, Vj)
    lt_A = log_tau(graph, list(A))
    lt_B = log_tau(graph, list(B))
    rec["log_tau_ratio"] = lt_i + lt_j - lt_A - lt_B
    la = forest_log_alpha(tgt, asn, new, d1, d2, Vi, Vj, A, B,
                          trees[d1], trees[d2], T_A, T_B,
                          nodes, lo, hi, lt_i, lt_j, lt_A, lt_B)
    rec["log_alpha"] = la
    if math.log(random.random() + 1e-300) < la:
        rec["accepted"] = True
        new_trees = dict(trees)
        new_trees[d1] = T_A
        new_trees[d2] = T_B
        cache.store[d1] = (frozenset(A), lt_A)
        cache.store[d2] = (frozenset(B), lt_B)
        return new, new_trees, rec
    return asn, trees, rec


def lifted_local_step(tgt, asn, trees, ideal, eps, alpha, beta):
    from recom.hybrid import local_step
    new, rec = local_step(tgt, asn, ideal, eps, alpha, beta)
    rec["kind"] = "local"
    if not rec["accepted"]:
        return asn, trees, rec
    changed = {asn[v] for v in asn if new[v] != asn[v]} | \
              {new[v] for v in asn if new[v] != asn[v]}
    new_trees = dict(trees)
    graph = tgt.graph
    for d in changed:
        nodes = [v for v in new if new[v] == d]
        ns = set(nodes)
        adj = {v: [u for u in graph.neighbors(v) if u in ns] for v in nodes}
        new_trees[d] = [frozenset(e) for e in wilson_ust(nodes, adj)]
    return new, new_trees, rec


# ─── runner (same return format as hybrid.run_hybrid) ────────────────────────

def run_forest_hybrid(tgt, init_asn, ideal, eps, n_steps, p_local, alpha, beta,
                      seed, state_id=None, log_every=0, label="",
                      max_tree_draws=MAX_TREE_DRAWS):
    """p_local = 0 is the forest move alone (the Forest ReCom baseline);
    p_local = 1 is the lifted Chain E alone."""
    random.seed(seed)
    np.random.seed(seed)
    asn = dict(init_asn)
    trees = initial_forest(tgt.graph, asn)
    cache = TauCache(tgt.graph)
    wc = np.empty(n_steps)
    nc = np.empty(n_steps, dtype=np.int32)
    kind = np.empty(n_steps, dtype=np.int8)
    feas = np.zeros(n_steps, dtype=bool)
    acc = np.zeros(n_steps, dtype=bool)
    flips = np.zeros(n_steps, dtype=np.int32)
    fast = np.zeros(n_steps, dtype=bool)
    draws = np.zeros(n_steps, dtype=np.int16)
    ltr = np.full(n_steps, np.nan)
    ids = np.full(n_steps, -1, dtype=np.int64)
    t0 = time.perf_counter()
    for t in range(n_steps):
        if random.random() < p_local:
            asn, trees, r = lifted_local_step(tgt, asn, trees, ideal, eps, alpha, beta)
            kind[t] = 1
            flips[t] = r["flips"]
            fast[t] = r["fast"]
        else:
            asn, trees, r = forest_step(tgt, asn, trees, ideal, eps, cache, max_tree_draws)
            kind[t] = 0
            draws[t] = r["draws"]
            ltr[t] = r["log_tau_ratio"]
        feas[t] = r["feasible"]
        acc[t] = r["accepted"]
        wc[t] = tgt.wcut(asn)
        nc[t] = tgt.ncut(asn)
        if state_id is not None:
            ids[t] = state_id(asn)
        if log_every and (t + 1) % log_every == 0:
            k0 = kind[:t + 1] == 0
            k1 = kind[:t + 1] == 1
            print(f"  [{label}] {t+1}/{n_steps}  {time.perf_counter()-t0:6.1f}s  "
                  f"wcut {wc[t]:.2f}  acc(forest) "
                  f"{acc[:t+1][k0].mean() if k0.any() else float('nan'):.3f}  "
                  f"acc(local) {acc[:t+1][k1].mean() if k1.any() else float('nan'):.3f}",
                  flush=True)
    return dict(wcut=wc, ncut=nc, kind=kind, feasible=feas, accepted=acc,
                flips=flips, fast=fast, draws=draws, log_tau_ratio=ltr, ids=ids,
                wall=time.perf_counter() - t0, final=asn, final_trees=trees)


# ─── analytic kernel check on the lifted 2x3 fiber ───────────────────────────

def _spanning_trees(graph, nodes):
    """All spanning trees of graph[nodes], brute force (tiny graphs only)."""
    import networkx as nx
    nodes = list(nodes)
    if len(nodes) == 1:
        return [[]]
    sub = graph.subgraph(nodes)
    E = [frozenset(e) for e in sub.edges()]
    out = []
    for S in combinations(E, len(nodes) - 1):
        H = nx.Graph()
        H.add_nodes_from(nodes)
        H.add_edges_from(tuple(e) for e in S)
        if nx.is_connected(H):
            out.append(list(S))
    return out


def db_check_forest_2x3(kernel_fn=None, lam=0.10, alpha=1.0, beta=10.0,
                        eps=1.0 / 3.0, p_local=0.5, density=None,
                        target="pi", max_tree_draws=MAX_TREE_DRAWS, verbose=True):
    """Assemble K_forest, K_local (lifted) and K_mix analytically on the
    lifted 2x3 fiber and report the maximum relative violation of
    pi~(F) K(F,F') = pi~(F') K(F',F) for each, plus the maximum relative
    error between the closed-form proposal mass of the forest move and
    the mass obtained by enumerating every (tree, balanced edge, coin)
    draw. The retry enters as the state-independent factor c_H. Returns
    a dict of the maxima."""
    import networkx as nx
    from recom.qp_model import kernel_uniform
    from recom.hybrid import solve_qp_fast
    DELTA = 1e-12
    if kernel_fn is None:
        kernel_fn = kernel_uniform

    G = nx.grid_graph([3, 2])
    for v in G.nodes():
        G.nodes[v]["population"] = 1
        G.nodes[v]["density"] = 0.0 if density is None else float(density(v))
    cells = sorted(G.nodes())
    ideal = 3.0
    lo, hi = (1 - eps) * ideal, (1 + eps) * ideal
    tgt = make_target(target, G, kernel_fn, lam)

    # labeled plans
    plans = []
    for size in range(int(math.ceil(lo - 1e-9)), int(math.floor(hi + 1e-9)) + 1):
        for S in combinations(cells, size):
            S, T = set(S), set(cells) - set(S)
            if T and nx.is_connected(G.subgraph(S)) and nx.is_connected(G.subgraph(T)):
                plans.append({c: (0 if c in S else 1) for c in cells})
    # lifted states: (plan index, tree of district 0, tree of district 1)
    states = []
    for pidx, a in enumerate(plans):
        S = [c for c in cells if a[c] == 0]
        T = [c for c in cells if a[c] == 1]
        for tS in _spanning_trees(G, S):
            for tT in _spanning_trees(G, T):
                states.append((pidx, frozenset(tS), frozenset(tT)))
    n = len(states)
    key = {s: i for i, s in enumerate(states)}

    def lt(nodes):
        return log_tau(G, list(nodes))

    logpi_t = np.empty(n)
    for i, (pidx, _, _) in enumerate(states):
        a = plans[pidx]
        S = [c for c in cells if a[c] == 0]
        T = [c for c in cells if a[c] == 1]
        logpi_t[i] = tgt.log_pi(a) - lt(S) - lt(T)
    pi_t = np.exp(logpi_t - logpi_t.max())
    pi_t /= pi_t.sum()

    # forest proposal mass by enumeration of every draw (independent of F)
    all_trees = _spanning_trees(G, cells)
    tauH = len(all_trees)
    q_enum = np.zeros(n)
    n_bal_trees = 0
    for T in all_trees:
        bal = balanced_edges(T, cells, tgt.pop, lo, hi)
        if bal:
            n_bal_trees += 1
        for e in bal:
            A, B = _split(T, cells, e)
            for lab in (0, 1):
                X, Y = (A, B) if lab == 0 else (B, A)
                a = {c: (0 if c in X else 1) for c in cells}
                pidx = next(k for k, pl in enumerate(plans) if pl == a)
                tX = frozenset(f for f in T if f != e and next(iter(f)) in X)
                tY = frozenset(f for f in T if f != e and next(iter(f)) in Y)
                q_enum[key[(pidx, tX, tY)]] += 1.0 / (tauH * len(bal) * 2)
    p_H = n_bal_trees / tauH
    c_H = (1.0 - (1.0 - p_H) ** max_tree_draws) / p_H
    q_enum *= c_H
    # closed form
    q_cf = np.empty(n)
    for i, (pidx, tS, tT) in enumerate(states):
        a = plans[pidx]
        S = {c for c in cells if a[c] == 0}
        T = {c for c in cells if a[c] == 1}
        q_cf[i] = c_H * proposal_sum(G, list(tS), list(tT), S, T, cells, tgt.pop, lo, hi) / (2 * tauH)
    q_err = float(np.max(np.abs(q_cf - q_enum) / q_enum))

    # forest kernel, using the implementation's own log-alpha
    K_f = np.zeros((n, n))
    for i, (pi_i, tS_i, tT_i) in enumerate(states):
        a = plans[pi_i]
        Vi = [c for c in cells if a[c] == 0]
        Vj = [c for c in cells if a[c] == 1]
        for j, (pi_j, tS_j, tT_j) in enumerate(states):
            if i == j:
                continue
            b = plans[pi_j]
            A = {c for c in cells if b[c] == 0}
            B = {c for c in cells if b[c] == 1}
            la = forest_log_alpha(tgt, a, b, 0, 1, Vi, Vj, A, B,
                                  list(tS_i), list(tT_i), list(tS_j), list(tT_j),
                                  cells, lo, hi, lt(Vi), lt(Vj), lt(A), lt(B))
            K_f[i, j] = q_enum[j] * math.exp(min(0.0, la))
        K_f[i, i] = 1.0 - K_f[i].sum()

    # lifted local kernel: K_E on plans divided by tau(A) tau(B)
    def bits(a):
        return np.array([1.0 if a[c] == 0 else 0.0 for c in cells])
    p_of = []
    for a in plans:
        x, _ = solve_qp_fast(G, cells, a, 0, alpha, beta, eps, kernel_fn, ideal)
        p_of.append(np.clip(np.asarray(x), DELTA, 1 - DELTA))
    m = len(plans)
    KE = np.zeros((m, m))
    logpi_p = np.array([tgt.log_pi(a) for a in plans])
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            bj, bi = bits(plans[j]), bits(plans[i])
            lq_f = float(np.sum(bj * np.log(p_of[i]) + (1 - bj) * np.log(1 - p_of[i])))
            lq_r = float(np.sum(bi * np.log(p_of[j]) + (1 - bi) * np.log(1 - p_of[j])))
            la = min(0.0, (logpi_p[j] - logpi_p[i]) + lq_r - lq_f)
            KE[i, j] = math.exp(lq_f + la)
    K_l = np.zeros((n, n))
    tau_of = []
    for a in plans:
        S = [c for c in cells if a[c] == 0]
        T = [c for c in cells if a[c] == 1]
        tau_of.append(math.exp(lt(S) + lt(T)))
    for i, (pi_i, _, _) in enumerate(states):
        for j, (pi_j, _, _) in enumerate(states):
            if pi_i == pi_j:
                continue
            K_l[i, j] = KE[pi_i, pi_j] / tau_of[pi_j]
        K_l[i, i] = 1.0 - K_l[i].sum()

    K_m = p_local * K_l + (1 - p_local) * K_f

    def viol(K):
        F = pi_t[:, None] * K
        den = np.maximum(np.abs(F), np.abs(F.T))
        msk = den > 0
        rel = np.zeros_like(F)
        rel[msk] = np.abs(F - F.T)[msk] / den[msk]
        return float(rel.max()), float(np.abs(K.sum(axis=1) - 1).max()), float(K.diagonal().min())

    out = {"q_closed_form_vs_enumerated": q_err}
    for name, K in (("forest", K_f), ("local", K_l), ("mixture", K_m)):
        v, rs, dmin = viol(K)
        out[name] = v
        if verbose:
            print(f"  {name:8s} max rel DB violation {v:.2e}   "
                  f"max |row sum - 1| {rs:.1e}   min diagonal {dmin:.4f}")
    if verbose:
        print(f"  ({len(plans)} labeled plans, {n} lifted states, tau(H) = {tauH}, "
              f"p_H = {p_H:.3f}, retry factor c_H = {c_H:.4f}, "
              f"closed-form proposal mass vs enumeration: max rel err {q_err:.1e})")
    return out


if __name__ == "__main__":
    from recom.qp_model import make_kernel_density
    for target in ("pi", "tree"):
        print(f"lifted 2x3 fiber, target = {target}, uniform kernel, lam = 0.10")
        db_check_forest_2x3(target=target)
        print(f"lifted 2x3 fiber, target = {target}, density kernel (rho = x, sigma = 1), lam = 1.0")
        db_check_forest_2x3(kernel_fn=make_kernel_density(sigma=1.0), lam=1.0,
                            density=lambda v: float(v[0]), target=target)
