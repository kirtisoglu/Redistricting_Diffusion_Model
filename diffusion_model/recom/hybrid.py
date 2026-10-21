"""The reversible hybrid: an exact tree move, the exact diffusion move, and
their mixture, all reversible for a kernel-weighted compactness target.

TARGET.  pi_w(P) ∝ exp(-lam * w(boundary(P))), where w(boundary(P)) is the
sum of kernel weights over cut edges. With the uniform kernel this is the
paper's tempered target; with the perimeter or density kernel it is a
geography-weighted compactness distribution. The kernel is a property of
the TARGET here, and the local move's Laplacian uses the same kernel, so
the proposal is matched to the target's geometry by construction.

STATE SPACE.  Labeled plans, as in Section 4.6 of the paper. Both moves are
bijections on labeled plans, and pi_w is label-invariant, so forgetting
labels afterward recovers the unlabeled law.

THE TREE MOVE (exact).  Pick an adjacent pair uniformly from the boundary
pairs. Draw a uniform spanning tree T of the merged region H by Wilson's
algorithm. Pick one edge of T uniformly among ALL n_H - 1 tree edges, not
among balanced ones. If the two components are both inside the global
balance window, assign the pair's labels to them by a fair coin and
propose; otherwise reject. Every spanning tree that yields the components
{A, B} is a spanning tree of A plus a spanning tree of B plus one crossing
edge, so the proposal density has the closed form

    q(P -> P') = tau(A) tau(B) |d(A,B)| / ( 2 |bp(P)| tau(H) (n_H - 1) ),

with tau the spanning-tree count by the Matrix-Tree theorem and |d(A,B)|
the number of edges between A and B. The reverse move shares H, so tau(H)
and n_H - 1 cancel in the Hastings ratio, which becomes

    [tau(V_i) tau(V_j) |d(V_i,V_j)|] / [tau(A) tau(B) |d(A,B)|]  *  |bp(P)| / |bp(P')|.

The fair coin is what makes the move a bijection on labeled plans. A
deterministic labeling rule would make some reverse moves impossible and
break reversibility. Choosing among all tree edges rather than balanced
ones is what keeps the density closed-form; GerryChain's recom chooses
among balanced edges, which introduces a state-dependent normalization
that is why Chain A in the paper is only approximately correct.

THE LOCAL MOVE (exact).  Chain E of the paper, with the weighted cut in
the acceptance ratio. With the uniform kernel it is identical to
qp_mh_proposal in qp_model.py line for line.

THE HYBRID.  At each step, a local move with probability p_local, else a
tree move. A mixture of pi_w-reversible kernels is pi_w-reversible.

VERIFICATION.  db_check_hybrid() assembles all three kernels analytically
on the 18-state 2x3 fiber and checks pi K = (pi K)^T entrywise, under both
the uniform kernel and a synthetic density kernel.
"""

import math
import random
import time

import numpy as np

from helpers.metrics import border_pairs_set
from recom.qp_model import (_solve_qp, _randomized_round,
                            _pair_split_feasible, kernel_uniform,
                            build_laplacian)

DELTA = 1e-12          # identical to qp_mh_proposal's clip


def solve_qp_fast(graph, nodes, asn, d1, alpha, beta, eps, kernel_fn, ideal):
    """The QP optimum by Lemma 4.1 of the paper when it applies, Gurobi
    otherwise. The unconstrained minimizer (I + tau L)^{-1} x0 lies in the
    unit cube automatically and is the constrained optimum whenever it
    satisfies the balance window, so in that case one linear solve replaces
    the QP solver. Both paths are deterministic functions of (state, pair),
    which is all the detailed-balance argument uses. Returns (x, fast)."""
    L, _ = build_laplacian(graph, nodes, kernel_fn=kernel_fn)
    x0 = np.array([1.0 if asn[v] == d1 else 0.0 for v in nodes])
    x = np.linalg.solve(np.eye(len(nodes)) + (alpha / beta) * L, x0)
    pop = np.array([float(graph.nodes[v]["population"]) for v in nodes])
    lo = max((1 - eps) * ideal, pop.sum() - (1 + eps) * ideal)
    hi = min((1 + eps) * ideal, pop.sum() - (1 - eps) * ideal)
    px = float(pop @ x)
    if lo - 1e-9 <= px <= hi + 1e-9:
        return np.clip(x, 0.0, 1.0), True        # clip guards 1e-16 roundoff only
    xq = _solve_qp(graph, nodes, asn, d1, alpha=alpha, beta=beta, epsilon=eps,
                   kernel_fn=kernel_fn, ideal_pop=ideal)
    return (None if xq is None else np.asarray(xq)), False


# ─── shared diagnostic ───────────────────────────────────────────────────────

def ess_fft(x):
    """Geyer initial-positive-sequence effective sample size with an FFT
    autocovariance (O(n log n)). Shared by the Iowa scripts so that none of
    them has to import a script that parses its own command line."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    if n < 8 or x.var() == 0:
        return float(n)
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, nfft)
    acf = np.fft.irfft(f * np.conj(f), nfft)[:n]
    acf = acf / acf[0]
    tau = 1.0
    for m in range(1, n // 2):
        g = acf[2 * m - 1] + acf[2 * m]
        if g <= 0:
            break
        tau += 2 * g
    return max(n / tau, 1.0)


# ─── target ──────────────────────────────────────────────────────────────────

class Target:
    """pi_w(P) ∝ exp(-lam * sum_{cut edges} w(u,v)), with edge weights
    precomputed once from kernel_fn(graph, u, v)."""

    def __init__(self, graph, kernel_fn, lam):
        self.graph, self.kernel_fn, self.lam = graph, kernel_fn, lam
        self.edges = list(graph.edges())
        self.W = {}
        for u, v in self.edges:
            w = float(kernel_fn(graph, u, v))
            self.W[(u, v)] = w
            self.W[(v, u)] = w
        self.pop = {v: float(graph.nodes[v]["population"]) for v in graph.nodes()}

    def wcut(self, asn):
        return sum(self.W[e] for e in self.edges if asn[e[0]] != asn[e[1]])

    def ncut(self, asn):
        return sum(1 for u, v in self.edges if asn[u] != asn[v])

    def log_pi(self, asn):
        return -self.lam * self.wcut(asn)


# ─── primitives ──────────────────────────────────────────────────────────────

def wilson_ust(nodes, adj):
    """Uniform spanning tree of the graph (nodes, adj) by Wilson's algorithm
    (loop-erased random walks). Returns a list of edges."""
    root = random.choice(nodes)
    in_tree = {root}
    nxt = {root: None}
    for s in nodes:
        u = s
        while u not in in_tree:
            nxt[u] = random.choice(adj[u])
            u = nxt[u]
        u = s
        while u not in in_tree:
            in_tree.add(u)
            u = nxt[u]
    return [(v, nxt[v]) for v in nodes if nxt[v] is not None]


def log_tau(graph, nodes):
    """log of the number of spanning trees of the induced subgraph on
    `nodes` (Matrix-Tree theorem, unweighted). 0 for a single node."""
    n = len(nodes)
    if n <= 1:
        return 0.0
    idx = {v: i for i, v in enumerate(nodes)}
    L = np.zeros((n, n))
    for u in nodes:
        i = idx[u]
        for w in graph.neighbors(u):
            j = idx.get(w)
            if j is not None:
                L[i, j] -= 1.0
                L[i, i] += 1.0
    sign, ld = np.linalg.slogdet(L[1:, 1:])
    if sign <= 0:
        raise ValueError("district is disconnected (zero spanning trees)")
    return float(ld)


def _cross(graph, S1, S2):
    S2 = set(S2)
    return sum(1 for u in S1 for w in graph.neighbors(u) if w in S2)


# ─── the two moves ───────────────────────────────────────────────────────────

def tree_step(tgt, asn, ideal, eps):
    graph = tgt.graph
    bp = sorted(border_pairs_set(asn, graph))
    d1, d2 = random.choice(bp)
    nodes = [v for v in asn if asn[v] in (d1, d2)]
    nodeset = set(nodes)
    adj = {v: [u for u in graph.neighbors(v) if u in nodeset] for v in nodes}
    rec = dict(kind="tree", feasible=False, accepted=False, log_alpha=np.nan)

    T = wilson_ust(nodes, adj)
    e = random.choice(T)
    tadj = {v: [] for v in nodes}
    for a, b in T:
        if (a, b) == e or (b, a) == e:
            continue
        tadj[a].append(b)
        tadj[b].append(a)
    A = set()
    stack = [e[0]]
    while stack:
        v = stack.pop()
        if v in A:
            continue
        A.add(v)
        stack.extend(tadj[v])
    B = nodeset - A

    lo, hi = (1 - eps) * ideal, (1 + eps) * ideal
    popA = sum(tgt.pop[v] for v in A)
    popB = sum(tgt.pop[v] for v in B)
    if not (lo <= popA <= hi and lo <= popB <= hi):
        return asn, rec
    rec["feasible"] = True

    if random.random() < 0.5:          # fair coin: bijection on labeled plans
        A, B = B, A
    new = dict(asn)
    for v in A:
        new[v] = d1
    for v in B:
        new[v] = d2

    Vi = [v for v in nodes if asn[v] == d1]
    Vj = [v for v in nodes if asn[v] == d2]
    lq_ratio = (log_tau(graph, Vi) + log_tau(graph, Vj)
                + math.log(_cross(graph, Vi, Vj))
                - log_tau(graph, list(A)) - log_tau(graph, list(B))
                - math.log(_cross(graph, A, B)))
    bp_new = border_pairs_set(new, graph)
    la = ((tgt.log_pi(new) - tgt.log_pi(asn)) + lq_ratio
          + math.log(len(bp)) - math.log(len(bp_new)))
    rec["log_alpha"] = la
    if math.log(random.random() + 1e-300) < la:
        rec["accepted"] = True
        return new, rec
    return asn, rec


def local_step(tgt, asn, ideal, eps, alpha, beta):
    """Chain E with the weighted cut in the acceptance ratio. Mirrors
    qp_mh_proposal exactly; with kernel_uniform it IS qp_mh_proposal."""
    graph = tgt.graph
    bp = sorted(border_pairs_set(asn, graph))
    d1, d2 = random.choice(bp)
    nodes = [v for v, d in asn.items() if d in (d1, d2)]
    rec = dict(kind="local", feasible=False, accepted=False, flips=0,
               log_alpha=np.nan, fast=False)

    xf, fast_f = solve_qp_fast(graph, nodes, asn, d1, alpha, beta, eps,
                               tgt.kernel_fn, ideal)
    if xf is None:
        return asn, rec
    prop, lq_f, _ = _randomized_round(xf, nodes, d1, d2, graph, asn)
    rec["flips"] = int(sum(1 for v in nodes if prop[v] != asn[v]))
    if not _pair_split_feasible(prop, nodes, d1, d2, graph, eps, ideal_pop=ideal):
        rec["fast"] = fast_f
        return asn, rec
    rec["feasible"] = True
    xr, fast_r = solve_qp_fast(graph, nodes, prop, d1, alpha, beta, eps,
                               tgt.kernel_fn, ideal)
    rec["fast"] = bool(fast_f and fast_r)
    if xr is None:
        return asn, rec
    xr_c = np.clip(np.asarray(xr), DELTA, 1 - DELTA)
    x0 = np.array([1.0 if asn[v] == d1 else 0.0 for v in nodes])
    lq_r = float(np.sum(x0 * np.log(xr_c) + (1 - x0) * np.log(1 - xr_c)))
    bp_new = border_pairs_set(prop, graph)
    la = ((tgt.log_pi(prop) - tgt.log_pi(asn)) + (lq_r - lq_f)
          + math.log(len(bp)) - math.log(max(len(bp_new), 1)))
    rec["log_alpha"] = la
    if math.log(random.random() + 1e-300) < la:
        rec["accepted"] = True
        return prop, rec
    return asn, rec


# ─── runner ──────────────────────────────────────────────────────────────────

def run_hybrid(tgt, init_asn, ideal, eps, n_steps, p_local, alpha, beta,
               seed, state_id=None, log_every=0, label=""):
    """p_local = 0 is the tree move alone; p_local = 1 is Chain E alone."""
    random.seed(seed)
    np.random.seed(seed)
    asn = dict(init_asn)
    wc = np.empty(n_steps)
    nc = np.empty(n_steps, dtype=np.int32)
    kind = np.empty(n_steps, dtype=np.int8)
    feas = np.zeros(n_steps, dtype=bool)
    acc = np.zeros(n_steps, dtype=bool)
    flips = np.zeros(n_steps, dtype=np.int32)
    fast = np.zeros(n_steps, dtype=bool)
    ids = np.full(n_steps, -1, dtype=np.int64)
    t0 = time.perf_counter()
    for t in range(n_steps):
        if random.random() < p_local:
            asn, r = local_step(tgt, asn, ideal, eps, alpha, beta)
            kind[t] = 1
            flips[t] = r["flips"]
            fast[t] = r["fast"]
        else:
            asn, r = tree_step(tgt, asn, ideal, eps)
            kind[t] = 0
        feas[t] = r["feasible"]
        acc[t] = r["accepted"]
        wc[t] = tgt.wcut(asn)
        nc[t] = tgt.ncut(asn)
        if state_id is not None:
            ids[t] = state_id(asn)
        if log_every and (t + 1) % log_every == 0:
            print(f"  [{label}] {t+1}/{n_steps}  {time.perf_counter()-t0:6.1f}s  "
                  f"wcut {wc[t]:.2f}  acc(tree) {acc[kind[:t+1]==0].mean() if (kind[:t+1]==0).any() else float('nan'):.3f}  "
                  f"acc(local) {acc[kind[:t+1]==1].mean() if (kind[:t+1]==1).any() else float('nan'):.3f}")
    return dict(wcut=wc, ncut=nc, kind=kind, feasible=feas, accepted=acc,
                flips=flips, fast=fast, ids=ids,
                wall=time.perf_counter() - t0, final=asn)


# ─── analytic kernel check on the 2x3 fiber ──────────────────────────────────

def db_check_hybrid(kernel_fn=kernel_uniform, lam=0.10, alpha=1.0, beta=10.0,
                    eps=1.0 / 3.0, p_local=0.5, density=None, verbose=True):
    """Assemble K_tree, K_local and K_mix analytically on the 18 labeled
    states of the 2x3 fiber and report the maximum relative violation of
    pi(P) K(P,P') = pi(P') K(P',P) for each. Returns the three maxima."""
    import networkx as nx
    from itertools import combinations

    G = nx.grid_graph([3, 2])
    for v in G.nodes():
        G.nodes[v]["population"] = 1
        G.nodes[v]["density"] = 0.0 if density is None else float(density(v))
    cells = sorted(G.nodes())
    ideal = 3.0
    tgt = Target(G, kernel_fn, lam)

    lo = int(math.ceil((1 - eps) * ideal - 1e-9))
    hi = int(math.floor((1 + eps) * ideal + 1e-9))
    states = []
    for size in range(lo, hi + 1):
        for S in combinations(cells, size):
            S, T = set(S), set(cells) - set(S)
            if T and nx.is_connected(G.subgraph(S)) and nx.is_connected(G.subgraph(T)):
                states.append({c: (0 if c in S else 1) for c in cells})
    n = len(states)
    logpi = np.array([tgt.log_pi(a) for a in states])
    pi = np.exp(logpi - logpi.max())
    pi /= pi.sum()

    # Tree kernel. Here k = 2, so H is the whole graph and |bp| = 1 always.
    lt_H = log_tau(G, cells)
    nH = len(cells)
    lq = np.empty(n)                      # log q(. -> state j), same for every source
    for j, a in enumerate(states):
        S = [c for c in cells if a[c] == 0]
        T = [c for c in cells if a[c] == 1]
        lq[j] = (log_tau(G, S) + log_tau(G, T) + math.log(_cross(G, S, T))
                 - math.log(2.0) - lt_H - math.log(nH - 1))
    K_tree = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            la = min(0.0, (logpi[j] - logpi[i]) + (lq[i] - lq[j]))
            K_tree[i, j] = math.exp(lq[j] + la)
        K_tree[i, i] = 1.0 - K_tree[i].sum()

    # Local kernel (Chain E with the weighted target), as in db_check_2x3.py.
    def bits(a):
        return np.array([1.0 if a[c] == 0 else 0.0 for c in cells])

    p_of = []
    n_fast = 0
    for a in states:
        x, fast = solve_qp_fast(G, cells, a, 0, alpha, beta, eps, kernel_fn, ideal)
        n_fast += int(fast)
        p_of.append(np.clip(np.asarray(x), DELTA, 1 - DELTA))
    K_loc = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            bj, bi = bits(states[j]), bits(states[i])
            lq_f = float(np.sum(bj * np.log(p_of[i]) + (1 - bj) * np.log(1 - p_of[i])))
            lq_r = float(np.sum(bi * np.log(p_of[j]) + (1 - bi) * np.log(1 - p_of[j])))
            la = min(0.0, (logpi[j] - logpi[i]) + lq_r - lq_f)
            K_loc[i, j] = math.exp(lq_f + la)
        K_loc[i, i] = 1.0 - K_loc[i].sum()

    K_mix = p_local * K_loc + (1 - p_local) * K_tree

    def viol(K):
        F = pi[:, None] * K
        den = np.maximum(np.abs(F), np.abs(F.T))
        m = den > 0
        rel = np.zeros_like(F)
        rel[m] = np.abs(F - F.T)[m] / den[m]
        return float(rel.max()), float(np.abs(K.sum(axis=1) - 1).max()), float(K.diagonal().min())

    out = {}
    for name, K in (("tree", K_tree), ("local", K_loc), ("mixture", K_mix)):
        v, rs, dmin = viol(K)
        out[name] = v
        if verbose:
            print(f"  {name:8s} max rel DB violation {v:.2e}   "
                  f"max |row sum - 1| {rs:.1e}   min diagonal {dmin:.4f}")
    if verbose:
        print(f"  ({n} labeled states, tau(H) = {math.exp(lt_H):.0f}, "
              f"sum_j q(.->j) = {np.exp(lq).sum():.4f} of 1 is balanced-feasible, "
              f"local solves by linear solve: {n_fast}/{n})")
    return out


if __name__ == "__main__":
    print("2x3 fiber, uniform kernel, lam = 0.10")
    db_check_hybrid()
    print("2x3 fiber, density kernel (rho = x coordinate, sigma = 1), lam = 1.0")
    from recom.qp_model import make_kernel_density
    db_check_hybrid(kernel_fn=make_kernel_density(sigma=1.0), lam=1.0,
                    density=lambda v: float(v[0]))
