"""
qp_model.py — Quadratic Programming redistricting model.

Contains
--------
build_laplacian(graph, nodes, attr)
    Unnormalised graph Laplacian L = D − W with kernel w(e) = exp(-|Δattr|).
    x^T L x = Σ w(e)·(x_u−x_v)² is the weighted cut; L is PSD (convex QP).

make_partition(gc_graph, assignment)
    Wrap a district assignment in a GerryChain Partition object.

_solve_qp(graph, nodes, assignment, d1_id, alpha, beta, epsilon)
    Solve the continuous QP relaxation for one district pair.
    Objective: min  α·x^T L x  +  β·‖x−x₀‖²  (strictly convex, no NonConvex flag).
    Minimising the weighted cut pushes boundaries to cross low-weight edges,
    directly increasing the Density Score.
    Returns x_star ∈ [0,1]^n or None on failure.

_randomized_round(x_star, nodes, d1_id, d2_id, graph, assignment, max_tries)
    Bernoulli rounding with connectivity check.
    Returns (new_assignment, log_q_fwd, bits) or (None, None, None).

qp_mh_proposal(graph, assignment, alpha, beta, epsilon, lam)
    One QP-MH step with exact Metropolis-Hastings correction.
    Returns (new_assignment, accepted).

run_qp_only(graph, gc_graph, initial_assignment, steps, alpha, beta, epsilon, lam)
    Run the standalone QP-MH chain for `steps` steps.
    Returns (metrics_list, times_list, accepts_list, final_assignment).

tune_qp_parameters(graph, gc_graph, initial_assignment, param_grid, steps)
    Grid search over (alpha, beta, epsilon, lam).
    Evaluates each combo by mean density_score over `steps` steps.
    Returns a sorted list of (mean_ds, params_dict) — best first.

Parameter guide
---------------
alpha   : weight of the Laplacian term (weighted-cut signal).
          Higher → QP minimises weighted cut more aggressively (stronger
          density alignment); lower → solution stays closer to x0 (β dominates).
          Typical range: [0.1, 5.0]

beta    : weight of the quadratic proximity term  β‖x − x₀‖².
          Higher → proposed partition stays close to current.  Lower → larger
          jumps but lower acceptance.  Typical range: [1.0, 50.0]

epsilon : population balance tolerance (fraction of ideal district pop).
          Typical range: [0.05, 0.20]

lam     : MH target temperature  π(x) ∝ exp(−λ · cut_edges(x)).
          Higher → strongly prefers fewer cut edges (less exploration).
          Typical range: [0.01, 0.20]
"""

import random
import time
import sys
from pathlib import Path
import numpy as np
import networkx as nx

import gurobipy as gp
from gurobipy import GRB

from gerrychain import Partition
from gerrychain.updaters import Tally, cut_edges

_DIFFUSION_ROOT = Path(__file__).resolve().parent.parent
if str(_DIFFUSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_DIFFUSION_ROOT))

from helpers.metrics import n_cut_edges, border_pairs_set, density_score, record_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Laplacian & partition helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_laplacian(graph, nodes, attr="density", kernel_fn=None,
                    weight_jitter=False):
    """
    Returns (L, idx) where L = D − W is the unnormalised graph Laplacian.

    Kernel choice:
      kernel_fn != None → w(u,v) = kernel_fn(graph, u, v)
      kernel_fn is None → legacy  w(u,v) = exp(-|Δattr|).

    weight_jitter=True  → multiply each edge weight by U[1, 2]
                          (SpecReCom trick from Davies et al. 2025 to
                          break the determinism of spectral / QP local
                          steps and improve coverage).

    Identity (any choice):  x^T L x = Σ_{e=(u,v)} w(u,v)·(x_u − x_v)².
    """
    n   = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    sub = graph.subgraph(nodes)
    W   = np.zeros((n, n))
    for u, v in sub.edges():
        if kernel_fn is not None:
            w = kernel_fn(graph, u, v)
        else:
            w = np.exp(-abs(graph.nodes[u].get(attr, 0)
                            - graph.nodes[v].get(attr, 0)))
        if weight_jitter:
            w *= np.random.uniform(1.0, 2.0)
        i, j = idx[u], idx[v]
        W[i, j] = w; W[j, i] = w
    L = np.diag(W.sum(axis=1)) - W      # L = D − W  (PSD)
    return L, idx


def build_laplacian_sym(graph, nodes, attr="density", kernel_fn=None,
                        weight_jitter=False):
    """Symmetric normalised Laplacian  L_sym = I − D^{-1/2} W D^{-1/2}.
    Same kernel-choice and weight-jitter semantics as `build_laplacian`."""
    n   = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    sub = graph.subgraph(nodes)
    W   = np.zeros((n, n))
    for u, v in sub.edges():
        if kernel_fn is not None:
            w = kernel_fn(graph, u, v)
        else:
            w = np.exp(-abs(graph.nodes[u].get(attr, 0)
                            - graph.nodes[v].get(attr, 0)))
        if weight_jitter:
            w *= np.random.uniform(1.0, 2.0)
        i, j = idx[u], idx[v]
        W[i, j] = w; W[j, i] = w
    d = W.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    Wn = (d_inv_sqrt[:, None] * W) * d_inv_sqrt[None, :]
    L_sym = np.eye(n) - Wn
    return L_sym, idx


# ─── Built-in kernels ───────────────────────────────────────────────────────

def kernel_uniform(graph, u, v):
    """Unweighted: w(u,v) = 1 for every edge."""
    return 1.0


def kernel_perimeter(graph, u, v):
    """w(u,v) = shared boundary length between u and v.
       Falls back to 1.0 if `shared_perim` is not on the edge."""
    return float(graph.edges[u, v].get("shared_perim", 1.0))


def make_kernel_density(sigma=1.0):
    """w(u,v) = shared_perim · exp(-(ρ_u − ρ_v)² / σ²)
       where ρ = graph.nodes[v]['density'].
       Discourages cuts inside same-density regions; encourages cuts
       across density transitions (urban↔rural)."""
    def kfn(graph, u, v):
        p = float(graph.edges[u, v].get("shared_perim", 1.0))
        rho_u = float(graph.nodes[u].get("density", 0.0))
        rho_v = float(graph.nodes[v].get("density", 0.0))
        return p * np.exp(-((rho_u - rho_v) ** 2) / (sigma ** 2))
    return kfn


KERNELS = {
    "uniform":   kernel_uniform,
    "perimeter": kernel_perimeter,
    # "density" is constructed via make_kernel_density(sigma) per call
}


def make_partition(gc_graph, assignment):
    return Partition(
        gc_graph,
        assignment=assignment,
        updaters={"population": Tally("population"), "cut_edges": cut_edges},
    )


# ─────────────────────────────────────────────────────────────────────────────
# QP solver
# ─────────────────────────────────────────────────────────────────────────────

def _solve_qp(graph, nodes, assignment, d1_id,
              alpha=1.0, beta=10.0, epsilon=0.10):
    """
    Minimise   α · x^T L x  +  β · ‖x − x₀‖²
    subject to  (1−ε)·p̄ ≤ popᵀx ≤ (1+ε)·p̄,   x ∈ [0,1]^n

    x^T L x = Σ_{e=(u,v)} w(e)·(x_u−x_v)²  is the weighted cut.
    Minimising it pushes boundaries to cross low-weight (cross-density)
    edges — directly increasing the Density Score.

    L is PSD  ⟹  objective is CONVEX.  No NonConvex flag needed.
    Gurobi solves this as a standard convex QP, which is significantly
    faster than the non-convex formulation.

    x₀  = current indicator (1 if node in D1, 0 if in D2).
    p̄   = half the total population of the merged pair.

    Returns x_star ∈ [0,1]^n or None on solver failure.
    """
    n      = len(nodes)
    L, _   = build_laplacian(graph, nodes, attr="density")
    x0     = np.array([1.0 if assignment[v] == d1_id else 0.0 for v in nodes])
    pop    = np.array([graph.nodes[v]["population"] for v in nodes])
    p_bar  = pop.sum() / 2.0

    try:
        m = gp.Model("qp")
        m.setParam("OutputFlag", 0)
        x = m.addVars(n, lb=0.0, ub=1.0, name="x")

        obj = gp.QuadExpr()
        for i in range(n):
            for j in range(i, n):
                val = L[i, j]
                if abs(val) > 1e-12:
                    if i == j:
                        obj += alpha * val * x[i] * x[i]            # α L diagonal
                    else:
                        obj += 2 * alpha * val * x[i] * x[j]        # α L off-diagonal
        for i in range(n):
            obj += beta * (x[i] * x[i] - 2.0 * x0[i] * x[i])       # β ‖x − x₀‖² (const ignored)

        m.setObjective(obj, GRB.MINIMIZE)
        pe = gp.LinExpr(pop.tolist(), [x[i] for i in range(n)])
        m.addConstr(pe >= (1 - epsilon) * p_bar)
        m.addConstr(pe <= (1 + epsilon) * p_bar)
        m.optimize()

        if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            return np.array([x[i].X for i in range(n)])
    except gp.GurobiError:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Randomized rounding
# ─────────────────────────────────────────────────────────────────────────────

def _randomized_round(x_star, nodes, d1_id, d2_id, graph, assignment,
                      max_tries=20):
    """
    Bernoulli(x*_v) rounding with district-connectivity check.

    Returns (new_assignment, log_q_fwd, bits) on success,
            (None, None, None)               after max_tries failures.
    """
    n = len(nodes)
    for _ in range(max_tries):
        bits    = (np.random.rand(n) < x_star).astype(float)
        new_asn = dict(assignment)
        for i, v in enumerate(nodes):
            new_asn[v] = d1_id if bits[i] == 1 else d2_id

        nodes1 = [v for v, d in new_asn.items() if d == d1_id]
        nodes2 = [v for v, d in new_asn.items() if d == d2_id]
        if not nodes1 or not nodes2:
            continue
        sub1 = graph.subgraph(nodes1)
        sub2 = graph.subgraph(nodes2)
        if nx.is_connected(sub1) and nx.is_connected(sub2):
            eps      = 1e-12
            x_star_c = np.clip(x_star, eps, 1 - eps)
            log_q    = np.sum(bits * np.log(x_star_c)
                              + (1 - bits) * np.log(1 - x_star_c))
            return new_asn, log_q, bits
    return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# QP-MH proposal
# ─────────────────────────────────────────────────────────────────────────────

def qp_mh_proposal(graph, assignment,
                   alpha=1.0, beta=10.0, epsilon=0.10, lam=0.05):
    """
    One QP-MH step.

    MH target : π(x) ∝ exp(−λ · cut_edges(x))
    Acceptance : log α = −λ·Δcut
                        + log q(x|x′,pair) − log q(x′|x,pair)
                        + log|bp(x)|       − log|bp(x′)|

    The QP proposal (min α·xᵀLx + β‖x−x₀‖²) generates candidates that
    minimise the weighted cut (= maximise density alignment).  The MH step
    then accepts/rejects them against the cut-edge target π, ensuring detailed balance.

    Returns (new_assignment, accepted_bool).
    """
    bp_cur   = border_pairs_set(assignment, graph)
    n_bp_cur = len(bp_cur)
    if n_bp_cur == 0:
        return dict(assignment), False

    d1_id, d2_id = random.choice(sorted(bp_cur))
    nodes        = [n for n, d in assignment.items() if d in (d1_id, d2_id)]

    # Forward QP + randomized round
    x_star_fwd = _solve_qp(graph, nodes, assignment, d1_id, alpha, beta, epsilon)
    if x_star_fwd is None:
        return dict(assignment), False

    proposed, log_q_fwd, _ = _randomized_round(
        x_star_fwd, nodes, d1_id, d2_id, graph, assignment)
    if proposed is None:
        return dict(assignment), False

    # Reverse QP (exact MH ratio)
    x_star_rev = _solve_qp(graph, nodes, proposed, d1_id, alpha, beta, epsilon)
    if x_star_rev is None:
        return dict(assignment), False

    eps       = 1e-12
    x_rev_c   = np.clip(x_star_rev, eps, 1 - eps)
    orig_bits = np.array([1.0 if assignment[v] == d1_id else 0.0 for v in nodes])
    log_q_rev = np.sum(orig_bits * np.log(x_rev_c)
                       + (1 - orig_bits) * np.log(1 - x_rev_c))

    cut_cur      = n_cut_edges(graph, assignment)
    cut_prop     = n_cut_edges(graph, proposed)
    bp_prop      = border_pairs_set(proposed, graph)
    n_bp_prop    = max(len(bp_prop), 1)

    log_alpha = (
        -lam * (cut_prop - cut_cur)
        + (log_q_rev - log_q_fwd)
        + (np.log(n_bp_cur) - np.log(n_bp_prop))
    )

    if np.log(random.random() + 1e-300) < log_alpha:
        return proposed, True
    return dict(assignment), False


# ─────────────────────────────────────────────────────────────────────────────
# QP diffusion step  (notes-aligned: L_sym + threshold rounding + repair)
# ─────────────────────────────────────────────────────────────────────────────

def _solve_qp_sym(graph, nodes, assignment, d1_id,
                  alpha=1.0, beta=1.0, epsilon=0.05, attr="density",
                  normalize=True, kernel_fn=None, weight_jitter=False):
    """Solve  min α·xᵀL x + β·‖x − x₀‖²
              s.t. (1−ε)·p̄ ≤ pᵀx ≤ (1+ε)·p̄,  x ∈ [0,1]^n.

    L is the (un)normalised Laplacian on the merged pair, with edge
    weights from `kernel_fn(graph, u, v)` if provided, else from
    `attr`-difference exponentials (legacy).  weight_jitter=True
    multiplies each weight by U[1, 2] (SpecReCom diversity trick).
    """
    n      = len(nodes)
    if normalize:
        L, _ = build_laplacian_sym(graph, nodes, attr=attr,
                                   kernel_fn=kernel_fn,
                                   weight_jitter=weight_jitter)
    else:
        L, _ = build_laplacian(graph, nodes, attr=attr,
                               kernel_fn=kernel_fn,
                               weight_jitter=weight_jitter)
    x0     = np.array([1.0 if assignment[v] == d1_id else 0.0 for v in nodes])
    pop    = np.array([graph.nodes[v]["population"] for v in nodes])
    p_bar  = pop.sum() / 2.0

    try:
        m = gp.Model("qp_sym")
        m.setParam("OutputFlag", 0)
        x = m.addVars(n, lb=0.0, ub=1.0, name="x")

        obj = gp.QuadExpr()
        for i in range(n):
            for j in range(i, n):
                val = L[i, j]
                if abs(val) > 1e-12:
                    if i == j:
                        obj += alpha * val * x[i] * x[i]
                    else:
                        obj += 2 * alpha * val * x[i] * x[j]
        for i in range(n):
            obj += beta * (x[i] * x[i] - 2.0 * x0[i] * x[i])

        m.setObjective(obj, GRB.MINIMIZE)
        pe = gp.LinExpr(pop.tolist(), [x[i] for i in range(n)])
        m.addConstr(pe >= (1 - epsilon) * p_bar)
        m.addConstr(pe <= (1 + epsilon) * p_bar)
        m.optimize()

        if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            return np.array([x[i].X for i in range(n)])
    except gp.GurobiError:
        pass
    return None


def _cheeger_sweep_round(x_star, nodes, d1_id, d2_id,
                         graph, assignment, epsilon, repair=True):
    """Cheeger sweep: among all sorted-x_star thresholds, return the
    assignment minimising cut on the merged pair subject to ε balance.

    For each threshold k = 1..n-1 we try the cut "top-k by x_star → D1".
    With `repair=True` (default), thresholds whose induced sides are
    disconnected are passed through orphan-flip repair before scoring;
    only thresholds for which repair fails are skipped.  With
    `repair=False`, disconnected thresholds are skipped (faithful to
    SpecReCom / Davies et al. 2025).

    Returns the assignment with min cut among feasible (possibly
    repaired) candidates, or None if none is feasible.
    """
    n   = len(nodes)
    pop = np.array([graph.nodes[v]["population"] for v in nodes])
    p_bar = pop.sum() / 2.0
    pop_lo = (1 - epsilon) * p_bar
    pop_hi = (1 + epsilon) * p_bar

    order = np.argsort(-x_star, kind="stable")
    pop_cum = np.cumsum(pop[order])

    sub = graph.subgraph(nodes)
    sub_edges = list(sub.edges())

    best_asn = None
    best_cuts = float("inf")

    for k in range(1, n):
        pop1 = pop_cum[k - 1]
        if pop1 < pop_lo or pop1 > pop_hi:
            continue

        side1 = [nodes[order[j]] for j in range(k)]
        side2 = [nodes[order[j]] for j in range(k, n)]

        cand_asn = dict(assignment)
        for v in side1:
            cand_asn[v] = d1_id
        for v in side2:
            cand_asn[v] = d2_id

        connected = (nx.is_connected(sub.subgraph(side1))
                     and nx.is_connected(sub.subgraph(side2)))
        if not connected:
            if not repair:
                continue
            cand_asn = _orphan_flip_repair(
                cand_asn, nodes, d1_id, d2_id, graph, epsilon)
            if cand_asn is None:
                continue

        side1_set = {v for v in nodes if cand_asn[v] == d1_id}
        cuts = sum(1 for u, v in sub_edges
                   if (u in side1_set) != (v in side1_set))

        if cuts < best_cuts:
            best_cuts = cuts
            best_asn = cand_asn

    return best_asn


def _orphan_flip_repair(new_asn, nodes, d1_id, d2_id, graph, epsilon):
    """Orphan-flip contiguity repair: for each side, keep the largest
    connected component and flip smaller components to the opposite
    side.  Returns repaired assignment if both sides end up connected
    and ε-balanced, else None.

    Used by the IQP local step (and by `_threshold_round_and_repair`
    via its rounding fallback).
    """
    pop_total = sum(graph.nodes[v]["population"] for v in nodes)
    p_bar = pop_total / 2.0

    for d_id, opp in ((d1_id, d2_id), (d2_id, d1_id)):
        side = [v for v in nodes if new_asn[v] == d_id]
        if not side:
            continue
        sub_side = graph.subgraph(side)
        if not nx.is_connected(sub_side):
            comps = sorted(nx.connected_components(sub_side),
                           key=len, reverse=True)
            for c in comps[1:]:
                for v in c:
                    new_asn[v] = opp

    nodes1 = [v for v in nodes if new_asn[v] == d1_id]
    nodes2 = [v for v in nodes if new_asn[v] == d2_id]
    if not nodes1 or not nodes2:
        return None
    if not nx.is_connected(graph.subgraph(nodes1)):
        return None
    if not nx.is_connected(graph.subgraph(nodes2)):
        return None

    pop1 = sum(graph.nodes[v]["population"] for v in nodes1)
    if not ((1 - epsilon) * p_bar <= pop1 <= (1 + epsilon) * p_bar):
        return None
    return new_asn


def _threshold_round_and_repair(x_star, nodes, d1_id, d2_id,
                                graph, assignment, epsilon,
                                threshold="cheeger"):
    """Round x_star and ensure (ε balance, connectivity).

    threshold:
      "cheeger" → SpecReCom-style sweep, min-cut feasible (RECOMMENDED)
      "median"  → t = median(x_star), then orphan-flip repair
      "balance" → balance-adjusted t so cumulative pop ≈ p̄, then repair
      0.5       → t = 0.5, then repair

    For the "cheeger" mode we attempt the strict sweep first.  If no
    feasible cut exists (rare, but possible on tight ε), we fall back
    to "balance" with repair.
    """
    if threshold == "cheeger":
        result = _cheeger_sweep_round(
            x_star, nodes, d1_id, d2_id, graph, assignment, epsilon)
        if result is not None:
            return result
        threshold = "balance"  # fallback

    n   = len(nodes)
    pop = np.array([graph.nodes[v]["population"] for v in nodes])
    p_bar = pop.sum() / 2.0

    if threshold == "median":
        t = float(np.median(x_star))
    elif threshold == "balance":
        order = np.argsort(-x_star, kind="stable")
        cum = np.cumsum(pop[order])
        k = int(np.argmin(np.abs(cum - p_bar))) + 1
        if k >= n:
            t = float(x_star[order[-1]]) - 1.0
        else:
            t = (float(x_star[order[k - 1]]) + float(x_star[order[k]])) / 2.0
    else:
        t = float(threshold)

    bits = (x_star >= t).astype(int)
    new_asn = dict(assignment)
    for i, v in enumerate(nodes):
        new_asn[v] = d1_id if bits[i] == 1 else d2_id

    # Orphan-flip repair
    for d_id, opp in ((d1_id, d2_id), (d2_id, d1_id)):
        side = [v for v in nodes if new_asn[v] == d_id]
        if not side:
            continue
        sub_side = graph.subgraph(side)
        if not nx.is_connected(sub_side):
            comps = sorted(nx.connected_components(sub_side), key=len, reverse=True)
            for c in comps[1:]:
                for v in c:
                    new_asn[v] = opp

    nodes1 = [v for v in nodes if new_asn[v] == d1_id]
    nodes2 = [v for v in nodes if new_asn[v] == d2_id]
    if not nodes1 or not nodes2:
        return None
    if not nx.is_connected(graph.subgraph(nodes1)):
        return None
    if not nx.is_connected(graph.subgraph(nodes2)):
        return None

    pop1 = sum(graph.nodes[v]["population"] for v in nodes1)
    if not ((1 - epsilon) * p_bar <= pop1 <= (1 + epsilon) * p_bar):
        return None

    return new_asn


def qp_diffusion_step(graph, assignment, alpha=1.0, beta=1.0,
                      epsilon=0.05, threshold="median", attr="density",
                      normalize=True, kernel_fn=None,
                      weight_jitter=False, forbid_pairs=None):
    """One QP-diffusion proposal step.

    Pipeline (matching notes Algorithm 2 + §4.3):
      1. Pick a random adjacent district pair (D1, D2).
      2. Solve QP: min α·xᵀL_sym x + β·‖x − x₀‖²  on H = G[D1 ∪ D2].
      3. Threshold round x_star to a binary assignment.
      4. Repair contiguity (absorb orphan components).
      5. Verify ε population balance; return new assignment or revert.

    No Metropolis ratio — the step is deterministic given the chosen
    pair and threshold.  Ergodicity of the overall sampler comes from
    mixing with ReCom (see iowa_target_matched.py).
    """
    bp = border_pairs_set(assignment, graph)
    if forbid_pairs:
        bp = bp - set(forbid_pairs)
    if not bp:
        return dict(assignment), False, None
    d1_id, d2_id = random.choice(sorted(bp))
    chosen_pair = (d1_id, d2_id)
    nodes = [n for n, d in assignment.items() if d in (d1_id, d2_id)]

    x_star = _solve_qp_sym(graph, nodes, assignment, d1_id,
                           alpha=alpha, beta=beta, epsilon=epsilon,
                           attr=attr, normalize=normalize,
                           kernel_fn=kernel_fn,
                           weight_jitter=weight_jitter)
    if x_star is None:
        return dict(assignment), False, chosen_pair

    new_asn = _threshold_round_and_repair(
        x_star, nodes, d1_id, d2_id, graph, assignment, epsilon,
        threshold=threshold,
    )
    if new_asn is None:
        return dict(assignment), False, chosen_pair

    return new_asn, True, chosen_pair


# ─────────────────────────────────────────────────────────────────────────────
# SpecReCom step  (Davies et al. 2025: Fiedler vector + Cheeger sweep)
# ─────────────────────────────────────────────────────────────────────────────

def spec_recom_step(graph, assignment, epsilon=0.05, attr="density",
                    normalize=False):
    """SpecReCom step (Davies, Job, Kampbell, Kim, Seo 2025).

    Faithful to Algorithm 2 of the SpecReCom paper: pick a random
    adjacent district pair, **randomise the merged-pair edge weights
    uniformly in [1, 2]** (lines 4-6 of their algorithm), compute the
    Fiedler vector, and apply Cheeger sweep at threshold 0 (split
    on sign of Fiedler entries) — or via our cheeger sweep at any
    sorted-x threshold for population balance.

    The edge-weight randomisation is what breaks the determinism of
    pure spectral bisection (without it, the same merged pair would
    always yield the same partition).
    """
    bp = border_pairs_set(assignment, graph)
    if not bp:
        return dict(assignment), False
    d1_id, d2_id = random.choice(sorted(bp))
    nodes = [n for n, d in assignment.items() if d in (d1_id, d2_id)]
    if len(nodes) < 2:
        return dict(assignment), False

    # weight_jitter=True is faithful to Davies et al. 2025 alg. 2 lines 4-6
    if normalize:
        L, _ = build_laplacian_sym(graph, nodes, attr=attr,
                                   weight_jitter=True)
    else:
        L, _ = build_laplacian(graph, nodes, attr=attr,
                               weight_jitter=True)

    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return dict(assignment), False

    # Fiedler vector = eigenvector of 2nd-smallest eigenvalue.
    # eigh returns sorted ascending; index 0 is the trivial constant
    # eigenvector at λ=0; index 1 is the Fiedler vector.
    phi = eigvecs[:, 1]

    # Faithful to Davies et al. 2025: no repair. Repair helps when the
    # method has a fidelity term (β > 0); SpecReCom has none, and repair
    # tends to accept lower-quality cuts on average without that anchor.
    new_asn = _cheeger_sweep_round(
        phi, nodes, d1_id, d2_id, graph, assignment, epsilon,
        repair=False)
    if new_asn is None:
        return dict(assignment), False
    return new_asn, True


# ─────────────────────────────────────────────────────────────────────────────
# Integer QP local step  (MIQP — integer version of the continuous QP)
# ─────────────────────────────────────────────────────────────────────────────

def iqp_local_step(graph, assignment, alpha=1.0, beta=1.0,
                   epsilon=0.05, attr="density",
                   normalize=False, kernel_fn=None,
                   time_limit=5.0, weight_jitter=False,
                   forbid_pairs=None):
    """Integer-domain version of the continuous QP-diffusion model.

    Solves
        min  α·xᵀL x  +  β·‖x − x₀‖²
        s.t. (1−ε)·p̄ ≤ pᵀx ≤ (1+ε)·p̄,
             x ∈ {0,1}^n.

    Same objective, same constraints, same kernel as `qp_diffusion_step`
    — only the variable domain changes. No auxiliary cut variables, no
    MIP linearization. Gurobi solves it as a convex MIQP (L is PSD).

    The β·‖x − x₀‖² term anchors the solution near the current
    (contiguous) partition, which (i) drastically reduces post-hoc
    connectivity rejections, (ii) tightens the QP relaxation used at
    each branch-and-bound node, often speeding up the solve.
    """
    bp = border_pairs_set(assignment, graph)
    if forbid_pairs:
        bp = bp - set(forbid_pairs)
    if not bp:
        return dict(assignment), False, None

    d1_id, d2_id = random.choice(sorted(bp))
    chosen_pair = (d1_id, d2_id)
    nodes = [n for n, d in assignment.items() if d in (d1_id, d2_id)]
    n = len(nodes)
    if n < 2:
        return dict(assignment), False, chosen_pair

    if normalize:
        L, _ = build_laplacian_sym(graph, nodes, attr=attr,
                                   kernel_fn=kernel_fn,
                                   weight_jitter=weight_jitter)
    else:
        L, _ = build_laplacian(graph, nodes, attr=attr,
                               kernel_fn=kernel_fn,
                               weight_jitter=weight_jitter)

    x0 = np.array([1.0 if assignment[v] == d1_id else 0.0 for v in nodes])
    pop = np.array([graph.nodes[v]["population"] for v in nodes])
    p_bar = pop.sum() / 2.0

    try:
        m = gp.Model("iqp")
        m.setParam("OutputFlag", 0)
        m.setParam("TimeLimit", time_limit)

        x = m.addVars(n, vtype=GRB.BINARY, name="x")

        obj = gp.QuadExpr()
        for i in range(n):
            for j in range(i, n):
                val = L[i, j]
                if abs(val) > 1e-12:
                    if i == j:
                        obj += alpha * val * x[i] * x[i]
                    else:
                        obj += 2 * alpha * val * x[i] * x[j]
        for i in range(n):
            obj += beta * (x[i] * x[i] - 2.0 * x0[i] * x[i])

        m.setObjective(obj, GRB.MINIMIZE)
        pe = gp.LinExpr(pop.tolist(), [x[i] for i in range(n)])
        m.addConstr(pe >= (1 - epsilon) * p_bar)
        m.addConstr(pe <= (1 + epsilon) * p_bar)
        m.optimize()

        if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
            return dict(assignment), False, chosen_pair
        if m.SolCount == 0:
            return dict(assignment), False, chosen_pair
    except gp.GurobiError:
        return dict(assignment), False, chosen_pair

    new_asn = dict(assignment)
    for i, v in enumerate(nodes):
        new_asn[v] = d1_id if x[i].X > 0.5 else d2_id

    repaired = _orphan_flip_repair(new_asn, nodes, d1_id, d2_id,
                                   graph, epsilon)
    if repaired is None:
        return dict(assignment), False, chosen_pair
    return repaired, True, chosen_pair


# ─────────────────────────────────────────────────────────────────────────────
# Standalone QP-only runner
# ─────────────────────────────────────────────────────────────────────────────

def run_qp_only(graph, gc_graph, initial_assignment, steps=50,
                alpha=1.0, beta=10.0, epsilon=0.10, lam=0.05,
                verbose=True):
    """
    Run the QP-MH chain in isolation for `steps` steps.

    Parameters
    ----------
    graph              : networkx.Graph  (with 'density' and 'population' attrs)
    gc_graph           : gerrychain.Graph
    initial_assignment : dict {node → district_id}
    steps              : number of QP-MH steps
    alpha, beta, epsilon, lam : QP-MH hyperparameters (see module docstring)
    verbose            : print per-step diagnostics

    Returns
    -------
    metrics  : list of dicts (density_score, pp, pop_dev per step)
    times    : list of wall-clock seconds per step
    accepts  : list of 0/1 per step
    final    : final assignment dict
    """
    asn     = dict(initial_assignment)
    metrics = []
    times   = []
    accepts = []

    for s in range(steps):
        t0       = time.perf_counter()
        asn, acc = qp_mh_proposal(graph, asn, alpha=alpha, beta=beta,
                                  epsilon=epsilon, lam=lam)
        elapsed  = time.perf_counter() - t0
        m        = record_metrics(graph, asn)
        metrics.append(m)
        times.append(elapsed)
        accepts.append(int(acc))

        if verbose:
            print(f"  [QP-only] step {s+1:3d}  "
                  f"DS={m['density_score']:.4f}  PP={m['pp']:.4f}  "
                  f"acc={acc}  t={elapsed:.3f}s")

    ar = 100 * np.mean(accepts) if accepts else float("nan")
    if verbose:
        print(f"  [QP-only] accept rate: {ar:.1f}%  "
              f"mean step: {1e3*np.mean(times):.1f}ms\n")

    return metrics, times, accepts, asn


# ─────────────────────────────────────────────────────────────────────────────
# Parameter tuning via grid search
# ─────────────────────────────────────────────────────────────────────────────

def tune_qp_parameters(graph, gc_graph, initial_assignment,
                        param_grid=None, steps=30, verbose=True):
    """
    Grid search over QP-MH hyperparameters.

    Parameters
    ----------
    graph, gc_graph, initial_assignment : as above
    param_grid : dict with lists of values for each hyperparameter.
        Default grid:
            alpha   : [0.5, 1.0, 2.0]
            beta    : [5.0, 10.0, 20.0]
            epsilon : [0.05, 0.10, 0.15]
            lam     : [0.02, 0.05, 0.10]
    steps   : QP-MH steps per combo (short runs — for speed)
    verbose : print each combo result

    Returns
    -------
    results : list of (mean_density_score, params_dict), sorted best-first.

    Notes
    -----
    mean_density_score measures how well the chain's output aligns districts
    with the density kernel.  Higher is better.

    Tie-breaking: among combos with similar DS, lower pop_dev is preferred.
    The returned list is sorted by (−mean_DS, mean_pop_dev).
    """
    if param_grid is None:
        param_grid = {
            "alpha":   [0.5, 1.0, 2.0],
            "beta":    [5.0, 10.0, 20.0],
            "epsilon": [0.05, 0.10, 0.15],
            "lam":     [0.02, 0.05, 0.10],
        }

    # Build all combos (Cartesian product without itertools to keep deps minimal)
    keys = list(param_grid.keys())
    combos = [{}]
    for k in keys:
        combos = [dict(**c, **{k: v}) for c in combos for v in param_grid[k]]

    if verbose:
        print(f"Tuning QP parameters: {len(combos)} combos × {steps} steps each")
        print("-" * 72)

    results = []
    for i, params in enumerate(combos):
        metrics, _, accepts, _ = run_qp_only(
            graph, gc_graph, initial_assignment,
            steps=steps, verbose=False, **params
        )
        ds_vals  = [m["density_score"] for m in metrics]
        pd_vals  = [m["pop_dev"]       for m in metrics]
        mean_ds  = float(np.mean(ds_vals))
        mean_pd  = float(np.mean(pd_vals))
        acc_rate = 100 * np.mean(accepts)

        results.append((mean_ds, mean_pd, params))

        if verbose:
            print(f"  [{i+1:3d}/{len(combos)}] "
                  f"α={params['alpha']:.1f}  β={params['beta']:.1f}  "
                  f"ε={params['epsilon']:.2f}  λ={params['lam']:.2f}  │  "
                  f"mean DS={mean_ds:.4f}  pop_dev={mean_pd:.4f}  "
                  f"acc={acc_rate:.1f}%")

    # Sort: highest DS first; break ties by lowest pop_dev
    results.sort(key=lambda r: (-r[0], r[1]))

    if verbose:
        print("\n── Top 5 parameter sets ─────────────────────────────────────────────")
        for rank, (ds, pd, p) in enumerate(results[:5], 1):
            print(f"  #{rank}  mean DS={ds:.4f}  pop_dev={pd:.4f}  "
                  f"α={p['alpha']}  β={p['beta']}  ε={p['epsilon']}  λ={p['lam']}")

    return [(ds, p) for ds, _, p in results]


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    warnings.filterwarnings("ignore")

    from grid_setup import graph, gc_graph, initial_assignment, ns

    PLOTS_DIR = Path(__file__).parent.parent / "plots"
    PLOTS_DIR.mkdir(exist_ok=True)

    # ── 1. Parameter tuning ───────────────────────────────────────────────────
    print("=" * 72)
    print("QP-MH Parameter Tuning")
    print("=" * 72)
    tuning_results = tune_qp_parameters(
        graph, gc_graph, initial_assignment,
        param_grid={
            "alpha":   [0.5, 1.0, 2.0],
            "beta":    [5.0, 10.0, 20.0],
            "epsilon": [0.05, 0.10],
            "lam":     [0.02, 0.05, 0.10],
        },
        steps=30,
        verbose=True,
    )

    best_ds, best_params = tuning_results[0]
    print(f"\nBest params: {best_params}  (mean DS={best_ds:.4f})")

    # ── 2. Full run with best params ──────────────────────────────────────────
    print("\n" + "=" * 72)
    print("QP-MH Full Run (100 steps, best parameters)")
    print("=" * 72)
    metrics, times, accepts, final_asn = run_qp_only(
        graph, gc_graph, initial_assignment,
        steps=100, verbose=True, **best_params
    )

    # ── 3. Tuning heatmap: mean DS by (alpha, beta) for best epsilon+lam ─────
    best_eps = best_params["epsilon"]
    best_lam = best_params["lam"]
    alphas   = [0.5, 1.0, 2.0, 3.0]
    betas    = [5.0, 10.0, 20.0, 30.0]

    heatmap = np.zeros((len(alphas), len(betas)))
    print("\nGenerating alpha×beta heatmap …")
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            m, _, _, _ = run_qp_only(
                graph, gc_graph, initial_assignment,
                steps=20, verbose=False,
                alpha=a, beta=b, epsilon=best_eps, lam=best_lam,
            )
            heatmap[i, j] = np.mean([x["density_score"] for x in m])

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(heatmap, aspect="auto", origin="lower",
                   cmap="RdYlGn", vmin=heatmap.min(), vmax=heatmap.max())
    ax.set_xticks(range(len(betas)));  ax.set_xticklabels(betas)
    ax.set_yticks(range(len(alphas))); ax.set_yticklabels(alphas)
    ax.set_xlabel("beta  (proximity weight)");  ax.set_ylabel("alpha  (weighted-cut weight)")
    ax.set_title(
        f"Mean Density Score — QP-MH\n"
        f"ε={best_eps}  λ={best_lam}  (20 steps each)\n"
        r"$DS(x)=\Sigma_{\rm intra}w(e)/\Sigma_{\rm all}w(e)$"
    )
    plt.colorbar(im, ax=ax, label="Mean DS")
    for i in range(len(alphas)):
        for j in range(len(betas)):
            ax.text(j, i, f"{heatmap[i,j]:.3f}", ha="center", va="center",
                    fontsize=8, color="black")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "qp_tuning_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: qp_tuning_heatmap.png")

    # ── 4. Full-run density-score trace plot ──────────────────────────────────
    ds_trace = [m["density_score"] for m in metrics]
    fig, ax  = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(ds_trace) + 1), ds_trace, "b-", alpha=0.5, lw=0.8)
    cum_mean = np.cumsum(ds_trace) / np.arange(1, len(ds_trace) + 1)
    ax.plot(range(1, len(ds_trace) + 1), cum_mean, "b--", lw=2,
            label=f"Running mean  (final={cum_mean[-1]:.4f})")
    ax.set_xlabel("Step"); ax.set_ylabel("Density Score")
    ax.set_title(
        f"QP-MH  Density Score Trace  "
        f"(α={best_params['alpha']} β={best_params['beta']} "
        f"ε={best_params['epsilon']} λ={best_params['lam']})\n"
        r"$DS(x)=\Sigma_{\rm intra}w(e)/\Sigma_{\rm all}w(e)$,"
        r"  $w(e)=\exp(-|\Delta\mathrm{density}|)$"
    )
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "qp_trace.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: qp_trace.png")

    print("\nDone.")
