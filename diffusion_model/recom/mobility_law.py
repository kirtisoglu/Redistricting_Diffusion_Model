"""The second-order mobility law for Chain E, derived and checked.

Section 4.7 established that on an exact-balance fiber every single-unit
proposal is infeasible and that feasible proposals are accepted with
probability close to one. So the chain moves through coordinated draws,
the cheapest of which is a two-unit SWAP: one unit u from district i
and one unit v from district j change sides together, which preserves
both populations exactly. The mobility of the chain is then, to leading
order, twice the probability that the Bernoulli rounding produces
exactly one feasible swap.

THE DERIVATION. Rounding flips unit v independently with probability
p_v = |x*_v - x0_v| (Proposition 4.2). The probability that exactly the
pair {u, v} flips is

    p_u p_v prod_{w != u, v} (1 - p_w),

so the two-swap channel contributes

    M_2 = 2 * sum_{u in V_i, v in V_j, swap(u,v) feasible} p_u p_v prod_{w != u,v} (1 - p_w)

units per step, with the sum over cross-boundary pairs whose swap keeps
both districts connected. This is exact for the two-unit channel, with
no small-tau assumption; it omits only four-unit and larger coordinated
draws and the (near-unity) Metropolis acceptance. For small tau,
p_v = tau b_v + O(tau^2) with b_v the kernel weight of v's cross-boundary
edges (Corollary 4.1), so

    M_2 = 2 tau^2 S_2 + O(tau^3),    S_2 = sum_{feasible swaps} b_u b_v,

a closed-form coefficient computable from the current plan.

THE CHECK. Run Chain E at several tau, and at every step compute M_2 and
2 tau^2 S_2 for the pair actually chosen under the plan actually visited,
from the same x* the proposal used. Compare their chain averages with the
measured units moved per step. If the exact channel matches the
measurement at small tau and falls below it as tau grows, the law is
confirmed and its range of validity is read off directly: the gap is the
four-unit channel appearing.

Uses the linear-solve path of hybrid.solve_qp_fast, which on this fiber
is always the QP optimum (Lemma 4.1).

Writes: iowa_hybrid_test_out/mobility_law_5x5.npz
Usage:  python3 recom/mobility_law.py [n_steps_per_tau]
"""

import math
import random
import sys
from pathlib import Path

import numpy as np
import networkx as nx

_HERE = Path(__file__).resolve().parent
_DIFF = _HERE.parent
_REPO = _DIFF.parent
if str(_DIFF) not in sys.path:
    sys.path.insert(0, str(_DIFF))

from recom.problems import setup_problem
from recom.qp_model import _randomized_round, _pair_split_feasible, kernel_uniform
from recom.hybrid import solve_qp_fast, DELTA
from helpers.metrics import border_pairs_set, n_cut_edges

TAUS = [0.01, 0.03, 0.1, 0.3]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
LAM = 0.10
SEED = 42


def swap_feasible(graph, Vi, Vj, u, v):
    """Both districts stay connected after u -> j and v -> i."""
    Vi2 = (Vi - {u}) | {v}
    Vj2 = (Vj - {v}) | {u}
    return (nx.is_connected(graph.subgraph(Vi2))
            and nx.is_connected(graph.subgraph(Vj2)))


def channel_terms(graph, nodes, asn, d1, d2, x_star, tau, eps, ideal, n_bp,
                  cut_cur):
    """Return (M_2 with the Metropolis factor, M_2 without it, S_2).

    The Metropolis factor matters. A swap can lengthen the boundary, and
    the reverse proposal density at the swapped plan differs from the
    forward one, so a genuine two-swap is accepted with probability
    noticeably below one even though the chain's overall acceptance given
    feasibility is near one (that statistic is dominated by zero-flip
    proposals, which are trivially accepted). Each candidate swap needs
    one reverse linear solve, which Lemma 4.1 makes cheap."""
    x0 = np.array([1.0 if asn[v] == d1 else 0.0 for v in nodes])
    xc = np.clip(x_star, DELTA, 1 - DELTA)
    p = np.abs(xc - x0)
    one_minus = 1.0 - p
    log_all = float(np.sum(np.log(one_minus)))
    idx = {v: i for i, v in enumerate(nodes)}
    Vi = {v for v in nodes if asn[v] == d1}
    Vj = {v for v in nodes if asn[v] == d2}
    nodeset = set(nodes)
    b = {v: sum(1 for w in graph.neighbors(v) if w in nodeset and asn[w] != asn[v])
         for v in nodes}
    M2_acc, M2_raw, S2 = 0.0, 0.0, 0.0
    for u in Vi:
        if b[u] == 0:
            continue
        for v in Vj:
            if b[v] == 0:
                continue
            if not swap_feasible(graph, Vi, Vj, u, v):
                continue
            iu, iv = idx[u], idx[v]
            lq_f = (math.log(p[iu]) + math.log(p[iv]) + log_all
                    - math.log(one_minus[iu]) - math.log(one_minus[iv]))
            prob = math.exp(lq_f)
            M2_raw += prob
            S2 += b[u] * b[v]
            # Metropolis factor for this specific swap.
            prop = dict(asn)
            prop[u], prop[v] = d2, d1
            xr, _ = solve_qp_fast(graph, nodes, prop, d1, tau, 1.0, eps,
                                  kernel_uniform, ideal)
            xr_c = np.clip(np.asarray(xr), DELTA, 1 - DELTA)
            lq_r = float(np.sum(x0 * np.log(xr_c) + (1 - x0) * np.log(1 - xr_c)))
            dcut = n_cut_edges(graph, prop) - cut_cur
            n_bp_p = max(len(border_pairs_set(prop, graph)), 1)
            la = min(0.0, -LAM * dcut + (lq_r - lq_f)
                     + math.log(n_bp) - math.log(n_bp_p))
            M2_acc += prob * math.exp(la)
    return 2.0 * M2_acc, 2.0 * M2_raw, S2


def main():
    p = setup_problem("5x5", _REPO)
    graph, eps, ideal = p["graph"], p["eps"], p["ideal"]
    print(f"5x5 fiber, {N} steps per tau, alpha = tau, beta = 1\n")
    print(f"{'tau':>6} {'measured':>9} {'M_2 w/ accept':>14} {'M_2 no accept':>14} "
          f"{'2t^2 S_2':>9} {'w/acc / meas':>13} {'<S_2>':>7}")
    out = {}
    for tau in TAUS:
        random.seed(SEED)
        np.random.seed(SEED)
        asn = dict(p["init"])
        moved, M2s, M2raw, S2s = [], [], [], []
        for _ in range(N):
            bp = sorted(border_pairs_set(asn, graph))
            d1, d2 = random.choice(bp)
            nodes = [v for v, d in asn.items() if d in (d1, d2)]
            xf, _ = solve_qp_fast(graph, nodes, asn, d1, tau, 1.0, eps,
                                  kernel_uniform, ideal)
            cut_cur = n_cut_edges(graph, asn)
            m2, m2r, s2 = channel_terms(graph, nodes, asn, d1, d2, np.asarray(xf),
                                        tau, eps, ideal, len(bp), cut_cur)
            M2s.append(m2)
            M2raw.append(m2r)
            S2s.append(s2)
            prop, lq_f, _ = _randomized_round(xf, nodes, d1, d2, graph, asn)
            flips = sum(1 for v in nodes if prop[v] != asn[v])
            if not _pair_split_feasible(prop, nodes, d1, d2, graph, eps, ideal_pop=ideal):
                moved.append(0)
                continue
            xr, _ = solve_qp_fast(graph, nodes, prop, d1, tau, 1.0, eps,
                                  kernel_uniform, ideal)
            xr_c = np.clip(np.asarray(xr), DELTA, 1 - DELTA)
            x0 = np.array([1.0 if asn[v] == d1 else 0.0 for v in nodes])
            lq_r = float(np.sum(x0 * np.log(xr_c) + (1 - x0) * np.log(1 - xr_c)))
            dcut = n_cut_edges(graph, prop) - n_cut_edges(graph, asn)
            bp_p = border_pairs_set(prop, graph)
            la = -LAM * dcut + (lq_r - lq_f) + math.log(len(bp)) - math.log(max(len(bp_p), 1))
            if math.log(random.random() + 1e-300) < la:
                asn = prop
                moved.append(flips)
            else:
                moved.append(0)
        mv = float(np.mean(moved)); m2 = float(np.mean(M2s))
        m2r = float(np.mean(M2raw)); s2 = float(np.mean(S2s))
        small = 2 * tau * tau * s2
        print(f"{tau:6.2f} {mv:9.5f} {m2:14.5f} {m2r:14.5f} {small:9.5f} "
              f"{(m2 / mv if mv > 0 else float('nan')):13.2f} {s2:7.2f}")
        out[f"tau{tau}"] = np.array([mv, m2, m2r, small, s2])
    np.savez(_HERE / "iowa_hybrid_test_out" / "mobility_law_5x5.npz",
             taus=np.array(TAUS), n_steps=N, **out)
    print("\nSaved mobility_law_5x5.npz")
    print("READ: M_2 / measured near 1 means the two-swap channel accounts for the "
          "chain's mobility; falling below 1 as tau grows is the four-unit channel. "
          "The small-tau column should track M_2 at tau <= 0.03 and overshoot after, "
          "as the move-size law itself does.")


if __name__ == "__main__":
    main()
