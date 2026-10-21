"""Acceptance and mobility of Chain E against the diffusion timestep.

The move-size law says how far a proposal reaches. It says nothing about
how often the proposal is accepted, and mobility is the product of the
two. This sweep measures the second factor and separates its two causes,
which turn out to be different in kind:

  * infeasibility: the Bernoulli draw breaks contiguity or balance and is
    rejected with probability one, before any coin is flipped;
  * the Metropolis ratio: the draw is a valid plan and loses the coin.

THE MECHANISM THIS IS BUILT TO EXPOSE. On the exact-balance 5x5 fiber
every single-unit move is infeasible, because it produces districts of
sizes 4 and 6. The small-tau law says the leading-order proposal IS a
single-unit move. So at small tau the chain generates almost nothing but
infeasible proposals, and it can move only through the O(tau^2)
coordinated multi-unit draws. That is the mechanism behind "6 of 4006
states in 1000 steps", stated as a consequence of the theory rather than
an observation. It also predicts an interior optimum in tau: too small
and nothing is feasible, too large and the draws are incoherent and fail
contiguity. Units moved per step should peak in between, and that peak is
the tuning rule's real target.

Reimplements one Chain E step from the same primitives qp_model.py uses,
purely so the proposal can be logged before accept/reject. The paper's
qp_mh_proposal is untouched. alpha = tau, beta = 1, since only the ratio
enters the optimum.

Writes: iowa_hybrid_test_out/acceptance_sweep_5x5.npz
Usage:  python3 recom/acceptance_sweep.py [n_steps_per_tau]
"""

import random
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_DIFF = _HERE.parent
_REPO = _DIFF.parent
if str(_DIFF) not in sys.path:
    sys.path.insert(0, str(_DIFF))

from recom.problems import setup_problem
from recom.qp_model import (_solve_qp, _randomized_round,
                            _pair_split_feasible, kernel_uniform)
from helpers.metrics import border_pairs_set, n_cut_edges

TAUS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
LAM = 0.10
DELTA = 1e-12          # identical to qp_mh_proposal's clip
SEED = 42


def instrumented_step(graph, asn, tau, eps, ideal):
    """One Chain E step, returning (next_assignment, record). Mirrors
    qp_mh_proposal line for line, with the proposal logged."""
    bp = border_pairs_set(asn, graph)
    d1, d2 = random.choice(sorted(bp))
    nodes = [v for v, d in asn.items() if d in (d1, d2)]
    rec = dict(flips=0, feasible=False, accepted=False, solver_fail=False,
               log_alpha=np.nan, wcut=np.nan)

    x0 = np.array([1.0 if asn[v] == d1 else 0.0 for v in nodes])
    # boundary of this pair, for the law 2*tau*|boundary|
    sub = graph.subgraph(nodes)
    rec["wcut"] = float(sum(1 for u, v in sub.edges() if asn[u] != asn[v]))

    xf = _solve_qp(graph, nodes, asn, d1, alpha=tau, beta=1.0, epsilon=eps,
                   kernel_fn=kernel_uniform, ideal_pop=ideal)
    if xf is None:
        rec["solver_fail"] = True
        return asn, rec

    prop, lq_f, _ = _randomized_round(xf, nodes, d1, d2, graph, asn)
    rec["flips"] = int(sum(1 for v in nodes if prop[v] != asn[v]))

    if not _pair_split_feasible(prop, nodes, d1, d2, graph, eps,
                                ideal_pop=ideal):
        return asn, rec
    rec["feasible"] = True

    xr = _solve_qp(graph, nodes, prop, d1, alpha=tau, beta=1.0, epsilon=eps,
                   kernel_fn=kernel_uniform, ideal_pop=ideal)
    if xr is None:
        rec["solver_fail"] = True
        return asn, rec
    xr_c = np.clip(np.asarray(xr), DELTA, 1 - DELTA)
    lq_r = float(np.sum(x0 * np.log(xr_c) + (1 - x0) * np.log(1 - xr_c)))

    dcut = n_cut_edges(graph, prop) - n_cut_edges(graph, asn)
    bp_p = border_pairs_set(prop, graph)
    la = (-LAM * dcut + (lq_r - lq_f)
          + np.log(len(bp)) - np.log(max(len(bp_p), 1)))
    rec["log_alpha"] = float(la)
    if np.log(random.random() + 1e-300) < la:
        rec["accepted"] = True
        return prop, rec
    return asn, rec


def main():
    p = setup_problem("5x5", _REPO)
    graph, eps, ideal = p["graph"], p["eps"], p["ideal"]
    lookup = p["fiber_lookup"]

    def state_id(asn):
        canon = frozenset(frozenset(v for v, d in asn.items() if d == k)
                          for k in set(asn.values()))
        return lookup.get(canon, -1)

    print(f"5x5 fiber, eps={eps}, {N} steps per tau, alpha=tau, beta=1\n")
    hdr = (f"{'tau':>6} {'E[flips]':>9} {'2t|bd|':>7} {'P(1flip)':>9} "
           f"{'feas|1':>7} {'P(feas)':>8} {'acc|feas':>9} {'accept':>7} "
           f"{'moved/step':>11} {'states':>7}")
    print(hdr)
    out = {}
    for tau in TAUS:
        random.seed(SEED)
        np.random.seed(SEED)
        asn = dict(p["init"])
        recs = []
        seen = {state_id(asn)}
        for _ in range(N):
            asn, r = instrumented_step(graph, asn, tau, eps, ideal)
            recs.append(r)
            if r["accepted"]:
                seen.add(state_id(asn))
        fl = np.array([r["flips"] for r in recs], dtype=float)
        fe = np.array([r["feasible"] for r in recs], dtype=float)
        ac = np.array([r["accepted"] for r in recs], dtype=float)
        wc = np.array([r["wcut"] for r in recs], dtype=float)
        one = fl == 1
        p_feas = fe.mean()
        acc_given_feas = ac[fe == 1].mean() if fe.sum() > 0 else np.nan
        feas_given_one = fe[one].mean() if one.sum() > 0 else np.nan
        moved = float((ac * fl).mean())
        print(f"{tau:6.2f} {fl.mean():9.3f} {2*tau*np.nanmean(wc):7.3f} "
              f"{one.mean():9.3f} {feas_given_one:7.3f} {p_feas:8.3f} "
              f"{acc_given_feas:9.3f} {ac.mean():7.3f} {moved:11.4f} "
              f"{len(seen):7d}")
        out[f"tau{tau}"] = np.array([
            fl.mean(), 2 * tau * np.nanmean(wc), one.mean(), feas_given_one,
            p_feas, acc_given_feas, ac.mean(), moved, len(seen)])
        out[f"tau{tau}_flips"] = fl
        out[f"tau{tau}_feasible"] = fe
        out[f"tau{tau}_accepted"] = ac

    print("\nColumns: E[flips] is the mean proposal size (the law's exact "
          "quantity); 2t|bd| the small-tau law; P(1flip) the share of "
          "single-unit proposals; feas|1 their feasibility rate; moved/step "
          "= acceptance x accepted size, the mobility the design rule "
          "should maximize; states = distinct fiber states reached.")
    np.savez(_HERE / "iowa_hybrid_test_out" / "acceptance_sweep_5x5.npz",
             taus=np.array(TAUS), n_steps=N, seed=SEED, lam=LAM, **out)
    print("Saved acceptance_sweep_5x5.npz")


if __name__ == "__main__":
    main()
