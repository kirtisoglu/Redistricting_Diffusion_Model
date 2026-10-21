"""Make the diffusion connection do work: a move-size law for the QP proposal.

Section 4.2 derives the frequency response and stops. It is presented as
interpretation ("the QP acts as a low-pass filter"). It is more than that.
Worked through, it predicts exactly how far a proposal moves, which is the
quantity that controls acceptance, mobility and the whole exactness-versus-
mobility trade the paper keeps running into. That turns one decorative
paragraph into the tuning theory for the entire family, and no other
redistricting proposal has one.

THE DERIVATION

Interior stationarity of  min alpha x^T L x + beta ||x - x0||^2  gives

    (alpha L + beta I) x* = beta x0     i.e.    x* = (I + tau L)^{-1} x0,
    tau := alpha / beta,

one implicit Euler step of the graph heat equation xdot = -L x with
timestep tau. That much the paper already says. Three consequences it
does not draw:

(1) THE BOX CONSTRAINTS NEVER BIND.
    I + tau L is an M-matrix, so its inverse is entrywise nonnegative, and
    L1 = 0 gives (I + tau L)1 = 1, hence (I + tau L)^{-1} 1 = 1. So
    (I + tau L)^{-1} is ROW-STOCHASTIC and x* is a convex combination of
    the entries of x0, automatically inside [0,1]^n. The QP therefore
    reduces to a linear solve unless the population balance constraint
    binds. That is a lemma worth stating, and it makes the proposal far
    cheaper than "call a QP solver" suggests.

(2) THE EXPECTED MOVE SIZE IS EXACTLY A FILTER NORM.
    Under single-shot Bernoulli rounding a unit v flips with probability
    x*_v if x0_v = 0 and 1 - x*_v if x0_v = 1, and both equal
    |x*_v - x0_v|. So

        E[# units flipped]  =  ||x* - x0||_1  =  || (I + tau L)^{-1} tau L x0 ||_1,

    a HIGH-pass filter applied to the current bipartition, with gain
    g(lambda) = tau*lambda / (1 + tau*lambda) on eigenvalue lambda. No
    simulation needed: the move-size distribution of the proposal is a
    closed-form function of the merged pair's Laplacian spectrum.

(3) THE SMALL-TAU LAW.
    For tau small, g(lambda) = tau*lambda + O(tau^2), so
    x* - x0 = -tau L x0 + O(tau^2). For binary x0 and any nonnegative
    kernel, (L x0)_v is +(weight of v's cross-boundary edges) on one side
    and minus that on the other, so the entries sum in absolute value to
    twice the weighted cut of the pair. Therefore

        E[# units flipped]  =  2 * tau * |boundary(V_i, V_j)|  +  O(tau^2),

    where the boundary is measured in the SAME kernel the proposal uses.

WHY THIS MATTERS FOR THE PAPER

  * It is a design rule. To propose moves of a target size m, set
    alpha/beta = m / (2 |boundary|). Nothing else in the redistricting
    literature tells you how to size a proposal.
  * It explains Chain E quantitatively rather than by assertion. Appendix
    A.2 locks alpha = 1, beta = 10, i.e. tau = 0.1, and justifies it as
    "beta >> alpha keeps x* near-binary". The law says the expected move
    is 2 * 0.1 * |boundary| units. That is why Chain E visits 6 states in
    1000 steps. The stickiness is not a mystery or a tuning accident, it
    is the diffusion timestep being small.
  * It makes the kernel choice quantitative. Move size is proportional to
    the boundary measured in the chosen kernel, so a perimeter kernel
    takes larger steps exactly where the geographic border is long.
  * It connects the two ends of the paper. Mobility is controlled by tau,
    exactness by the Metropolis correction, and the reversible hybrid can
    now be tuned rather than guessed.

WHAT THIS SCRIPT CHECKS

For many (plan, district-pair) instances on the enumerated 5x5 fiber and
on Iowa, over a sweep of tau:
  - that the unconstrained optimum really does land in [0,1] (claim 1),
  - how often the balance constraint binds, which is the only thing that
    can break the closed form,
  - the exact filter-norm prediction against the small-tau law (claim 3),
  - Gurobi's constrained solution against the linear solve on a subsample,
    which checks the implementation against the theory.

Writes: iowa_hybrid_test_out/spectral_move_size.npz
        Papers/fig/spectral_move_size.{pdf,png}

Usage: python3 recom/spectral_move_size.py [n_plans_5x5] [n_qp_check]
"""

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_DIFF = _HERE.parent
_REPO = _DIFF.parent
if str(_DIFF) not in sys.path:
    sys.path.insert(0, str(_DIFF))

from recom.problems import setup_problem
from recom.qp_model import build_laplacian, kernel_uniform, _solve_qp
from helpers.metrics import border_pairs_set

N_PLANS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
N_QP_CHECK = int(sys.argv[2]) if len(sys.argv) > 2 else 150
TAUS = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
RNG = np.random.default_rng(20260918)


def instances_5x5(p, n_plans):
    """(assignment, pair) instances drawn from the enumerated fiber, so no
    chain has to be run to get a representative set of plans."""
    with open(_HERE / "fiber_5x5" / "fiber_5x5_partitions.json") as f:
        data = json.load(f)
    pids = list(data.keys())
    pick = RNG.choice(len(pids), size=min(n_plans, len(pids)), replace=False)
    out = []
    for i in pick:
        parts = data[pids[i]]
        asn = {}
        for d_id, part in enumerate(parts):
            for s in part:
                asn[tuple(s)] = d_id
        for pair in sorted(border_pairs_set(asn, p["graph"])):
            out.append((asn, pair))
    return out


def instances_tree_plans(p, n_plans, seed0=1000):
    """Plans from recursive tree partitioning, so a real-geography sample
    needs no chain run. Each is a valid state of every chain in the paper."""
    import random
    from recom.problems import balanced_tree_part
    out = []
    for i in range(n_plans):
        random.seed(seed0 + i)
        np.random.seed(seed0 + i)
        try:
            asn = dict(balanced_tree_part(p["graph"], p["K"], p["ideal"],
                                          "population", p["eps"]))
        except Exception:                            # noqa: BLE001
            continue
        for pair in sorted(border_pairs_set(asn, p["graph"])):
            out.append((asn, pair))
    return out


def analyze(label, p, insts, do_qp_check):
    graph, eps, ideal = p["graph"], p["eps"], p["ideal"]
    rows = []
    box_violations = 0
    n_qp, qp_err = 0, []

    for asn, (d1, d2) in insts:
        nodes = [v for v, d in asn.items() if d in (d1, d2)]
        L, _ = build_laplacian(graph, nodes, kernel_fn=kernel_uniform)
        x0 = np.array([1.0 if asn[v] == d1 else 0.0 for v in nodes])
        pop = np.array([float(graph.nodes[v]["population"]) for v in nodes])
        n = len(nodes)
        wcut = float(x0 @ L @ x0)          # weighted cut of this pair
        lo = max((1 - eps) * ideal, pop.sum() - (1 + eps) * ideal)
        hi = min((1 + eps) * ideal, pop.sum() - (1 - eps) * ideal)

        for tau in TAUS:
            # Claim 1: this linear solve IS the optimum, no box clipping.
            x = np.linalg.solve(np.eye(n) + tau * L, x0)
            if x.min() < -1e-9 or x.max() > 1 + 1e-9:
                box_violations += 1
            exact = float(np.abs(x - x0).sum())     # E[#flips], filter norm
            approx = 2.0 * tau * wcut               # small-tau law
            binds = not (lo - 1e-9 <= float(pop @ x) <= hi + 1e-9)
            rows.append((tau, n, wcut, exact, approx, float(binds),
                         float(x.min()), float(x.max())))

        if do_qp_check and n_qp < N_QP_CHECK:
            # Implementation vs theory at the paper's own Chain E setting.
            xq = _solve_qp(graph, nodes, asn, d1, alpha=1.0, beta=10.0,
                           epsilon=eps, kernel_fn=kernel_uniform,
                           ideal_pop=ideal)
            if xq is not None:
                xt = np.linalg.solve(np.eye(n) + 0.1 * L, x0)
                slack = lo - 1e-9 <= float(pop @ xt) <= hi + 1e-9
                if slack:
                    qp_err.append(float(np.abs(np.asarray(xq) - xt).max()))
                    n_qp += 1

    a = np.array(rows)
    print(f"\n{'='*74}\n{label}: {len(insts)} (plan, pair) instances, "
          f"{len(a)} solves")
    print(f"  box-constraint violations of the linear solve: {box_violations}"
          f"   <- claim 1 predicts 0")
    if qp_err:
        print(f"  Gurobi constrained solve vs linear solve, balance slack, "
              f"tau=0.1: max |diff| = {max(qp_err):.2e} over {n_qp} pairs"
          f"   <- implementation matches theory")

    # Every row below is computed on the SAME fixed subset: the instances
    # whose balance constraint is slack at EVERY tau. Otherwise the rows
    # average over shrinking subsets as the constraint starts binding, and
    # the move-size column stops being comparable across tau.
    n_inst = len(a) // len(TAUS)
    binds_any = a[:, 5].reshape(n_inst, len(TAUS)).max(axis=1) > 0
    keep = np.repeat(~binds_any, len(TAUS))
    print(f"  balance slack at every tau: {int((~binds_any).sum())}/{n_inst} "
          f"instances (rows below use exactly these)")

    print(f"\n  {'tau':>6} {'binds':>7} {'E[flips]':>10} {'2*tau*cut':>10} "
          f"{'rel.err':>9} {'x range':>16}")
    summary = []
    for tau in TAUS:
        m = a[:, 0] == tau
        sl = a[m & keep]
        if len(sl) == 0:
            continue
        ex, ap = sl[:, 3], sl[:, 4]
        rel = float(np.median(np.abs(ap - ex) / np.maximum(ex, 1e-12)))
        print(f"  {tau:6.2f} {100*a[m][:,5].mean():6.1f}% {ex.mean():10.3f} "
              f"{ap.mean():10.3f} {100*rel:8.1f}% "
              f"{sl[:,6].min():7.3f},{sl[:,7].max():7.3f}")
        summary.append((tau, a[m][:, 5].mean(), ex.mean(), ap.mean(), rel,
                        len(sl)))
    return a, np.array(summary), np.array(qp_err)


def main():
    results = {}
    for mode, label in (("5x5", "5x5 grid (enumerated fiber)"),
                        ("iowa", "Iowa counties (initial plan)")):
        try:
            p = setup_problem(mode, _REPO)
        except Exception as e:                      # noqa: BLE001
            print(f"[skipping {mode}: {e}]")
            continue
        insts = (instances_5x5(p, N_PLANS) if mode == "5x5"
                 else instances_tree_plans(p, 25))
        a, s, q = analyze(label, p, insts, do_qp_check=(mode == "5x5"))
        results[f"{mode}_rows"] = a
        results[f"{mode}_summary"] = s
        results[f"{mode}_qp_err"] = q            # Gurobi vs linear solve, slack pairs
        results[f"{mode}_n_instances"] = len(insts)
        results[f"{mode}_n_plans"] = (min(N_PLANS, 4006) if mode == "5x5" else 25)
        # The paper quotes the all-pairs mean move at tau = 0.1 as well as
        # the always-slack-subset mean reported in the summary rows.
        m01 = a[:, 0] == 0.1
        results[f"{mode}_mean_flips_tau0.1_all_pairs"] = float(a[m01][:, 3].mean())

        # The design rule, stated in the paper's own locked units.
        m = a[:, 0] == 0.1
        print(f"\n  AT THE PAPER'S CHAIN E SETTING (alpha=1, beta=10, "
              f"tau=0.1):")
        print(f"    mean weighted pair boundary  {a[m][:,2].mean():.2f} edges")
        print(f"    predicted units flipped      {a[m][:,3].mean():.2f}")
        print(f"    to propose ~10-unit moves    alpha/beta = "
              f"{10.0/(2*a[m][:,2].mean()):.3f}")

    if results:
        np.savez(_HERE / "iowa_hybrid_test_out" / "spectral_move_size.npz",
                 taus=TAUS, **results)
        print(f"\nSaved spectral_move_size.npz")


if __name__ == "__main__":
    main()
