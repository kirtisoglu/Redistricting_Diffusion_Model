"""Long-run ground-truth validation on the enumerated 5x5 fiber.

Runs Chain A (tempered ReCom, symmetric-q) and Chain E (exact QP-MH)
for many steps and compares the empirical cut distribution against the
exactly computable tempered target pi(P) ∝ exp(-lam*cuts(P)) over all
4006 partitions:

  * empirical mean cuts vs E_pi[cuts]  (converges fast if unbiased)
  * TV(empirical, pi) over the fiber   (needs ESS >> 4006)

Chain A tests whether the symmetric-q bias persists as N -> infinity;
Chain E tests that the exactly reversible kernel converges to its
stated target (theory says it must; mixing speed is the question).

Usage: python3 long_run_5x5.py [n_steps_A] [n_steps_E] [seed]
"""

import random
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
_DIFF = _REPO / "diffusion_model"
if str(_DIFF) not in sys.path:
    sys.path.insert(0, str(_DIFF))

from recom.problems import setup_problem
from recom.chains import run_chain, make_tempered_step, make_qp_mh_step
from recom.diagnostics import (target_distribution, expected_target_cuts,
                               kl_to_target, tv_to_uniform)
from recom.qp_model import make_partition, kernel_uniform

N_A = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
N_E = int(sys.argv[2]) if len(sys.argv) > 2 else 500_000
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 42
LAM = 0.10
BURN_FRAC = 0.1


def tv_to_target(visits, fiber_cuts, lam):
    ids, p = target_distribution(fiber_cuts, lam)
    n_total = sum(visits.values())
    q = np.array([visits.get(i, 0) for i in ids], dtype=float) / n_total
    return 0.5 * float(np.abs(q - p).sum())


def main():
    p = setup_problem("5x5", _REPO)
    graph, eps, ideal, init = p["graph"], p["eps"], p["ideal"], p["init"]
    fiber_cuts = p["fiber_cuts"]
    e_pi = expected_target_cuts(fiber_cuts, LAM)
    print(f"5x5 fiber: {len(fiber_cuts)} partitions, "
          f"E_pi[cuts] = {e_pi:.4f} at lam={LAM}")

    out = {}
    for tag, n_steps, factory in [
        ("A", N_A, lambda: make_tempered_step(graph, ideal, eps, LAM)),
        ("E", N_E, lambda: make_qp_mh_step(graph, ideal, eps, LAM,
                                           alpha=1.0, beta=10.0,
                                           kernel_fn=kernel_uniform)),
    ]:
        print(f"\n== Chain {tag}: {n_steps} steps ==")
        random.seed(SEED * 1000 + ord(tag))
        np.random.seed(SEED * 1000 + ord(tag))
        res = run_chain(tag, make_partition(graph, init), factory(),
                        n_steps, p["fiber_lookup"])
        burn = int(n_steps * BURN_FRAC)
        cuts = res["cuts"][burn:]
        # restrict visit counts to post-burn-in by rebuilding from cuts?
        # visits accumulated over the whole run; burn-in fraction is small
        # relative to n_steps, so report both raw-visit TV and post-burn mean.
        visits = res["visits"]
        tvt = tv_to_target(visits, fiber_cuts, LAM)
        tvu = tv_to_uniform(visits, len(fiber_cuts))
        kl, n_in = kl_to_target(visits, fiber_cuts, LAM)
        unique = len(visits)
        print(f"  mean cuts (post-burn) = {cuts.mean():.4f}  "
              f"(target {e_pi:.4f}; gap {cuts.mean()-e_pi:+.4f})")
        print(f"  std = {cuts.std():.4f}  unique states = {unique}/4006")
        print(f"  TV(emp, pi_target) = {tvt:.4f}   "
              f"TV(emp, uniform) = {tvu:.4f}   KL(emp||pi) = {kl:.4f}")
        out[tag] = dict(cuts=res["cuts"],
                        visits_ids=np.array(list(visits.keys())),
                        visits_counts=np.array(list(visits.values())))

    np.savez(Path(__file__).parent / "iowa_hybrid_test_out" /
             f"long_run_5x5_seed{SEED}.npz",
             A_cuts=out["A"]["cuts"], E_cuts=out["E"]["cuts"],
             A_visit_ids=out["A"]["visits_ids"],
             A_visit_counts=out["A"]["visits_counts"],
             E_visit_ids=out["E"]["visits_ids"],
             E_visit_counts=out["E"]["visits_counts"],
             lam=LAM, e_pi=e_pi, seed=SEED)
    print("\nSaved long_run npz.")


if __name__ == "__main__":
    main()
