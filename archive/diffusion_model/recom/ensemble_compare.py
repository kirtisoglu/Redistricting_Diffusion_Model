"""Ensemble-quality comparison: Chain C vs SA-with-restarts on Iowa.

Both methods get a similar wall-clock budget. Chain C runs once and its
post-burn trajectory is the ensemble. SA restarts run the annealed
ReCom chain from the same initial plan with different seeds and keep
each run's final plan. We compare the two ensembles on

  * number of distinct plans,
  * cut-count distribution (mean, best),
  * diversity, measured as mean pairwise county-reassignment distance
    between distinct plans (best district relabeling),
  * distinct plans per second of compute.

Usage: python3 recom/ensemble_compare.py [n_restarts] [seed]
"""

import math
import random
import sys
import time
from itertools import permutations
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
_DIFF = _REPO / "diffusion_model"
if str(_DIFF) not in sys.path:
    sys.path.insert(0, str(_DIFF))

from recom.problems import setup_problem, cut_edges_count, assignment_canonical
from recom.chains import recom_valid, run_chain, make_iqp_hybrid_step
from recom.qp_model import make_partition, kernel_uniform

N_RESTARTS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42
LAM_SCHED = lambda t, T: 0.05 * (5.0 / 0.05) ** (t / T)
SA_STEPS = 500
C_STEPS = 500
BURN = 100
K = 4


def plan_distance(a, b):
    best = None
    for perm in permutations(range(K)):
        d = sum(1 for n in a if perm[a[n]] != b[n])
        best = d if best is None else min(best, d)
    return best


def ensemble_stats(name, plans, cuts, wall):
    """plans: list of assignment dicts (distinct); cuts: aligned cut counts."""
    n = len(plans)
    dists = []
    if n > 1:
        idx = list(range(n))
        random.seed(0)
        pairs = [(i, j) for i in idx for j in idx if i < j]
        if len(pairs) > 300:
            pairs = random.sample(pairs, 300)
        dists = [plan_distance(plans[i], plans[j]) for i, j in pairs]
    print(f"{name:<16} distinct={n:>4}  best={min(cuts)}  "
          f"mean={np.mean(cuts):.2f}  "
          f"pairwise_dist={np.mean(dists):.1f} counties" if dists else
          f"{name:<16} distinct={n:>4}  best={min(cuts)}  "
          f"mean={np.mean(cuts):.2f}  pairwise_dist=n/a")
    print(f"{'':<16} wall={wall:.1f}s  distinct_per_min={60*n/wall:.1f}")
    return dict(n=n, best=int(min(cuts)), mean=float(np.mean(cuts)),
                pair_dist=float(np.mean(dists)) if dists else None,
                wall=wall)


def main():
    p = setup_problem("iowa", _REPO)
    graph, eps, ideal, init = p["graph"], p["eps"], p["ideal"], p["init"]

    # ── Chain C, capturing every distinct post-burn plan ────────────────
    random.seed(SEED * 1000 + 3)
    np.random.seed(SEED * 1000 + 3)
    seen = {}
    base = make_iqp_hybrid_step(graph, ideal, eps, 0.10, 10, 10.0, 1.0,
                                kernel_uniform, False, 5.0)
    step_count = [0]

    def step(state):
        out = base(state)
        step_count[0] += 1
        if step_count[0] > BURN:
            key = assignment_canonical(out[1])
            if key not in seen:
                seen[key] = (dict(out[1]), out[2])
        return out

    t0 = time.perf_counter()
    run_chain("C", make_partition(graph, init), step, C_STEPS, None)
    c_wall = time.perf_counter() - t0
    c_plans = [v[0] for v in seen.values()]
    c_cuts = [v[1] for v in seen.values()]
    c_stats = ensemble_stats("Chain C", c_plans, c_cuts, c_wall)

    # ── SA restarts, keeping each run's final plan ──────────────────────
    sa_plans, sa_cuts = {}, {}
    t0 = time.perf_counter()
    for r in range(N_RESTARTS):
        random.seed(SEED * 10000 + r)
        np.random.seed(SEED * 10000 + r)
        part = make_partition(graph, init)
        cur = cut_edges_count(graph, dict(part.assignment))
        for t in range(SA_STEPS):
            lam = LAM_SCHED(t, SA_STEPS)
            prop = recom_valid(part, graph, ideal, eps)
            new = cut_edges_count(graph, dict(prop.assignment))
            if math.log(random.random() + 1e-300) < -lam * (new - cur):
                part, cur = prop, new
        asn = dict(part.assignment)
        key = assignment_canonical(asn)
        if key not in sa_plans:
            sa_plans[key] = asn
            sa_cuts[key] = cur
    sa_wall = time.perf_counter() - t0
    sa_stats = ensemble_stats(f"SA x{N_RESTARTS}",
                              list(sa_plans.values()),
                              list(sa_cuts.values()), sa_wall)

    np.savez(Path(__file__).parent / "iowa_hybrid_test_out" /
             f"ensemble_compare_seed{SEED}.npz",
             c_cuts=np.array(c_cuts), sa_cuts=np.array(list(sa_cuts.values())),
             c_stats=str(c_stats), sa_stats=str(sa_stats),
             n_restarts=N_RESTARTS, seed=SEED)
    print("\nSaved ensemble_compare npz.")


if __name__ == "__main__":
    main()
