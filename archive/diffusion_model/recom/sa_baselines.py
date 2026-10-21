"""Single-run optimizer baselines on Iowa (Table: optimizer baselines).

Runs, from the shared deterministic initial plan:
  * SA over balance-validated ReCom proposals, geometric lambda
    0.05 -> 5 over 500 steps (and a 5000-step extension),
  * hill climbing (accept only non-worsening moves), 500 steps.

Reports best cuts, the frozen plateau (final cut count), and wall-clock.
Usage: python3 recom/sa_baselines.py [seed]
"""

import math
import random
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
_DIFF = _REPO / "diffusion_model"
if str(_DIFF) not in sys.path:
    sys.path.insert(0, str(_DIFF))

from recom.problems import setup_problem, cut_edges_count
from recom.chains import recom_valid
from recom.qp_model import make_partition

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
LAM_SCHED = lambda t, T: 0.05 * (5.0 / 0.05) ** (t / max(T - 1, 1))


def run(graph, ideal, eps, init, steps, mode):
    part = make_partition(graph, init)
    cur = cut_edges_count(graph, dict(part.assignment))
    trace = []
    t0 = time.perf_counter()
    for t in range(steps):
        prop = recom_valid(part, graph, ideal, eps)
        new = cut_edges_count(graph, dict(prop.assignment))
        if mode == "hill":
            accept = new <= cur
        else:
            lam = LAM_SCHED(t, steps)
            accept = math.log(random.random() + 1e-300) < -lam * (new - cur)
        if accept:
            part, cur = prop, new
        trace.append(cur)
    wall = time.perf_counter() - t0
    return np.array(trace), wall


def main():
    p = setup_problem("iowa", _REPO)
    graph, eps, ideal, init = p["graph"], p["eps"], p["ideal"], p["init"]

    out = {}
    for tag, steps, mode in [("SA_500", 500, "sa"),
                             ("HILL_500", 500, "hill")]:
        bests, finals, walls = [], [], []
        for s in range(SEED, SEED + 10):
            random.seed(s * 1000 + 42)
            np.random.seed(s * 1000 + 42)
            trace, wall = run(graph, ideal, eps, init, steps, mode)
            bests.append(int(trace.min()))
            finals.append(int(trace[-1]))
            walls.append(wall)
            if s == SEED:
                out[tag] = trace
        out[f"{tag}_bests"] = np.array(bests)
        out[f"{tag}_finals"] = np.array(finals)
        out[f"{tag}_walls"] = np.array(walls)
        print(f"{tag:<10} bests(10 seeds)={bests}  finals={finals}  "
              f"wall/run={np.mean(walls):.1f}s")

    random.seed(SEED * 1000 + 42)
    np.random.seed(SEED * 1000 + 42)
    trace, wall = run(graph, ideal, eps, init, 5000, "sa")
    out["SA_5000"] = trace
    print(f"SA_5000    best={trace.min()}  final(plateau)={trace[-1]}  "
          f"wall={wall:.1f}s")

    np.savez(Path(__file__).parent / "iowa_hybrid_test_out" /
             f"sa_baselines_seed{SEED}.npz", **out, seed=SEED)
    print("Saved sa_baselines npz.")


if __name__ == "__main__":
    main()
