"""Trajectory-mean kernel metrics for the multikernel table.

Reruns Chain C under each kernel (deterministic streams) computing the
geographic boundary length and the density-gap ratio of every
post-burn plan, so the multikernel table can report trajectory means
instead of noisy single-plan snapshots.

Usage: python3 recom/kernel_traj_metrics.py [seed]
"""

import json
import random
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
_DIFF = _REPO / "diffusion_model"
if str(_DIFF) not in sys.path:
    sys.path.insert(0, str(_DIFF))

from recom.problems import setup_problem
from recom.chains import run_chain, make_iqp_hybrid_step
from recom.qp_model import (make_partition, kernel_uniform,
                            kernel_perimeter, make_kernel_density)

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42

with open(_REPO / "IA_counties" / "IA_counties.json") as f:
    g = json.load(f)
dens = {}
for n in g["nodes"]:
    dens[n["id"]] = n["TOTPOP"] / max(float(n.get("ALAND10", 1)), 1.0)
mean_rho = np.mean(list(dens.values()))
dens = {k: v / mean_rho for k, v in dens.items()}
edges = [(u, e["id"], float(e.get("shared_perim", 1.0)))
         for u, adj in enumerate(g["adjacency"]) for e in adj if u < e["id"]]
ALL_GAP = np.mean([abs(dens[u] - dens[v]) for u, v, _ in edges])


def metrics(asn):
    gp_, gaps = 0.0, []
    for u, v, sp in edges:
        if asn[u] != asn[v]:
            gp_ += sp
            gaps.append(abs(dens[u] - dens[v]))
    return gp_, float(np.mean(gaps)) / ALL_GAP


def main():
    p = setup_problem("iowa", _REPO)
    graph, eps, ideal, init = p["graph"], p["eps"], p["ideal"], p["init"]
    out = {}
    for kname, kfn in [("uniform", kernel_uniform),
                       ("perimeter", kernel_perimeter),
                       ("density", make_kernel_density(1.0))]:
        random.seed(SEED * 1000 + 3)
        np.random.seed(SEED * 1000 + 3)
        base = make_iqp_hybrid_step(graph, ideal, eps, 0.10, 10, 10.0, 1.0,
                                    kfn, False, 5.0)
        rows = []

        def step(state, _b=base, _r=rows):
            o = _b(state)
            _r.append(metrics(o[1]))
            return o

        run_chain(kname, make_partition(graph, init), step, 500, None)
        a = np.array(rows[100:])
        out[kname] = a
        print(f"{kname:<10} traj geo_perim={a[:, 0].mean():.2f}"
              f"+-{a[:, 0].std():.2f}  "
              f"gap_ratio={a[:, 1].mean():.2f}+-{a[:, 1].std():.2f}")
    np.savez(Path(__file__).parent / "iowa_hybrid_test_out" /
             f"kernel_traj_metrics_seed{SEED}.npz",
             **{f"{k}_gp": v[:, 0] for k, v in out.items()},
             **{f"{k}_gr": v[:, 1] for k, v in out.items()})
    print("saved kernel_traj_metrics npz")


if __name__ == "__main__":
    main()
