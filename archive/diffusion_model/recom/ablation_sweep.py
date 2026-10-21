"""Chain C ablation sweep on Iowa (corrected balance constraint).

Regenerates, with current code, the three ablation tables in the paper:
  * tab:diversity-iowa  — baseline / +jitter / +tabu at n_iqp=5
  * tab:sensitivity     — (n_iqp, alpha, beta) grid, uniform kernel
  * app:sweeps          — full n x alpha x beta x kernel grid

Each configuration runs Chain C (tempered ReCom + IQP local) for
N_STEPS steps from the shared deterministic initial plan, on the same
seeded RNG stream as the headline Chain C, so rows are comparable to
tab:iowa-headline. Coverage is the fraction of distinct plans visited.

Usage: python3 recom/ablation_sweep.py [seed]
Output: iowa_hybrid_test_out/ablation_sweep_seed{SEED}.npz + stdout table.
"""

import random
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
_DIFF = _REPO / "diffusion_model"
if str(_DIFF) not in sys.path:
    sys.path.insert(0, str(_DIFF))

from recom.problems import setup_problem, assignment_canonical
from recom.chains import run_chain, make_iqp_hybrid_step
from recom.qp_model import (make_partition, kernel_uniform,
                            kernel_perimeter, make_kernel_density)

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
LAM = 0.10
N_STEPS = 500
BURN = 100
IQP_TIME_LIMIT = 5.0

KFNS = {
    "unif": kernel_uniform,
    "perim": kernel_perimeter,
    "dens": make_kernel_density(1.0),
}

configs = []
for kname in ["unif", "perim", "dens"]:
    for n in [3, 5, 10]:
        for a in [10.0, 50.0]:
            for b in [0.0, 1.0]:
                configs.append(dict(
                    tag=f"{kname}_n{n}_a{int(a)}_b{int(b)}",
                    kernel=kname, n=n, alpha=a, beta=b,
                    jitter=False, tabu=0))
configs.append(dict(tag="unif_n5_a10_b1_jitter", kernel="unif",
                    n=5, alpha=10.0, beta=1.0, jitter=True, tabu=0))
configs.append(dict(tag="unif_n5_a10_b1_tabu3", kernel="unif",
                    n=5, alpha=10.0, beta=1.0, jitter=False, tabu=3))


def main():
    p = setup_problem("iowa", _REPO)
    graph, eps, ideal, init = p["graph"], p["eps"], p["ideal"], p["init"]

    # A reference for the Delta column (same init, same code state).
    ref = np.load(Path(__file__).parent / "iowa_hybrid_test_out" /
                  "target_matched_iowa_seed42.npz", allow_pickle=True)
    a_mean = float(ref["A"][BURN:].mean())
    print(f"A reference mean (post-burn): {a_mean:.2f}")
    print(f"{len(configs)} configurations, {N_STEPS} steps each\n")

    out, rows = {}, []
    for ci, cfg in enumerate(configs):
        random.seed(SEED * 1000 + 3)      # headline Chain C stream
        np.random.seed(SEED * 1000 + 3)
        seen = set()
        base_step = make_iqp_hybrid_step(
            graph, ideal, eps, LAM, cfg["n"], cfg["alpha"], cfg["beta"],
            KFNS[cfg["kernel"]], False, IQP_TIME_LIMIT,
            weight_jitter=cfg["jitter"], tabu_size=cfg["tabu"])

        def step(state, _base=base_step, _seen=seen):
            outp = _base(state)
            _seen.add(assignment_canonical(outp[1]))
            return outp

        t0 = time.perf_counter()
        res = run_chain(cfg["tag"], make_partition(graph, init), step,
                        N_STEPS, None, graph=graph, ideal=ideal)
        wall = time.perf_counter() - t0
        cuts = res["cuts"][BURN:]
        cov = len(seen) / N_STEPS
        dev = float(res["pop_dev"].max())
        delta = 100 * (cuts.mean() - a_mean) / a_mean
        rows.append((cfg["tag"], cuts.mean(), cuts.std(), delta,
                     cov, wall, dev))
        out[f"{cfg['tag']}_cuts"] = res["cuts"]
        print(f"[{ci+1}/{len(configs)}] {cfg['tag']:<24} "
              f"mean={cuts.mean():6.2f} std={cuts.std():5.2f} "
              f"dvsA={delta:+6.1f}% cov={100*cov:4.1f}% "
              f"wall={wall:6.1f}s maxdev={dev:.4f}")

    print("\n" + "=" * 84)
    print(f"{'config':<26}{'mean':>7}{'std':>7}{'dvsA%':>8}"
          f"{'cov%':>7}{'wall_s':>8}{'maxdev':>8}")
    for r in rows:
        print(f"{r[0]:<26}{r[1]:>7.2f}{r[2]:>7.2f}{r[3]:>8.1f}"
              f"{100*r[4]:>7.1f}{r[5]:>8.1f}{r[6]:>8.4f}")

    np.savez(Path(__file__).parent / "iowa_hybrid_test_out" /
             f"ablation_sweep_seed{SEED}.npz",
             tags=np.array([r[0] for r in rows]),
             means=np.array([r[1] for r in rows]),
             stds=np.array([r[2] for r in rows]),
             deltas=np.array([r[3] for r in rows]),
             coverages=np.array([r[4] for r in rows]),
             walls=np.array([r[5] for r in rows]),
             maxdevs=np.array([r[6] for r in rows]),
             a_mean=a_mean, lam=LAM, seed=SEED, n_steps=N_STEPS,
             burn=BURN, **out)
    print("\nSaved ablation_sweep npz.")


if __name__ == "__main__":
    main()
