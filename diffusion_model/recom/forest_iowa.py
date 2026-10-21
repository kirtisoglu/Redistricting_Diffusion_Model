"""The kill test on Iowa against the Forest ReCom baseline.

Question: does the diffusion local move (lifted Chain E) raise the
effective samples per second of an exact sampler whose global move is the
forest-lifted tree move with balanced-edge selection, under the paper's
three kernel-weighted targets?

Per kernel (uniform, perimeter, density; lam scaled as in hybrid_iowa.py so
the three targets are equally tempered at the start plan), chains from the
same start:
    forest             forest move alone (the baseline)
    f-hybrid p=...     forest move + lifted Chain E, for each p in P_LOCALS
    tree-part          the paper's partition-level tree move, short mode
                       only, for the baseline-vs-baseline comparison

Reported: acceptance by move type, units moved per local step, share of
local solves on the linear path, half-trace (short) or quarter-trace
(long) means of the weighted cut as the convergence check, ESS of the
weighted-cut trace, wall time, ESS/s, and the ratio to the forest move.

Two modes.
  short   n_steps default 200000, all three kernels, P_LOCALS = (0.1, 0.3),
          plus tree-part. Writes forest_iowa_short_tau{TAU}_seed{SEED}.npz.
  long    n_steps default 2000000, perimeter and density only, one chain
          per invocation so runs can be parallelised:
              python3 recom/forest_iowa.py long forest   [n] [seed]
              python3 recom/forest_iowa.py long hybrid   [n] [seed]   (p = 0.1)
          Writes forest_iowa_long_{forest|hybrid}_seed{SEED}.npz.

KILL CRITERION (pre-registered): on the long runs, ESS/s(f-hybrid) /
ESS/s(forest) >= 1.5 under both the perimeter and the density target on
both seeds, with the two chains' quarter means on the same level. Short
runs are for direction and timing only; the tree move's ESS in a short run
is too small to quote a ratio.

Usage:  python3 recom/forest_iowa.py short [n_steps] [seed] [tau]
        python3 recom/forest_iowa.py long forest|hybrid [n_steps] [seed]
"""

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_DIFF = _HERE.parent
_REPO = _DIFF.parent
if str(_DIFF) not in sys.path:
    sys.path.insert(0, str(_DIFF))

from recom.problems import setup_problem
from recom.qp_model import kernel_uniform, kernel_perimeter, make_kernel_density
from recom.hybrid import Target, run_hybrid
from recom.forest_recom import run_forest_hybrid
from recom.kill_stats import chain_summary

MODE = sys.argv[1] if len(sys.argv) > 1 else "short"
TAU_DEFAULT = 0.1
BURN = 0.1
OUT_DIR = _HERE / "iowa_hybrid_test_out"


def targets(graph, init, kernels):
    """lam per kernel so that lam * w(boundary) at the start plan equals
    the uniform case's 0.1 * |boundary|."""
    n0 = Target(graph, kernel_uniform, 0.10).ncut(init)
    out = []
    for kname, kfn in kernels:
        w0 = Target(graph, kfn, 1.0).wcut(init)
        out.append((kname, kfn, 0.10 * n0 / w0))
    return out


def print_row(kname, lam, cname, s, base):
    ratio = s["ess_s"] / base if base else float("nan")
    print(f"{kname:>10} {lam:7.3f} {cname:>14} {s['acc_global']:9.4f} {s['acc_local']:8.3f} "
          f"{s['mv_local']:7.3f} {s['fast']:5.2f} {s['h1']:8.2f} {s['h2']:8.2f} "
          f"{s['ess']:7.0f} {s['wall']:8.1f} {s['ess_s']:7.3f}   ratio vs forest {ratio:.2f}",
          flush=True)


def store_res(store, tag, res, lam):
    for key in ("wcut", "ncut", "kind", "accepted", "fast", "flips"):
        store[f"{tag}_{key}"] = res[key]
    store[f"{tag}_wall"] = res["wall"]
    store[f"{tag}_lam"] = lam


def run_short():
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
    SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    TAU = float(sys.argv[4]) if len(sys.argv) > 4 else TAU_DEFAULT
    P_LOCALS = (0.1, 0.3)
    p = setup_problem("iowa", _REPO)
    graph, eps, ideal, init = p["graph"], p["eps"], p["ideal"], p["init"]
    cfgs = targets(graph, init, (("uniform", kernel_uniform),
                                 ("perimeter", kernel_perimeter),
                                 ("density", make_kernel_density(sigma=1.0))))
    store = {"n_steps": N, "seed": SEED, "tau": TAU, "mode": "short"}
    print(f"Iowa, k=4, eps={eps}, {N} steps per chain, tau = {TAU}, seed {SEED}\n")
    print(f"{'kernel':>10} {'lam':>7} {'chain':>14} {'acc glob':>9} {'acc loc':>8} "
          f"{'mv/loc':>7} {'fast':>5} {'wcut h1':>8} {'wcut h2':>8} {'ESS':>7} "
          f"{'wall s':>8} {'ESS/s':>7}")
    for kname, kfn, lam in cfgs:
        tgt = Target(graph, kfn, lam)
        res = run_forest_hybrid(tgt, init, ideal, eps, N, 0.0, TAU, 1.0, SEED, label=f"{kname} forest")
        s = chain_summary(res, BURN)
        base = s["ess_s"]
        print_row(kname, lam, "forest", s, base)
        store_res(store, f"{kname}_forest", res, lam)
        for pl in P_LOCALS:
            res = run_forest_hybrid(tgt, init, ideal, eps, N, pl, TAU, 1.0, SEED, label=f"{kname} f-hybrid {pl}")
            s = chain_summary(res, BURN)
            print_row(kname, lam, f"f-hybrid p={pl}", s, base)
            store_res(store, f"{kname}_fhybrid_p{pl}", res, lam)
        res = run_hybrid(tgt, init, ideal, eps, N, 0.0, TAU, 1.0, SEED, label=f"{kname} tree-part")
        s = chain_summary(res, BURN)
        print_row(kname, lam, "tree-part", s, base)
        store_res(store, f"{kname}_treepart", res, lam)
        print()
    out = OUT_DIR / f"forest_iowa_short_tau{TAU:g}_seed{SEED}.npz"
    np.savez(out, **store)
    print(f"Saved {out.name}")


def run_long():
    WHICH = sys.argv[2] if len(sys.argv) > 2 else "forest"
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 2_000_000
    SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    TAU = TAU_DEFAULT
    P_LOCAL = 0.0 if WHICH == "forest" else 0.1
    p = setup_problem("iowa", _REPO)
    graph, eps, ideal, init = p["graph"], p["eps"], p["ideal"], p["init"]
    cfgs = targets(graph, init, (("perimeter", kernel_perimeter),
                                 ("density", make_kernel_density(sigma=1.0))))
    store = {"n_steps": N, "seed": SEED, "tau": TAU, "p_local": P_LOCAL, "mode": "long"}
    for kname, kfn, lam in cfgs:
        tgt = Target(graph, kfn, lam)
        res = run_forest_hybrid(tgt, init, ideal, eps, N, P_LOCAL, TAU, 1.0, SEED,
                                log_every=N // 20, label=f"{kname} {WHICH}")
        s = chain_summary(res, BURN)
        qm = " ".join(f"{v:.2f}" for v in s["quarters"])
        print(f"{kname:>10} lam={lam:.3f} {WHICH} {N} steps seed {SEED}: "
              f"acc forest {s['acc_global']:.4f} acc loc {s['acc_local']:.3f} "
              f"quarter means {qm}  ESS {s['ess']:.0f}  wall {s['wall']:.0f}s  "
              f"ESS/s {s['ess_s']:.4f}", flush=True)
        store_res(store, kname, res, lam)
    out = OUT_DIR / f"forest_iowa_long_{WHICH}_seed{SEED}.npz"
    np.savez(out, **store)
    print(f"Saved {out.name}")


if __name__ == "__main__":
    if MODE == "short":
        run_short()
    elif MODE == "long":
        run_long()
    else:
        raise SystemExit(__doc__)
