"""Long Iowa runs for the matched kill test: the exact tree move alone, and
the hybrid at p_loc = 0.1, each for 2e6 steps under the perimeter and
density targets (lam scaled as in hybrid_iowa.py).

These are the rows of the paper's Iowa table marked "long runs". The
short runs understate the tree move's autocorrelation time, which is
near 1e5 steps on Iowa, so only runs of this length compare two chains at
equilibrium. Quarter means are printed as the convergence check.

Writes: iowa_hybrid_test_out/hybrid_iowa_treeonly_2e6.npz      (mode tree)
        iowa_hybrid_test_out/hybrid_iowa_hybrid_p0.1_2e6.npz   (mode hybrid)
Usage:  python3 recom/hybrid_iowa_long.py tree|hybrid [n_steps] [seed]
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
from recom.hybrid import Target, run_hybrid, ess_fft

MODE = sys.argv[1] if len(sys.argv) > 1 else "tree"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2_000_000
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 42
P_LOCAL = 0.0 if MODE == "tree" else 0.1
TAU = 0.1
_SUFFIX = "" if SEED == 42 else f"_seed{SEED}"      # seed 42 files predate this flag
OUT = (_HERE / "iowa_hybrid_test_out" /
       (f"hybrid_iowa_treeonly_2e6{_SUFFIX}.npz" if MODE == "tree"
        else f"hybrid_iowa_hybrid_p0.1_2e6{_SUFFIX}.npz"))


def main():
    p = setup_problem("iowa", _REPO)
    graph, eps, ideal, init = p["graph"], p["eps"], p["ideal"], p["init"]
    n0 = Target(graph, kernel_uniform, 0.1).ncut(init)
    out = {}
    for kname, kfn in (("perimeter", kernel_perimeter),
                       ("density", make_kernel_density(sigma=1.0))):
        w0 = Target(graph, kfn, 1.0).wcut(init)
        lam = 0.1 * n0 / w0
        tgt = Target(graph, kfn, lam)
        res = run_hybrid(tgt, init, ideal, eps, N, P_LOCAL, TAU, 1.0, SEED)
        wc = res["wcut"][N // 10:]
        q = len(wc) // 4
        k = res["kind"][N // 10:]
        a = res["accepted"][N // 10:]
        ess = ess_fft(wc)
        acc_t = a[k == 0].mean() if (k == 0).any() else float("nan")
        acc_l = a[k == 1].mean() if (k == 1).any() else float("nan")
        print(f"{kname:>10} lam={lam:.3f} {MODE} {N} steps: acc tree {acc_t:.4f} "
              f"acc loc {acc_l:.3f}  quarter means {wc[:q].mean():.2f} "
              f"{wc[q:2*q].mean():.2f} {wc[2*q:3*q].mean():.2f} {wc[3*q:].mean():.2f}  "
              f"ESS {ess:.0f}  wall {res['wall']:.0f}s  ESS/s {ess/res['wall']:.3f}")
        for key in ("wcut", "kind", "accepted", "fast", "flips"):
            out[f"{kname}_{key}"] = res[key]
        out[f"{kname}_lam"] = lam
        out[f"{kname}_wall"] = res["wall"]
    np.savez(OUT, n_steps=N, seed=SEED, p_local=P_LOCAL, tau=TAU, **out)
    print(f"Saved {OUT.name}")


if __name__ == "__main__":
    main()
