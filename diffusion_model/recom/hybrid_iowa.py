"""Kill test on Iowa: does the diffusion local move earn its place next to
an exact tree move, under the paper's own kernels?

No ground truth exists on Iowa, so the comparison is purely mobility at
matched wall-clock: effective sample size per second of the weighted-cut
trace for the tree move alone against the hybrid, for the uniform,
perimeter and density kernels. lam for the non-uniform kernels is scaled
so that lam * w(boundary) at the initial plan equals the uniform case's
0.1 * |boundary|, so the three targets are comparably tempered.

Both chains are exactly reversible for their target (Section 4.6 and
hybrid.py), so what is being compared is mixing, not correctness.

Writes: iowa_hybrid_test_out/hybrid_iowa.npz
Usage:  python3 recom/hybrid_iowa.py [n_steps] [seed]
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

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42
TAU = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1   # paper's locked setting
P_LOCALS = (0.1, 0.3)
ALPHA, BETA = TAU, 1.0
OUT_TAG = f"_tau{TAU:g}"
BURN = 0.1


def main():
    p = setup_problem("iowa", _REPO)
    graph, eps, ideal, init = p["graph"], p["eps"], p["ideal"], p["init"]
    base = Target(graph, kernel_uniform, 0.10)
    n0, w0 = base.ncut(init), None
    configs = []
    for kname, kfn in (("uniform", kernel_uniform),
                       ("perimeter", kernel_perimeter),
                       ("density", make_kernel_density(sigma=1.0))):
        t = Target(graph, kfn, 1.0)
        w0 = t.wcut(init)
        lam = 0.10 * n0 / w0            # equal tempering at the start plan
        configs.append((kname, kfn, lam))

    store = {"n_steps": N, "seed": SEED, "tau": TAU}
    print(f"Iowa, k=4, eps={eps}, {N} steps per chain, tau = {TAU}\n")
    print(f"{'kernel':>10} {'lam':>7} {'chain':>12} {'acc tree':>9} {'acc loc':>8} "
          f"{'mv/loc':>7} {'fast':>5} {'wcut h1':>8} {'wcut h2':>8} {'ESS':>7} "
          f"{'wall s':>7} {'ESS/s':>7}")
    for kname, kfn, lam in configs:
        tgt = Target(graph, kfn, lam)
        base_rate = None
        for cname, pl in [("tree-only", 0.0)] + [(f"hybrid p={q}", q) for q in P_LOCALS]:
            res = run_hybrid(tgt, init, ideal, eps, N, pl, ALPHA, BETA, SEED,
                             log_every=0, label=f"{kname} {cname}")
            burn = int(N * BURN)
            wc = res["wcut"][burn:]
            k = res["kind"][burn:]; a = res["accepted"][burn:]
            acc_t = a[k == 0].mean() if (k == 0).any() else np.nan
            acc_l = a[k == 1].mean() if (k == 1).any() else np.nan
            mv_l = (a[k == 1] * res["flips"][burn:][k == 1]).mean() if (k == 1).any() else np.nan
            fast = res["fast"][burn:][k == 1].mean() if (k == 1).any() else np.nan
            h = len(wc) // 2
            ess = ess_fft(wc); ess_s = ess / res["wall"]
            if base_rate is None:
                base_rate = ess_s
            print(f"{kname:>10} {lam:7.3f} {cname:>12} {acc_t:9.4f} {acc_l:8.3f} "
                  f"{mv_l:7.3f} {fast:5.2f} {wc[:h].mean():8.2f} {wc[h:].mean():8.2f} "
                  f"{ess:7.0f} {res['wall']:7.1f} {ess_s:7.2f}   "
                  f"ratio vs tree-only {ess_s/base_rate:.2f}")
            tag = f"{kname}_{cname.replace(' ', '_').replace('=', '')}"
            store[f"{tag}_wcut"] = res["wcut"]; store[f"{tag}_ncut"] = res["ncut"]
            store[f"{tag}_kind"] = res["kind"]; store[f"{tag}_accepted"] = res["accepted"]
            store[f"{tag}_wall"] = res["wall"]; store[f"{tag}_lam"] = lam
            store[f"{tag}_fast"] = res["fast"]; store[f"{tag}_flips"] = res["flips"]
        print()
    np.savez(_HERE / "iowa_hybrid_test_out" / f"hybrid_iowa{OUT_TAG}.npz", **store)
    print(f"Saved hybrid_iowa{OUT_TAG}.npz")


if __name__ == "__main__":
    main()
