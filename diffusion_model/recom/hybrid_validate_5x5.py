"""Validate the exact tree move and the hybrid against enumerated targets on
the 5x5 fiber, and run the kill test.

For each configuration (kernel, lam), two chains from the same start:
    tree-only   p_local = 0
    hybrid      p_local = P_LOCAL
and the exact target pi_w over all 4006 partitions, computed from the fiber
JSON with the same kernel. Reported per chain:

  * total variation of the cut-count marginal to pi_w's marginal, with the
    independent-sampler noise floor at the chain's effective sample size,
    and the excess in floor standard deviations (the calibration statistic
    of Section 5.1);
  * batch-means Hotelling T^2 on the 4 free marginal coordinates;
  * full-state TV with its floor, for completeness;
  * acceptance by move type, units moved per local step;
  * ESS of the weighted-cut trace and ESS per second, which is the kill
    criterion: hybrid / tree-only >= 1.5 means the local kernel earns its
    place, below 1 means it is a passenger.

The density configuration is a synthetic city: rho(x,y) = 3 exp(-r^2/2)
from the grid center, kernel w = exp(-(rho_u - rho_v)^2), so pi_w at
lam = 1 prefers boundaries that follow the density gradient. The tree
proposal is blind to that kernel and the local proposal is built from it,
which is the regime where an informed local move should matter if it ever
does.

Writes: iowa_hybrid_test_out/hybrid_validate_5x5.npz
Usage:  python3 recom/hybrid_validate_5x5.py [n_steps] [seed]
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import f as f_dist

_HERE = Path(__file__).resolve().parent
_DIFF = _HERE.parent
_REPO = _DIFF.parent
if str(_DIFF) not in sys.path:
    sys.path.insert(0, str(_DIFF))

from recom.problems import setup_problem, assignment_canonical
from recom.qp_model import kernel_uniform, make_kernel_density
from recom.hybrid import Target, run_hybrid

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40_000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42
P_LOCAL = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
TAU = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1   # paper's locked setting
ALPHA, BETA = TAU, 1.0         # only the ratio enters the optimum
OUT_TAG = f"_p{P_LOCAL:g}_tau{TAU:g}"
BURN = 0.1
RNG = np.random.default_rng(20261013)


def ess_fft(x):
    x = np.asarray(x, float); x = x - x.mean(); n = len(x)
    if n < 8 or x.var() == 0:
        return float(n)
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, nfft)
    acf = np.fft.irfft(f * np.conj(f), nfft)[:n]
    acf /= acf[0]
    tau = 1.0
    for m in range(1, n // 2):
        g = acf[2 * m - 1] + acf[2 * m]
        if g <= 0:
            break
        tau += 2 * g
    return max(n / tau, 1.0)


def tv(q, p):
    return 0.5 * float(np.abs(q - p).sum())


def floor_tv(p, n_eff, n_boot=300):
    n = max(int(round(n_eff)), 1)
    v = np.array([tv(RNG.multinomial(n, p) / n, p) for _ in range(n_boot)])
    return float(v.mean()), float(v.std(ddof=1))


def hotelling_marginal(ids_post, cuts_by_id, vals, p_marg, n_blocks=100):
    """Batch-means Hotelling T^2 on the cut-count marginal."""
    c = cuts_by_id[ids_post]
    iat = max(len(c) / ess_fft((c == v).astype(float)) for v in vals)
    L = max(int(50 * iat), len(c) // n_blocks)
    B = len(c) // L
    if B <= len(vals) + 3:
        L = len(c) // (len(vals) + 10); B = len(c) // L
    blocks = c[:B * L].reshape(B, L)
    P = np.stack([np.array([(b == v).mean() for v in vals]) for b in blocks])
    Y = (P - p_marg)[:, :-1]
    ybar = Y.mean(axis=0); S = np.cov(Y, rowvar=False, ddof=1)
    p = Y.shape[1]
    T2 = B * float(ybar @ np.linalg.solve(S, ybar))
    F = (B - p) / (p * (B - 1)) * T2
    return float(f_dist.sf(F, p, B - p)), B, L, iat


def main():
    p = setup_problem("5x5", _REPO)
    graph, eps, ideal, init = p["graph"], p["eps"], p["ideal"], p["init"]
    lookup = p["fiber_lookup"]
    n_fiber = len(lookup)

    # synthetic city for the density kernel
    for (x, y) in graph.nodes():
        graph.nodes[(x, y)]["density"] = 3.0 * np.exp(-((x - 2) ** 2 + (y - 2) ** 2) / 2.0)

    with open(_HERE / "fiber_5x5" / "fiber_5x5_partitions.json") as f:
        data = json.load(f)
    plans = {}
    for pid, parts in data.items():
        a = {}
        for d, part in enumerate(parts):
            for s in part:
                a[tuple(s)] = d
        plans[int(pid)] = a
    ids_sorted = np.array(sorted(plans))
    pos = {i: j for j, i in enumerate(ids_sorted)}

    def state_id(asn):
        return lookup.get(assignment_canonical(asn), -1)

    configs = [
        ("uniform", kernel_uniform, 0.10),
        ("uniform", kernel_uniform, 1.00),
        ("density", make_kernel_density(sigma=1.0), 1.00),
    ]
    store = {"n_steps": N, "seed": SEED, "p_local": P_LOCAL, "tau": TAU}
    print(f"5x5 fiber, {n_fiber} states, {N} steps per chain, "
          f"p_local = {P_LOCAL}, tau = {TAU}\n")
    hdr = (f"{'config':>18} {'chain':>9} {'TV marg':>8} {'floor':>7} {'z':>7} "
           f"{'Hotel p':>8} {'TV full':>8} {'floor':>7} {'acc tree':>9} "
           f"{'acc loc':>8} {'mv/loc':>7} {'ESS':>7} {'wall s':>7} {'ESS/s':>7}")
    print(hdr)
    rows = []
    for kname, kfn, lam in configs:
        tgt = Target(graph, kfn, lam)
        wc_all = np.array([tgt.wcut(plans[i]) for i in ids_sorted])
        nc_all = np.array([tgt.ncut(plans[i]) for i in ids_sorted])
        logpi = -lam * wc_all
        pi = np.exp(logpi - logpi.max()); pi /= pi.sum()
        vals = np.unique(nc_all)
        p_marg = np.array([pi[nc_all == v].sum() for v in vals])
        cuts_by_pos = nc_all
        tag = f"{kname} lam={lam:g}"
        store[f"{tag}_pi"] = pi
        for cname, pl in (("tree-only", 0.0), ("hybrid", P_LOCAL)):
            res = run_hybrid(tgt, init, ideal, eps, N, pl, ALPHA, BETA, SEED,
                             state_id=state_id, log_every=0, label=f"{tag} {cname}")
            burn = int(N * BURN)
            ids_post = np.array([pos[i] for i in res["ids"][burn:]])
            wc_post = res["wcut"][burn:]
            ess = ess_fft(wc_post)
            # marginal
            emp_m = np.array([(cuts_by_pos[ids_post] == v).mean() for v in vals])
            tvm = tv(emp_m, p_marg)
            fm, fs = floor_tv(p_marg, ess)
            z = (tvm - fm) / fs if fs > 0 else np.inf
            hp, B, L, iat = hotelling_marginal(ids_post, cuts_by_pos, vals, p_marg)
            # full state
            q = np.bincount(ids_post, minlength=n_fiber) / len(ids_post)
            tvf = tv(q, pi)
            ff, _ = floor_tv(pi, ess, n_boot=60)
            k = res["kind"][burn:]; a = res["accepted"][burn:]; fe = res["feasible"][burn:]
            acc_t = a[k == 0].mean() if (k == 0).any() else np.nan
            acc_l = a[k == 1].mean() if (k == 1).any() else np.nan
            mv_l = (a[k == 1] * res["flips"][burn:][k == 1]).mean() if (k == 1).any() else np.nan
            ess_s = ess / res["wall"]
            print(f"{tag:>18} {cname:>9} {tvm:8.4f} {fm:7.4f} {z:+7.1f} {hp:8.3f} "
                  f"{tvf:8.4f} {ff:7.4f} {acc_t:9.3f} {acc_l:8.3f} {mv_l:7.3f} "
                  f"{ess:7.0f} {res['wall']:7.1f} {ess_s:7.1f}")
            rows.append((tag, cname, tvm, fm, z, hp, tvf, ff, acc_t, acc_l, mv_l, ess, res["wall"], ess_s))
            store[f"{tag}_{cname}_wcut"] = res["wcut"]
            store[f"{tag}_{cname}_ids"] = res["ids"]
            store[f"{tag}_{cname}_kind"] = res["kind"]
            store[f"{tag}_{cname}_accepted"] = res["accepted"]
            store[f"{tag}_{cname}_wall"] = res["wall"]
            store[f"{tag}_{cname}_flips"] = res["flips"]
            store[f"{tag}_{cname}_fast"] = res["fast"]
        r_t = [r for r in rows if r[0] == tag and r[1] == "tree-only"][0]
        r_h = [r for r in rows if r[0] == tag and r[1] == "hybrid"][0]
        print(f"{'':>18} kill ratio ESS/s hybrid / tree-only = {r_h[13]/r_t[13]:.2f}"
              f"   (>= 1.5 passes, < 1 means the local move is a passenger)\n")

    np.savez(_HERE / "iowa_hybrid_test_out" / f"hybrid_validate_5x5{OUT_TAG}.npz", **store)
    print(f"Saved hybrid_validate_5x5{OUT_TAG}.npz")
    print("\nREAD: a chain targeting pi_w correctly has z near 0 and Hotelling p "
          "well above 0.05 for BOTH tree-only and hybrid. If tree-only passes and "
          "hybrid fails, the mixture is wrong. If both fail, the move is wrong.")


if __name__ == "__main__":
    main()
