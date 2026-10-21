"""Statistics shared by the kill-test runners: effective sample size, total
variation with its independent-sampler noise floor, and the batch-means
Hotelling test on the cut-count marginal. Lifted verbatim from
hybrid_validate_5x5.py so the forest-move runners report the same
quantities with the same code."""

import numpy as np
from scipy.stats import f as f_dist

from recom.hybrid import ess_fft  # noqa: F401

RNG = np.random.default_rng(20261013)


def tv(q, p):
    return 0.5 * float(np.abs(q - p).sum())


def floor_tv(p, n_eff, n_boot=300):
    n = max(int(round(n_eff)), 1)
    v = np.array([tv(RNG.multinomial(n, p) / n, p) for _ in range(n_boot)])
    return float(v.mean()), float(v.std(ddof=1))


def merge_bins(vals, p_marg, min_mass=0.02):
    """Group consecutive cut values so every bin carries target mass at
    least min_mass (merging from the tail). A bin whose expected count per
    block is below one makes the batch covariance singular and the
    Hotelling statistic meaningless; under a sharp tree-weighted target
    the top cut value can have mass 3e-4. Returns (bin index per value,
    mass per bin)."""
    bins = []
    cur, mass = [], 0.0
    for i, m in enumerate(p_marg):
        cur.append(i)
        mass += m
        if mass >= min_mass:
            bins.append(cur)
            cur, mass = [], 0.0
    if cur:
        if bins:
            bins[-1].extend(cur)
        else:
            bins.append(cur)
    idx = np.empty(len(vals), dtype=int)
    for b, members in enumerate(bins):
        for i in members:
            idx[i] = b
    pm = np.array([p_marg[members].sum() for members in bins])
    return idx, pm


def hotelling_marginal(ids_post, cuts_by_id, vals, p_marg, n_blocks=100):
    """Batch-means Hotelling T^2 on the cut-count marginal, after merging
    bins of negligible target mass (see merge_bins)."""
    c = cuts_by_id[ids_post]
    bin_of_val, pm = merge_bins(np.asarray(vals), np.asarray(p_marg))
    val_to_bin = {v: bin_of_val[i] for i, v in enumerate(vals)}
    cb = np.array([val_to_bin[v] for v in c])
    nb = len(pm)
    if nb < 2:
        return float("nan"), 0, 0, float("nan")
    iat = max(len(cb) / ess_fft((cb == b).astype(float)) for b in range(nb))
    L = max(int(50 * iat), len(cb) // n_blocks)
    B = len(cb) // L
    if B <= nb + 3:
        L = len(cb) // (nb + 10)
        B = len(cb) // L
    blocks = cb[:B * L].reshape(B, L)
    P = np.stack([np.array([(b == k).mean() for k in range(nb)]) for b in blocks])
    Y = (P - pm)[:, :-1]
    ybar = Y.mean(axis=0)
    S = np.cov(Y, rowvar=False, ddof=1)
    p = Y.shape[1]
    T2 = B * float(ybar @ np.linalg.solve(S, ybar))
    F = (B - p) / (p * (B - 1)) * T2
    return float(f_dist.sf(F, p, B - p)), B, L, iat


def chain_summary(res, burn_frac=0.1):
    """Acceptance by move type, units moved per local step, linear-solve
    share, half-trace means, ESS and ESS per second of the weighted cut."""
    n = len(res["wcut"])
    burn = int(n * burn_frac)
    wc = res["wcut"][burn:]
    k = res["kind"][burn:]
    a = res["accepted"][burn:]
    k0, k1 = k == 0, k == 1
    acc_g = a[k0].mean() if k0.any() else np.nan
    acc_l = a[k1].mean() if k1.any() else np.nan
    mv_l = (a[k1] * res["flips"][burn:][k1]).mean() if k1.any() else np.nan
    fast = res["fast"][burn:][k1].mean() if k1.any() else np.nan
    h = len(wc) // 2
    q = len(wc) // 4
    ess = ess_fft(wc)
    return dict(acc_global=acc_g, acc_local=acc_l, mv_local=mv_l, fast=fast,
                h1=wc[:h].mean(), h2=wc[h:].mean(),
                quarters=[wc[i * q:(i + 1) * q].mean() for i in range(4)],
                ess=ess, wall=res["wall"], ess_s=ess / res["wall"])
