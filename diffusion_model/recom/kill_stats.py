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


def hotelling_marginal(ids_post, cuts_by_id, vals, p_marg, n_blocks=100):
    """Batch-means Hotelling T^2 on the cut-count marginal."""
    c = cuts_by_id[ids_post]
    iat = max(len(c) / ess_fft((c == v).astype(float)) for v in vals)
    L = max(int(50 * iat), len(c) // n_blocks)
    B = len(c) // L
    if B <= len(vals) + 3:
        L = len(c) // (len(vals) + 10)
        B = len(c) // L
    blocks = c[:B * L].reshape(B, L)
    P = np.stack([np.array([(b == v).mean() for v in vals]) for b in blocks])
    Y = (P - p_marg)[:, :-1]
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
