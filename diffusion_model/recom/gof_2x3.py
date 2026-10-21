"""Correlation-aware goodness-of-fit for Chain E on the 2x3 fiber.

WHY THIS EXISTS. long_run_2x3.py thinned the chain by N/ESS and ran a
chi-square test, which rejected at p = 1e-4. That test is not valid and
its rejection should not be reported. The ESS it thinned by was computed
from the CUT-COUNT trace, which takes only two values on this fiber, so
it describes the autocorrelation of a coarse two-valued summary, not of
the 18-state occupancy process the test is about. Thinning an 18-cell
multinomial by a two-cell summary's correlation time leaves substantial
dependence, and residual positive dependence inflates a chi-square
statistic exactly the way we observed. The diagnostic was under-specified,
not the sampler.

WHAT THIS DOES INSTEAD. Batch means, which is the standard way to test a
distributional hypothesis on a correlated chain. Split the post-burn-in
trace into B non-overlapping blocks whose length greatly exceeds the
integrated autocorrelation time. Each block yields an occupancy vector.
The between-block scatter then estimates the sampling covariance of the
overall occupancy vector WITH the chain's dependence built in, and no
thinning or independence assumption is needed.

Two tests are reported.

  * Hotelling T^2 on the 17 free coordinates of the occupancy vector,
    with the covariance estimated from the blocks. This is the rigorous
    joint test.
  * A per-state z table, so that if there is a real discrepancy you can
    see which states carry it rather than only that something is off.

Plus a moving-block bootstrap confidence interval for total variation,
which answers the question the point estimate cannot: is TV = 0.025
distinguishable from zero given this chain's correlation?

Reads:  iowa_hybrid_test_out/long_run_2x3_seed<SEED>.npz
Writes: nothing. Prints only.

Usage: python3 recom/gof_2x3.py [seed] [n_blocks]
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import f as f_dist

HERE = Path(__file__).resolve().parent
OUT = HERE / "iowa_hybrid_test_out"

SRC_NAME = sys.argv[1] if len(sys.argv) > 1 else "long_run_2x3_seed42.npz"
N_BLOCKS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
BURN_FRAC = 0.1
N_BOOT = 2000
RNG = np.random.default_rng(20260918)


def iat_fft(x):
    """Integrated autocorrelation time, Geyer initial-positive-sequence,
    with an FFT autocovariance so this is O(n log n) rather than O(n^2)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    if n < 8 or x.var() == 0:
        return 1.0
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, nfft)
    acf = np.fft.irfft(f * np.conj(f), nfft)[:n]
    if acf[0] <= 0:
        return 1.0
    acf = acf / acf[0]
    tau = 1.0
    for m in range(1, n // 2):
        g = acf[2 * m - 1] + acf[2 * m]
        if g <= 0:
            break
        tau += 2.0 * g
    return max(tau, 1.0)


def main():
    src = OUT / SRC_NAME
    if not src.exists():
        raise SystemExit(f"missing {src.name}; run long_run_2x3.py first "
                         f"(usage: gof_2x3.py <npz filename> [n_blocks])")
    d = np.load(src)
    trace = d["trace"].astype(int)
    pi = d["pi"]
    cuts = d["cuts"]
    k = len(pi)

    burn = int(len(trace) * BURN_FRAC)
    post = trace[burn:]
    n = len(post)

    print(f"2x3 fiber, {k} states, {n} retained steps")

    # The block length must beat the STATE-OCCUPANCY correlation time, not
    # the cut trace's. On this fiber the cut count takes two values, so its
    # correlation time badly understates the 18-state process, which is what
    # sank the thinned chi-square in long_run_2x3.py.
    iat_cut = n / float(d["ess"])
    iats = np.array([iat_fft((post == s).astype(float)) for s in range(k)])
    iat_max = float(iats.max())
    print(f"IAT from the cut trace        {iat_cut:8.1f}")
    print(f"IAT per state, max over {k:2d}    {iat_max:8.1f}  "
          f"(median {float(np.median(iats)):.1f})")
    if iat_max > 2 * iat_cut:
        print(f"  -> the cut trace understates dependence by "
              f"{iat_max/iat_cut:.1f}x. Blocks are sized from the state IAT.")

    L = max(int(50 * iat_max), n // N_BLOCKS)
    B = n // L
    if B <= k + 4:                       # keep the covariance estimable
        L = n // (k + 20)
        B = n // L
        print(f"  -> capped block length to keep {B} blocks for a "
              f"{k-1}-dimensional covariance")
    post = post[:B * L]
    print(f"batch means: {B} blocks of {L} steps, block/IAT = "
          f"{L/iat_max:.0f}")

    # Occupancy vector per block.
    blocks = post.reshape(B, L)
    P = np.stack([np.bincount(b, minlength=k) / L for b in blocks])
    phat = P.mean(axis=0)

    tv = 0.5 * float(np.abs(phat - pi).sum())
    print(f"\noverall TV(empirical, pi) = {tv:.5f}")
    print(f"mean cuts {float((phat*cuts).sum()):.6f}  "
          f"target {float((pi*cuts).sum()):.6f}")

    # Hotelling T^2 on the 17 free coordinates.
    Y = (P - pi)[:, :-1]
    ybar = Y.mean(axis=0)
    S = np.cov(Y, rowvar=False, ddof=1)
    p = k - 1
    if B > p:
        T2 = B * float(ybar @ np.linalg.solve(S, ybar))
        Fstat = (B - p) / (p * (B - 1)) * T2
        pval = float(f_dist.sf(Fstat, p, B - p))
        print(f"\nHotelling T^2 = {T2:.2f}, F({p},{B-p}) = {Fstat:.3f}, "
              f"p = {pval:.4f}")
        if pval > 0.05:
            print("  Not rejected. The occupancy distribution is "
                  "statistically indistinguishable from the exact target "
                  "once the chain's own dependence is accounted for.")
        else:
            print("  REJECTED. Investigate before making any claim: check "
                  "block length against the state-occupancy correlation "
                  "time, then the sampler itself.")
    else:
        print(f"\n[need more blocks than states: B={B}, p={p}]")

    # Per-state z, so a real discrepancy is locatable.
    se = P.std(axis=0, ddof=1) / np.sqrt(B)
    z = (phat - pi) / np.where(se > 0, se, np.inf)
    order = np.argsort(-np.abs(z))
    print(f"\n  {'state':>5} {'cuts':>5} {'pi':>9} {'empirical':>10} "
          f"{'z':>7}")
    for s in order[:6]:
        print(f"  {s:>5d} {int(cuts[s]):>5d} {pi[s]:9.5f} {phat[s]:10.5f} "
              f"{z[s]:+7.2f}")
    print(f"  (largest |z| of {k} states; +-2 is the usual eyeball "
          f"threshold, and with {k} states one or two near 2 is expected)")

    # Moving-block bootstrap CI for TV, preserving dependence.
    idx = np.arange(B)
    tvs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = RNG.choice(idx, size=B, replace=True)
        pb = P[pick].mean(axis=0)
        tvs[i] = 0.5 * np.abs(pb - pi).sum()
    lo, hi = np.quantile(tvs, [0.025, 0.975])
    print(f"\nblock-bootstrap 95% CI for TV: [{lo:.5f}, {hi:.5f}]")
    print(f"  A CI that excludes 0 is expected even for a perfect sampler, "
          f"because TV is a nonnegative statistic with positive bias at "
          f"finite n. The Hotelling test above is the one that answers "
          f"whether the distribution differs.")


if __name__ == "__main__":
    main()
