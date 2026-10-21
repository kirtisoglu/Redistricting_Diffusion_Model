"""Exact convergence validation of Chain E on the 18-state 2x3 fiber.

Why this fiber and not the 5x5. On the 5x5 fiber Chain E has 4006 states
to cover, and after 1e5 steps it has visited 403 of them at TV 0.90
against an independent-sampler floor of about 0.085. So the paper's
end-to-end evidence reduces to one scalar, the mean, landing 0.17 from
its target while the distribution is nowhere near it. That is a weak
claim and a referee will say so.

On the 2x3 fiber there are 18 states. A long run can genuinely cover it,
TV can genuinely approach zero, and the claim "Chain E converges to its
stated target" becomes a real, falsifiable measurement rather than a
one-dimensional coincidence. This is the same state space whose
transition kernel db_check_2x3.py already verifies to 1.5e-15, so the
two checks compose: the kernel satisfies detailed balance exactly, and
the chain it generates does reach pi.

It also delivers a hypothesis test rather than a distance. With 18 cells
and a chain thinned to its integrated autocorrelation time, a chi-square
goodness-of-fit test against the exactly known pi is available. "We
cannot reject that Chain E's output is pi (p = ...)" is a far stronger
sentence than "the empirical mean is within 1% of the target mean".

NOTE ON LABELING, which the paper needs to state in Section 4.5. The
state space enumerated here is LABELED assignments, matching both the
implementation and db_check_2x3.py. Chain E's proposal map is a bijection
on labeled plans; on unlabeled partitions it is two-to-one, because a
Bernoulli draw and its complement give the same partition with the
district names swapped, and the transition probability would then need
both terms. The proof in Section 4.5 is written with set notation, which
reads as unlabeled. Since pi depends only on the cut count it is
label-invariant, so sampling labeled plans and forgetting the labels
gives the right answer, but the paper has to say which space it is on.

Parameters match Chain E as locked in Appendix A.2 and as used by
db_check_2x3.py, so nothing here is tuned for this test.

Writes one new file:
    iowa_hybrid_test_out/long_run_2x3_seed<SEED>.npz

THIS SCRIPT RUNS A CHAIN. It is the one Tier 1 item that needs compute.
Budget: two small Gurobi solves per step. Start with 20000 steps to
time it before committing to the full run.

Usage: python3 recom/long_run_2x3.py [n_steps] [seed]
"""

import random
import sys
import time
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.stats import chisquare

_HERE = Path(__file__).resolve().parent
_DIFF = _HERE.parent
if str(_DIFF) not in sys.path:
    sys.path.insert(0, str(_DIFF))

from recom.qp_model import qp_mh_proposal, kernel_uniform

N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42

# Locked Chain E parameters (Appendix A.2), and the tolerance used by
# db_check_2x3.py so the two checks share a state space.
ALPHA, BETA, EPS, LAM = 1.0, 10.0, 1.0 / 3.0, 0.10
BURN_FRAC = 0.1


def ess_geyer(x):
    """Geyer initial-positive-sequence ESS with an FFT autocovariance, so
    it is O(n log n). The direct np.correlate version is O(n^2) and takes
    longer than the chain itself past a few hundred thousand steps."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    x = x - x.mean()
    if n < 8 or x.var() == 0:
        return float(n)
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, nfft)
    acf = np.fft.irfft(f * np.conj(f), nfft)[:n]
    acf = acf / acf[0]
    tau = 1.0
    for m in range(1, n // 2):
        gamma = acf[2 * m - 1] + acf[2 * m]
        if gamma <= 0:
            break
        tau += 2.0 * gamma
    return max(n / tau, 1.0)


def tv_floor(p, n):
    """Leading-order E[TV] for n independent draws from p."""
    n = np.maximum(np.asarray(n, dtype=float), 1.0)
    return 0.5 * np.sqrt(2.0 * p * (1.0 - p) / np.pi).sum() / np.sqrt(n)


def build_fiber():
    """All valid LABELED 2-district plans of the 2x3 grid, matching
    db_check_2x3.py exactly."""
    G = nx.grid_graph([3, 2])
    for n in G.nodes():
        G.nodes[n]["population"] = 1
        G.nodes[n]["density"] = 0
    cells = sorted(G.nodes())
    ideal = 3.0
    lo = int(np.ceil((1 - EPS) * ideal - 1e-9))
    hi = int(np.floor((1 + EPS) * ideal + 1e-9))

    states = []
    for size in range(lo, hi + 1):
        for S in combinations(cells, size):
            S, T = set(S), set(cells) - set(S)
            if not T:
                continue
            if nx.is_connected(G.subgraph(S)) and nx.is_connected(G.subgraph(T)):
                states.append({c: (0 if c in S else 1) for c in cells})

    def key(asn):
        return tuple(asn[c] for c in cells)

    index = {key(a): i for i, a in enumerate(states)}
    cuts = np.array([sum(1 for u, v in G.edges() if a[u] != a[v])
                     for a in states], dtype=float)
    return G, cells, ideal, states, index, key, cuts


def main():
    G, cells, ideal, states, index, key, cuts = build_fiber()
    n_states = len(states)
    print(f"2x3 fiber: {n_states} valid labeled states "
          f"(sizes within eps={EPS:.3f} of {ideal})")
    print(f"cut counts: {sorted(set(cuts.astype(int).tolist()))}")

    logp = -LAM * cuts
    logp -= logp.max()
    pi = np.exp(logp)
    pi /= pi.sum()
    e_pi = float((pi * cuts).sum())
    print(f"exact target: E_pi[cuts] = {e_pi:.6f} at lam = {LAM}")

    random.seed(SEED)
    np.random.seed(SEED)
    asn = dict(states[0])

    visits = np.zeros(n_states, dtype=np.int64)
    trace = np.empty(N_STEPS, dtype=np.int32)
    n_accept = 0
    t0 = time.perf_counter()

    for t in range(N_STEPS):
        asn, accepted = qp_mh_proposal(
            G, asn, alpha=ALPHA, beta=BETA, epsilon=EPS, lam=LAM,
            kernel_fn=kernel_uniform, ideal_pop=ideal)
        n_accept += bool(accepted)
        k = key(asn)
        if k not in index:
            raise SystemExit(
                f"step {t}: chain reached a state outside the enumerated "
                f"fiber ({k}). The chain's validity test and this "
                f"enumeration disagree, which is itself a finding.")
        j = index[k]
        trace[t] = j
        visits[j] += 1
        if (t + 1) % 10_000 == 0:
            el = time.perf_counter() - t0
            print(f"  {t+1:>8d} steps  {el:7.1f}s  "
                  f"accept {n_accept/(t+1):.3f}  "
                  f"visited {int((visits > 0).sum())}/{n_states}")

    wall = time.perf_counter() - t0
    burn = int(N_STEPS * BURN_FRAC)
    post = trace[burn:]
    post_cuts = cuts[post]
    ess = ess_geyer(post_cuts)

    print(f"\nwall {wall:.1f}s, acceptance rate {n_accept/N_STEPS:.4f}")
    print(f"(the paper reports no acceptance rates anywhere; this is one)")

    grid = np.unique(np.logspace(2, np.log10(len(post)), 40).astype(int))
    tvs = np.empty(len(grid))
    for i, n in enumerate(grid):
        q = np.bincount(post[:n], minlength=n_states).astype(float) / n
        tvs[i] = 0.5 * np.abs(q - pi).sum()

    q_full = np.bincount(post, minlength=n_states).astype(float) / len(post)
    tv_final = 0.5 * float(np.abs(q_full - pi).sum())
    floor_raw = float(tv_floor(pi, len(post)))
    floor_ess = float(tv_floor(pi, ess))

    print(f"\ncoverage      {int((visits > 0).sum())}/{n_states}")
    print(f"mean cuts     {post_cuts.mean():.6f}   target {e_pi:.6f}   "
          f"gap {post_cuts.mean()-e_pi:+.6f}")
    print(f"ESS(cuts)     {ess:.1f} of {len(post)} retained")
    print(f"TV(emp, pi)   {tv_final:.6f}")
    print(f"  floor at n   {floor_raw:.6f}   (independent draws)")
    print(f"  floor at ESS {floor_ess:.6f}   (conservative)")

    # Goodness of fit on a thinned sample, so the independence assumption
    # the test needs is approximately satisfied.
    thin = max(int(round(len(post) / ess)), 1)
    sample = post[::thin]
    obs = np.bincount(sample, minlength=n_states).astype(float)
    exp = pi * obs.sum()
    keep = exp >= 5.0
    if keep.sum() >= 2 and (~keep).sum() == 0:
        stat, pval = chisquare(obs, exp)
        print(f"\nchi-square GOF on {len(sample)} thinned draws "
              f"(thin = {thin}): stat {stat:.2f}, df {n_states-1}, "
              f"p = {pval:.4f}")
        print("  p well above 0.05 means the output is statistically "
              "indistinguishable from the exact target. That is the "
              "sentence Section 5.1 should contain.")
    else:
        stat, pval = float("nan"), float("nan")
        print(f"\nchi-square skipped: {int((~keep).sum())} cells have "
              f"expected count below 5 after thinning. Run more steps.")

    # Filename carries the step count so runs of different length never
    # overwrite each other. gof_2x3.py takes the filename as its argument.
    dest = (_HERE / "iowa_hybrid_test_out" /
            f"long_run_2x3_n{N_STEPS}_seed{SEED}.npz")
    np.savez(dest, trace=trace, visits=visits, cuts=cuts, pi=pi,
             tv_grid=grid, tv_curve=tvs, tv_final=tv_final,
             floor_raw=floor_raw, floor_ess=floor_ess, ess=ess,
             chi2_stat=stat, chi2_p=pval, accept_rate=n_accept / N_STEPS,
             e_pi=e_pi, mean_cuts=float(post_cuts.mean()), wall=wall,
             n_steps=N_STEPS, seed=SEED, lam=LAM, alpha=ALPHA, beta=BETA,
             eps=EPS)
    print(f"\nSaved {dest.name}")


if __name__ == "__main__":
    main()
