"""Distributional ground-truth calibration on the enumerated 5x5 fiber.

This is the experiment the paper is missing.  Section 5.1 currently shows
that two *scalars* agree: Chain A's mean cut count (16.73) sits 0.05 from
the Matrix-Tree tree-weighted-and-tempered mean (16.78) and 1.83 from the
nominal tempered mean (18.56).  A mean is a one-dimensional summary, and
many distributions share one.  Because the fiber is fully enumerated, the
*distributions* are directly comparable, which turns a suggestive
coincidence into a falsifiable claim.

The claim under test:

    tree-based recombination under a symmetric-q Metropolis filter
    converges to the spanning-tree-weighted analogue of its nominal
    target, not to the nominal target itself.

The test: compare each chain's empirical distribution against three
exactly computable reference measures over all 4006 partitions,

    pi_nominal   (P) ∝ exp(-lam * cuts(P))
    pi_tree      (P) ∝ prod_i tau(V_i)
    pi_tree_temp (P) ∝ prod_i tau(V_i) * exp(-lam * cuts(P))

and report total variation, KL and chi-square to each.  If the claim
holds, distance to pi_tree_temp collapses to the sampling-noise floor
while distance to pi_nominal plateaus well above it.

TWO METHODOLOGICAL POINTS, both of which the current draft needs.

1.  A raw TV number is uninterpretable.  With 4006 states even a perfect
    independent sampler shows large TV unless the draw count greatly
    exceeds the state count, which is exactly why the paper's TV column
    "mostly restates the coverage column".  So every distance here is
    reported against a *parametric bootstrap noise floor*: the same
    distance computed for genuinely independent draws from the reference
    measure, at the same sample size.  A chain that has converged sits at
    its floor.  A chain converging to something else sits above it by
    many floor standard deviations.

2.  Chain draws are autocorrelated, so the honest floor uses the
    effective sample size, not the raw step count.  Both are reported.
    The truth lies between them, and the conclusion should be stated
    only if it survives the conservative (ESS-based) floor.

The headline statistic is the CUT-COUNT MARGINAL, not the full state
distribution.  The fiber's cut counts take ~20 distinct values, so with
1.8e5 retained draws the marginal is estimated to three decimals and the
noise floor is tiny.  That comparison is immune to the coverage problem
that makes full-state TV uninformative, and it is the one a referee can
check by eye against the figure.

Reads (never writes) these existing artifacts:
    iowa_hybrid_test_out/long_run_5x5_seed<SEED>.npz
    iowa_hybrid_test_out/matrix_tree_targets.npz
    fiber_5x5/fiber_5x5_partitions.json
    iowa_hybrid_test_out/target_matched_5x5_seed<SEED>.npz   (optional)

Writes one new file:
    iowa_hybrid_test_out/calibration_distance_seed<SEED>.npz

Usage: python3 recom/calibration_distance.py [seed] [n_boot]
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "iowa_hybrid_test_out"

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
N_BOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 400
BURN_FRAC = 0.1
RNG = np.random.default_rng(20260918)


# ─── distances ───────────────────────────────────────────────────────────────

def tv(q, p):
    return 0.5 * float(np.abs(q - p).sum())


def kl(q, p):
    m = q > 0
    return float((q[m] * (np.log(q[m]) - np.log(p[m]))).sum())


def chi2(q, p, n):
    m = p > 0
    return float(n * ((q[m] - p[m]) ** 2 / p[m]).sum())


def ess_geyer(x):
    """Geyer initial-positive-sequence ESS. Same estimator the paper uses
    (iowa_target_matched.ess_geyer), duplicated here so this script has no
    import-time dependency on the orchestrator."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    x = x - x.mean()
    var = np.dot(x, x) / n
    if var == 0 or n < 4:
        return float(n)
    acf = np.correlate(x, x, mode="full")[n - 1:] / (n * var)
    tau = 1.0
    for m in range(1, n // 2):
        gamma = acf[2 * m - 1] + acf[2 * m]
        if gamma <= 0:
            break
        tau += 2.0 * gamma
    return max(n / tau, 1.0)


def noise_floor(p, n_draws, n_boot, rng):
    """Distribution of TV(empirical, p) for n_draws INDEPENDENT draws from p.

    This is the yardstick: a converged chain cannot do better than this, and
    a chain targeting p should land inside it. Returns (mean, sd, q025, q975).
    """
    n = max(int(round(n_draws)), 1)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        vals[b] = tv(rng.multinomial(n, p) / n, p)
    return (float(vals.mean()), float(vals.std(ddof=1)),
            float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))


# ─── reference measures ──────────────────────────────────────────────────────

def load_references():
    """Build the three reference measures over the fiber, aligned to the
    state ids used by the chain runners.

    Alignment matters and is easy to get wrong. problems._setup_5x5 keys the
    fiber by int(pid) where pid is the JSON key, while matrix_tree_targets.py
    builds positional arrays by iterating the same JSON in file order. So
    position j in the Matrix-Tree arrays corresponds to state id
    int(list(json)[j]). We rebuild that map from the JSON rather than assuming
    the keys happen to be 0..4005 in order.
    """
    with open(HERE / "fiber_5x5" / "fiber_5x5_partitions.json") as f:
        keys = list(json.load(f).keys())
    ids = np.array([int(k) for k in keys])

    mt = np.load(OUT / "matrix_tree_targets.npz")
    cuts = np.asarray(mt["cuts"], dtype=float)
    logw = np.asarray(mt["log_tree_weight"], dtype=float)
    lam = float(mt["lam"])
    if len(cuts) != len(ids):
        raise SystemExit(
            f"fiber size mismatch: {len(ids)} partitions in JSON vs "
            f"{len(cuts)} rows in matrix_tree_targets.npz")

    def normalize(logp):
        logp = logp - logp.max()
        p = np.exp(logp)
        return p / p.sum()

    refs = {
        "nominal":   normalize(-lam * cuts),
        "tree":      normalize(logw),
        "tree_temp": normalize(logw - lam * cuts),
    }
    return ids, cuts, lam, refs


def marginalize(p, cuts):
    """Push a state measure forward to the cut-count marginal."""
    vals = np.unique(cuts)
    return vals, np.array([p[cuts == v].sum() for v in vals])


# ─── reporting ───────────────────────────────────────────────────────────────

def report_block(title, q, refs, n_eff_raw, n_eff_ess, n_boot, rng,
                 store, prefix):
    print(f"\n{title}")
    print(f"  {'reference':<12} {'TV':>8} {'floor(n)':>20} "
          f"{'floor(ESS)':>20} {'KL':>9}")
    for name, p in refs.items():
        d_tv = tv(q, p)
        f_raw = noise_floor(p, n_eff_raw, n_boot, rng)
        f_ess = noise_floor(p, n_eff_ess, n_boot, rng)
        z_ess = (d_tv - f_ess[0]) / f_ess[1] if f_ess[1] > 0 else np.inf
        print(f"  {name:<12} {d_tv:8.4f} "
              f"{f_raw[0]:8.4f}±{f_raw[1]:.4f}      "
              f"{f_ess[0]:8.4f}±{f_ess[1]:.4f}   "
              f"{kl(q, p):9.4f}   z(ESS)={z_ess:+8.1f}")
        store[f"{prefix}_{name}_tv"] = d_tv
        store[f"{prefix}_{name}_kl"] = kl(q, p)
        store[f"{prefix}_{name}_floor_raw"] = np.array(f_raw)
        store[f"{prefix}_{name}_floor_ess"] = np.array(f_ess)
        store[f"{prefix}_{name}_z_ess"] = z_ess


def tv_trajectory(trace, ref_marginals, vals, grid):
    """TV of the running cut-marginal to each reference, as a function of
    how many steps have been used. Feeds panel (c) of the figure and shows
    the qualitative signature directly: convergence to one reference,
    a plateau against another."""
    out = {k: np.empty(len(grid)) for k in ref_marginals}
    for i, t in enumerate(grid):
        emp = np.array([(trace[:t] == v).sum() for v in vals], dtype=float)
        emp /= emp.sum()
        for k, pm in ref_marginals.items():
            out[k][i] = tv(emp, pm)
    return out


def main():
    ids, cuts, lam, refs = load_references()
    n_fiber = len(ids)
    pos_of_id = {int(s): j for j, s in enumerate(ids)}
    vals, ref_marg = None, {}
    for name, p in refs.items():
        vals, m = marginalize(p, cuts)
        ref_marg[name] = m

    print(f"5x5 fiber: {n_fiber} partitions, lam = {lam}")
    print(f"reference means: " + "  ".join(
        f"{k}={float((p*cuts).sum()):.4f}" for k, p in refs.items()))
    print(f"cut values present: {vals.astype(int).tolist()}")

    store = {"lam": lam, "cut_values": vals, "n_fiber": n_fiber}
    for name, p in refs.items():
        store[f"ref_{name}_marginal"] = ref_marg[name]
        store[f"ref_{name}_mean"] = float((p * cuts).sum())

    lr_path = OUT / f"long_run_5x5_seed{SEED}.npz"
    if not lr_path.exists():
        raise SystemExit(f"missing {lr_path}; run long_run_5x5.py first")
    lr = np.load(lr_path)

    for tag in ("A", "E"):
        trace = np.asarray(lr[f"{tag}_cuts"], dtype=float)
        burn = int(len(trace) * BURN_FRAC)
        post = trace[burn:]
        ess = ess_geyer(post)
        print(f"\n{'='*70}\nChain {tag}: {len(trace)} steps, "
              f"{len(post)} retained, ESS(cuts) = {ess:.1f}, "
              f"mean = {post.mean():.4f}")

        # (1) HEADLINE: cut-count marginal. Well estimated, tiny noise floor.
        emp_m = np.array([(post == v).sum() for v in vals], dtype=float)
        emp_m /= emp_m.sum()
        report_block("  cut-count marginal", emp_m, ref_marg,
                     len(post), ess, N_BOOT, RNG, store, f"{tag}_marg")
        store[f"{tag}_emp_marginal"] = emp_m
        store[f"{tag}_mean"] = float(post.mean())
        store[f"{tag}_ess"] = ess

        # (2) Full state distribution. Included for completeness, but with
        #     4006 states and correlated draws the floor is large; read the
        #     z-scores, never the raw TV.
        vid = np.asarray(lr[f"{tag}_visit_ids"])
        vct = np.asarray(lr[f"{tag}_visit_counts"], dtype=float)
        q = np.zeros(n_fiber)
        for s, c in zip(vid, vct):
            q[pos_of_id[int(s)]] = c
        n_vis = q.sum()
        q /= n_vis
        print(f"  [full-state visits span the whole run including burn-in; "
              f"{int((q > 0).sum())}/{n_fiber} states, n={int(n_vis)}]")
        report_block("  full state distribution", q, refs,
                     n_vis, ess, N_BOOT, RNG, store, f"{tag}_full")
        store[f"{tag}_coverage"] = int((q > 0).sum())

        # (3) TV against run length, for the figure and for showing that the
        #     gap to the nominal target is a plateau rather than slow drift.
        grid = np.unique(np.logspace(2, np.log10(len(post)), 40).astype(int))
        traj = tv_trajectory(post, ref_marg, vals, grid)
        store[f"{tag}_tv_grid"] = grid
        for k, v in traj.items():
            store[f"{tag}_tv_traj_{k}"] = v

    # Short-run chains: cut marginals only. 900 retained draws is too few to
    # be decisive, so these are reported as directional, and a long V run is
    # the obvious gap (long_run_5x5.py only ever ran A and E).
    sr_path = OUT / f"target_matched_5x5_seed{SEED}.npz"
    if sr_path.exists():
        sr = np.load(sr_path)
        print(f"\n{'='*70}\nShort-run chains (1000 steps, directional only)")
        for tag in ("V", "A", "B", "C", "D", "E"):
            if tag not in sr:
                continue
            post = np.asarray(sr[tag], dtype=float)[100:]
            emp_m = np.array([(post == v).sum() for v in vals], dtype=float)
            emp_m /= emp_m.sum()
            line = "  ".join(f"{k}={tv(emp_m, m):.4f}"
                             for k, m in ref_marg.items())
            print(f"  {tag}: mean={post.mean():7.4f}  TV(cut marginal) -> {line}")
            store[f"short_{tag}_marg_tv"] = np.array(
                [tv(emp_m, ref_marg[k]) for k in refs])
    else:
        print(f"\n[no short-run file at {sr_path}]")

    dest = OUT / f"calibration_distance_seed{SEED}.npz"
    np.savez(dest, **store)
    print(f"\nSaved {dest.name}")
    print("\nHOW TO READ THIS. The claim is supported if, for Chain A, the "
          "cut-marginal TV to tree_temp sits inside its ESS noise floor "
          "while the TV to nominal sits many floor standard deviations "
          "above it. If both sit above their floors, the chain has not "
          "converged to either and the section needs a longer run, not a "
          "stronger claim.")


if __name__ == "__main__":
    main()
