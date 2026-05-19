"""Diagnostic measurements and plot helpers for redistricting experiments.

Public API:
    target_distribution(fiber_cuts, lam)
    expected_target_cuts(fiber_cuts, lam)
    kl_to_target(visits, fiber_cuts, lam)
    tv_to_uniform(visits, n_fiber)
    kernel_correctness_table(chain_results, fiber_cuts, lam, n_steps)
    bias_bound_table(chain_results, n_fiber)
    island_neck_table(chain_results)
    cuts_compare_plot(chain_results, label, init_cuts, lam,
                      e_pi_cuts, burn, out_path)
    island_trace_plot(chain_results, out_path, suptitle)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ─── Target distribution and divergences ────────────────────────────────────

def target_distribution(fiber_cuts, lam):
    """π(P) ∝ exp(−λ · cuts(P)) over the enumerated fiber. Returns
    (ids, p) with p summing to 1."""
    ids = list(fiber_cuts.keys())
    cuts_arr = np.array([fiber_cuts[i] for i in ids])
    log_p = -lam * cuts_arr
    log_p -= log_p.max()
    p = np.exp(log_p)
    p /= p.sum()
    return ids, p


def expected_target_cuts(fiber_cuts, lam):
    ids, p = target_distribution(fiber_cuts, lam)
    return float((p * np.array([fiber_cuts[i] for i in ids])).sum())


def kl_to_target(visits, fiber_cuts, lam):
    """KL(empirical || π_target). Returns (kl, n_total_visits)."""
    ids, p = target_distribution(fiber_cuts, lam)
    n_total = sum(visits.values())
    if n_total == 0:
        return float("nan"), 0
    q = np.array([visits.get(i, 0) for i in ids], dtype=float) / n_total
    mask = q > 0
    return float((q[mask] * (np.log(q[mask]) - np.log(p[mask]))).sum()), n_total


def tv_to_uniform(visits, n_fiber):
    """TV distance between empirical visit distribution and uniform-on-fiber."""
    n_total = sum(visits.values())
    if n_total == 0:
        return float("nan")
    q = np.zeros(n_fiber)
    for pid, count in visits.items():
        q[pid] = count / n_total
    u = np.full(n_fiber, 1.0 / n_fiber)
    return 0.5 * float(np.abs(q - u).sum())


# ─── Diagnostic tables (pretty-printed to stdout) ───────────────────────────

def kernel_correctness_table(chain_results, fiber_cuts, lam, n_steps,
                             burn):
    """Print KL-to-target table.  chain_results: list of (label, dict)."""
    e_pi = expected_target_cuts(fiber_cuts, lam)
    print("\n  KERNEL-CORRECTNESS CHECK (KL to target, fully enumerated fiber):")
    for label, ch in chain_results:
        kl, n_in = kl_to_target(ch["visits"], fiber_cuts, lam)
        emp_mean = ch["cuts"][burn:].mean()
        print(f"    {label:24s}  in-fiber visits = {n_in}/{n_steps}  "
              f"empirical_mean_cuts = {emp_mean:.2f}  "
              f"E_π[cuts] = {e_pi:.2f}  KL = {kl:.4f}")
    print("\n    Interpretation: a CORRECT chain has empirical_mean → E_π[cuts] "
          "and KL → 0 as N → ∞.")
    print(f"    Current N={n_steps} is small; large gaps "
          f"(KL > 0.5 or |empirical−E_π| > 1) signal a biased kernel.")
    return e_pi


def bias_bound_table(chain_results, n_fiber):
    """Print TV-to-uniform table.  chain_results: list of (label, dict)."""
    print("\n  BIAS BOUND (TV distance to uniform-on-fiber):")
    rows = []
    for label, ch in chain_results:
        n_total = sum(ch["visits"].values())
        unique = len(ch["visits"])
        if n_total == 0:
            continue
        tv = tv_to_uniform(ch["visits"], n_fiber)
        cov = unique / n_fiber
        rows.append((label, unique, n_fiber, cov, tv))
        print(f"    {label:24s}  unique = {unique:>4}/{n_fiber}  "
              f"coverage = {cov:.1%}  TV(emp, uniform) = {tv:.4f}")
    print("    (TV → 0 means uniform sampling; TV near 1 means concentrated.)")
    return rows


def island_neck_table(chain_results):
    """Print islands-visited and jump-rate per chain."""
    print("\n  ISLAND-NECK DIAGNOSTIC:")
    rows = []
    for label, ch in chain_results:
        trace = ch.get("island_trace")
        if not trace:
            continue
        valid_pairs = sum(1 for i in range(1, len(trace))
                          if trace[i] is not None and trace[i - 1] is not None)
        jumps = sum(1 for i in range(1, len(trace))
                    if trace[i] is not None and trace[i - 1] is not None
                    and trace[i] != trace[i - 1])
        jr = jumps / max(valid_pairs, 1)
        n_isl = len({i for i in trace if i is not None})
        rows.append((label, n_isl, jumps, valid_pairs, jr))
        print(f"    {label:24s}  islands visited = {n_isl}  "
              f"jump rate = {jr:.2%}  jumps = {jumps}/{valid_pairs}")
    return rows


# ─── Plot helpers ───────────────────────────────────────────────────────────

def cuts_compare_plot(chain_results, label, init_cuts, lam,
                      e_pi_cuts, burn, out_path):
    """Two-panel plot: cuts trace + post-burn-in histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["C0", "C2", "C1", "C3", "C4", "C5"]

    for (lbl, ch), c in zip(chain_results, colors):
        post = ch["cuts"][burn:]
        axes[0].plot(ch["cuts"], label=f"{lbl} (μ={post.mean():.1f})",
                     alpha=0.85, lw=1.0, color=c)
    axes[0].axvline(burn, color="k", ls=":", lw=0.7, label="burn-in")
    axes[0].axhline(init_cuts, color="gray", ls="--", lw=0.7, label="initial")
    if e_pi_cuts is not None:
        axes[0].axhline(e_pi_cuts, color="red", ls=":", lw=1.2,
                        label=f"E_π[cuts]={e_pi_cuts:.1f}")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("cut edges")
    axes[0].set_title(f"{label}, target λ={lam}")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    posts = [ch["cuts"][burn:] for _, ch in chain_results]
    lo = min(p.min() for p in posts)
    hi = max(p.max() for p in posts)
    bins = np.linspace(lo - 0.5, hi + 0.5, 30)
    for (lbl, ch), c in zip(chain_results, colors):
        post = ch["cuts"][burn:]
        axes[1].hist(post, bins=bins, alpha=0.45, label=lbl, color=c, density=True)
        axes[1].axvline(post.mean(), color=c, ls="--", lw=1.2)
    if e_pi_cuts is not None:
        axes[1].axvline(e_pi_cuts, color="red", ls=":", lw=1.5,
                        label="E_π[cuts]")
    axes[1].set_xlabel("cut edges"); axes[1].set_ylabel("density")
    axes[1].set_title("Post burn-in distributions")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def island_trace_plot(chain_results, out_path, suptitle):
    """One row per chain showing step-vs-island scatter."""
    n = len(chain_results)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2 + 1.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    colors = ["C0", "C2", "C1", "C3", "C4", "C5"]
    for (lbl, ch), c, ax in zip(chain_results, colors, axes):
        trace = ch.get("island_trace") or []
        xs, ys = [], []
        for i, isl in enumerate(trace):
            if isl is not None:
                xs.append(i); ys.append(isl)
        ax.scatter(xs, ys, s=4, c=c, alpha=0.7)
        valid_pairs = sum(1 for i in range(1, len(trace))
                          if trace[i] is not None and trace[i - 1] is not None)
        jumps = sum(1 for i in range(1, len(trace))
                    if trace[i] is not None and trace[i - 1] is not None
                    and trace[i] != trace[i - 1])
        ax.set_ylabel(lbl, fontsize=9)
        ax.set_title(f"island visits — jump rate {jumps}/{valid_pairs} = "
                     f"{jumps/max(valid_pairs,1):.1%}", fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("step")
    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
