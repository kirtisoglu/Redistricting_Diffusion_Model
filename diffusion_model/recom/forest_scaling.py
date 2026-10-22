"""The scaling experiment: how the exact tree move's acceptance under a
non-tree target falls with the size of the merged pair, and how the local
move's contribution grows with it.

MECHANISM UNDER TEST.  Metropolized to pi_w, the forest move's Hastings
ratio carries tau(V_i) tau(V_j) / (tau(A) tau(B)), the ratio of
spanning-tree counts of the current split to the proposed one. A uniform
spanning tree proposes splits with many spanning trees, so at pi_w's
equilibrium this ratio is typically far below one, and by Procaccia and
Tucker-Foltz its logarithm scales with the difference in boundary length
between the two splits. Under the tree-weighted target the ratio cancels.
So the prediction is: forest acceptance under pi_w decays with the
boundary length of the pair, hence with the grid size, while under the
tree-weighted target it stays flat; and the local move, whose ratio
involves only a one-unit boundary change, becomes the component that
moves.

DESIGN.  N x N grids, k = 2, so the merged pair is the whole graph and
there is no pair-selection noise. Unit populations, tolerance EPS (fixed
in relative terms, so single flips are feasible at every N). Uniform
kernel, temperature LAM. For each N and each target family (pi, tree):
forest move alone and the hybrid at P_LOCAL, from a vertical-split start.
Reported per chain: P(tree has a balanced edge), forest acceptance after
burn-in, mean and standard error of the log tree-count ratio over
feasible proposals, mean cut, quarter means (the equilibrium check), ESS,
ESS per second and the hybrid / forest ratio.

The hybrid is what equilibrates at large N under pi_w (the forest move
alone may not within the budget); the forest move's acceptance is read
from the hybrid's trace, where every forest proposal is made from a
pi_w-distributed state. The forest-only row is kept for the ESS ratio.

Writes: iowa_hybrid_test_out/forest_scaling_lam{LAM}_seed{SEED}.npz
        Papers/fig/forest_scaling_lam{LAM}.{pdf,png}   (with --fig)
Usage:  python3 recom/forest_scaling.py [n_steps] [seed] [lam] [N list] [p_local] [--fig]
Default: 50000 42 1.0 6,8,10,12,16,20 0.3
         The tree-target runs use n_steps // 4.
"""

import sys
from pathlib import Path

import numpy as np
import networkx as nx

_HERE = Path(__file__).resolve().parent
_DIFF = _HERE.parent
_REPO = _DIFF.parent
if str(_DIFF) not in sys.path:
    sys.path.insert(0, str(_DIFF))

from recom.qp_model import kernel_uniform
from recom.forest_recom import run_forest_hybrid, make_target
from recom.kill_stats import chain_summary

argv = [a for a in sys.argv[1:] if not a.startswith("--")]
FIG = "--fig" in sys.argv
N_STEPS = int(argv[0]) if len(argv) > 0 else 50_000
SEED = int(argv[1]) if len(argv) > 1 else 42
LAM = float(argv[2]) if len(argv) > 2 else 1.0
N_LIST = [int(x) for x in argv[3].split(",")] if len(argv) > 3 else [6, 8, 10, 12, 16, 20]
P_LOCAL = float(argv[4]) if len(argv) > 4 else 0.3
EPS = 0.10
TAU = 0.1
BURN = 0.1
OUT = _HERE / "iowa_hybrid_test_out" / f"forest_scaling_lam{LAM:g}_seed{SEED}.npz"
FIG_OUT = _REPO / "Papers" / "fig" / f"forest_scaling_lam{LAM:g}"


def grid_problem(N):
    G = nx.grid_graph([N, N])
    for v in G.nodes():
        G.nodes[v]["population"] = 1
        G.nodes[v]["density"] = 0.0
    init = {v: (0 if v[0] < N // 2 else 1) for v in G.nodes()}
    return G, init, N * N / 2.0


def forest_stats(res, burn_frac):
    n = len(res["wcut"])
    b = int(n * burn_frac)
    k = res["kind"][b:]
    f = res["feasible"][b:]
    ltr = res["log_tau_ratio"][b:][k == 0]
    ltr = ltr[~np.isnan(ltr)]
    return dict(p_feasible=float(f[k == 0].mean()) if (k == 0).any() else np.nan,
                ltr_mean=float(ltr.mean()) if len(ltr) else np.nan,
                ltr_se=float(ltr.std(ddof=1) / np.sqrt(len(ltr))) if len(ltr) > 1 else np.nan,
                ncut_mean=float(res["ncut"][b:].mean()))


def run():
    store = {"n_steps": N_STEPS, "seed": SEED, "lam": LAM, "eps": EPS, "p_local": P_LOCAL,
             "tau": TAU, "N_list": np.array(N_LIST)}
    print(f"grids k=2, eps={EPS}, lam={LAM}, p_local={P_LOCAL}, tau={TAU}, seed {SEED}, "
          f"{N_STEPS} steps under pi and {N_STEPS // 4} under tree\n")
    print(f"{'N':>3} {'target':>6} {'chain':>7} {'P(feas)':>7} {'acc f':>6} {'acc l':>6} "
          f"{'log tau-ratio':>14} {'cut':>6} {'quarter means':>28} {'ESS':>6} {'wall s':>7} "
          f"{'ESS/s':>7} {'ratio':>6}")
    for N in N_LIST:
        G, init, ideal = grid_problem(N)
        for target in ("pi", "tree"):
            tgt = make_target(target, G, kernel_uniform, LAM)
            steps = N_STEPS if target == "pi" else max(N_STEPS // 4, 5000)
            base = None
            for cname, pl in (("forest", 0.0), ("hybrid", P_LOCAL)):
                res = run_forest_hybrid(tgt, init, ideal, EPS, steps, pl, TAU, 1.0, SEED,
                                        label=f"N={N} {target} {cname}")
                s = chain_summary(res, BURN)
                fs = forest_stats(res, BURN)
                if base is None:
                    base = s["ess_s"]
                qm = " ".join(f"{v:6.2f}" for v in s["quarters"])
                print(f"{N:>3} {target:>6} {cname:>7} {fs['p_feasible']:7.3f} {s['acc_global']:6.3f} "
                      f"{s['acc_local']:6.3f} {fs['ltr_mean']:+8.2f} ± {fs['ltr_se']:4.2f} "
                      f"{fs['ncut_mean']:6.1f} {qm:>28} {s['ess']:6.0f} {s['wall']:7.1f} "
                      f"{s['ess_s']:7.2f} {s['ess_s'] / base:6.2f}", flush=True)
                tag = f"N{N}_{target}_{cname}"
                for key in ("wcut", "ncut", "kind", "accepted", "feasible", "log_tau_ratio", "flips"):
                    store[f"{tag}_{key}"] = res[key]
                store[f"{tag}_wall"] = res["wall"]
                store[f"{tag}_steps"] = steps
        print(flush=True)
    np.savez(OUT, **store)
    print(f"Saved {OUT.name}")


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load(OUT, allow_pickle=True)
    Ns = list(z["N_list"])
    col = {"pi": "#D55E00", "tree": "#0072B2"}
    lab = {"pi": r"target $\pi_w$", "tree": "tree-weighted target"}

    def stat(N, target, cname):
        tag = f"N{N}_{target}_{cname}"
        res = {k: z[f"{tag}_{k}"] for k in ("wcut", "ncut", "kind", "accepted", "feasible",
                                             "log_tau_ratio", "flips")}
        res["wall"] = float(z[f"{tag}_wall"])
        res["fast"] = np.zeros(len(res["wcut"]), dtype=bool)
        return chain_summary(res, BURN), forest_stats(res, BURN)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for target in ("pi", "tree"):
        acc, ltr, ltr_se, ratio, cut = [], [], [], [], []
        for N in Ns:
            sh, fh = stat(N, target, "hybrid")
            sf, _ = stat(N, target, "forest")
            acc.append(sh["acc_global"])
            ltr.append(-fh["ltr_mean"])
            ltr_se.append(fh["ltr_se"])
            ratio.append(sh["ess_s"] / sf["ess_s"])
            cut.append(fh["ncut_mean"])
        ls = "-" if target == "pi" else "--"
        ax[0].plot(Ns, acc, ls, marker="o", color=col[target], label=lab[target])
        if target == "pi":
            ax[1].errorbar(Ns, ltr, yerr=ltr_se, fmt="o-", color=col[target], label=lab[target])
            ax2 = ax[1].twinx()
            ax2.plot(Ns, cut, ":", marker="s", color="#999999", label="mean cut length under $\\pi_w$")
            ax2.set_ylabel("mean cut length")
            ax2.legend(loc="lower right", frameon=False)
        ax[2].plot(Ns, ratio, ls, marker="o", color=col[target], label=lab[target])
    ax[0].set_yscale("log")
    ax[0].set_xlabel("grid side $N$ (merged pair has $N^2$ units)")
    ax[0].set_ylabel("forest-move acceptance at equilibrium")
    ax[0].set_title("(a) Acceptance of the exact tree move")
    ax[0].legend(frameon=False)
    ax[1].set_xlabel("grid side $N$")
    ax[1].set_ylabel(r"$-\,\mathbb{E}\,\log[\tau(V_i)\tau(V_j)/\tau(A)\tau(B)]$")
    ax[1].set_title("(b) The tree-count penalty")
    ax[2].axhline(1.5, color="#999999", lw=1, ls=":")
    ax[2].axhline(1.0, color="#999999", lw=1, ls="-")
    ax[2].set_xlabel("grid side $N$")
    ax[2].set_ylabel("ESS/s hybrid / forest alone")
    ax[2].set_title("(c) What the local move adds")
    ax[2].legend(frameon=False)
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(f"Exact tree move against grid size, $k=2$, $\\lambda={LAM:g}$, "
                 f"tolerance {EPS:g}, uniform kernel", y=1.02)
    fig.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(FIG_OUT) + ".pdf", bbox_inches="tight")
    fig.savefig(str(FIG_OUT) + ".png", dpi=200, bbox_inches="tight")
    print(f"Saved {FIG_OUT}.pdf and .png")


if __name__ == "__main__":
    if not FIG or not OUT.exists():
        run()
    if FIG:
        figure()
