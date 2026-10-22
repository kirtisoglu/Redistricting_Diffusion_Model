"""The scaling experiment: how the exact tree move's acceptance falls with
the size of the merged pair as the target moves away from the measure the
tree proposes, and how the local move's contribution grows with it.

MECHANISM UNDER TEST.  The forest move is an independence sampler on the
merged pair: it proposes a balanced split from the untempered
spanning-tree measure, whatever the target. Its Hastings ratio is

    [pi(P')/pi(P)] * [tau(V_i)tau(V_j) / (tau(A)tau(B))] * [S ratio] * [bp ratio],

and only the first two terms matter (the prototype of 2026-10-22 put the
S and bp terms within a factor of two of one). Under pi_w the second term
is the tree-count penalty, negative on average and growing with the
boundary length; under the tree-weighted target it cancels, and the first
term carries the whole mismatch. Either way the acceptance at equilibrium
is governed by how far the target's conditional on the pair sits from the
untempered tree measure, and that distance grows with the pair size at
fixed temperature. The local move's ratio involves a one-unit boundary
change and does not. Prediction: forest acceptance decays with N for
every target except one close to the untempered tree measure (the
tree-weighted target at a mild temperature, which is the Iowa control),
fastest for strongly tempered targets, which is the compactness regime
practitioners ask for; the hybrid / forest ESS ratio grows with N in
the same order.

DESIGN.  N x N grids, k = 2, so the merged pair is the whole graph and
there is no pair-selection noise. Unit populations, tolerance EPS (fixed
in relative terms, so single flips are feasible at every N). Uniform
kernel. Targets as "family:lam" pairs, default
    pi:1  pi:2  tree:0.1  tree:1
For each N and target: forest move alone and the hybrid at P_LOCAL, from
a vertical-split start. Reported per chain: P(tree has a balanced edge),
forest acceptance after burn-in, mean and standard error of the log
tree-count ratio and of the log target ratio over feasible forest
proposals, mean cut, quarter means (the equilibrium check), ESS, ESS per
second and the hybrid / forest ratio. The forest move's acceptance and
the two terms are read from the hybrid's trace, where every forest
proposal is made from a target-distributed state even when the forest
move alone would not equilibrate within the budget.

COST.  A forest step costs about |boundary| x N^2 Python operations for
the crossing-edge sums, so a step grows like N^3: on the cloud machine
about 1.2 ms at N = 6, 4 ms at N = 10 and 10 ms at N = 14 for the hybrid
at p = 0.3. With the defaults (50,000 steps, doubled for N >= 16) expect
about 2 minutes per chain at N = 6, 10 at N = 14, 45 at N = 20; four
targets times two chains times six sizes is about 6 hours in total, or
2 hours if the three largest sizes run in parallel.

Writes: iowa_hybrid_test_out/forest_scaling_seed{SEED}.npz
        Papers/fig/forest_scaling.{pdf,png}   (with --fig; --fig alone
        redraws from the stored artifact without rerunning)
Usage:  python3 recom/forest_scaling.py [n_steps] [seed] [N list] [targets] [p_local] [--fig]
Default: 50000 42 6,8,10,12,16,20 pi:1,pi:2,tree:0.1,tree:1 0.3
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
N_LIST = [int(x) for x in argv[2].split(",")] if len(argv) > 2 else [6, 8, 10, 12, 16, 20]
TARGETS = ([(t.split(":")[0], float(t.split(":")[1])) for t in argv[3].split(",")]
           if len(argv) > 3 else [("pi", 1.0), ("pi", 2.0), ("tree", 0.1), ("tree", 1.0)])
P_LOCAL = float(argv[4]) if len(argv) > 4 else 0.3
EPS = 0.10
TAU = 0.1
BURN = 0.1
OUT = _HERE / "iowa_hybrid_test_out" / f"forest_scaling_seed{SEED}.npz"
FIG_OUT = _REPO / "Papers" / "fig" / "forest_scaling"
FOREST_KEYS = ("wcut", "ncut", "kind", "accepted", "feasible", "log_tau_ratio",
               "log_pi_ratio", "flips")


def grid_problem(N):
    G = nx.grid_graph([N, N])
    for v in G.nodes():
        G.nodes[v]["population"] = 1
        G.nodes[v]["density"] = 0.0
    init = {v: (0 if v[0] < N // 2 else 1) for v in G.nodes()}
    return G, init, N * N / 2.0


def steps_for(N):
    return N_STEPS * (2 if N >= 16 else 1)


def tag_of(N, fam, lam, cname):
    return f"N{N}_{fam}{lam:g}_{cname}"


def forest_stats(res, burn_frac):
    n = len(res["wcut"])
    b = int(n * burn_frac)
    k = res["kind"][b:]
    f = res["feasible"][b:]
    out = dict(p_feasible=float(f[k == 0].mean()) if (k == 0).any() else np.nan,
               ncut_mean=float(res["ncut"][b:].mean()))
    for name in ("log_tau_ratio", "log_pi_ratio"):
        x = res[name][b:][k == 0]
        x = x[~np.isnan(x)]
        out[name] = float(x.mean()) if len(x) else np.nan
        out[name + "_se"] = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan
    return out


def run():
    store = {"n_steps": N_STEPS, "seed": SEED, "eps": EPS, "p_local": P_LOCAL, "tau": TAU,
             "N_list": np.array(N_LIST),
             "targets": np.array([f"{f}:{l:g}" for f, l in TARGETS])}
    tl = " ".join(f"{f}:{l:g}" for f, l in TARGETS)
    print(f"grids k=2, eps={EPS}, p_local={P_LOCAL}, tau={TAU}, seed {SEED}, "
          f"{N_STEPS} steps (x2 for N >= 16), targets {tl}\n")
    print(f"{'N':>3} {'target':>8} {'chain':>6} {'P(feas)':>7} {'acc f':>6} {'acc l':>6} "
          f"{'log tau-ratio':>14} {'log pi-ratio':>14} {'cut':>6} {'quarter means':>28} "
          f"{'ESS':>6} {'wall s':>7} {'ESS/s':>7} {'ratio':>6}")
    for N in N_LIST:
        G, init, ideal = grid_problem(N)
        steps = steps_for(N)
        for fam, lam in TARGETS:
            tgt = make_target(fam, G, kernel_uniform, lam)
            base = None
            for cname, pl in (("forest", 0.0), ("hybrid", P_LOCAL)):
                res = run_forest_hybrid(tgt, init, ideal, EPS, steps, pl, TAU, 1.0, SEED,
                                        label=f"N={N} {fam}:{lam:g} {cname}")
                s = chain_summary(res, BURN)
                fs = forest_stats(res, BURN)
                if base is None:
                    base = s["ess_s"]
                qm = " ".join(f"{v:6.2f}" for v in s["quarters"])
                print(f"{N:>3} {fam + ':' + format(lam, 'g'):>8} {cname:>6} {fs['p_feasible']:7.3f} "
                      f"{s['acc_global']:6.3f} {s['acc_local']:6.3f} "
                      f"{fs['log_tau_ratio']:+7.2f} ± {fs['log_tau_ratio_se']:4.2f} "
                      f"{fs['log_pi_ratio']:+7.2f} ± {fs['log_pi_ratio_se']:4.2f} "
                      f"{fs['ncut_mean']:6.1f} {qm:>28} {s['ess']:6.0f} {s['wall']:7.1f} "
                      f"{s['ess_s']:7.2f} {s['ess_s'] / base:6.2f}", flush=True)
                tag = tag_of(N, fam, lam, cname)
                for key in FOREST_KEYS:
                    store[f"{tag}_{key}"] = res[key]
                store[f"{tag}_wall"] = res["wall"]
                store[f"{tag}_steps"] = steps
        print(flush=True)
    np.savez_compressed(OUT, **store)          # uncompressed is 107 MiB, over GitHub's 100 MiB limit
    print(f"Saved {OUT.name}")


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load(OUT, allow_pickle=True)
    Ns = [int(n) for n in z["N_list"]]
    targets = [(t.split(":")[0], float(t.split(":")[1])) for t in z["targets"]]
    style = {("pi", 1.0): ("#D55E00", "-", "o"), ("pi", 2.0): ("#CC79A7", "-", "s"),
             ("tree", 0.1): ("#0072B2", "--", "^"), ("tree", 1.0): ("#009E73", "--", "D")}
    fallback = ["#E69F00", "#56B4E9", "#F0E442", "#000000"]

    def label(fam, lam):
        return (r"$\pi_w$" if fam == "pi" else "tree-weighted") + rf", $\lambda={lam:g}$"

    def stat(N, fam, lam, cname):
        tag = tag_of(N, fam, lam, cname)
        res = {k: z[f"{tag}_{k}"] for k in FOREST_KEYS}
        res["wall"] = float(z[f"{tag}_wall"])
        res["fast"] = np.zeros(len(res["wcut"]), dtype=bool)
        return chain_summary(res, BURN), forest_stats(res, BURN)

    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.3))
    for i, (fam, lam) in enumerate(targets):
        col, ls, mk = style.get((fam, lam), (fallback[i % 4], "-", "o"))
        acc, tau_t, pi_t, ratio = [], [], [], []
        for N in Ns:
            sh, fh = stat(N, fam, lam, "hybrid")
            sf, _ = stat(N, fam, lam, "forest")
            acc.append(sh["acc_global"])
            tau_t.append(-fh["log_tau_ratio"])
            pi_t.append(-fh["log_pi_ratio"])
            ratio.append(sh["ess_s"] / sf["ess_s"])
        ax[0].plot(Ns, acc, ls, marker=mk, color=col, label=label(fam, lam))
        ax[1].plot(Ns, np.array(tau_t) + np.array(pi_t), ls, marker=mk, color=col,
                   label=label(fam, lam))
        ax[2].plot(Ns, ratio, ls, marker=mk, color=col, label=label(fam, lam))
    ax[0].set_yscale("log")
    ax[0].set_xlabel("grid side $N$ (the merged pair has $N^2$ units)")
    ax[0].set_ylabel("forest-move acceptance at equilibrium")
    ax[0].set_title("(a) The exact tree move freezes as the pair grows")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].set_xlabel("grid side $N$")
    ax[1].set_ylabel(r"$-\,\mathbb{E}[\log(\pi\ \mathrm{ratio}) + \log(\tau\ \mathrm{ratio})]$")
    ax[1].set_title("(b) The mismatch with the tree measure, per proposal")
    ax[1].legend(frameon=False, fontsize=8)
    ax[2].axhline(1.5, color="#999999", lw=1, ls=":")
    ax[2].axhline(1.0, color="#999999", lw=1, ls="-")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("grid side $N$")
    ax[2].set_ylabel("ESS/s, hybrid / forest move alone")
    ax[2].set_title("(c) What the local move adds")
    ax[2].legend(frameon=False, fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(rf"$k=2$ grids, tolerance {EPS:g}, uniform kernel, hybrid at $p_{{loc}}={P_LOCAL:g}$, "
                 rf"$\tau={TAU:g}$", y=1.02)
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
