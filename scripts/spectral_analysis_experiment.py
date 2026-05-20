"""Spectral analysis of the QP-diffusion model on Iowa merged pairs.

Generates Figure 8 of the paper (frequency response curve) plus three
companion panels that empirically validate §5 (Spectral analysis):

  Panel A. Closed-form frequency response  H(λ) = β / (β + α λ)
           overlaid with the empirical |x*_hat_k| / |x0_hat_k| ratios
           on a real Iowa merged pair.
  Panel B. Energy descent: bar of E(x0), E(x*) for several pairs.
  Panel C. x0's Fourier spectrum (where the indicator lives in L's
           eigenbasis) — answers §5.4 ("are the first eigenvectors
           noise?").
  Panel D. Cheeger bound: λ2, sweep-cut Φ(S_Fiedler), upper bound
           sqrt(2 λ2 d_max) across sampled pairs.

Usage
-----
    python scripts/spectral_analysis_experiment.py
        [--n-pairs 8] [--alpha 10] [--beta 1] [--kernel uniform]

Writes:
    plots/spectral_frequency_response.png   (Fig 8 + companion panels)
    plots/spectral_summary.json             (numerical summary for §5)

Reproducibility
---------------
Uses Iowa's k=4 ε=0.05 partition built with a deterministic
recursive_tree_part fallback (no gerrychain dependency required for
this script). Seed = 42.
"""
from __future__ import annotations

import argparse
import json
import sys
import random
from pathlib import Path
from typing import Optional

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make `recom.spectral_analysis` importable.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO / "diffusion_model"))


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight Iowa loader (no gerrychain dependency)
# ─────────────────────────────────────────────────────────────────────────────

def load_iowa(json_path: Path) -> nx.Graph:
    """Build a networkx graph from the IA_counties JSON dialect.

    Format: {nodes: [{id, TOTPOP, ALAND10, ...}, ...],
             adjacency: [[{id, shared_perim}, ...], ...]}.
    """
    with open(json_path) as f:
        data = json.load(f)
    G = nx.Graph()
    for nd in data["nodes"]:
        nid = nd["id"]
        G.add_node(nid, **{k: v for k, v in nd.items() if k != "id"})
    for u, neighbours in enumerate(data["adjacency"]):
        for nb in neighbours:
            v = nb["id"]
            if not G.has_edge(u, v):
                G.add_edge(u, v, shared_perim=float(nb.get("shared_perim",
                                                           1.0)))
    # Cache population + (TOTPOP / area) density
    densities = []
    for n in G.nodes():
        G.nodes[n]["population"] = int(G.nodes[n].get("TOTPOP", 1))
        area = max(float(G.nodes[n].get("ALAND10", 1.0)), 1.0)
        rho = G.nodes[n]["population"] / area
        G.nodes[n]["density_raw"] = rho
        densities.append(rho)
    mean_rho = float(np.mean(densities)) or 1.0
    for n in G.nodes():
        G.nodes[n]["density"] = G.nodes[n]["density_raw"] / mean_rho
    return G


def random_balanced_partition(G: nx.Graph, k: int, eps: float,
                              seed: int = 42, max_tries: int = 200
                              ) -> Optional[dict]:
    """Tiny stand-in for recursive_tree_part. Picks k random seeds,
    grows BFS regions, tries until the largest district is within
    (1+eps) times mean population.

    For Iowa (99 counties), k=4, eps=0.05 this converges in <50 tries.
    """
    rng = random.Random(seed)
    nodes = list(G.nodes())
    pop = {n: float(G.nodes[n]["population"]) for n in nodes}
    total = sum(pop.values())
    ideal = total / k
    pop_lo, pop_hi = (1 - eps) * ideal, (1 + eps) * ideal

    for trial in range(max_tries):
        seeds = rng.sample(nodes, k)
        asn = {n: -1 for n in nodes}
        # Multi-source BFS: at each round, each district claims one
        # frontier node at random (population-aware).
        district_pop = {d: 0.0 for d in range(k)}
        for d, s in enumerate(seeds):
            asn[s] = d
            district_pop[d] = pop[s]
        frontier = {d: [v for v in G.neighbors(s)] for d, s in enumerate(seeds)}
        order = list(range(k))
        # Pop-priority round-robin
        while any(d for d in nodes if asn[d] == -1):
            order.sort(key=lambda d: district_pop[d])  # smallest first
            progress = False
            for d in order:
                if district_pop[d] >= ideal:
                    continue
                # Take an unclaimed neighbor of district d.
                cand = [v for v in frontier[d] if asn[v] == -1]
                if not cand:
                    # No neighbors — try any unclaimed node adjacent to
                    # the district.
                    members = {n for n, a in asn.items() if a == d}
                    cand = [v for n in members for v in G.neighbors(n)
                            if asn[v] == -1]
                if not cand:
                    continue
                v = rng.choice(cand)
                asn[v] = d
                district_pop[d] += pop[v]
                frontier[d].extend(u for u in G.neighbors(v) if asn[u] == -1)
                progress = True
            if not progress:
                # Drop unclaimed into nearest district by population.
                for n in nodes:
                    if asn[n] == -1:
                        # neighbor districts
                        neigh_d = [asn[u] for u in G.neighbors(n)
                                   if asn[u] != -1]
                        if neigh_d:
                            d = min(neigh_d, key=lambda d: district_pop[d])
                            asn[n] = d
                            district_pop[d] += pop[n]
                break

        # Fill any remaining unclaimed (rare).
        for n in nodes:
            if asn[n] == -1:
                neigh_d = [asn[u] for u in G.neighbors(n) if asn[u] != -1]
                d = min(neigh_d, key=lambda d: district_pop[d]) if neigh_d \
                    else min(district_pop, key=district_pop.get)
                asn[n] = d
                district_pop[d] += pop[n]

        ok = all(pop_lo <= district_pop[d] <= pop_hi for d in range(k))
        # Connectivity check
        if ok:
            for d in range(k):
                sub = G.subgraph([n for n, a in asn.items() if a == d])
                if sub.number_of_nodes() == 0 or not nx.is_connected(sub):
                    ok = False
                    break
        if ok:
            return asn
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Laplacian builder for the merged pair (mirrors qp_model.build_laplacian)
# ─────────────────────────────────────────────────────────────────────────────

def kernel_value(G, u, v, kernel: str, density_sigma: float = 1.0) -> float:
    if kernel == "uniform":
        return 1.0
    if kernel == "perimeter":
        return float(G.edges[u, v].get("shared_perim", 1.0))
    if kernel == "density":
        p = float(G.edges[u, v].get("shared_perim", 1.0))
        ru = float(G.nodes[u].get("density", 0.0))
        rv = float(G.nodes[v].get("density", 0.0))
        return p * np.exp(-((ru - rv) ** 2) / (density_sigma ** 2))
    raise ValueError(f"unknown kernel {kernel!r}")


def merged_pair_laplacian(G, asn: dict, d1: int, d2: int,
                          kernel: str = "uniform"):
    """Return (L, x0, nodes) for H = G[V_{d1} ∪ V_{d2}]."""
    nodes = [n for n, d in asn.items() if d in (d1, d2)]
    nlist = nodes
    n = len(nlist)
    idx = {v: i for i, v in enumerate(nlist)}
    sub = G.subgraph(nlist)
    W = np.zeros((n, n))
    for u, v in sub.edges():
        w = kernel_value(G, u, v, kernel)
        i, j = idx[u], idx[v]
        W[i, j] = w; W[j, i] = w
    L = np.diag(W.sum(axis=1)) - W
    x0 = np.array([1.0 if asn[v] == d1 else 0.0 for v in nlist])
    return L, x0, nlist


def border_pairs(G, asn) -> list:
    seen = set()
    for u, v in G.edges():
        if asn[u] != asn[v]:
            seen.add(tuple(sorted((asn[u], asn[v]))))
    return sorted(seen)


# ─────────────────────────────────────────────────────────────────────────────
# Spectral analysis (inlined from recom/spectral_analysis.py — kept in
# sync; the module-level version is the canonical one)
# ─────────────────────────────────────────────────────────────────────────────

from recom.spectral_analysis import (             # noqa: E402
    unconstrained_qp_optimum, qp_energy, energy_descent,
    frequency_response, filter_transfer, cheeger_bound,
    eigenvector_signal_diagnostic, box_activity_fraction,
)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment driver
# ─────────────────────────────────────────────────────────────────────────────

def run(n_pairs: int = 8, alpha: float = 10.0, beta: float = 1.0,
        kernel: str = "uniform", seed: int = 42) -> dict:
    """Run the spectral analysis on n_pairs sampled Iowa merged pairs."""
    random.seed(seed); np.random.seed(seed)
    iowa_json = _REPO / "IA_counties" / "IA_counties.json"
    print(f"loading Iowa from {iowa_json} ...")
    G = load_iowa(iowa_json)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("building k=4, ε=0.05 partition ...")
    asn = random_balanced_partition(G, k=4, eps=0.05, seed=seed)
    if asn is None:
        raise RuntimeError("Could not build a balanced k=4 partition.")
    bp = border_pairs(G, asn)
    print(f"  {len(bp)} border pairs:", bp)

    # Run on every available adjacent pair, up to n_pairs
    pairs_to_use = bp[:n_pairs]
    results = []
    for d1, d2 in pairs_to_use:
        L, x0, nodes = merged_pair_laplacian(G, asn, d1, d2, kernel=kernel)
        n = L.shape[0]

        E0, E_star, x_star = energy_descent(L, x0, alpha, beta)
        eigvals, x0_hat, x_star_hat, x_star_recon = frequency_response(
            L, x0, alpha, beta)
        recon_err = float(np.max(np.abs(x_star - x_star_recon)))
        cb = cheeger_bound(L)
        diag = eigenvector_signal_diagnostic(L, x0)
        box = box_activity_fraction(L, x0, alpha, beta)

        rec = {
            "pair": (int(d1), int(d2)), "n": int(n),
            "E0": E0, "E_star": E_star,
            "energy_drop": E0 - E_star,
            "energy_drop_frac": (E0 - E_star) / E0 if E0 > 0 else 0.0,
            "lambda2": cb["lambda2"], "d_max": cb["d_max"],
            "cheeger_upper": cb["cheeger_upper"],
            "sweep_cut_fiedler": cb["sweep_cut_fiedler"],
            "recon_err": recon_err,
            "frac_in_low10_nontrivial": diag["frac_in_low10_nontrivial"],
            "trivial_mode_energy": diag["trivial_mode_energy"],
            "x0_norm_sq": diag["x0_norm_sq"],
            "box_min": box["min"], "box_max": box["max"],
            "n_interior": box["n_strictly_interior"],
            # arrays needed for plots
            "_eigvals": eigvals, "_x0_hat": x0_hat,
            "_x_star_hat": x_star_hat, "_energy": diag["energy"],
        }
        results.append(rec)
        print(f"  pair ({d1},{d2}) n={n:3d}  "
              f"E0={E0:6.2f}->E*={E_star:6.2f}  "
              f"λ2={cb['lambda2']:.4f}  sweep={cb['sweep_cut_fiedler']:.4f}  "
              f"low10={diag['frac_in_low10_nontrivial']:.0%}  "
              f"x*∈[{box['min']:.3f},{box['max']:.3f}]")

    return {"asn": asn, "results": results, "G": G,
            "alpha": alpha, "beta": beta, "kernel": kernel}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def make_figure(run_data: dict, out_path: Path) -> None:
    res = run_data["results"]
    alpha = run_data["alpha"]; beta = run_data["beta"]

    fig = plt.figure(figsize=(13.5, 9.5))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.27)

    # ── Panel A: analytic H(λ) overlaid with empirical ratios ──
    axA = fig.add_subplot(gs[0, 0])
    # smooth analytic curve
    lam_max = max(r["_eigvals"].max() for r in res)
    lam_grid = np.linspace(0.0, lam_max * 1.02, 400)
    H_grid = filter_transfer(lam_grid, alpha, beta)
    axA.plot(lam_grid, H_grid, color="black", lw=2.4,
             label=r"$H(\lambda)=\beta/(\beta+\alpha\lambda)$, analytic",
             zorder=10)
    # empirical points: |x*_hat_k| / |x0_hat_k| for modes where x0_hat_k ≠ 0
    for ridx, r in enumerate(res):
        evs = r["_eigvals"]
        x0h = r["_x0_hat"]
        xsh = r["_x_star_hat"]
        # Avoid divide-by-zero: keep modes where |x0_hat_k| > 1e-9 max
        mask = np.abs(x0h) > 1e-9 * np.max(np.abs(x0h))
        H_emp = np.where(mask, xsh / np.where(mask, x0h, 1.0), np.nan)
        axA.scatter(evs[mask], H_emp[mask], s=14, alpha=0.55,
                    label=(f"pair {r['pair']}" if ridx < 4 else None))
    axA.set_xlabel(r"eigenvalue $\lambda_k$ of $L$")
    axA.set_ylabel(r"transfer $\hat x^*_k / \hat x_{0,k}$")
    axA.set_title(f"(A) Closed-form frequency response  "
                  f"(α={alpha}, β={beta})")
    axA.set_ylim(-0.02, 1.05)
    axA.grid(alpha=0.3)
    axA.legend(fontsize=8, loc="upper right")

    # ── Panel B: energy descent ──
    axB = fig.add_subplot(gs[0, 1])
    idx = np.arange(len(res))
    E0s = [r["E0"] for r in res]
    Es = [r["E_star"] for r in res]
    width = 0.38
    axB.bar(idx - width/2, E0s, width, label=r"$E(x_0)$",
            color="#7aa6c2", edgecolor="black", lw=0.6)
    axB.bar(idx + width/2, Es, width, label=r"$E(x^*)$",
            color="#cc7766", edgecolor="black", lw=0.6)
    for i, (e0, es) in enumerate(zip(E0s, Es)):
        if e0 > 0:
            frac = (e0 - es) / e0
            axB.text(i, max(e0, es) * 1.02, f"−{frac:.0%}",
                     ha="center", fontsize=8)
    axB.set_xticks(idx)
    axB.set_xticklabels([f"{r['pair'][0]},{r['pair'][1]}" for r in res],
                        fontsize=8)
    axB.set_xlabel("merged pair (d1,d2)")
    axB.set_ylabel(r"energy  $\alpha x^\top L x + \beta\|x{-}x_0\|^2$")
    axB.set_title("(B) Energy descent  (Lemma 6.1)")
    axB.legend(fontsize=9)
    axB.grid(alpha=0.3, axis="y")

    # ── Panel C: x0's spectrum — "are the first eigenvectors noise?" ──
    axC = fig.add_subplot(gs[1, 0])
    # plot cumulative spectrum for each pair, drop trivial λ=0
    for ridx, r in enumerate(res):
        E = r["_energy"][1:]
        if E.sum() == 0:
            continue
        cum = np.cumsum(E) / E.sum()
        k_axis = (np.arange(1, len(E) + 1)) / len(E)
        axC.plot(k_axis, cum, alpha=0.7,
                 label=(f"pair {r['pair']}" if ridx < 4 else None))
    axC.axvline(0.10, color="black", ls="--", lw=0.8,
                label="lowest 10% of modes")
    axC.set_xlabel("fraction of (nontrivial) modes,"
                   r" lowest $\lambda$ first")
    axC.set_ylabel(r"cumulative $|\hat x_{0,k}|^2$ (normalised)")
    axC.set_title("(C) Where the indicator lives "
                  "(§5.4)")
    axC.set_xlim(0, 1); axC.set_ylim(0, 1.02)
    axC.grid(alpha=0.3)
    axC.legend(fontsize=8, loc="lower right")

    # ── Panel D: Cheeger bound across pairs ──
    axD = fig.add_subplot(gs[1, 1])
    sweep = np.array([r["sweep_cut_fiedler"] for r in res])
    upper = np.array([r["cheeger_upper"] for r in res])
    lam2 = np.array([r["lambda2"] for r in res])
    half_lam2 = np.maximum(lam2 / 2.0, 1e-6)   # log axis, clip floor
    idx = np.arange(len(res))
    axD.scatter(idx, half_lam2, marker="v", s=70, color="#3a6ea5",
                label=r"$\lambda_2/2$  (lower bound)", zorder=3)
    axD.scatter(idx, sweep, marker="o", s=70, color="#cc7766",
                label=r"$\Phi(S_{\mathrm{Fiedler}})$  (sweep)", zorder=4)
    # Only plot upper bound where it's finite (disconnected weighted
    # subgraphs give NaN under the density kernel).
    finite_upper = np.isfinite(upper)
    axD.scatter(idx[finite_upper], upper[finite_upper], marker="^", s=70,
                color="#444444",
                label=r"$\sqrt{2\lambda_2 d_{\max}}$  (upper)", zorder=3)
    for i in idx:
        top = upper[i] if np.isfinite(upper[i]) else sweep[i]
        axD.plot([i, i], [half_lam2[i], top], color="gray",
                 alpha=0.45, lw=1.0, zorder=1)
    axD.set_xticks(idx)
    axD.set_xticklabels([f"{r['pair'][0]},{r['pair'][1]}" for r in res],
                        fontsize=8)
    axD.set_xlabel("merged pair (d1,d2)")
    axD.set_ylabel("cut value / bound")
    axD.set_title("(D) Cheeger inequality on the rounded cut "
                  "(§5.3)")
    axD.legend(fontsize=8, loc="upper left")
    axD.set_yscale("log")
    axD.grid(alpha=0.3, which="both")

    fig.suptitle(
        f"Spectral analysis of the QP-diffusion  "
        f"(Iowa, k=4, ε=0.05, kernel={run_data['kernel']})",
        fontsize=13, y=0.995)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def write_summary(run_data: dict, out_path: Path) -> None:
    """JSON dump of every numerical fact behind the figure."""
    summary = {
        "alpha": run_data["alpha"],
        "beta": run_data["beta"],
        "kernel": run_data["kernel"],
        "n_pairs": len(run_data["results"]),
        "pairs": [],
    }
    for r in run_data["results"]:
        summary["pairs"].append({
            "pair": r["pair"], "n": r["n"],
            "E0": r["E0"], "E_star": r["E_star"],
            "energy_drop_frac": r["energy_drop_frac"],
            "lambda2": r["lambda2"], "d_max": r["d_max"],
            "cheeger_upper": r["cheeger_upper"],
            "sweep_cut_fiedler": r["sweep_cut_fiedler"],
            "recon_err": r["recon_err"],
            "frac_in_low10_nontrivial": r["frac_in_low10_nontrivial"],
            "x0_norm_sq": r["x0_norm_sq"],
            "box_min": r["box_min"], "box_max": r["box_max"],
            "n_interior": r["n_interior"],
        })
    # roll-ups
    arr = lambda key: np.array([r[key] for r in run_data["results"]])
    summary["aggregate"] = {
        "mean_energy_drop_frac": float(arr("energy_drop_frac").mean()),
        "min_energy_drop_frac": float(arr("energy_drop_frac").min()),
        "max_recon_err": float(arr("recon_err").max()),
        "mean_frac_low10": float(arr("frac_in_low10_nontrivial").mean()),
        "mean_lambda2": float(arr("lambda2").mean()),
        "mean_sweep_cut": float(arr("sweep_cut_fiedler").mean()),
        "mean_cheeger_upper": float(arr("cheeger_upper").mean()),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--kernel", default="uniform",
                    choices=["uniform", "perimeter", "density"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-plot", default=None)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    out_plot = Path(args.out_plot) if args.out_plot else \
        _REPO / "plots" / "spectral_frequency_response.png"
    out_json = Path(args.out_json) if args.out_json else \
        _REPO / "plots" / "spectral_summary.json"

    run_data = run(n_pairs=args.n_pairs, alpha=args.alpha,
                   beta=args.beta, kernel=args.kernel, seed=args.seed)
    make_figure(run_data, out_plot)
    write_summary(run_data, out_json)

    # Headline numbers for the writeup
    agg = json.load(open(out_json))["aggregate"]
    print("\n── HEADLINE NUMBERS FOR §5 ──")
    print(f"  Energy descent (mean fractional drop): "
          f"{agg['mean_energy_drop_frac']:.1%}")
    print(f"  Spectral ≡ matrix-inverse (max abs err): "
          f"{agg['max_recon_err']:.2e}")
    print(f"  Indicator energy in lowest 10% of nontrivial modes (mean): "
          f"{agg['mean_frac_low10']:.1%}")
    print(f"  Cheeger bound:  λ2={agg['mean_lambda2']:.4f},  "
          f"sweep={agg['mean_sweep_cut']:.4f},  "
          f"upper={agg['mean_cheeger_upper']:.4f}")


if __name__ == "__main__":
    main()
