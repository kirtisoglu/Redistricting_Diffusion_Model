"""Regenerate the multikernel Iowa map figure from saved final plans.

Loads the final Chain C plan of the uniform, perimeter, and density
kernel runs (target_matched_iowa*_seed{SEED}.npz), draws them on the
county shapefile, annotates unweighted cut count and geographic
boundary length, and prints pairwise county disagreements under the
best district relabeling (numbers quoted in the paper).

Usage: python3 recom/multikernel_figure.py [seed] [shapefile_path]
"""

import json
import sys
from itertools import permutations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

HERE = Path(__file__).resolve().parent
_REPO = HERE.parent.parent
OUT = HERE / "iowa_hybrid_test_out"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
SHP = Path(sys.argv[2]) if len(sys.argv) > 2 else \
    Path.home() / "Downloads" / "IA_counties" / "IA_counties.shp"
K = 4
COLORS = ["#c78b8b", "#8ba7c7", "#8bc78f", "#e0b774"]


def load_final(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    nodes = [int(s) for s in d["C_final_nodes"]]
    dists = [int(x) for x in d["C_final_dists"]]
    return dict(zip(nodes, dists))


def relabel_to(ref, plan):
    """Relabel plan's districts to best match ref; returns new dict."""
    la = sorted(set(ref.values()))
    lb = sorted(set(plan.values()))
    ia = {d: i for i, d in enumerate(la)}
    ib = {d: i for i, d in enumerate(lb)}
    nodes = list(ref.keys())
    av = [ia[ref[n]] for n in nodes]
    bv = [ib[plan[n]] for n in nodes]
    best_perm, best_d = None, None
    for perm in permutations(range(K)):
        d = sum(1 for x, y in zip(av, bv) if x != perm[y])
        if best_d is None or d < best_d:
            best_d, best_perm = d, perm
    return {n: best_perm[ib[plan[n]]] for n in nodes}, best_d


def main():
    with open(_REPO / "IA_counties" / "IA_counties.json") as f:
        g = json.load(f)
    geoid_of = {n["id"]: n["GEOID10"] for n in g["nodes"]}
    edges = [(u, e["id"], float(e.get("shared_perim", 1.0)))
             for u, adj in enumerate(g["adjacency"]) for e in adj
             if u < e["id"]]

    plans = {}
    for kname, path in [
        ("uniform", OUT / f"target_matched_iowa_seed{SEED}.npz"),
        ("perimeter", OUT / f"target_matched_iowa_perimeter_seed{SEED}.npz"),
        ("density", OUT / f"target_matched_iowa_density_seed{SEED}.npz"),
    ]:
        plans[kname] = load_final(path)

    ref = {n: plans["uniform"][n] - min(plans["uniform"].values())
           for n in plans["uniform"]}
    aligned = {"uniform": ref}
    print("Pairwise county disagreements (best relabeling):")
    disagreements = {}
    for kname in ("perimeter", "density"):
        aligned[kname], d = relabel_to(ref, plans[kname])
        disagreements[("uniform", kname)] = d
        print(f"  uniform vs {kname}: {d}")
    _, d_pd = relabel_to(aligned["perimeter"], plans["density"])
    disagreements[("perimeter", "density")] = d_pd
    print(f"  perimeter vs density: {d_pd}")

    gdf = gpd.read_file(SHP)
    gdf["GEOID10"] = gdf["GEOID10"].astype(str)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    for ax, kname in zip(axes, ("uniform", "perimeter", "density")):
        asn = aligned[kname]
        cuts = sum(1 for u, v, _ in edges if asn[u] != asn[v])
        blen = sum(sp for u, v, sp in edges if asn[u] != asn[v])
        colmap = {str(geoid_of[n]): COLORS[asn[n]] for n in asn}
        gdf["__c"] = gdf["GEOID10"].map(colmap)
        gdf.plot(ax=ax, color=gdf["__c"], edgecolor="white", linewidth=0.6)
        ax.set_title(f"{kname} kernel ({cuts} cut edges, "
                     f"boundary {blen:.1f})", fontsize=11)
        ax.set_axis_off()
    plt.tight_layout()
    out_png = OUT / f"multikernel_iowa_seed{SEED}.png"
    plt.savefig(out_png, dpi=170, bbox_inches="tight")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
