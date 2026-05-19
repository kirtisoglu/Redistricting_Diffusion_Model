"""Degree statistics of the 5x5 fiber graph."""

from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev

import matplotlib.pyplot as plt
import networkx as nx


def main():
    here = Path(__file__).parent
    F = nx.read_edgelist(here / "fiber_5x5_edges.txt", nodetype=int)
    F.add_nodes_from(range(4006))

    degrees = [d for _, d in F.degree()]
    dist = Counter(degrees)

    print(f"Nodes: {F.number_of_nodes()}, edges: {F.number_of_edges()}")
    print(f"Degree min={min(degrees)}, max={max(degrees)}, "
          f"mean={mean(degrees):.3f}, median={median(degrees)}, "
          f"stdev={pstdev(degrees):.3f}")
    print("\nDegree distribution:")
    print(f"  {'deg':>3}  {'count':>6}  {'%':>6}")
    for d in sorted(dist):
        c = dist[d]
        pct = 100 * c / len(degrees)
        bar = "#" * int(pct)
        print(f"  {d:>3}  {c:>6}  {pct:>5.2f}  {bar}")

    print("\nMost-extreme partitions:")
    by_deg = sorted(F.degree(), key=lambda x: x[1])
    print(f"  lowest degree: {by_deg[:5]}")
    print(f"  highest degree: {by_deg[-5:]}")

    plt.figure(figsize=(8, 5))
    ks = sorted(dist)
    counts = [dist[k] for k in ks]
    plt.bar(ks, counts, color="steelblue", edgecolor="black")
    plt.xlabel("degree (number of flip-swap neighbors)")
    plt.ylabel("number of partitions")
    plt.title(
        f"Degree distribution of 5x5 fiber graph\n"
        f"(4006 partitions, mean={mean(degrees):.2f}, "
        f"min={min(degrees)}, max={max(degrees)})"
    )
    plt.xticks(ks)
    for k, c in zip(ks, counts):
        plt.text(k, c, str(c), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    out = here / "fiber_5x5_degree_dist.png"
    plt.savefig(out, dpi=150)
    print(f"\nDegree-distribution plot saved to {out}")


if __name__ == "__main__":
    main()
