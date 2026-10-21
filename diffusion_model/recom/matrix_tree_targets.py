"""Matrix-Tree quasi-target expectations on the enumerated 5x5 fiber.

For each of the 4006 exact-balance partitions, computes the
spanning-tree-count weight prod_i tau(V_i) (Matrix-Tree theorem on
each district's induced subgraph) and reports E[cuts] under

  * the tree-weighted measure  w(P) = prod_i tau(V_i),
  * the tree-weighted-and-tempered measure w(P) * exp(-lam*cuts(P)),
  * the nominal tempered measure exp(-lam*cuts(P)),

the three reference values quoted in the ground-truth calibration.

Usage: python3 recom/matrix_tree_targets.py [lam]
"""

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

LAM = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10
HERE = Path(__file__).resolve().parent


def n_spanning_trees(g):
    if g.number_of_nodes() == 1:
        return 1.0
    L = nx.laplacian_matrix(g).todense().astype(float)
    return float(round(np.linalg.det(L[1:, 1:])))


def main():
    G = nx.grid_graph([5, 5])
    with open(HERE / "fiber_5x5" / "fiber_5x5_partitions.json") as f:
        data = json.load(f)

    cuts, logw = [], []
    for pid, parts in data.items():
        asn = {}
        for d_id, part in enumerate(parts):
            for s in part:
                asn[tuple(s)] = d_id
        cuts.append(sum(1 for u, v in G.edges() if asn[u] != asn[v]))
        w = 1.0
        for part in parts:
            w *= n_spanning_trees(G.subgraph([tuple(s) for s in part]))
        logw.append(np.log(w))

    cuts = np.array(cuts, dtype=float)
    logw = np.array(logw)

    def expect(logweights):
        lw = logweights - logweights.max()
        p = np.exp(lw)
        p /= p.sum()
        return float((p * cuts).sum())

    e_tree = expect(logw)
    e_tree_temp = expect(logw - LAM * cuts)
    e_temp = expect(-LAM * cuts)
    print(f"fiber size: {len(cuts)}")
    print(f"E[cuts], tree-weighted measure:              {e_tree:.4f}")
    print(f"E[cuts], tree-weighted + tempered (lam={LAM}): {e_tree_temp:.4f}")
    print(f"E[cuts], nominal tempered (lam={LAM}):         {e_temp:.4f}")

    np.savez(HERE / "iowa_hybrid_test_out" / "matrix_tree_targets.npz",
             cuts=cuts, log_tree_weight=logw, lam=LAM,
             e_tree=e_tree, e_tree_tempered=e_tree_temp,
             e_tempered=e_temp)
    print("Saved matrix_tree_targets.npz")


if __name__ == "__main__":
    main()
