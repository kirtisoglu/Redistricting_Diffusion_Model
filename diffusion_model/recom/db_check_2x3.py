"""Machine-precision detailed-balance check of Chain E on the 2x3 grid.

Enumerates all labeled 2-district plans of the 2x3 grid (18 valid
states: 3+3 split, both sides connected), assembles Chain E's full
transition kernel analytically from the implementation's own QP solve
and clipped-Bernoulli density, and reports the maximum relative
violation of pi(P) K(P,P') = pi(P') K(P',P).

Usage: python3 recom/db_check_2x3.py
"""

import sys
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
_DIFF = _REPO / "diffusion_model"
if str(_DIFF) not in sys.path:
    sys.path.insert(0, str(_DIFF))

from recom.qp_model import _solve_qp, kernel_uniform

ALPHA, BETA, EPS, LAM = 1.0, 10.0, 1.0 / 3.0, 0.10
DELTA = 1e-12


def main():
    G = nx.grid_graph([3, 2])   # 2x3 grid, 6 cells
    for n in G.nodes():
        G.nodes[n]["population"] = 1
        G.nodes[n]["density"] = 0
    cells = sorted(G.nodes())
    ideal = 3.0

    # Labeled states: district 0 gets S, district 1 the complement,
    # both connected, sizes within (1 +- EPS) * ideal.
    lo = int(np.ceil((1 - EPS) * ideal - 1e-9))
    hi = int(np.floor((1 + EPS) * ideal + 1e-9))
    states = []
    for size in range(lo, hi + 1):
        for S in combinations(cells, size):
            S = set(S)
            T = set(cells) - S
            if not T:
                continue
            if nx.is_connected(G.subgraph(S)) and \
               nx.is_connected(G.subgraph(T)):
                states.append({c: (0 if c in S else 1) for c in cells})
    print(f"valid labeled states: {len(states)} (sizes {lo}..{hi})")

    def cuts(asn):
        return sum(1 for u, v in G.edges() if asn[u] != asn[v])

    def bits(asn):
        return np.array([1.0 if asn[c] == 0 else 0.0 for c in cells])

    def log_q(p_clip, b):
        return float(np.sum(b * np.log(p_clip) + (1 - b) * np.log(1 - p_clip)))

    # One deterministic QP solve per state (pair is always {0,1}).
    p_of = []
    for asn in states:
        x = _solve_qp(G, cells, asn, 0, ALPHA, BETA, EPS,
                      kernel_fn=kernel_uniform, ideal_pop=ideal)
        p_of.append(np.clip(x, DELTA, 1 - DELTA))

    n = len(states)
    c = np.array([cuts(a) for a in states], dtype=float)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            lq_f = log_q(p_of[i], bits(states[j]))
            lq_r = log_q(p_of[j], bits(states[i]))
            log_a = min(0.0, -LAM * (c[j] - c[i]) + lq_r - lq_f)
            K[i, j] = np.exp(lq_f + log_a)
        K[i, i] = 1.0 - K[i].sum()

    log_pi = -LAM * c
    pi = np.exp(log_pi - log_pi.max())
    pi /= pi.sum()
    F = pi[:, None] * K
    denom = np.maximum(np.abs(F), np.abs(F.T))
    mask = denom > 0
    rel = np.zeros_like(F)
    rel[mask] = np.abs(F - F.T)[mask] / denom[mask]
    print(f"max relative detailed-balance violation: {rel.max():.2e}")
    print(f"rows sum to 1 (max |row sum - 1|): "
          f"{np.abs(K.sum(axis=1) - 1).max():.2e}")
    print(f"min diagonal entry: {K.diagonal().min():.4f}")


if __name__ == "__main__":
    main()
