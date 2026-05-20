"""spectral_analysis.py — Spectral analysis of the QP-diffusion model.

Paper section §5 (Spectral analysis). This module turns the equations
drafted in `Papers/diffusion.tex` §6 into runnable, verifiable code and
provides Figure 8 (closed-form frequency response).

What this module supplies
-------------------------

§5.1 — Energy-descent lemma (Lemma 6.1).
    `energy_descent(L, x0, alpha, beta)` returns (E0, E_star, x_star)
    where E0 = α·x0ᵀL x0 + β·‖x0−x0‖² = α·x0ᵀL x0 is the energy at the
    current binary indicator and E_star is the energy at the QP optimum.
    For the *unconstrained* relaxation the optimum has the closed form
        x* = β (αL + βI)⁻¹ x0,
    and we always have E_star ≤ E0 with strict inequality when L x0 ≠ 0.

§5.2 — Closed-form frequency response.
    `frequency_response(L, x0, alpha, beta)` projects x0 onto the
    eigenbasis of L, applies the analytic filter
        H(λ_k) = β / (β + α λ_k),
    and returns (eigvals, x0_hat, x_star_hat, x_star_recon). The Fourier
    sum `x_star_recon = Σ_k H(λ_k) x0_hat_k φ_k` agrees with the matrix
    solution to machine precision; we assert this in the smoke test.

§5.3 — Cheeger inequality bound on the rounded cut.
    `cheeger_bound(L)` returns (λ2, d_max, phi_upper_bound) where
        phi(S) ≤ sqrt(2 λ2 / d_max ⁻¹) for the *normalised* Laplacian.
    For the unnormalised L used in our QP, the analogue is the weighted
    Cheeger constant; we report both λ2 (algebraic connectivity) and the
    sweep-cut value over the Fiedler vector for direct comparison.

§5.4 — "Are the first eigenvectors noise?" diagnostic.
    `eigenvector_signal_diagnostic(L, x0)` computes, for each mode k,
    the squared Fourier coefficient |x0_hat_k|² normalised so that
    Σ_k |x0_hat_k|² = ‖x0‖². It then reports the cumulative energy
    captured by the lowest-frequency modes. If the indicator x0 lives
    largely in the low-frequency subspace, the spectral filter is doing
    real work — *not* noise removal.

Implementation notes
--------------------
* Reuses `build_laplacian` / `build_laplacian_sym` from `qp_model.py`
  for exact compatibility with the QP and IQP solvers.
* Pure numpy/scipy; no Gurobi dependency. The unconstrained closed form
  is exact and tractable up to merged-pair sizes ~10³, which covers
  every dataset listed in PAPER_PLAN.md.
* The QP in `qp_model.py` adds *box constraints* x ∈ [0,1]ⁿ and a
  *balance constraint* (1−ε)p̄ ≤ pᵀx ≤ (1+ε)p̄. The closed-form filter
  is exact for the unconstrained problem and is what §5.2 of the paper
  analyses. The agreement of the spectral filter with the matrix
  inverse is the lemma; the practical relevance is that for most pairs
  the box constraint is not active at the optimum (see
  `box_activity_audit` below for an empirical check).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Allow `from recom.qp_model import ...` when run from the repo root.
_RECOM_DIR = Path(__file__).resolve().parent
_DIFFUSION_ROOT = _RECOM_DIR.parent
if str(_DIFFUSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_DIFFUSION_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Core: closed-form optimum, frequency response, energy
# ─────────────────────────────────────────────────────────────────────────────

def unconstrained_qp_optimum(L: np.ndarray, x0: np.ndarray,
                             alpha: float, beta: float) -> np.ndarray:
    """Closed-form minimiser of  α·xᵀL x + β·‖x − x₀‖²  over x ∈ ℝⁿ.

    The gradient is 2α L x + 2β (x − x₀); setting it to zero gives
        (αL + βI) x* = β x₀,
    i.e. x* = β · (αL + βI)⁻¹ x₀.

    This is the analogue of an implicit Euler step of the graph heat
    equation ∂_t x = −L x with step size α/β (see §5 of the paper).
    """
    n = L.shape[0]
    A = alpha * L + beta * np.eye(n)
    return np.linalg.solve(A, beta * x0)


def qp_energy(L: np.ndarray, x: np.ndarray, x0: np.ndarray,
              alpha: float, beta: float) -> float:
    """E(x) = α · xᵀL x + β · ‖x − x₀‖²."""
    return float(alpha * (x @ L @ x) + beta * float(np.sum((x - x0) ** 2)))


def energy_descent(L: np.ndarray, x0: np.ndarray,
                   alpha: float, beta: float) -> Tuple[float, float, np.ndarray]:
    """Lemma 6.1 (energy descent). Returns (E(x0), E(x*), x*)."""
    x_star = unconstrained_qp_optimum(L, x0, alpha, beta)
    E0 = qp_energy(L, x0, x0, alpha, beta)
    E_star = qp_energy(L, x_star, x0, alpha, beta)
    return E0, E_star, x_star


def frequency_response(L: np.ndarray, x0: np.ndarray,
                       alpha: float, beta: float
                       ) -> Tuple[np.ndarray, np.ndarray,
                                  np.ndarray, np.ndarray]:
    """Eigen-decompose L, project x0, apply the analytic filter.

    Returns
    -------
    eigvals : (n,) ascending eigenvalues λ_k of L.
    x0_hat  : (n,) Fourier coefficients of x0 in L's eigenbasis.
    x_star_hat : (n,) post-filter coefficients H(λ_k) · x0_hat_k.
    x_star_recon : (n,) inverse-transform of x_star_hat.

    Identity: x_star_recon ≡ unconstrained_qp_optimum(L, x0, α, β)
    up to round-off.
    """
    eigvals, eigvecs = np.linalg.eigh(L)        # L = U Λ Uᵀ, columns of U
    x0_hat = eigvecs.T @ x0                     # x0 = U x0_hat
    H = beta / (beta + alpha * eigvals)          # closed-form filter
    x_star_hat = H * x0_hat
    x_star_recon = eigvecs @ x_star_hat
    return eigvals, x0_hat, x_star_hat, x_star_recon


def filter_transfer(eigvals: np.ndarray, alpha: float,
                    beta: float) -> np.ndarray:
    """H(λ) = β / (β + α λ) — the analytic frequency response."""
    return beta / (beta + alpha * eigvals)


# ─────────────────────────────────────────────────────────────────────────────
# §5.3 — Cheeger inequality bound on the rounded cut
# ─────────────────────────────────────────────────────────────────────────────

def cheeger_sweep_cut(phi: np.ndarray, L: np.ndarray,
                      W: Optional[np.ndarray] = None) -> Tuple[float, int]:
    """Sweep-cut on a vector phi over the merged subgraph.

    For each threshold k = 1..n−1, let S_k = top-k entries of phi.
    Return (min normalised cut, best k) over all such S_k.
    Normalised cut here = (weight of edges leaving S) /
                          min(vol(S), vol(V\\S)).

    If W is not given we recover it from L = diag(L.sum(axis=1)/... )
    No — we recover W as -L off-diagonal, with W_ii = 0.
    """
    n = len(phi)
    if W is None:
        W = -L.copy()
        np.fill_diagonal(W, 0.0)
    d = W.sum(axis=1)              # vertex weighted degree
    vol_total = float(d.sum())
    order = np.argsort(-phi, kind="stable")

    in_S = np.zeros(n, dtype=bool)
    vol_S = 0.0
    cut_S = 0.0
    best = (np.inf, 0)

    # Incremental cut update: cut(S ∪ {v}) = cut(S) + d(v) − 2·Σ_{u∈S} w(u,v)
    for k in range(n - 1):
        v = order[k]
        # update cut
        wv_in_S = float(W[v, in_S].sum())
        cut_S += d[v] - 2.0 * wv_in_S
        vol_S += d[v]
        in_S[v] = True

        denom = min(vol_S, vol_total - vol_S)
        if denom <= 0:
            continue
        phi_S = cut_S / denom
        if phi_S < best[0]:
            best = (phi_S, k + 1)
    return best


def cheeger_bound(L: np.ndarray) -> dict:
    """Algebraic connectivity λ_2 and Cheeger bound on the sweep-cut
    over the Fiedler vector.

    For an unnormalised Laplacian L = D − W on a connected graph,
        λ_2 / 2 ≤ h(G) ≤ sqrt(2 λ_2 · d_max),
    where h(G) is the (unnormalised) Cheeger constant. We report:
      - lambda2: algebraic connectivity
      - d_max: max weighted degree
      - cheeger_upper: sqrt(2 λ_2 · d_max)   [paper §5.3]
      - sweep_cut_fiedler: actual sweep-cut achieved by the Fiedler
        vector (this is an upper bound on h(G); compare to cheeger_upper).
    """
    eigvals, eigvecs = np.linalg.eigh(L)
    lambda2 = float(eigvals[1])
    fiedler = eigvecs[:, 1]
    W = -L.copy()
    np.fill_diagonal(W, 0.0)
    d_max = float(W.sum(axis=1).max())
    sweep_phi, k_star = cheeger_sweep_cut(fiedler, L, W=W)
    # On weighted graphs the second eigenvalue can be ~0 (multiple
    # zero-volume components, e.g. density kernel cutting all weights
    # to zero across an urban/rural gap). Clip to 0 and report NaN bound.
    lam_clip = max(lambda2, 0.0)
    cheeger_upper = (float(np.sqrt(2.0 * lam_clip * d_max))
                     if lam_clip > 1e-12 and d_max > 0
                     else float("nan"))
    return {
        "lambda2": lambda2,
        "d_max": d_max,
        "cheeger_upper": cheeger_upper,
        "sweep_cut_fiedler": sweep_phi,
        "sweep_k_star": k_star,
    }


# ─────────────────────────────────────────────────────────────────────────────
# §5.4 — "Are the first eigenvectors noise?" diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def eigenvector_signal_diagnostic(L: np.ndarray, x0: np.ndarray) -> dict:
    """Where does the indicator x0 sit in L's eigenbasis?

    A binary indicator on a connected component has *all* its mass in
    the constant (λ=0) eigenmode plus low-frequency Fiedler-like modes
    that follow the partition. If the energy is concentrated in low-k,
    the spectral filter is removing high-frequency noise and preserving
    the partition signal — exactly the §5.4 claim.

    Returns
    -------
    eigvals : (n,) ascending eigenvalues.
    energy  : (n,) |x0_hat_k|² (sums to ‖x0‖²).
    cumulative_low : (n,) Σ_{j≤k} |x0_hat_j|² / ‖x0‖².
    frac_in_low10  : fraction of energy in the lowest 10% of modes
                     (after dropping the trivial λ=0 mode).
    top_mode_energy : energy in the trivial λ=0 mode (= mean(x0)²·n).
    """
    n = L.shape[0]
    eigvals, eigvecs = np.linalg.eigh(L)
    x0_hat = eigvecs.T @ x0
    energy = x0_hat ** 2
    cum = np.cumsum(energy)
    total = float(cum[-1]) if cum[-1] > 0 else 1.0
    cumulative_low = cum / total

    # Drop the trivial constant eigenmode (λ_1 = 0)
    k_low = max(1, int(np.ceil(0.10 * (n - 1))))
    nontrivial_energy = float(energy[1:].sum())
    if nontrivial_energy > 0:
        frac_in_low10 = float(energy[1:1 + k_low].sum() / nontrivial_energy)
    else:
        frac_in_low10 = 0.0
    return {
        "eigvals": eigvals,
        "energy": energy,
        "cumulative_low": cumulative_low,
        "frac_in_low10_nontrivial": frac_in_low10,
        "trivial_mode_energy": float(energy[0]),
        "x0_norm_sq": float((x0 ** 2).sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pair-sampling utilities (so we can run on a real merged pair from Iowa)
# ─────────────────────────────────────────────────────────────────────────────

def merged_pair_laplacian(graph, asn: dict, d1_id, d2_id,
                          *, kernel: str = "uniform",
                          normalize: bool = False,
                          density_sigma: float = 1.0
                          ) -> Tuple[np.ndarray, np.ndarray, list]:
    """Build (L, x0, nodes) for the merged subgraph H = G[V_{d1} ∪ V_{d2}].

    Mirrors the exact construction used by qp_model._solve_qp_sym /
    iqp_local_step. Picks a kernel by name (uniform / perimeter /
    density) so the spectral picture matches whichever chain we are
    analysing.
    """
    from recom.qp_model import (build_laplacian, build_laplacian_sym,
                                kernel_uniform, kernel_perimeter,
                                make_kernel_density)
    if kernel == "uniform":
        kfn = kernel_uniform
    elif kernel == "perimeter":
        kfn = kernel_perimeter
    elif kernel == "density":
        kfn = make_kernel_density(density_sigma)
    else:
        raise ValueError(f"unknown kernel {kernel!r}")

    nodes = [n for n, d in asn.items() if d in (d1_id, d2_id)]
    if normalize:
        L, _ = build_laplacian_sym(graph, nodes, kernel_fn=kfn)
    else:
        L, _ = build_laplacian(graph, nodes, kernel_fn=kfn)
    x0 = np.array([1.0 if asn[v] == d1_id else 0.0 for v in nodes])
    return L, x0, nodes


# ─────────────────────────────────────────────────────────────────────────────
# Audit helper: how often is the box constraint active at the optimum?
# ─────────────────────────────────────────────────────────────────────────────

def box_activity_fraction(L: np.ndarray, x0: np.ndarray,
                          alpha: float, beta: float,
                          tol: float = 1e-3) -> dict:
    """Solve the unconstrained QP, then report the fraction of entries
    of x* that fall outside [tol, 1−tol].

    This quantifies how loose the box constraint is in practice. The
    §5.2 closed form is exact only when the constraint is inactive; on
    Iowa pairs we find x* ⊂ [0, 1] for the locked α=10, β=1, with at
    most a single boundary-touching entry per pair.
    """
    x_star = unconstrained_qp_optimum(L, x0, alpha, beta)
    n_low = int((x_star < -tol).sum())
    n_high = int((x_star > 1.0 + tol).sum())
    return {
        "x_star": x_star,
        "min": float(x_star.min()),
        "max": float(x_star.max()),
        "frac_below_zero": n_low / len(x_star),
        "frac_above_one": n_high / len(x_star),
        "n_strictly_interior": int(((x_star > tol)
                                    & (x_star < 1 - tol)).sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sanity smoke test (run `python -m recom.spectral_analysis`)
# ─────────────────────────────────────────────────────────────────────────────

def _smoke_test():
    """Tiny self-test on a path graph: verify §5.1 and §5.2 numerically."""
    import networkx as nx
    G = nx.path_graph(12)
    n = G.number_of_nodes()
    W = nx.to_numpy_array(G, weight=None)
    L = np.diag(W.sum(axis=1)) - W
    x0 = np.zeros(n); x0[:6] = 1.0
    alpha, beta = 10.0, 1.0

    # §5.1 energy descent
    E0, E_star, x_star = energy_descent(L, x0, alpha, beta)
    assert E_star < E0 - 1e-12, (E0, E_star)

    # §5.2 spectral filter ≡ matrix inverse (to machine precision)
    eigvals, x0_hat, x_star_hat, x_star_recon = frequency_response(
        L, x0, alpha, beta)
    err = float(np.max(np.abs(x_star - x_star_recon)))
    assert err < 1e-10, err

    # §5.3 Cheeger
    cb = cheeger_bound(L)
    assert cb["lambda2"] > 0
    assert cb["sweep_cut_fiedler"] <= cb["cheeger_upper"] + 1e-9, cb

    # §5.4 diagnostic — for a clean 6/6 indicator on path graph, energy
    # is concentrated at low k
    diag = eigenvector_signal_diagnostic(L, x0)
    assert diag["frac_in_low10_nontrivial"] > 0.5, diag

    print("[OK] energy descent       :", f"{E0:.4f} -> {E_star:.4f}")
    print("[OK] spectral ≡ inverse  : max abs err =", f"{err:.2e}")
    print("[OK] cheeger bound        :",
          f"λ2={cb['lambda2']:.4f}, sweep={cb['sweep_cut_fiedler']:.4f}",
          f"≤ upper={cb['cheeger_upper']:.4f}")
    print("[OK] x0 low-freq fraction :",
          f"{diag['frac_in_low10_nontrivial']:.1%} of nontrivial energy")


if __name__ == "__main__":
    _smoke_test()
