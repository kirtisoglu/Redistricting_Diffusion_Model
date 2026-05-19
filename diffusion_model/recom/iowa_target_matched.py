"""Target-matched comparison orchestrator.

Compares five chains on a chosen problem:
  V — Vanilla ReCom
  A — Tempered ReCom  (MH on cuts, symmetric-q approximation)
  B — ReCom + QP-diffusion mixture  (continuous local refinement)
  C — Tempered ReCom + IQP local refinement  (integer version of B)
  D — ReCom + SpecReCom mixture  (Davies et al. 2025 baseline)

B and C share the same model min α·xᵀLx + β·‖x − x₀‖² with kernel
weights w_e; only the variable domain differs ([0,1] for B,
{0,1} for C).  C subsumes our prior MIP-cut-min: setting α=1, β=0,
kernel=uniform recovers raw cut minimisation.

Modes:
  MODE = "5x5"   — 5×5 grid (k=5), full fiber enumeration → kernel-correctness
                   check (KL to π_target), bias bound (TV to uniform),
                   island-neck step-trace.
  MODE = "iowa"  — Iowa counties (k=4), no ground-truth fiber.
  MODE = "nc"    — North Carolina precinct (NC_VTD, 2692 nodes).

Implementation lives in problems.py / chains.py / diagnostics.py;
this file is the orchestrator.
"""

import random
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, ttest_ind

_REPO = Path(__file__).resolve().parent.parent.parent
_DIFF = _REPO / "diffusion_model"
if str(_DIFF) not in sys.path:
    sys.path.insert(0, str(_DIFF))

from recom.problems import setup_problem, load_fiber_islands, cut_edges_count
from recom.chains import (run_chain,
                          make_vanilla_step, make_tempered_step,
                          make_qp_diffusion_mix_step,
                          make_iqp_hybrid_step, make_specrecom_mix_step)
from recom.diagnostics import (kernel_correctness_table, bias_bound_table,
                               island_neck_table, cuts_compare_plot,
                               island_trace_plot, expected_target_cuts)
from recom.qp_model import (make_partition, kernel_uniform, kernel_perimeter,
                            make_kernel_density)


# ─── Config ──────────────────────────────────────────────────────────────────
MODE = "5x5_eps02"    # "iowa" or "5x5" or "5x5_eps02"
SEED = 42
LAM = 0.10
N_STEPS = 1000 if MODE in ("5x5", "5x5_eps02") else 500
BURN = 100 if MODE in ("5x5", "5x5_eps02") else 100

P_RECOM        = 0.5      # mixture coefficient for B and D chains
QP_ALPHA       = 10.0
QP_BETA        = 1.0
QP_THRESHOLD   = "cheeger"
QP_NORMALIZE   = False    # unnormalised L: xᵀLx = weighted cut on binary
QP_KERNEL      = "uniform"  # "uniform" | "perimeter" | "density"
QP_KERNEL_SIGMA = 1.0       # σ for the density kernel

# Diversity controls applied to chains B and C only:
WEIGHT_JITTER  = False
TABU_SIZE      = 0

# Chain C: integer-domain version of the QP-diffusion model (IQP, MIQP
# in Gurobi).  Same α, β, kernel as the continuous QP; only the
# variable domain changes from [0,1] to {0,1}.  Subsumes the legacy
# MIP-cut-min (set α=1, β=0, kernel=uniform).
IQP_PER_RECOM   = 10
IQP_TIME_LIMIT  = 5.0
IQP_ALPHA       = 10.0
IQP_BETA        = 1.0

random.seed(SEED)
np.random.seed(SEED)

HERE = Path(__file__).resolve().parent
FIBER_EDGES = HERE / "fiber_5x5" / "fiber_5x5_edges.txt"
FIBER_EPS02_EDGES = HERE / "fiber_5x5" / "fiber_5x5_eps02_edges.txt"
FIBER_EPS02_COMMS = HERE / "fiber_5x5" / "fiber_5x5_eps02_communities.npz"
OUT = HERE / "iowa_hybrid_test_out"
OUT.mkdir(exist_ok=True)


def main():
    p = setup_problem(MODE, _REPO)
    graph, eps, ideal, init = p["graph"], p["eps"], p["ideal"], p["init"]
    label = p["label"]
    fiber_lookup = p["fiber_lookup"]
    fiber_cuts = p["fiber_cuts"]
    init_cuts = cut_edges_count(graph, init)

    island_of = None
    if MODE == "5x5":
        try:
            island_of, comms = load_fiber_islands(FIBER_EDGES)
            sizes = sorted([len(c) for c in comms], reverse=True)
            print(f"  Loaded {len(comms)} islands, sizes {sizes}")
        except Exception as e:
            print(f"  (island load failed: {e})")
    elif MODE == "5x5_eps02":
        try:
            island_of, comms = load_fiber_islands(
                FIBER_EPS02_EDGES, n_nodes=193128,
                prebuilt_npz=FIBER_EPS02_COMMS)
            sizes = sorted([len(c) for c in comms], reverse=True)
            print(f"  Loaded {len(comms)} islands, sizes (top 10) {sizes[:10]}")
        except Exception as e:
            print(f"  (island load failed: {e})")

    e_pi_cuts = (expected_target_cuts(fiber_cuts, LAM)
                 if fiber_cuts is not None else None)
    print(f"  {label}\n  initial cuts = {init_cuts}\n"
          f"  target: π(P) ∝ exp(−{LAM} · cut_edges(P))")
    if e_pi_cuts is not None:
        print(f"  E_π[cuts] under target = {e_pi_cuts:.3f}")

    # ── Chain definitions ──────────────────────────────────────────────────
    print(f"\n── V: Vanilla ReCom ({N_STEPS} steps) ──")
    V = run_chain("V", make_partition(graph, init),
                  make_vanilla_step(graph, ideal, eps),
                  N_STEPS, fiber_lookup, island_of)

    print(f"\n── A: Tempered ReCom λ={LAM} ({N_STEPS} steps) ──")
    A = run_chain("A", make_partition(graph, init),
                  make_tempered_step(graph, ideal, eps, LAM),
                  N_STEPS, fiber_lookup, island_of)
    A_acc = np.array([e[0] for e in A["extras"]])

    if QP_KERNEL == "uniform":
        kfn = kernel_uniform
    elif QP_KERNEL == "perimeter":
        kfn = kernel_perimeter
    elif QP_KERNEL == "density":
        kfn = make_kernel_density(QP_KERNEL_SIGMA)
    else:
        raise ValueError(f"unknown QP_KERNEL {QP_KERNEL!r}")
    print(f"\n── B: ReCom/QP-diffusion mixture (α={QP_ALPHA}, β={QP_BETA}, "
          f"thr={QP_THRESHOLD!r}, normalize={QP_NORMALIZE}, "
          f"kernel={QP_KERNEL!r}, jitter={WEIGHT_JITTER}, "
          f"tabu={TABU_SIZE}) ──")
    B = run_chain("B", make_partition(graph, init),
                  make_qp_diffusion_mix_step(graph, ideal, eps, LAM,
                                             P_RECOM, QP_ALPHA, QP_BETA,
                                             QP_THRESHOLD, QP_NORMALIZE,
                                             kernel_fn=kfn,
                                             weight_jitter=WEIGHT_JITTER,
                                             tabu_size=TABU_SIZE),
                  N_STEPS, fiber_lookup, island_of)
    B_kinds = [e[0] for e in B["extras"]]
    B_oks   = np.array([e[1] for e in B["extras"]])
    B_n_qp = sum(1 for k in B_kinds if k == "Q")
    B_qp_success = sum(1 for k, o in zip(B_kinds, B_oks) if k == "Q" and o)
    B_qp_rate = B_qp_success / max(B_n_qp, 1)

    print(f"\n── C: Tempered ReCom + IQP local "
          f"({IQP_PER_RECOM} per step, α={IQP_ALPHA}, β={IQP_BETA}, "
          f"kernel={QP_KERNEL!r}, jitter={WEIGHT_JITTER}, "
          f"tabu={TABU_SIZE}) ──")
    C = run_chain("C", make_partition(graph, init),
                  make_iqp_hybrid_step(graph, ideal, eps, LAM,
                                       IQP_PER_RECOM, IQP_ALPHA, IQP_BETA,
                                       kfn, QP_NORMALIZE, IQP_TIME_LIMIT,
                                       weight_jitter=WEIGHT_JITTER,
                                       tabu_size=TABU_SIZE),
                  N_STEPS, fiber_lookup, island_of)
    C_recom_acc = np.array([e[0] for e in C["extras"]])
    C_iqp_imp   = np.array([e[1] for e in C["extras"]])

    print(f"\n── D: ReCom/SpecReCom mixture (Davies 2025) ──")
    D = run_chain("D", make_partition(graph, init),
                  make_specrecom_mix_step(graph, ideal, eps, P_RECOM),
                  N_STEPS, fiber_lookup, island_of)
    D_kinds = [e[0] for e in D["extras"]]
    D_oks   = np.array([e[1] for e in D["extras"]])
    D_n_spec = sum(1 for k in D_kinds if k == "S")
    D_spec_success = sum(1 for k, o in zip(D_kinds, D_oks) if k == "S" and o)
    D_spec_rate = D_spec_success / max(D_n_spec, 1)

    # ── Compactness table ──────────────────────────────────────────────────
    A_b, B_b, C_b, D_b = (A["cuts"][BURN:], B["cuts"][BURN:],
                          C["cuts"][BURN:], D["cuts"][BURN:])
    V_b = V["cuts"][BURN:]

    def cmp(ref, new):
        d = new.mean() - ref.mean()
        return d, 100 * d / ref.mean(), ks_2samp(ref, new).pvalue

    print("\n" + "=" * 72)
    print(f"RESULTS — target-matched at λ={LAM}, after burn-in {BURN}")
    print("=" * 72)
    print(f"  {label}")
    print(f"  initial cuts:                              {init_cuts}")
    print(f"  V: Vanilla ReCom (reference):              "
          f"mean={V_b.mean():6.2f}  std={V_b.std():4.2f}")
    print(f"  A: Tempered ReCom λ={LAM}:                   "
          f"mean={A_b.mean():6.2f}  std={A_b.std():4.2f}  "
          f"recom_acc={A_acc[BURN:].mean():.2%}")
    print(f"  B: ReCom/QP-diffusion mixture:             "
          f"mean={B_b.mean():6.2f}  std={B_b.std():4.2f}  "
          f"qp_share={B_n_qp/N_STEPS:.0%}  qp_success={B_qp_rate:.0%}")
    print(f"  C: Tempered ReCom + IQP local:             "
          f"mean={C_b.mean():6.2f}  std={C_b.std():4.2f}  "
          f"iqp_imp={C_iqp_imp[BURN:].mean():.2f}/{IQP_PER_RECOM}")
    print(f"  D: ReCom/SpecReCom mixture:                "
          f"mean={D_b.mean():6.2f}  std={D_b.std():4.2f}  "
          f"spec_share={D_n_spec/N_STEPS:.0%}  spec_success={D_spec_rate:.0%}")
    for tag, post in [("B", B_b), ("C", C_b), ("D", D_b)]:
        d, pct, p = cmp(A_b, post)
        print(f"  Δ({tag} − A) = {d:+.2f}  ({pct:+.1f}%)   KS p = {p:.2e}")
    print(f"  Wall clock: " + "  ".join(
        f"{tag}={ch['times'].sum():.1f}s" for tag, ch
        in [("V", V), ("A", A), ("B", B), ("C", C), ("D", D)]))

    # ── Diagnostic tables (5×5 only) ───────────────────────────────────────
    chains_for_tables = [
        ("V: Vanilla ReCom", V),
        ("A: Tempered ReCom", A),
        ("B: QP-diffusion mix", B),
        ("C: ReCom + IQP local", C),
        ("D: ReCom + SpecReCom", D),
    ]
    if fiber_lookup is not None:
        kernel_correctness_table(chains_for_tables, fiber_cuts, LAM,
                                 N_STEPS, BURN)
        bias_bound_table(chains_for_tables, len(fiber_lookup))
        if island_of is not None:
            island_neck_table(chains_for_tables)

    # ── Plots ──────────────────────────────────────────────────────────────
    suffix = MODE
    cuts_compare_plot(chains_for_tables, label, init_cuts, LAM,
                      e_pi_cuts, BURN, OUT / f"target_matched_{suffix}.png")
    print(f"\n  Plot: {OUT / f'target_matched_{suffix}.png'}")
    if island_of is not None:
        island_trace_plot(chains_for_tables, OUT / f"island_trace_{suffix}.png",
                          f"{label} — chain step-trace by island")
        print(f"  Island trace: {OUT / f'island_trace_{suffix}.png'}")

    np.savez(OUT / f"target_matched_{suffix}.npz",
             V=V["cuts"], A=A["cuts"], B=B["cuts"], C=C["cuts"], D=D["cuts"],
             A_acc=A_acc, C_recom_acc=C_recom_acc, C_iqp_imp=C_iqp_imp,
             B_oks=B_oks, B_kinds=np.array(B_kinds),
             D_oks=D_oks, D_kinds=np.array(D_kinds),
             V_t=V["times"], A_t=A["times"], B_t=B["times"],
             C_t=C["times"], D_t=D["times"],
             init_cuts=init_cuts, lam=LAM)


if __name__ == "__main__":
    main()
