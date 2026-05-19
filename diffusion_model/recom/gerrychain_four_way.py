"""
Gerrychain: QP-MH + ReCom + Hybrid  (QP+ReCom vs ReCom comparison)
===================================================================

Key additions
-------------
1. Randomized rounding  — each node v is assigned to D1 with probability x*_v
   (Bernoulli), making q(x′|x) fully computable and the MH correction exact.

Target for QP-MH:
     π(x) ∝ exp(−λ · cut_edges(x))

Chains compared
---------------
  A. ReCom       (20 steps,  global jumps, no gradient)
  B. QP-MH       (20 steps,  local gradient, randomized rounding, exact MH)
  C. Hybrid      (10 rounds, 1 ReCom + 3 QP-MH, annealed λ)

Diagnostics: ACF, ESS, τ_int, running mean, TV proxy — all on Density Score trace.
Wall-clock timing per step is also recorded.
"""

# ─────────────────────────────────────────────────────────────────────────────
import random, warnings, time, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")

from gerrychain.proposals import recom

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Allow imports from sibling packages (helpers/)
# ─────────────────────────────────────────────────────────────────────────────
_DIFFUSION_ROOT = Path(__file__).resolve().parent.parent
if str(_DIFFUSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_DIFFUSION_ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# 1.  GRAPH SETUP  (grid, initial partition, density kernel — from grid_setup)
# ─────────────────────────────────────────────────────────────────────────────
from helpers.grid_setup import (
    graph, gc_graph,
    reference_assignment, initial_assignment,
    pop_target, ns,
    plot_boundary_nodes,
)

from helpers.metrics import (
    density_score, record_metrics,
)

from recom.qp_model import (
    make_partition,
    qp_mh_proposal,
)

import helpers.plots as _plots

PLOTS_DIR = Path(__file__).resolve().parent.parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ── initial plot ──────────────────────────────────────────────────────────────
_plots.plot_initial_partition(graph, initial_assignment, ns, PLOTS_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  RUN ALL CHAINS  (20 steps each)
# ─────────────────────────────────────────────────────────────────────────────

def run_chain(name, step_fn, steps, init_asn):
    """Generic runner. step_fn(asn) → (new_asn, accepted_bool)."""
    asn     = dict(init_asn)
    metrics = []
    times   = []
    accepts = []
    for s in range(steps):
        t0       = time.perf_counter()
        asn, acc = step_fn(asn)
        elapsed  = time.perf_counter() - t0
        m        = record_metrics(graph, asn)
        metrics.append(m)
        times.append(elapsed)
        accepts.append(int(acc))
        print(f"  [{name}] step {s+1:2d}  DS={m['density_score']:.4f}  "
              f"PP={m['pp']:.4f}  acc={acc}  t={elapsed:.3f}s")
    ar = 100 * np.mean(accepts) if accepts else float('nan')
    print(f"  [{name}] accept rate: {ar:.1f}%  "
          f"mean step time: {1000*np.mean(times):.1f}ms\n")
    return metrics, times, accepts, asn


# ── ReCom ─────────────────────────────────────────────────────────────────────
print("\n── ReCom (20 steps) ─────────────────────────────────────────────────────")
_part_recom = make_partition(gc_graph, initial_assignment)

def recom_step(_):
    global _part_recom
    _part_recom = recom(_part_recom, "population", pop_target, 0.10, node_repeats=1)
    return dict(_part_recom.assignment), True   # ReCom always "accepts"

recom_metrics, recom_times, _, recom_final = run_chain("ReCom", recom_step, 20, initial_assignment)


# ── QP-MH ─────────────────────────────────────────────────────────────────────
print("\n── QP-MH  (20 steps) ────────────────────────────────────────────────────")

def qp_mh_step(asn):
    return qp_mh_proposal(graph, asn, alpha=0.5, beta=20.0, epsilon=0.05, lam=0.05)

qp_metrics, qp_times, qp_accepts, qp_final = run_chain("QP-MH", qp_mh_step, 20, initial_assignment)


# ── Hybrid  (ReCom + multi-step QP-MH with annealed λ, 10 rounds) ────────────
#
# Improvements over the naive 1-ReCom + 1-QP design:
#
#  1. Multi-step local refinement: after each ReCom global jump, run
#     QP_STEPS_PER_RECOM QP-MH steps to exploit the gradient signal before
#     the next jump.  This lets the QP refine rather than just nudge.
#
#  2. Annealed λ: start with a low λ (broad acceptance, large moves) and
#     increase toward LAM_FINAL over the refinement sub-chain.  The schedule
#     log-linearly interpolates  λ_t = LAM_START * (LAM_FINAL/LAM_START)^{t/T}.
#     This prevents the QP sub-chain from getting stuck at the ReCom output.
#
print("\n── Hybrid (10 rounds: 1 ReCom + 3 QP-MH, annealed λ) ───────────────────")

QP_STEPS_PER_RECOM = 3      # QP-MH steps after each ReCom jump
LAM_START          = 0.01   # broad acceptance right after the jump
LAM_FINAL          = 0.10   # tighter toward end of refinement sub-chain

_part_hyb   = make_partition(gc_graph, initial_assignment)
_hyb_round  = [0]           # mutable counter accessible inside closure

def hybrid_step(_):
    global _part_hyb
    _part_hyb = recom(_part_hyb, "population", pop_target, 0.10, node_repeats=1)
    asn = dict(_part_hyb.assignment)

    # Annealed QP-MH refinement
    any_acc = False
    for t in range(QP_STEPS_PER_RECOM):
        frac = t / max(QP_STEPS_PER_RECOM - 1, 1)
        lam  = LAM_START * (LAM_FINAL / LAM_START) ** frac
        asn, acc = qp_mh_proposal(graph, asn, alpha=1.0, beta=10.0,
                                   epsilon=0.10, lam=lam)
        any_acc = any_acc or acc

    _part_hyb = make_partition(gc_graph, asn)
    return asn, any_acc

hyb_metrics, hyb_times, hyb_accepts, hyb_final = run_chain("Hybrid", hybrid_step, 10, initial_assignment)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  FINAL PARTITION PLOTS
# ─────────────────────────────────────────────────────────────────────────────
print("Plotting final partitions …")
_plots.plot_final_partitions(
    graph,
    assignments=[
        (recom_final, "ReCom (20 steps)"),
        (qp_final,    "QP-MH (20 steps)"),
        (hyb_final,   "Hybrid (10 rounds)"),
    ],
    ns=ns,
    plots_dir=PLOTS_DIR,
)


# ─────────────────────────────────────────────────────────────────────────────
# 8.5  BOUNDARY COMPARISON PLOT  (2 × 3 grid)
# ─────────────────────────────────────────────────────────────────────────────
print("Plotting boundary comparison …")
_plots.plot_boundary_comparison(
    reference_assignment, initial_assignment,
    recom_final, qp_final, hyb_final,
    plot_boundary_nodes,
    PLOTS_DIR,
)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  METRIC COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
print("Plotting metric comparison …")
_plots.plot_metric_comparison(recom_metrics, qp_metrics, hyb_metrics,
                              PLOTS_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# 10.  MIXING-TIME DIAGNOSTICS  (long runs: 200 steps)
# ─────────────────────────────────────────────────────────────────────────────
LONG    = 200
MAX_LAG = 60
print(f"\n── Long runs for mixing diagnostics  ({LONG} steps) ────────────────────")

def long_trace(step_fn, init_asn, steps=LONG):
    asn, trace, wtimes = dict(init_asn), [], []
    for _ in range(steps):
        t0      = time.perf_counter()
        asn, _  = step_fn(asn)
        wtimes.append(time.perf_counter() - t0)
        trace.append(density_score(graph, asn))
    return np.array(trace), np.array(wtimes)

print("  ReCom …")
_part_recom = make_partition(gc_graph, initial_assignment)
recom_trace, recom_wt = long_trace(recom_step, initial_assignment)

print("  QP-MH …")
qp_trace, qp_wt = long_trace(qp_mh_step, initial_assignment)

print("  Hybrid …")
_part_hyb = make_partition(gc_graph, initial_assignment)   # reset for long run
hyb_trace, hyb_wt = long_trace(hybrid_step, initial_assignment)


# ─────────────────────────────────────────────────────────────────────────────
# 11.  MIXING-TIME FIGURE  (6-panel)
# ─────────────────────────────────────────────────────────────────────────────
print("\nPlotting mixing-time figure …")
(tau_r, tau_q, tau_h,
 ess_r, ess_q, ess_h,
 eps_r, eps_q, eps_h) = _plots.plot_mixing_time(
    recom_trace, qp_trace, hyb_trace,
    recom_wt, qp_wt, hyb_wt,
    LONG, MAX_LAG, PLOTS_DIR,
)

print(f"\n  τ_int  →  ReCom={tau_r:.2f}  QP-MH={tau_q:.2f}  "
      f"Hybrid={tau_h:.2f}")
print(f"  ESS    →  ReCom={ess_r:.1f}   QP-MH={ess_q:.1f}  "
      f"Hybrid={ess_h:.1f}")
print(f"  ESS/s  →  ReCom={eps_r:.2f}   QP-MH={eps_q:.2f}  "
      f"Hybrid={eps_h:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 12.  SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("SUMMARY  (long runs, 200 steps each)")
print("=" * 68)
hdr = (f"{'Metric':<38} {'ReCom':>9} {'QP-MH':>9} {'Hybrid':>9}")
print(hdr); print("-" * 68)

rows = [
    ("Mean Density Score",  recom_trace.mean(), qp_trace.mean(), hyb_trace.mean(), "{:>9.4f}"),
    ("Std  Density Score",  recom_trace.std(),  qp_trace.std(),  hyb_trace.std(),  "{:>9.4f}"),
    ("τ_int (ACF)",         tau_r,              tau_q,           tau_h,            "{:>9.2f}"),
    ("ESS",                 ess_r,              ess_q,           ess_h,            "{:>9.1f}"),
    ("ESS / wall-second",   eps_r,              eps_q,           eps_h,            "{:>9.2f}"),
    ("Mean step time (ms)", 1e3*recom_wt.mean(), 1e3*qp_wt.mean(), 1e3*hyb_wt.mean(), "{:>9.1f}"),
]

for name, r, q, h, fmt in rows:
    print(f"  {name:<36} {fmt.format(r)} {fmt.format(q)} "
          f"{fmt.format(h)}")
print("=" * 68)

print("""
Density Score  DS(x) = Σ_{intra} w(e) / Σ_{all} w(e),   w(e) = exp(-|Δdensity|)
─────────────────────────────────────────────────────────────────────────────
Measures how well district boundaries align with the density kernel.
DS = 1 means all high-weight (same-density) edges are intra-district (perfect).
DS = 0 means all edges cross district boundaries.

Interpretation of ESS/second  (computed on Density Score trace)
─────────────────────────────────────────────────────────────────────────────
ESS/step  answers: "which chain decorrelates fastest?"
ESS/sec   answers: "which chain gives the most independent draws per unit
                    of compute?" — the practical metric for redistricting work.

Expected ordering (hypothesis)
  ESS/step  : Hybrid ≥ ReCom > QP-MH
              (Hybrid combines ReCom's large jumps with QP's gradient signal)
  ESS/sec   : ReCom > Hybrid > QP-MH
              (QP solve is expensive per step)

If ESS/sec(Hybrid) > ESS/sec(ReCom):  the gradient signal of QP pays off
                                       even after accounting for QP's cost.
─────────────────────────────────────────────────────────────────────────────
""")
print("Done. Output files: initial_partition.png, final_partitions.png,")
print("      boundary_comparison.png, metric_comparison.png, mixing_time_analysis.png")
