"""
QP-MEW comparison on Iowa Counties (k=5)
=========================================

Chains compared (balance energy only — no compactness):
  A. Plain MEW              — uniform proposals (baseline)
  B. QP-Hybrid (λ=0.5)     — cycle step + QP balance, low locality
  C. QP-Hybrid (λ=2)       — cycle step + QP balance, moderate
  D. QP-Hybrid (λ=10)      — cycle step + QP balance, high locality

Initial partition from GerryChain recursive_tree_part.
"""

import sys, time, random
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

_DIFFUSION_ROOT = Path(__file__).resolve().parent.parent
if str(_DIFFUSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_DIFFUSION_ROOT))

from mew.data_loader import iowa_counties, new_hampshire, initial_partition
from mew.mew_model import initialize_from_partition, mew_step, mew_wilson_step
from mew.qp_proposal import qp_hybrid_step
from helpers.metrics import n_cut_edges, mean_pop_deviation
from helpers.plots import acf, tau_and_ess, ess_per_second

PLOTS_DIR = Path(__file__).resolve().parent.parent.parent / "plots" / "mew"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)


# ═══════════════════════════════════════════════════════════════════════════════
#  1. SETUP
# ═══════════════════════════════════════════════════════════════════════════════

G = new_hampshire(k=5)
n_districts = G.graph["districts"]
total_pop = sum(G.nodes[v]["population"] for v in G)
pop_target = total_pop / n_districts
EPSILON = 0.10

print(f"Graph: {G.graph['name']}")
print(f"  n={G.number_of_nodes()}, e={G.number_of_edges()}, "
      f"d={n_districts}, pop={total_pop:,d}, target={pop_target:,.0f}")

# ── Initial partition via ReCom recursive_tree_part ─────────────────────────
init_asn = initial_partition(G, k=n_districts, epsilon=EPSILON)

init_cut = n_cut_edges(G, init_asn)
print(f"  Initial partition: {len(set(init_asn.values()))} districts, "
      f"cut edges: {init_cut}, "
      f"pop dev: {mean_pop_deviation(G, init_asn):.4f}\n")


# ── Energy: soft balance barrier + compactness gradient ─────────────────────

BETA_COMPACT = 0.1
MU_C = init_cut  # target the initial cut-edge count

def J_energy(graph, asn):
    """
    Soft balance barrier + compactness gradient.

    Balance: flat inside B, quadratic penalty outside.
    Compactness: -β(c - μ_c)² targets a cut-edge count.
    """
    # Balance: soft barrier at ε boundary
    pops = {}
    for node, d in asn.items():
        pops[d] = pops.get(d, 0) + graph.nodes[node]["population"]
    balance_penalty = 0.0
    for p in pops.values():
        dev = abs(p - pop_target) / (EPSILON * pop_target)
        if dev > 1.0:
            balance_penalty += (dev - 1.0) ** 2
    J_bal = -10.0 * balance_penalty  # strong push back into B

    # Compactness: gradient for exploration
    c = n_cut_edges(graph, asn)
    J_comp = -BETA_COMPACT * (c - MU_C) ** 2

    return J_bal + J_comp


# ═══════════════════════════════════════════════════════════════════════════════
#  2. RUN CHAINS
# ═══════════════════════════════════════════════════════════════════════════════

def run_chain(name, step_fn, steps):
    state = initialize_from_partition(G, init_asn)
    trace, pop_devs, wtimes = [], [], []
    accepts = 0

    for s in range(steps):
        t0 = time.perf_counter()
        state, acc = step_fn(state)
        elapsed = time.perf_counter() - t0
        wtimes.append(elapsed)
        accepts += int(acc)
        asn = state.get_assignment()
        trace.append(n_cut_edges(G, asn))
        pop_devs.append(mean_pop_deviation(G, asn))

        if (s + 1) % max(1, steps // 5) == 0:
            print(f"  [{name}] step {s+1:5d}  cut={trace[-1]:2d}  "
                  f"popdev={pop_devs[-1]:.4f}  "
                  f"acc={100*accepts/(s+1):.1f}%  t={1e3*elapsed:.2f}ms")

    trace = np.array(trace, dtype=float)
    pop_devs = np.array(pop_devs)
    wtimes = np.array(wtimes)
    ar = 100 * accepts / steps
    print(f"  [{name}] done — accept {ar:.1f}%, "
          f"mean step {1e3*wtimes.mean():.2f}ms\n")
    return trace, wtimes, accepts, pop_devs


STEPS = 5_000
MAX_LAG = 200

# ── Step functions ────────────────────────────────────────────────────────────

def plain_step(state):
    return mew_step(state, J_energy, pop_target, EPSILON)

def wilson_step(state):
    return mew_wilson_step(state, J_energy, pop_target, EPSILON)

def make_qp_step(lam):
    def step(state):
        return qp_hybrid_step(state, J_energy, pop_target, EPSILON,
                               k=n_districts, lam_mean=lam, lam_noise=0.3)
    return step

LAM_VALUES = [3, 7, 10, 15, 20]
chains = [("Plain MEW", plain_step), ("MEW+Wilson", wilson_step)]
chains += [(f"QP λ={lam}", make_qp_step(lam)) for lam in LAM_VALUES]

results = {}
for name, sfn in chains:
    random.seed(42); np.random.seed(42)
    print(f"── {name} ({STEPS:,d} steps) " + "─" * 40)
    tr, wt, acc, pd = run_chain(name, sfn, STEPS)
    results[name] = {"trace": tr, "wt": wt, "accepts": acc, "pop_dev": pd}


# ═══════════════════════════════════════════════════════════════════════════════
#  3. DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 92)
print(f"MIXING DIAGNOSTICS — Soft balance + compactness (μ_c={MU_C})   "
      f"({STEPS:,d} steps, {G.graph['name']}, k={n_districts})")
print("=" * 92)

hdr = (f"{'Chain':<16} {'Accept%':>8} {'Mean':>8} {'Std':>8} "
       f"{'τ_int':>8} {'ESS':>8} {'ESS/s':>8} {'ms/step':>8} {'PopDev':>8}")
print(hdr)
print("-" * 92)

diag = {}
for name, _ in chains:
    r = results[name]
    tr = r["trace"]
    wt = r["wt"]
    tau, ess_val = tau_and_ess(tr, MAX_LAG)
    eps_val = ess_per_second(ess_val, wt)
    diag[name] = {"tau": tau, "ess": ess_val, "eps": eps_val}
    print(f"  {name:<14} {100*r['accepts']/STEPS:>7.1f}% "
          f"{tr.mean():>8.1f} {tr.std():>8.2f} "
          f"{tau:>8.2f} {ess_val:>8.1f} {eps_val:>8.2f} "
          f"{1e3*wt.mean():>8.2f} {r['pop_dev'].mean():>8.4f}")

print("=" * 92)


# ═══════════════════════════════════════════════════════════════════════════════
#  4. PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

COLS = {name: f"C{j}" for j, (name, _) in enumerate(chains)}
N_CHAINS = len(chains)

# ── ESS/s vs λ summary plot ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

chain_names = [name for name, _ in chains]
x_pos = range(len(chain_names))

ax = axes[0]
ax.bar(x_pos, [diag[n]["eps"] for n in chain_names],
       color=[COLS[n] for n in chain_names])
ax.set_xticks(x_pos); ax.set_xticklabels(chain_names, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("ESS/s"); ax.set_title("Effective Samples per Second")

ax = axes[1]
ax.bar(x_pos, [diag[n]["tau"] for n in chain_names],
       color=[COLS[n] for n in chain_names])
ax.set_xticks(x_pos); ax.set_xticklabels(chain_names, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("τ_int"); ax.set_title("Integrated Autocorrelation Time")

ax = axes[2]
ax.bar(x_pos, [100*results[n]["accepts"]/STEPS for n in chain_names],
       color=[COLS[n] for n in chain_names])
ax.set_xticks(x_pos); ax.set_xticklabels(chain_names, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Accept %"); ax.set_title("Acceptance Rate")

fig.suptitle(f"QP λ sweep — {G.graph['name']} (k={n_districts}), {STEPS:,d} steps",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "qp_iowa_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved qp_iowa_comparison.png")

# ── Population deviation ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
for name, _ in chains:
    pd = results[name]["pop_dev"]
    ax.plot(np.convolve(pd, np.ones(50)/50, mode='valid'),
            lw=1.0, color=COLS[name], alpha=0.8, label=name)
ax.set_xlabel("Step"); ax.set_ylabel("Mean Pop Deviation (smoothed)")
ax.set_title(f"Population Balance — {G.graph['name']} (k={n_districts})")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "qp_iowa_pop_deviation.png", dpi=150)
plt.close(fig)
print("Saved qp_iowa_pop_deviation.png")

# ── Side-by-side district maps: Plain MEW vs best QP ──────────────────────────
pos = {v: (G.nodes[v]["C_X"], G.nodes[v]["C_Y"]) for v in G}

# Find best QP chain by ESS/s
best_qp_name = max(
    [n for n, _ in chains if n.startswith("QP")],
    key=lambda n: diag[n]["eps"]
)

# Get final assignments by re-running from results traces
# We stored states in run_chain; reconstruct final assignment from last accepted state
# Instead, just re-initialize and run a few steps to get a representative partition
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
cmap = plt.cm.Set3

for ax, name in zip(axes, ["Plain MEW", best_qp_name]):
    # Re-run a short chain to get a final state
    random.seed(42); np.random.seed(42)
    state = initialize_from_partition(G, init_asn)
    sfn = dict(chains)[name]
    for _ in range(STEPS):
        state, _ = sfn(state)
    asn = state.get_assignment()

    colors = [asn[v] for v in G.nodes()]
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, width=0.3)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=15,
                           node_color=colors, cmap=cmap, vmin=0, vmax=n_districts-1)
    cut = n_cut_edges(G, asn)
    dev = mean_pop_deviation(G, asn)
    ax.set_title(f"{name}\ncut={cut}, popdev={dev:.4f}", fontsize=11)
    ax.axis("off")

fig.suptitle(f"Final Partitions — {G.graph['name']} (k={n_districts})", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "qp_district_maps.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved qp_district_maps.png")

print(f"\nAll plots saved to {PLOTS_DIR}/")
