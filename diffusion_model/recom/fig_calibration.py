"""The figure the paper is missing: what the ReCom family actually samples.

Section 5.1 makes the paper's strongest claim and carries no figure at all.
This builds it, in three panels, from calibration_distance_seed<SEED>.npz.

  (a) Chain A's cut-count marginal against the three exactly computable
      reference measures. If the calibration claim holds, the bars sit on
      the tree-weighted-and-tempered curve and well clear of the nominal
      one. This is the panel a referee checks by eye.

  (b) Total variation of the running cut marginal against each reference,
      as a function of run length, with the sampling-noise floor shaded.
      This is the panel that distinguishes the paper's actual claim from
      the alternative reading. Convergence to the wrong distribution looks
      like one curve descending into the floor while another flattens onto
      a positive plateau. Slow mixing would look like both still falling.

  (c) Chain E's cut-count marginal against the nominal target, which is
      the distribution Chain E is proved reversible for. This is the
      end-to-end check of the detailed-balance construction.

Design notes, because journal figures get printed and photocopied.
Series colors are Okabe-Ito hues, validated for colorblind separation on
all pairs (worst pair dE 11.0 deutan, 8.6 tritan, 18.7 normal vision).
Every series also carries a distinct dash pattern, so identity survives
grayscale printing and never rests on color alone. The empirical data is
a neutral filled histogram, deliberately a different mark type from the
reference curves so the eye separates measurement from theory.

The noise floor is analytic, not bootstrapped: for n independent draws,
E|q_i - p_i| = sqrt(2 p_i (1 - p_i) / (pi n)) to leading order, so
E[TV] ~ 0.5 * sum_i sqrt(2 p_i (1-p_i) / (pi n)). It is evaluated at the
EFFECTIVE sample size, t * ESS / N, not the raw step count, which is the
conservative choice. A chain sitting inside that band is indistinguishable
from an exact sampler for this functional.

Reads:  iowa_hybrid_test_out/calibration_distance_seed<SEED>.npz
Writes: Papers/fig/calibration_5x5.{pdf,png}   (both new filenames)

Run calibration_distance.py first.

Usage: python3 recom/fig_calibration.py [seed]
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = HERE / "iowa_hybrid_test_out"
FIG = REPO / "Papers" / "fig"

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42

# Okabe-Ito, validated: all adjacent and all-pairs CVD checks pass.
REFS = [
    ("tree_temp", "tree-weighted + tempered", "#009E73", "-"),
    ("tree",      "tree-weighted",            "#0072B2", "--"),
    ("nominal",   r"nominal $\pi \propto e^{-\lambda|\partial P|}$",
                                              "#D55E00", "-."),
]
EMP_FILL = "#D9D9D6"
EMP_EDGE = "#4D4D4D"
INK = "#1A1A1A"
MUTED = "#767676"
ANNOT = "#4D4D4D"   # darker than MUTED so the in-panel numbers survive print


def tv_floor(p, n):
    """Leading-order E[TV] between p and the empirical measure of n
    independent draws from p."""
    n = np.maximum(np.asarray(n, dtype=float), 1.0)
    term = np.sqrt(2.0 * p * (1.0 - p) / np.pi)
    return 0.5 * term.sum() / np.sqrt(n)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3, width=0.8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)
    ax.grid(True, color="#E6E6E3", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def panel_marginal(ax, vals, emp, d, tag, refs_to_draw, title):
    ax.bar(vals, emp, width=0.82, color=EMP_FILL, edgecolor=EMP_EDGE,
           linewidth=0.7, zorder=2, label=f"Chain {tag} (empirical)")
    peak = float(emp.max())
    for key, label, color, dash in REFS:
        if key not in refs_to_draw:
            continue
        m = d[f"ref_{key}_marginal"]
        peak = max(peak, float(np.max(m)))
        ax.step(vals, m, where="mid", color=color,
                linestyle=dash, linewidth=2.0, zorder=3, label=label)
    # Headroom so the legend and the annotation never sit on the data.
    ax.set_ylim(0, peak * 1.55)
    ax.set_xlabel("cut edges", fontsize=9.5, color=INK)
    ax.set_ylabel("probability", fontsize=9.5, color=INK)
    ax.set_title(title, fontsize=10.5, color=INK, pad=8, loc="left")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper right",
              labelcolor=INK, handlelength=2.4)


def main():
    src = OUT / f"calibration_distance_seed{SEED}.npz"
    if not src.exists():
        raise SystemExit(f"missing {src.name}; run calibration_distance.py first")
    d = np.load(src)
    vals = d["cut_values"]
    FIG.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
    fig.patch.set_facecolor("white")

    # (a) Chain A against all three references.
    panel_marginal(
        axes[0], vals, d["A_emp_marginal"], d, "A",
        {"tree_temp", "tree", "nominal"},
        "(a) Tempered ReCom misses its nominal target")
    axes[0].annotate(
        f"mean {float(d['A_mean']):.2f}\n"
        f"tree+tempered {float(d['ref_tree_temp_mean']):.2f}\n"
        f"nominal {float(d['ref_nominal_mean']):.2f}",
        xy=(0.03, 0.97), xycoords="axes fraction", va="top", ha="left",
        fontsize=8, color=ANNOT, linespacing=1.5)

    # (b) TV against run length, with the noise floor.
    ax = axes[1]
    grid = d["A_tv_grid"].astype(float)
    # Fraction of retained draws that count as independent, so the floor is
    # drawn at the effective sample size rather than the raw step count.
    ess_ratio = float(d["A_ess"]) / grid[-1]
    floor = tv_floor(d["ref_tree_temp_marginal"], grid * ess_ratio)
    ax.fill_between(grid, 0, floor, color="#E6E6E3", zorder=1,
                    label="sampling-noise floor (at ESS)")
    for key, label, color, dash in REFS:
        y = np.maximum(d[f"A_tv_traj_{key}"], 1e-6)  # keep the log axis safe
        ax.plot(grid, y, color=color, linestyle=dash,
                linewidth=2.0, zorder=3, label=f"TV to {label}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("steps used (post burn-in)", fontsize=9.5, color=INK)
    ax.set_ylabel("TV, cut-count marginal", fontsize=9.5, color=INK)
    ax.set_title("(b) One distance plateaus, the other does not",
                 fontsize=10.5, color=INK, pad=8, loc="left")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK,
              handlelength=2.4)

    # (c) Chain E against the target it is proved reversible for.
    panel_marginal(
        axes[2], vals, d["E_emp_marginal"], d, "E", {"nominal"},
        "(c) The exact chain against its own target")
    axes[2].annotate(
        f"mean {float(d['E_mean']):.2f}\n"
        f"target {float(d['ref_nominal_mean']):.2f}\n"
        f"coverage {int(d['E_coverage'])}/{int(d['n_fiber'])}",
        xy=(0.03, 0.97), xycoords="axes fraction", va="top", ha="left",
        fontsize=8, color=ANNOT, linespacing=1.5)

    fig.tight_layout(pad=1.1)
    for ext in ("pdf", "png"):
        dest = FIG / f"calibration_5x5.{ext}"
        fig.savefig(dest, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"wrote {dest}")
    plt.close(fig)

    print("\nEYEBALL IT BEFORE USING IT. Check for label collisions, that "
          "panel (b)'s log-log axes do not clip a curve that touches zero, "
          "and that the legends do not cover data. The validator checked "
          "color, not layout.")
    print("\nIf panel (b) shows BOTH curves still descending, do not claim "
          "convergence to the wrong distribution. That picture is slow "
          "mixing, and the honest move is a longer run.")


if __name__ == "__main__":
    main()
