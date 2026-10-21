"""Figure for the move-size law: the diffusion connection, made load-bearing.

Three panels, from spectral_move_size.npz.

  (a,b) Expected units flipped per proposal against the diffusion timestep
        tau = alpha/beta, on the enumerated 5x5 fiber and on Iowa. Two
        curves per panel: the exact filter norm ||(I+tau L)^-1 tau L x0||_1,
        which IS the expectation, and the small-tau law 2*tau*|boundary|.
        They agree where the theory says they should and part company where
        the high-pass gain saturates. The paper's own locked Chain E
        setting, tau = 0.1, is marked, because that is where the argument
        lands: the chain is sticky by construction, not by accident.

  (c)   Fraction of merged pairs whose population-balance constraint is
        active. On the uniform grid it is never active, so the proposal is
        a pure diffusion step and a plain linear solve. On real county
        geography it takes over well before tau reaches 1, which is the
        honest limit on how far this dial can be pushed.

The saturation in (a) and (b) is the point most worth making. Move size is
bounded in tau, so mobility cannot be bought by raising alpha/beta. That is
a proof-shaped reason the reversible hybrid is necessary rather than merely
desirable, and it is a much stronger argument than observing that Chain E
happens to be slow.

Reads:  iowa_hybrid_test_out/spectral_move_size.npz
Writes: Papers/fig/spectral_move_size.{pdf,png}

Usage: python3 recom/fig_spectral.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = HERE / "iowa_hybrid_test_out"
FIG = REPO / "Papers" / "fig"

EXACT = "#0072B2"      # validated Okabe-Ito pair, distinct under all CVD types
APPROX = "#D55E00"
INK = "#1A1A1A"
MUTED = "#767676"
ANNOT = "#4D4D4D"
TAU_E = 0.1            # the paper's locked Chain E setting, alpha=1, beta=10


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


def panel_movesize(ax, s, title, note):
    tau, ex, ap = s[:, 0], s[:, 2], s[:, 3]
    ax.plot(tau, ap, color=APPROX, linestyle="--", linewidth=2.0,
            marker="s", markersize=5, zorder=3,
            label=r"small-$\tau$ law  $2\tau|\partial|$")
    ax.plot(tau, ex, color=EXACT, linestyle="-", linewidth=2.0,
            marker="o", markersize=5, zorder=4,
            label=r"exact  $\|(I+\tau L)^{-1}\tau L x_0\|_1$")
    ax.axvline(TAU_E, color=ANNOT, linewidth=1.0, linestyle=":", zorder=2)
    j = int(np.argmin(np.abs(tau - TAU_E)))
    ax.annotate(f"Chain E\n$\\tau=0.1$\n{ex[j]:.2f} units",
                xy=(TAU_E, ex[j]), xytext=(TAU_E * 1.5, ex[j] * 0.22),
                fontsize=8, color=ANNOT, linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=ANNOT, linewidth=0.8))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"diffusion timestep  $\tau=\alpha/\beta$",
                  fontsize=9.5, color=INK)
    ax.set_ylabel("expected units flipped", fontsize=9.5, color=INK)
    ax.set_title(title, fontsize=10.5, color=INK, pad=8, loc="left")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK,
              handlelength=2.6)
    ax.annotate(note, xy=(0.97, 0.04), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=8, color=ANNOT)


def main():
    src = OUT / "spectral_move_size.npz"
    if not src.exists():
        raise SystemExit("missing spectral_move_size.npz; "
                         "run spectral_move_size.py first")
    d = np.load(src)
    FIG.mkdir(parents=True, exist_ok=True)

    have = [m for m in ("5x5", "iowa") if f"{m}_summary" in d]
    fig, axes = plt.subplots(1, len(have) + 1, figsize=(4.5 * (len(have) + 1),
                                                        3.9))
    fig.patch.set_facecolor("white")

    titles = {"5x5": "(a) 5$\\times$5 grid, balance never binds",
              "iowa": "(b) Iowa counties, real geography"}
    for i, m in enumerate(have):
        s = d[f"{m}_summary"]
        sat = s[-1, 2] / s[0, 2]
        tratio = s[-1, 0] / s[0, 0]
        panel_movesize(
            axes[i], s, titles.get(m, m),
            f"$\\tau$ up {tratio:.0f}$\\times$, moves up {sat:.0f}$\\times$")

    ax = axes[-1]
    for m, color, dash in (("5x5", EXACT, "-"), ("iowa", APPROX, "--")):
        if f"{m}_summary" not in d:
            continue
        s = d[f"{m}_summary"]
        ax.plot(s[:, 0], 100 * s[:, 1], color=color, linestyle=dash,
                linewidth=2.0, marker="o", markersize=5, zorder=3,
                label="5$\\times$5 grid" if m == "5x5" else "Iowa counties")
    ax.axvline(TAU_E, color=ANNOT, linewidth=1.0, linestyle=":", zorder=2)
    ax.set_xscale("log")
    ax.set_ylim(-3, 103)
    ax.set_xlabel(r"diffusion timestep  $\tau=\alpha/\beta$",
                  fontsize=9.5, color=INK)
    ax.set_ylabel("merged pairs with balance active (%)",
                  fontsize=9.5, color=INK)
    ax.set_title("(c) What actually limits the move", fontsize=10.5,
                 color=INK, pad=8, loc="left")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK,
              handlelength=2.6)

    fig.tight_layout(pad=1.1)
    for ext in ("pdf", "png"):
        dest = FIG / f"spectral_move_size.{ext}"
        fig.savefig(dest, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"wrote {dest}")
    plt.close(fig)


if __name__ == "__main__":
    main()
