"""Ten-seed replication statistics from target_matched_*_seed*.npz.

For each mode (5x5, iowa) and each chain vs A: per-seed post-burn-in
means, paired differences, percent change, sign counts, and a paired
t-test across seeds. These are the between-seed numbers quoted in the
paper's statistical protocol paragraph.

Usage: python3 recom/ten_seed_stats.py
"""

from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

HERE = Path(__file__).resolve().parent
OUT = HERE / "iowa_hybrid_test_out"
SEEDS = list(range(42, 52))
BURN = 100


def main():
    for mode in ("5x5", "iowa"):
        print(f"\n===== {mode} : seeds {SEEDS[0]}-{SEEDS[-1]} =====")
        means = {ch: [] for ch in "VABCDE"}
        for s in SEEDS:
            f = OUT / f"target_matched_{mode}_seed{s}.npz"
            if not f.exists():
                print(f"  MISSING {f.name}")
                continue
            d = np.load(f, allow_pickle=True)
            for ch in "VABCDE":
                means[ch].append(float(d[ch][BURN:].mean()))
        a = np.array(means["A"])
        n = len(a)
        print(f"  n = {n} seeds")
        for ch in "VABCDE":
            m = np.array(means[ch])
            print(f"  {ch}: mean-of-means = {m.mean():.2f} +- {m.std():.2f}"
                  f"  (per-seed: {np.round(m, 2)})")
        for ch in "BCD":
            m = np.array(means[ch])
            diff = m - a
            t, p = ttest_rel(m, a)
            print(f"  {ch}-A: {diff.mean():+.1f} +- {diff.std():.1f} cuts "
                  f"({100 * diff.mean() / a.mean():+.1f}%), "
                  f"negative on {int((diff < 0).sum())}/{n} seeds, "
                  f"paired t p = {p:.2e}")


if __name__ == "__main__":
    main()
