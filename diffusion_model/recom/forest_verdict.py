"""Step 3 verdict for the Forest ReCom kill test, from the stored long runs.

For each (target family, kernel, seed): ESS per second of the weighted-cut
trace for the forest move and the forest hybrid, their ratio, quarter means,
and two checks the run sheet asks for or implies:

  * same level: the post-burn means of the two chains differ by z standard
    errors, each standard error from that chain's own ESS. |z| < 2 is "on
    the same level"; a ratio between chains at different levels is not a
    ratio of anything.
  * timing: the eight long runs shared five performance cores, so each
    pair's per-step cost ratio (hybrid / forest) is compared with the same
    ratio from the Step 2 short run, where both chains ran in one process
    one after the other. A large disagreement means scheduling, not the
    algorithm, moved the ESS-per-second ratio, and the cost-corrected ratio
    (ESS per step ratio times the short run's cost ratio) should be read
    instead.

Run from diffusion_model/: python3 <this file>
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from recom.hybrid import ess_fft

OUT = Path("recom/iowa_hybrid_test_out")
BURN = 0.1


def summary(d, k, n):
    wc = d[f"{k}_wcut"]
    b = int(len(wc) * BURN)
    w = wc[b:]
    q = len(w) // 4
    ess = ess_fft(w)
    wall = float(d[f"{k}_wall"])
    return dict(mean=float(w.mean()), sd=float(w.std()), ess=ess, wall=wall,
                ess_s=ess / wall, sec_step=wall / len(wc),
                ess_step=ess / len(w),
                quarters=[float(w[i * q:(i + 1) * q].mean()) for i in range(4)])


def short_cost_ratio(target, kernel, p="0.1"):
    f = OUT / f"forest_iowa_short_tau0.1_seed42_{target}.npz"
    if not f.exists():
        return np.nan
    d = np.load(f, allow_pickle=True)
    n = int(d["n_steps"])
    tf = float(d[f"{kernel}_forest_wall"]) / n
    th = float(d[f"{kernel}_fhybrid_p{p}_wall"]) / n
    return th / tf


rows = []
print(f"{'family':>9} {'kernel':>9} {'seed':>4} | {'ESS f':>6} {'ESS h':>6} {'ESS/s f':>8} "
      f"{'ESS/s h':>8} {'ratio':>6} | {'mean f':>7} {'mean h':>7} {'z':>6} | "
      f"{'cost h/f long':>13} {'short':>6} {'corr ratio':>10}")
for fam, ptag, p in (("pi", "", "0.1"), ("tree", "", "0.1"), ("pi", "_p0.3", "0.3")):
    label = fam if not ptag else f"{fam} p0.3"
    for kernel in ("perimeter", "density"):
        for seed in (42, 43):
            ff = OUT / f"forest_iowa_long_forest_seed{seed}_{fam}.npz"
            fh = OUT / f"forest_iowa_long_hybrid{ptag}_seed{seed}_{fam}.npz"
            if not (ff.exists() and fh.exists()):
                print(f"{label:>9} {kernel:>9} {seed:>4} | (pending)")
                continue
            df, dh = np.load(ff, allow_pickle=True), np.load(fh, allow_pickle=True)
            sf, sh = summary(df, kernel, int(df["n_steps"])), summary(dh, kernel, int(dh["n_steps"]))
            ratio = sh["ess_s"] / sf["ess_s"]
            se = np.sqrt(sf["sd"] ** 2 / sf["ess"] + sh["sd"] ** 2 / sh["ess"])
            z = (sh["mean"] - sf["mean"]) / se
            cost_long = sh["sec_step"] / sf["sec_step"]
            cost_short = short_cost_ratio(fam, kernel, p)
            corr = (sh["ess_step"] / sf["ess_step"]) / cost_short
            print(f"{label:>9} {kernel:>9} {seed:>4} | {sf['ess']:6.0f} {sh['ess']:6.0f} "
                  f"{sf['ess_s']:8.4f} {sh['ess_s']:8.4f} {ratio:6.2f} | "
                  f"{sf['mean']:7.2f} {sh['mean']:7.2f} {z:+6.2f} | "
                  f"{cost_long:13.2f} {cost_short:6.2f} {corr:10.2f}")
            print(f"{'':>23}   quarters forest {' '.join(f'{v:.2f}' for v in sf['quarters'])}"
                  f"   hybrid {' '.join(f'{v:.2f}' for v in sh['quarters'])}")
            rows.append((label, kernel, seed, sf["ess_s"], sh["ess_s"], ratio, z, corr))


def verdict(name, sel):
    if len(sel) != 4:
        print(f"\n{name}: {len(sel)} of 4 cells available, no verdict yet.")
        return None
    ok = [r[5] >= 1.5 and abs(r[6]) < 2 for r in sel]
    ok_c = [r[7] >= 1.5 and abs(r[6]) < 2 for r in sel]
    v = "PASS" if all(ok) else ("FAIL" if not any(ok) else "IN-BETWEEN")
    print(f"\n{name}: {sum(ok)} of 4 cells meet ratio >= 1.5 with |z| < 2 -> {v}; "
          f"cost-corrected: {sum(ok_c)} of 4.")
    return all(ok)


pre = verdict("Pre-registered criterion (pi, p_local = 0.1)", [r for r in rows if r[0] == "pi"])
p3 = verdict("p_local = 0.3 (added after the short runs)", [r for r in rows if r[0] == "pi p0.3"])
if pre is not None and p3 is not None:
    print(f"Amended criterion (either p clears it): {'PASS' if (pre or p3) else 'not passed'}. "
          f"Report the pre-registered verdict first; the amendment was made after seeing short runs.")
