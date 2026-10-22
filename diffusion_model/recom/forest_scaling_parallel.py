"""Run forest_scaling.py's experiment in parallel without changing its results.

forest_scaling.py runs, for every grid size N and target, the forest move
alone and then the hybrid, serially, about 6 to 11 hours in total. Each
chain reseeds both random generators at its start (run_forest_hybrid) and
builds its own tau cache, so a chain's trajectory depends only on
(N, target, chain, seed), not on what ran before it. This driver runs the
same chains as separate processes and merges them into exactly the
artifact the serial command writes.

Why pairs. Panel (c) of the figure is a ratio of ESS per second, so the
two chains it compares must see the same machine. The serial script runs
the forest chain, then the hybrid; split across processes that order
would give the hybrid a quieter machine whenever other jobs finish during
its run. Here every forest chain runs concurrently with its hybrid
partner, so both members of a pair always share the same contention.

Everything scientific (grid, target, step count, seed, p_local, tau,
burn-in, summary statistics, file keys) is imported from forest_scaling,
with sys.argv set to the run sheet's arguments before import.

Usage (from diffusion_model/):
    python3 recom/forest_scaling_parallel.py schedule [max_pairs]   # run all, then merge
    python3 recom/forest_scaling_parallel.py chain N fam:lam forest|hybrid
    python3 recom/forest_scaling_parallel.py merge
then draw the figure with the run sheet's own command plus --fig.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

RUN_SHEET_ARGS = ["50000", "42", "6,8,10,12,16,20", "pi:1,pi:2,tree:0.1,tree:1", "0.3"]
MODE = sys.argv[1] if len(sys.argv) > 1 else "schedule"
REST = sys.argv[2:]
sys.argv = ["forest_scaling.py"] + RUN_SHEET_ARGS        # forest_scaling parses argv at import

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import recom.forest_scaling as fs                         # noqa: E402

# Optional overrides, read from the environment so that the chain
# subprocesses inherit them. SCALING_TAU is the run sheet's "rerun with
# tau = 0.03" follow-up; SCALING_TARGETS restricts the targets. Any
# override writes to its own parts folder and artifact, never the main one.
_suffix = ""
if os.environ.get("SCALING_TAU"):
    fs.TAU = float(os.environ["SCALING_TAU"])
    _suffix += f"_tau{fs.TAU:g}"
if os.environ.get("SCALING_TARGETS"):
    fs.TARGETS = [(t.split(":")[0], float(t.split(":")[1]))
                  for t in os.environ["SCALING_TARGETS"].split(",")]
    _suffix += "_" + "_".join(f"{f}{l:g}" for f, l in fs.TARGETS)
if _suffix:
    fs.OUT = fs.OUT.with_name(fs.OUT.stem + _suffix + ".npz")
PARTS = fs.OUT.parent / f"{fs.OUT.stem}_parts"


def part_path(N, fam, lam, cname):
    return PARTS / f"{fs.tag_of(N, fam, lam, cname)}.npz"


def run_chain(N, fam, lam, cname):
    G, init, ideal = fs.grid_problem(N)
    steps = fs.steps_for(N)
    tgt = fs.make_target(fam, G, fs.kernel_uniform, lam)
    pl = 0.0 if cname == "forest" else fs.P_LOCAL
    res = fs.run_forest_hybrid(tgt, init, ideal, fs.EPS, steps, pl, fs.TAU, 1.0, fs.SEED,
                               label=f"N={N} {fam}:{lam:g} {cname}")
    tag = fs.tag_of(N, fam, lam, cname)
    store = {f"{tag}_{key}": res[key] for key in fs.FOREST_KEYS}
    store[f"{tag}_wall"] = res["wall"]
    store[f"{tag}_steps"] = steps
    PARTS.mkdir(parents=True, exist_ok=True)
    np.savez(part_path(N, fam, lam, cname), **store)
    print(f"saved {part_path(N, fam, lam, cname).name}  wall {res['wall']:.0f}s", flush=True)


def schedule(max_pairs):
    pairs = [(N, fam, lam) for N in sorted(fs.N_LIST, reverse=True) for fam, lam in fs.TARGETS
             if not (part_path(N, fam, lam, "forest").exists()
                     and part_path(N, fam, lam, "hybrid").exists())]
    PARTS.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1")
    active = []
    print(f"{len(pairs)} pairs to run, up to {max_pairs} at a time", flush=True)
    while pairs or active:
        while pairs and len(active) < max_pairs:
            N, fam, lam = pairs.pop(0)
            procs = []
            for cname in ("forest", "hybrid"):
                log = open(PARTS / f"{fs.tag_of(N, fam, lam, cname)}.log", "w")
                procs.append(subprocess.Popen(
                    ["nice", "-n", "19", sys.executable, __file__, "chain", str(N), f"{fam}:{lam:g}", cname],
                    stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(_HERE.parent)))
            active.append(((N, fam, lam), procs))
            print(f"{time.strftime('%H:%M:%S')} start  N={N} {fam}:{lam:g}", flush=True)
        time.sleep(5)
        still = []
        for key, procs in active:
            if all(p.poll() is not None for p in procs):
                codes = [p.returncode for p in procs]
                print(f"{time.strftime('%H:%M:%S')} done   N={key[0]} {key[1]}:{key[2]:g}  exit {codes}",
                      flush=True)
                if any(codes):
                    raise SystemExit(f"chain failed for {key}; see logs in {PARTS}")
            else:
                still.append((key, procs))
        active = still
    merge()


def merge():
    store = {"n_steps": fs.N_STEPS, "seed": fs.SEED, "eps": fs.EPS, "p_local": fs.P_LOCAL,
             "tau": fs.TAU, "N_list": np.array(fs.N_LIST),
             "targets": np.array([f"{f}:{l:g}" for f, l in fs.TARGETS])}
    missing = []
    for N in fs.N_LIST:
        for fam, lam in fs.TARGETS:
            for cname in ("forest", "hybrid"):
                f = part_path(N, fam, lam, cname)
                if not f.exists():
                    missing.append(f.name)
                    continue
                with np.load(f, allow_pickle=True) as z:
                    store.update({k: z[k] for k in z.files})
    if missing:
        raise SystemExit(f"missing parts: {missing}")

    # The table exactly as forest_scaling.run() prints it.
    tl = " ".join(f"{f}:{l:g}" for f, l in fs.TARGETS)
    print(f"grids k=2, eps={fs.EPS}, p_local={fs.P_LOCAL}, tau={fs.TAU}, seed {fs.SEED}, "
          f"{fs.N_STEPS} steps (x2 for N >= 16), targets {tl}\n")
    print(f"{'N':>3} {'target':>8} {'chain':>6} {'P(feas)':>7} {'acc f':>6} {'acc l':>6} "
          f"{'log tau-ratio':>14} {'log pi-ratio':>14} {'cut':>6} {'quarter means':>28} "
          f"{'ESS':>6} {'wall s':>7} {'ESS/s':>7} {'ratio':>6}")
    timing = []
    for N in fs.N_LIST:
        for fam, lam in fs.TARGETS:
            base = None
            per = {}
            for cname in ("forest", "hybrid"):
                tag = fs.tag_of(N, fam, lam, cname)
                res = {k: store[f"{tag}_{k}"] for k in fs.FOREST_KEYS}
                res["wall"] = float(store[f"{tag}_wall"])
                res["fast"] = np.zeros(len(res["wcut"]), dtype=bool)
                s = fs.chain_summary(res, fs.BURN)
                fstat = fs.forest_stats(res, fs.BURN)
                if base is None:
                    base = s["ess_s"]
                qm = " ".join(f"{v:6.2f}" for v in s["quarters"])
                print(f"{N:>3} {fam + ':' + format(lam, 'g'):>8} {cname:>6} {fstat['p_feasible']:7.3f} "
                      f"{s['acc_global']:6.3f} {s['acc_local']:6.3f} "
                      f"{fstat['log_tau_ratio']:+7.2f} ± {fstat['log_tau_ratio_se']:4.2f} "
                      f"{fstat['log_pi_ratio']:+7.2f} ± {fstat['log_pi_ratio_se']:4.2f} "
                      f"{fstat['ncut_mean']:6.1f} {qm:>28} {s['ess']:6.0f} {s['wall']:7.1f} "
                      f"{s['ess_s']:7.2f} {s['ess_s'] / base:6.2f}", flush=True)
                n = len(res["wcut"])
                per[cname] = (s["ess"] / (n - int(n * fs.BURN)), s["wall"] / n)
            timing.append((N, f"{fam}:{lam:g}", per))
        print(flush=True)
    np.savez_compressed(fs.OUT, **store)      # same arrays; uncompressed is over GitHub's 100 MiB limit
    print(f"Saved {fs.OUT.name}")

    print("\nTiming check (pairs ran concurrently): ESS per retained step and seconds per step.")
    print(f"{'N':>3} {'target':>8} {'ESS/step f':>10} {'ESS/step h':>10} {'per-step ratio':>14} "
          f"{'s/step f':>9} {'s/step h':>9} {'cost h/f':>8}")
    for N, t, per in timing:
        (ef, cf), (eh, ch) = per["forest"], per["hybrid"]
        print(f"{N:>3} {t:>8} {ef:10.4f} {eh:10.4f} {eh / ef:14.2f} {cf * 1e3:8.2f}m {ch * 1e3:8.2f}m "
              f"{ch / cf:8.2f}")


if __name__ == "__main__":
    if MODE == "chain":
        N, tgt, cname = int(REST[0]), REST[1], REST[2]
        fam, lam = tgt.split(":")
        run_chain(N, fam, float(lam), cname)
    elif MODE == "merge":
        merge()
    elif MODE == "schedule":
        schedule(int(REST[0]) if REST else 5)
    else:
        raise SystemExit(__doc__)
