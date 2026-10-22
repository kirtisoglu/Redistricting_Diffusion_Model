"""The precinct-scale run: North Carolina VTDs against the Forest ReCom baseline.

Section 5.7 of the paper. Same question as the Iowa kill test
(forest_iowa.py), on the 2692-precinct North Carolina dual graph: does the
lifted diffusion local move raise the effective samples per second (and per
step) of the exact sampler whose global move is the forest-lifted tree move,
under the perimeter and density targets? Section 5.5 predicts that the
forest move's acceptance collapses at precinct-scale pairs under a tempered
target, so the expected outcome is not a factor of two but a chain that
reaches equilibrium where the forest move alone does not.

Setup. K = 13 districts (the congressional count; pairs hold about 410
precincts), eps = 0.05, global balance window. lam per kernel is scaled as
on Iowa so that lam * w(boundary) at the start plan equals 0.10 * |boundary|.
The timestep is set by the move-size rule of Corollary 4.3,
    tau = m / (2 * mean_pairs |boundary_w(V_i, V_j)|),
so that a local proposal moves about m precincts in expectation; m = 2 by
default. The initial plan is a balanced recursive tree partition drawn from
the run's seed, so the forest chain and the hybrid of one seed share a start
and the two seeds have different starts.

Three modes.
  probe    short timing run (default 200 steps of the forest move alone and
           200 of the lifted local move alone, per kernel): ms per step,
           acceptance, Pr[balanced tree], the two Hastings terms of the
           forest move, the local move's acceptance, flips and linear-solve
           share, and the projected wall time of a full run.
  run      one chain per invocation so runs can be parallelised:
             python3 recom/forest_nc.py run forest --seed 42 --target pi --kernel perimeter
             python3 recom/forest_nc.py run hybrid --seed 42 --target pi --kernel perimeter
           Writes forest_nc_k{K}_{forest|hybrid}[_p{p}]_m{m}_seed{SEED}_{target}_{kernel}.npz
  verdict  reads the stored runs and prints, per (kernel, seed): forest
           acceptance, ESS, ESS per second and per step for both chains,
           the ratio, the level check (z of the post-burn means), quarter
           means, and the mean log tree-count term of the forest move.

Criterion (fixed before the runs, same as Iowa): ESS/s(hybrid) / ESS/s(forest)
>= 1.5 under both kernels on both seeds with |z| < 2, read beside the
per-step ratio. If the forest move alone does not converge (quarter means
drifting, ESS below 50) the ratio is not a ratio of anything and the report
is the convergence contrast itself.

Usage:
  python3 recom/forest_nc.py probe   [--k 13] [--m 2] [--steps 200] [--seed 42] [--kernel both]
  python3 recom/forest_nc.py run forest|hybrid [--n 200000] [--seed 42] [--target pi|tree]
                             [--p-local 0.1] [--m 2] [--k 13] [--kernel perimeter|density|both]
  python3 recom/forest_nc.py verdict [--k 13] [--m 2] [--target pi] [--p-local 0.1]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_DIFF = _HERE.parent
_REPO = _DIFF.parent
if str(_DIFF) not in sys.path:
    sys.path.insert(0, str(_DIFF))

from recom.problems import setup_problem
from recom.qp_model import kernel_uniform, kernel_perimeter, make_kernel_density
from recom.hybrid import Target, ess_fft
from recom.forest_recom import run_forest_hybrid, make_target
from recom.kill_stats import chain_summary

BURN = 0.1
OUT_DIR = _HERE / "iowa_hybrid_test_out"
KERNELS = {"perimeter": kernel_perimeter, "density": make_kernel_density(sigma=1.0)}


def lam_for(graph, init, kfn):
    """lam so that lam * w(boundary) at the start equals 0.10 * |boundary|."""
    n0 = Target(graph, kernel_uniform, 0.10).ncut(init)
    w0 = Target(graph, kfn, 1.0).wcut(init)
    return 0.10 * n0 / w0


def pair_boundaries(graph, asn, kfn):
    """Kernel-weighted boundary of every adjacent district pair."""
    out = {}
    for u, v in graph.edges():
        a, b = asn[u], asn[v]
        if a != b:
            key = (a, b) if a < b else (b, a)
            out[key] = out.get(key, 0.0) + kfn(graph, u, v)
    return out


def tau_rule(graph, asn, kfn, m):
    """Move-size rule: tau = m / (2 * mean weighted pair boundary)."""
    pb = pair_boundaries(graph, asn, kfn)
    mean_b = float(np.mean(list(pb.values())))
    return m / (2.0 * mean_b), mean_b, len(pb)


def kernels_of(arg):
    return list(KERNELS.items()) if arg == "both" else [(arg, KERNELS[arg])]


def pair_sizes(asn):
    sizes = {}
    for v, d in asn.items():
        sizes[d] = sizes.get(d, 0) + 1
    return sizes


def probe(a):
    p = setup_problem("nc", _REPO, K=a.k, seed=a.seed)
    graph, eps, ideal, init = p["graph"], p["eps"], p["ideal"], p["init"]
    sizes = pair_sizes(init)
    print(f"{p['label']}; districts of {min(sizes.values())}-{max(sizes.values())} precincts; "
          f"seed {a.seed}, m = {a.m}, {a.steps} steps per probe\n")
    for kname, kfn in kernels_of(a.kernel):
        lam = lam_for(graph, init, kfn)
        tau, mean_b, n_pairs = tau_rule(graph, init, kfn, a.m)
        tgt = make_target(a.target, graph, kfn, lam)
        print(f"[{kname}] lam = {lam:.4f}, {n_pairs} boundary pairs, mean weighted pair "
              f"boundary {mean_b:.2f}, tau = {tau:.5f}")
        rf = run_forest_hybrid(tgt, init, ideal, eps, a.steps, 0.0, tau, 1.0, a.seed,
                               label=f"{kname} forest")
        feas = rf["feasible"]
        ms = 1000 * rf["wall"] / a.steps
        print(f"  forest: {ms:7.1f} ms/step, Pr[balanced tree in <=20 draws] {feas.mean():.3f}, "
              f"mean draws {rf['draws'].mean():.1f}, acc {rf['accepted'].mean():.4f}, "
              f"acc|feasible {rf['accepted'][feas].mean() if feas.any() else float('nan'):.4f}")
        if feas.any():
            print(f"          mean log tree-count term {np.nanmean(rf['log_tau_ratio'][feas]):+.2f}, "
                  f"mean log target term {np.nanmean(rf['log_pi_ratio'][feas]):+.2f}, "
                  f"share tau-term < e^-2.3: {(rf['log_tau_ratio'][feas] < -2.3).mean():.2f}")
        print(f"          projected wall for 2e5 steps: {ms * 2e5 / 3600 / 1000:.1f} h")
        rl = run_forest_hybrid(tgt, init, ideal, eps, a.steps, 1.0, tau, 1.0, a.seed,
                               label=f"{kname} local")
        msl = 1000 * rl["wall"] / a.steps
        acc = rl["accepted"]
        print(f"  local:  {msl:7.1f} ms/step, feasible {rl['feasible'].mean():.3f}, acc {acc.mean():.3f}, "
              f"flips per proposal {rl['flips'].mean():.2f}, per accepted step "
              f"{(rl['flips'] * acc).mean():.2f}, linear-solve share {rl['fast'].mean():.2f}")
        print(f"          hybrid at p_local 0.1 costs {0.9 * ms + 0.1 * msl:.1f} ms/step, "
              f"ratio to forest {(0.9 * ms + 0.1 * msl) / ms:.2f}\n")


def run(a):
    p = setup_problem("nc", _REPO, K=a.k, seed=a.seed)
    graph, eps, ideal, init = p["graph"], p["eps"], p["ideal"], p["init"]
    p_local = 0.0 if a.which == "forest" else a.p_local
    store = {"n_steps": a.n, "seed": a.seed, "p_local": p_local, "m": a.m, "K": a.k,
             "target": a.target, "mode": "run", "init": np.array(sorted(init.items()))}
    for kname, kfn in kernels_of(a.kernel):
        lam = lam_for(graph, init, kfn)
        tau, mean_b, n_pairs = tau_rule(graph, init, kfn, a.m)
        tgt = make_target(a.target, graph, kfn, lam)
        print(f"[{kname}] {a.which} p_local={p_local:g} K={a.k} lam={lam:.4f} tau={tau:.5f} "
              f"(mean pair boundary {mean_b:.2f}, {n_pairs} pairs) {a.n} steps seed {a.seed} "
              f"target {a.target}", flush=True)
        res = run_forest_hybrid(tgt, init, ideal, eps, a.n, p_local, tau, 1.0, a.seed,
                                log_every=max(a.n // 20, 1), label=f"{kname} {a.which}")
        s = chain_summary(res, BURN)
        qm = " ".join(f"{v:.2f}" for v in s["quarters"])
        print(f"{kname:>10} {a.which}: acc forest {s['acc_global']:.4f} acc local {s['acc_local']:.3f} "
              f"flips/local {s['mv_local']:.2f} fast {s['fast']:.2f} quarter means {qm}  "
              f"ESS {s['ess']:.0f}  wall {s['wall']:.0f}s  ESS/s {s['ess_s']:.5f}", flush=True)
        for key in ("wcut", "ncut", "kind", "feasible", "accepted", "fast", "flips",
                    "draws", "log_tau_ratio", "log_pi_ratio"):
            store[f"{kname}_{key}"] = res[key]
        store[f"{kname}_wall"] = res["wall"]
        store[f"{kname}_lam"] = lam
        store[f"{kname}_tau"] = tau
        store[f"{kname}_final"] = np.array(sorted(res["final"].items()))
    ptag = "" if p_local in (0.0, 0.1) else f"_p{p_local:g}"
    ktag = "" if a.kernel == "both" else f"_{a.kernel}"
    out = OUT_DIR / f"forest_nc_k{a.k}_{a.which}{ptag}_m{a.m:g}_seed{a.seed}_{a.target}{ktag}.npz"
    np.savez_compressed(out, **store)
    print(f"Saved {out.name}")


def _find(a, which, seed, kernel):
    ptag = "" if a.p_local == 0.1 or which == "forest" else f"_p{a.p_local:g}"
    for ktag in (f"_{kernel}", ""):
        f = OUT_DIR / f"forest_nc_k{a.k}_{which}{ptag}_m{a.m:g}_seed{seed}_{a.target}{ktag}.npz"
        if f.exists():
            d = np.load(f, allow_pickle=True)
            if f"{kernel}_wcut" in d:
                return d
    return None


def _summary(d, kernel):
    wc = d[f"{kernel}_wcut"]
    b = int(len(wc) * BURN)
    w = wc[b:]
    q = len(w) // 4
    k = d[f"{kernel}_kind"][b:]
    acc = d[f"{kernel}_accepted"][b:]
    feas = d[f"{kernel}_feasible"][b:]
    k0 = k == 0
    ltr = d[f"{kernel}_log_tau_ratio"][b:][k0 & feas]
    ess = ess_fft(w)
    wall = float(d[f"{kernel}_wall"])
    return dict(mean=float(w.mean()), sd=float(w.std()), ess=ess, wall=wall,
                ess_s=ess / wall, ess_step=ess / len(w), sec_step=wall / len(wc),
                acc_f=float(acc[k0].mean()) if k0.any() else np.nan,
                feas_f=float(feas[k0].mean()) if k0.any() else np.nan,
                tau_term=float(np.nanmean(ltr)) if len(ltr) else np.nan,
                acc_l=float(acc[~k0].mean()) if (~k0).any() else np.nan,
                quarters=[float(w[i * q:(i + 1) * q].mean()) for i in range(4)])


def verdict(a):
    print(f"NC K={a.k}, m={a.m:g}, target {a.target}, p_local {a.p_local:g}\n")
    print(f"{'kernel':>9} {'seed':>4} | {'acc F':>7} {'feas F':>6} {'tau term':>8} | "
          f"{'ESS f':>6} {'ESS h':>6} {'ESS/s f':>9} {'ESS/s h':>9} {'ratio/s':>7} {'ratio/step':>10} | "
          f"{'mean f':>8} {'mean h':>8} {'z':>6}")
    rows = []
    for kernel in ("perimeter", "density"):
        for seed in (42, 43):
            df, dh = _find(a, "forest", seed, kernel), _find(a, "hybrid", seed, kernel)
            if df is None or dh is None:
                print(f"{kernel:>9} {seed:>4} | (pending: {'forest ' if df is None else ''}"
                      f"{'hybrid' if dh is None else ''})")
                continue
            sf, sh = _summary(df, kernel), _summary(dh, kernel)
            se = np.sqrt(sf["sd"] ** 2 / sf["ess"] + sh["sd"] ** 2 / sh["ess"])
            z = (sh["mean"] - sf["mean"]) / se
            r_s = sh["ess_s"] / sf["ess_s"]
            r_step = sh["ess_step"] / sf["ess_step"]
            print(f"{kernel:>9} {seed:>4} | {sf['acc_f']:7.4f} {sf['feas_f']:6.3f} {sf['tau_term']:+8.2f} | "
                  f"{sf['ess']:6.0f} {sh['ess']:6.0f} {sf['ess_s']:9.5f} {sh['ess_s']:9.5f} "
                  f"{r_s:7.2f} {r_step:10.2f} | {sf['mean']:8.2f} {sh['mean']:8.2f} {z:+6.2f}")
            print(f"{'':>16} quarters forest {' '.join(f'{v:.2f}' for v in sf['quarters'])}"
                  f"   hybrid {' '.join(f'{v:.2f}' for v in sh['quarters'])}"
                  f"   acc local {sh['acc_l']:.3f}   cost h/f {sh['sec_step'] / sf['sec_step']:.2f}")
            rows.append((kernel, seed, r_s, r_step, z, sf["ess"]))
    if len(rows) == 4:
        ok = [r[2] >= 1.5 and abs(r[4]) < 2 for r in rows]
        conv = [r[5] >= 50 for r in rows]
        print(f"\nCriterion: {sum(ok)} of 4 cells at ratio >= 1.5 with |z| < 2 -> "
              f"{'PASS' if all(ok) else ('FAIL' if not any(ok) else 'IN-BETWEEN')}.")
        if not all(conv):
            print(f"Forest ESS below 50 in {4 - sum(conv)} cells: the forest move alone has not "
                  f"converged there, and the report is the convergence contrast, not the ratio.")
    else:
        print(f"\n{len(rows)} of 4 cells available, no verdict yet.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["probe", "run", "verdict"])
    ap.add_argument("which", nargs="?", choices=["forest", "hybrid"])
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target", choices=["pi", "tree"], default="pi")
    ap.add_argument("--p-local", type=float, default=0.1)
    ap.add_argument("--m", type=float, default=2.0)
    ap.add_argument("--k", type=int, default=13)
    ap.add_argument("--kernel", choices=["perimeter", "density", "both"], default="both")
    a = ap.parse_args()
    if a.mode == "run" and a.which is None:
        ap.error("run needs forest or hybrid")
    {"probe": probe, "run": run, "verdict": verdict}[a.mode](a)


if __name__ == "__main__":
    main()
