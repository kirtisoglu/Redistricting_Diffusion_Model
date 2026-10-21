"""Correctness gate for the forest-lifted move on enumerated ground truth,
and the 5x5 half of the kill test.

Three chains from the same start, per target:
    tree-part    the paper's partition-level tree move (hybrid.py, p_local=0),
                 for reference and for the baseline-vs-baseline comparison
    forest       the forest-lifted move alone (forest_recom.py, p_local=0),
                 the Forest ReCom baseline
    f-hybrid     forest move + lifted Chain E at p_local = P_LOCAL
against the exact target pi_w over all 4,006 plans, for the uniform kernel
at lam = 0.1 and lam = 1 and the synthetic-city density kernel at lam = 1
(the three targets of the paper's Table "hybrid-5x5").

Reported per chain, as in hybrid_validate_5x5.py: TV of the cut-count
marginal to pi_w with its noise floor at the ESS and the excess z;
batch-means Hotelling p; full-state TV; acceptance by move type; ESS and
ESS per second.

GATE (must pass before any Iowa run is believed): forest and f-hybrid both
within |z| <= 2 and Hotelling p > 0.05 on all three targets. If forest
fails, the move is wrong. If forest passes and f-hybrid fails, the lift of
the local move is wrong.

KILL RATIO on this fiber: ESS/s(f-hybrid) / ESS/s(forest). Reported, not
decisive: under exact balance the local move's first-order channel is
closed (paper Section 4.6), so a passenger verdict here is expected.

The fifth argument selects the target: "pi" (default) is the paper's
pi_w, "tree" is the tree-weighted tempered measure prod tau(V_d) * pi_w,
on which the forest move pays no tree-count ratio and the local move
does; both are enumerated exactly here. The paper's tree-part chain is
run only under "pi", since hybrid.py has no tree-weighted target.

Writes: iowa_hybrid_test_out/forest_validate_5x5_p{P}_tau{TAU}_{target}.npz
Usage:  python3 recom/forest_validate_5x5.py [n_steps] [seed] [p_local] [tau] [pi|tree]
Default: 200000 42 0.2 0.1 pi
"""

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_DIFF = _HERE.parent
_REPO = _DIFF.parent
if str(_DIFF) not in sys.path:
    sys.path.insert(0, str(_DIFF))

from recom.problems import setup_problem, assignment_canonical
from recom.qp_model import kernel_uniform, make_kernel_density
from recom.hybrid import run_hybrid, log_tau
from recom.forest_recom import run_forest_hybrid, make_target
from recom.kill_stats import tv, floor_tv, hotelling_marginal, chain_summary

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42
P_LOCAL = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
TAU = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
TARGET = sys.argv[5] if len(sys.argv) > 5 else "pi"     # "pi" or "tree"
ALPHA, BETA = TAU, 1.0
OUT_TAG = f"_p{P_LOCAL:g}_tau{TAU:g}_{TARGET}"
BURN = 0.1


def main():
    p = setup_problem("5x5", _REPO)
    graph, eps, ideal, init = p["graph"], p["eps"], p["ideal"], p["init"]
    lookup = p["fiber_lookup"]
    n_fiber = len(lookup)
    for (x, y) in graph.nodes():
        graph.nodes[(x, y)]["density"] = 3.0 * np.exp(-((x - 2) ** 2 + (y - 2) ** 2) / 2.0)

    with open(_HERE / "fiber_5x5" / "fiber_5x5_partitions.json") as f:
        data = json.load(f)
    plans = {}
    for pid, parts in data.items():
        a = {}
        for d, part in enumerate(parts):
            for s in part:
                a[tuple(s)] = d
        plans[int(pid)] = a
    ids_sorted = np.array(sorted(plans))
    pos = {i: j for j, i in enumerate(ids_sorted)}
    # spanning-tree weight of every plan, for the tree-weighted target
    log_tw = np.zeros(len(ids_sorted))
    if TARGET == "tree":
        for j, i in enumerate(ids_sorted):
            a = plans[i]
            log_tw[j] = sum(log_tau(graph, [v for v in a if a[v] == d])
                            for d in set(a.values()))

    def state_id(asn):
        return lookup.get(assignment_canonical(asn), -1)

    configs = [
        ("uniform", kernel_uniform, 0.10),
        ("uniform", kernel_uniform, 1.00),
        ("density", make_kernel_density(sigma=1.0), 1.00),
    ]
    chains = ([
        ("tree-part", lambda tgt, tag: run_hybrid(tgt, init, ideal, eps, N, 0.0, ALPHA, BETA, SEED,
                                                  state_id=state_id, label=tag)),
    ] if TARGET == "pi" else []) + [
        ("forest", lambda tgt, tag: run_forest_hybrid(tgt, init, ideal, eps, N, 0.0, ALPHA, BETA, SEED,
                                                      state_id=state_id, label=tag)),
        ("f-hybrid", lambda tgt, tag: run_forest_hybrid(tgt, init, ideal, eps, N, P_LOCAL, ALPHA, BETA, SEED,
                                                        state_id=state_id, label=tag)),
    ]
    store = {"n_steps": N, "seed": SEED, "p_local": P_LOCAL, "tau": TAU, "target": TARGET}
    print(f"5x5 fiber, {n_fiber} states, {N} steps per chain, "
          f"p_local = {P_LOCAL}, tau = {TAU}, target = {TARGET}\n")
    print(f"{'config':>16} {'chain':>9} {'TV marg':>8} {'floor':>7} {'z':>7} "
          f"{'Hotel p':>8} {'TV full':>8} {'floor':>7} {'acc glob':>9} "
          f"{'acc loc':>8} {'mv/loc':>7} {'ESS':>7} {'wall s':>7} {'ESS/s':>7}")
    gate_ok = True
    for kname, kfn, lam in configs:
        tgt = make_target(TARGET, graph, kfn, lam)
        wc_all = np.array([tgt.wcut(plans[i]) for i in ids_sorted])
        nc_all = np.array([tgt.ncut(plans[i]) for i in ids_sorted])
        logpi = -lam * wc_all + log_tw
        pi = np.exp(logpi - logpi.max())
        pi /= pi.sum()
        vals = np.unique(nc_all)
        p_marg = np.array([pi[nc_all == v].sum() for v in vals])
        tag = f"{kname} lam={lam:g}"
        store[f"{tag}_pi"] = pi
        rates = {}
        for cname, runner in chains:
            res = runner(tgt, f"{tag} {cname}")
            burn = int(N * BURN)
            ids_post = np.array([pos[i] for i in res["ids"][burn:]])
            s = chain_summary(res, BURN)
            emp_m = np.array([(nc_all[ids_post] == v).mean() for v in vals])
            tvm = tv(emp_m, p_marg)
            fm, fs = floor_tv(p_marg, s["ess"])
            z = (tvm - fm) / fs if fs > 0 else np.inf
            hp, B, L, iat = hotelling_marginal(ids_post, nc_all, vals, p_marg)
            q = np.bincount(ids_post, minlength=n_fiber) / len(ids_post)
            tvf = tv(q, pi)
            ff, _ = floor_tv(pi, s["ess"], n_boot=60)
            print(f"{tag:>16} {cname:>9} {tvm:8.4f} {fm:7.4f} {z:+7.1f} {hp:8.3f} "
                  f"{tvf:8.4f} {ff:7.4f} {s['acc_global']:9.3f} {s['acc_local']:8.3f} "
                  f"{s['mv_local']:7.3f} {s['ess']:7.0f} {s['wall']:7.1f} {s['ess_s']:7.1f}",
                  flush=True)
            rates[cname] = s["ess_s"]
            if cname != "tree-part" and (abs(z) > 2 or hp < 0.05):
                gate_ok = False
            for key in ("wcut", "ids", "kind", "accepted", "flips", "fast"):
                store[f"{tag}_{cname}_{key}"] = res[key]
            store[f"{tag}_{cname}_wall"] = res["wall"]
            store[f"{tag}_{cname}_z"] = z
            store[f"{tag}_{cname}_hotelling_p"] = hp
        base_line = (f"forest / tree-part ESS/s = {rates['forest']/rates['tree-part']:.2f}   "
                     if "tree-part" in rates else "")
        print(f"{'':>16} {base_line}kill ratio f-hybrid / forest = "
              f"{rates['f-hybrid']/rates['forest']:.2f}\n", flush=True)

    np.savez(_HERE / "iowa_hybrid_test_out" / f"forest_validate_5x5{OUT_TAG}.npz", **store)
    print(f"Saved forest_validate_5x5{OUT_TAG}.npz")
    print("GATE " + ("PASSED: forest and f-hybrid sample their enumerated targets."
                     if gate_ok else
                     "FAILED: some forest or f-hybrid row has |z| > 2 or Hotelling p < 0.05. "
                     "Stop; do not run Iowa until this is understood."))


if __name__ == "__main__":
    main()
