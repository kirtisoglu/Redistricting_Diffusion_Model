"""Least-change refinement from the enacted plan (Iowa).

Question: starting from a reference plan (the enacted 2011 Iowa
congressional map), how many cut edges can each method remove while
reassigning at most D counties (best district relabeling)?

Methods, all starting at the reference plan:
  IQP     — anchored IQP local passes (alpha=10, beta=1, uniform kernel,
            global-ideal balance window), the paper's Chain C inner move
            with no outer ReCom step. Trajectory of (cuts, dist) recorded
            after every pass; best cuts at dist <= D reported per budget.
  SA-ReCom — simulated annealing over balance-validated ReCom proposals
            (geometric lambda 0.05 -> 5), rejecting any proposal with
            dist > D. One run per budget.
  SA-flip — simulated annealing over single-county boundary flips
            (contiguity + global balance window enforced), same schedule
            and cap. The fine-grained classical baseline.

Usage: python3 recom/least_change.py [n_seeds]
Writes iowa_hybrid_test_out/least_change_seeds.npz and prints the table.
"""

import math
import random
import sys
import time
from collections import Counter
from itertools import permutations
from pathlib import Path

import networkx as nx
import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
_DIFF = _REPO / "diffusion_model"
if str(_DIFF) not in sys.path:
    sys.path.insert(0, str(_DIFF))

from recom.problems import setup_problem, cut_edges_count
from recom.chains import recom_valid, _plan_balanced
from recom.qp_model import make_partition, kernel_uniform, iqp_local_step

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
SEEDS = list(range(42, 42 + N_SEEDS))
BUDGETS = [3, 5, 10, 15, 20, 99]
IQP_PASSES = 50
BETAS = [0.0, 0.3, 1.0, 3.0, 10.0, 30.0]   # the least-change dial
SA_STEPS = 2000
LAM_SCHED = lambda t, T: 0.05 * (5.0 / 0.05) ** (t / max(T - 1, 1))
ALPHA, TLIM = 10.0, 5.0
K = 4

OUT = Path(__file__).parent / "iowa_hybrid_test_out"
OUT.mkdir(exist_ok=True)


def plan_distance(a, b, labels_a, labels_b):
    """Counties reassigned under the best relabeling of b's districts."""
    ia = {d: i for i, d in enumerate(labels_a)}
    ib = {d: i for i, d in enumerate(labels_b)}
    best = None
    nodes = list(a.keys())
    av = [ia[a[n]] for n in nodes]
    bv = [ib[b[n]] for n in nodes]
    for perm in permutations(range(K)):
        d = sum(1 for x, y in zip(av, bv) if perm[x] != y)
        best = d if best is None else min(best, d)
    return best


def make_dist_fn(ref):
    labels_ref = sorted(set(ref.values()))
    def dist(asn):
        return plan_distance(asn, ref, sorted(set(asn.values())), labels_ref)
    return dist


def sa_recom_capped(graph, ideal, eps, ref, cap, dist, steps):
    part = make_partition(graph, ref)
    cur_asn = dict(part.assignment)
    cur = cut_edges_count(graph, cur_asn)
    best, best_at = cur, 0
    accepted = 0
    for t in range(steps):
        lam = LAM_SCHED(t, steps)
        prop = recom_valid(part, graph, ideal, eps)
        asn = dict(prop.assignment)
        if asn == cur_asn:
            continue
        if dist(asn) > cap:
            continue
        new = cut_edges_count(graph, asn)
        if math.log(random.random() + 1e-300) < -lam * (new - cur):
            part, cur_asn, cur = prop, asn, new
            accepted += 1
            if cur < best:
                best, best_at = cur, dist(asn)
    return best, best_at, accepted


def flip_candidates(graph, asn):
    """Cut edges as (mover, target_district) flip candidates."""
    out = []
    for u, v in graph.edges():
        if asn[u] != asn[v]:
            out.append((u, asn[v]))
            out.append((v, asn[u]))
    return out


def flip_ok(graph, asn, node, new_d, ideal, eps):
    """Contiguity of the donor district and global balance after the flip."""
    old_d = asn[node]
    donor_nodes = [n for n, d in asn.items() if d == old_d and n != node]
    if not donor_nodes:
        return False
    if not nx.is_connected(graph.subgraph(donor_nodes)):
        return False
    pops = Counter()
    for n, d in asn.items():
        pops[d] += graph.nodes[n]["population"]
    p = graph.nodes[node]["population"]
    pops[old_d] -= p
    pops[new_d] += p
    return all(abs(q - ideal) <= eps * ideal for q in pops.values())


def sa_flip_capped(graph, ideal, eps, ref, cap, dist, steps):
    asn = dict(ref)
    cur = cut_edges_count(graph, asn)
    best, best_at = cur, 0
    accepted = 0
    for t in range(steps):
        lam = LAM_SCHED(t, steps)
        cands = flip_candidates(graph, asn)
        node, new_d = random.choice(cands)
        if not flip_ok(graph, asn, node, new_d, ideal, eps):
            continue
        trial = dict(asn)
        trial[node] = new_d
        if dist(trial) > cap:
            continue
        new = cut_edges_count(graph, trial)
        if math.log(random.random() + 1e-300) < -lam * (new - cur):
            asn, cur = trial, new
            accepted += 1
            if cur < best:
                best, best_at = cur, dist(asn)
    return best, best_at, accepted


def iqp_trajectory(graph, ideal, eps, ref, dist, passes, beta):
    asn = dict(ref)
    traj = [(cut_edges_count(graph, asn), 0)]
    flips = []
    for _ in range(passes):
        prev = dict(asn)
        asn, ok, _pair = iqp_local_step(
            graph, asn, alpha=ALPHA, beta=beta, epsilon=eps,
            normalize=False, kernel_fn=kernel_uniform,
            time_limit=TLIM, ideal_pop=ideal)
        if ok:
            moved = sum(1 for n in prev if prev[n] != asn[n])
            if moved:
                flips.append(moved)
        traj.append((cut_edges_count(graph, asn), dist(asn)))
    return traj, flips


def main():
    p = setup_problem("iowa", _REPO)
    graph, eps, ideal = p["graph"], p["eps"], p["ideal"]
    ref = {n: graph.nodes[n]["CD"] for n in graph.nodes()}
    assert _plan_balanced(graph, ref, ideal, eps), "enacted plan unbalanced?"
    for d in sorted(set(ref.values())):
        nodes_d = [n for n, dd in ref.items() if dd == d]
        assert nx.is_connected(graph.subgraph(nodes_d)), f"CD {d} disconnected"
    ref_cuts = cut_edges_count(graph, ref)
    dist = make_dist_fn(ref)
    print(f"Enacted reference plan: {ref_cuts} cut edges, k={K}, "
          f"eps={eps}, {len(graph.nodes())} counties")

    res = {m: {D: [] for D in BUDGETS} for m in ("iqp", "sa_recom", "sa_flip")}
    walls = {m: [] for m in ("iqp", "sa_recom", "sa_flip")}
    iqp_flip_sizes, iqp_final_dist = [], []
    recom_move_sizes = []
    trajs = {}

    for seed in SEEDS:
        # IQP anchored refinement. beta is the least-change dial, so the
        # method's frontier is the envelope over a small beta sweep of
        # (cuts, distance) trajectories.
        t0 = time.perf_counter()
        env = []
        for beta in BETAS:
            random.seed(seed * 1000 + 7)
            np.random.seed(seed * 1000 + 7)
            traj, flips = iqp_trajectory(graph, ideal, eps, ref, dist,
                                         IQP_PASSES, beta)
            env.extend(traj)
            if beta == 1.0:
                iqp_flip_sizes.extend(flips)
                iqp_final_dist.append(traj[-1][1])
            trajs[f"traj_cuts_b{beta}_s{seed}"] = \
                np.array([c for c, _ in traj])
            trajs[f"traj_dist_b{beta}_s{seed}"] = \
                np.array([d for _, d in traj])
        walls["iqp"].append(time.perf_counter() - t0)
        for D in BUDGETS:
            vals = [c for c, d in env if d <= D]
            res["iqp"][D].append(min(vals))

        # uncapped ReCom move granularity (for the granularity table)
        random.seed(seed * 1000 + 8)
        np.random.seed(seed * 1000 + 8)
        part = make_partition(graph, ref)
        for _ in range(20):
            prop = recom_valid(part, graph, ideal, eps)
            asn = dict(prop.assignment)
            moved = sum(1 for n in ref if asn[n] != dict(part.assignment)[n])
            if moved:
                recom_move_sizes.append(moved)
            part = prop

        for D in BUDGETS:
            random.seed(seed * 1000 + 100 + D)
            np.random.seed(seed * 1000 + 100 + D)
            t0 = time.perf_counter()
            b, _, _ = sa_recom_capped(graph, ideal, eps, ref, D, dist, SA_STEPS)
            walls["sa_recom"].append(time.perf_counter() - t0)
            res["sa_recom"][D].append(b)

            random.seed(seed * 1000 + 200 + D)
            np.random.seed(seed * 1000 + 200 + D)
            t0 = time.perf_counter()
            b, _, _ = sa_flip_capped(graph, ideal, eps, ref, D, dist, SA_STEPS)
            walls["sa_flip"].append(time.perf_counter() - t0)
            res["sa_flip"][D].append(b)
        print(f"  seed {seed} done")

    print(f"\nLEAST-CHANGE TABLE (reference = enacted, {ref_cuts} cuts; "
          f"{N_SEEDS} seeds; best cuts within budget, mean+-sd)")
    hdr = f"{'D':>4} | {'IQP anchored':>16} | {'SA-ReCom capped':>16} | {'SA-flip capped':>16}"
    print(hdr)
    print("-" * len(hdr))
    for D in BUDGETS:
        row = [f"{D:>4}"]
        for m in ("iqp", "sa_recom", "sa_flip"):
            a = np.array(res[m][D], dtype=float)
            row.append(f"{a.mean():8.2f} +- {a.std():4.2f}")
        print(" | ".join(row))
    print(f"\nGranularity: IQP accepted-pass flip size "
          f"mean={np.mean(iqp_flip_sizes):.1f} "
          f"median={np.median(iqp_flip_sizes):.0f} "
          f"max={np.max(iqp_flip_sizes):.0f} (n={len(iqp_flip_sizes)}); "
          f"ReCom accepted-move size mean={np.mean(recom_move_sizes):.1f} "
          f"median={np.median(recom_move_sizes):.0f} "
          f"min={np.min(recom_move_sizes):.0f} (n={len(recom_move_sizes)})")
    print(f"IQP final distance from reference after {IQP_PASSES} passes: "
          f"mean={np.mean(iqp_final_dist):.1f} max={np.max(iqp_final_dist):.0f}")
    print(f"Wall-clock per run: IQP {np.mean(walls['iqp']):.1f}s, "
          f"SA-ReCom {np.mean(walls['sa_recom']):.1f}s, "
          f"SA-flip {np.mean(walls['sa_flip']):.1f}s")

    np.savez(OUT / "least_change_seeds.npz",
             budgets=np.array(BUDGETS),
             **{f"{m}_D{D}": np.array(res[m][D])
                for m in res for D in BUDGETS},
             iqp_flip_sizes=np.array(iqp_flip_sizes),
             recom_move_sizes=np.array(recom_move_sizes),
             iqp_final_dist=np.array(iqp_final_dist),
             iqp_wall=np.array(walls["iqp"]),
             sa_recom_wall=np.array(walls["sa_recom"]),
             sa_flip_wall=np.array(walls["sa_flip"]),
             **trajs,
             ref_cuts=ref_cuts, seeds=np.array(SEEDS),
             config=np.array([f"alpha={ALPHA}", f"betas={BETAS}",
                              f"iqp_passes={IQP_PASSES}",
                              f"sa_steps={SA_STEPS}",
                              "kernel=uniform", "balance=global_ideal",
                              "schedule=geometric 0.05->5"]))
    print("\nSaved least_change_seeds.npz")


if __name__ == "__main__":
    main()
