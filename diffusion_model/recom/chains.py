"""Chain runners and step-function factories.

Public API:
    run_chain(name, init_state, step_fn, n_steps, fiber_lookup, island_of)
    tempered_recom_step(part, graph, ideal, eps, lam)

    make_vanilla_step(graph, ideal, eps)
    make_tempered_step(graph, ideal, eps, lam)
    make_qp_diffusion_mix_step(...)
    make_iqp_hybrid_step(...)
    make_specrecom_mix_step(graph, ideal, eps, p_recom, normalize)

Diversity options for B and C step factories:
    weight_jitter (bool): per-step uniform [1,2] edge-weight randomisation
        (SpecReCom diversity trick — Davies et al. 2025 alg. 2 lines 4-6).
    tabu_size (int):     forbid the last `tabu_size` chosen pairs from
        being re-picked by the local kernel.  When the preceding ReCom
        step accepts (changes the partition) the tabu is cleared.
"""

import math
import random
import time
from collections import Counter, deque

import numpy as np

from gerrychain.proposals import recom

from recom.qp_model import (make_partition, qp_diffusion_step,
                            iqp_local_step, spec_recom_step,
                            qp_mh_proposal)
from recom.problems import cut_edges_count, assignment_canonical


# ─── Balance-validated ReCom proposal ────────────────────────────────────────

def _plan_balanced(graph, asn, ideal_pop, eps):
    """True iff every district population is within (1±eps)·ideal_pop."""
    pops = Counter()
    for v, d in asn.items():
        pops[d] += graph.nodes[v]["population"]
    return all(abs(p - ideal_pop) <= eps * ideal_pop for p in pops.values())


def recom_valid(part, graph, ideal_pop, eps, max_tries=20):
    """ReCom proposal with an explicit two-sided global balance check.

    gerrychain's raw `recom` proposal accepts a tree cut when EITHER side
    is within eps of ideal (find_balanced_edge_cuts_memoization); the
    complement side is unchecked, so calling the proposal without a
    Validator lets district populations drift outside the global
    tolerance.  This wrapper re-proposes until the full plan is valid
    (up to max_tries), else returns the current partition.
    """
    for _ in range(max_tries):
        proposed = recom(part, "population", ideal_pop, eps, node_repeats=1)
        if _plan_balanced(graph, dict(proposed.assignment), ideal_pop, eps):
            return proposed
    return part


# ─── Tempered ReCom (MH on −λ·Δcuts, symmetric-q approximation) ─────────────

def tempered_recom_step(part, graph, ideal_pop, eps, lam):
    cur = cut_edges_count(graph, dict(part.assignment))
    proposed = recom_valid(part, graph, ideal_pop, eps)
    new = cut_edges_count(graph, dict(proposed.assignment))
    if math.log(random.random() + 1e-300) < -lam * (new - cur):
        return proposed, True, new
    return part, False, cur


# ─── run_chain ───────────────────────────────────────────────────────────────

def run_chain(name, init_state, step_fn, n_steps, fiber_lookup, island_of=None,
              graph=None, ideal=None):
    """Generic chain runner. step_fn(state) -> (new_state, asn, cuts, *extra).

    If `graph` and `ideal` are given, records the per-step maximum
    relative deviation of any district population from the global ideal
    (`pop_dev` array) — the audit for the ε-balance guarantee.
    """
    state = init_state
    cuts, times, extras = [], [], []
    pop_devs = [] if (graph is not None and ideal is not None) else None
    visits = Counter() if fiber_lookup else None
    island_trace = [] if island_of is not None else None
    for s in range(n_steps):
        t0 = time.perf_counter()
        state, asn, c, *extra = step_fn(state)
        # Sever gerrychain's parent link: Partition objects reference
        # their ancestor, so a long chain otherwise keeps its entire
        # history alive and GC-thrashes on 10^5+ step runs.
        if hasattr(state, "parent"):
            state.parent = None
        times.append(time.perf_counter() - t0)
        cuts.append(c)
        extras.append(extra)
        if pop_devs is not None:
            dist_pop = Counter()
            for v, d in asn.items():
                dist_pop[d] += graph.nodes[v]["population"]
            pop_devs.append(max(abs(p - ideal) / ideal
                                for p in dist_pop.values()))
        cid = None
        if fiber_lookup is not None:
            cid = fiber_lookup.get(assignment_canonical(asn))
            if cid is not None:
                visits[cid] += 1
        if island_trace is not None:
            island_trace.append(island_of.get(cid) if cid is not None else None)
        if (s + 1) % 100 == 0:
            print(f"  [{name}] {s+1}/{n_steps}  cuts={c}")
    return {
        "cuts": np.array(cuts),
        "times": np.array(times),
        "extras": extras,
        "visits": visits,
        "island_trace": island_trace,
        "pop_dev": np.array(pop_devs) if pop_devs is not None else None,
        "final_asn": asn,
    }


# ─── Step-function factories ─────────────────────────────────────────────────

def make_vanilla_step(graph, ideal, eps):
    def step(part):
        new = recom_valid(part, graph, ideal, eps)
        c = cut_edges_count(graph, dict(new.assignment))
        return new, dict(new.assignment), c
    return step


def make_tempered_step(graph, ideal, eps, lam):
    def step(part):
        new, ok, c = tempered_recom_step(part, graph, ideal, eps, lam)
        return new, dict(new.assignment), c, int(ok)
    return step


def make_qp_diffusion_mix_step(graph, ideal, eps, p_recom,
                               alpha, beta, threshold, normalize,
                               kernel_fn=None,
                               weight_jitter=False, tabu_size=0):
    """Mixture chain: ReCom w.p. p_recom, QP-diffusion otherwise.

    NOTE: neither branch carries a Metropolis correction — the ReCom
    branch is vanilla (always accept) and the QP branch applies the
    rounded/repaired split unconditionally.  This chain is an
    exploratory heuristic with no stationarity claim; use
    `make_qp_mh_step` for the exactly reversible sampler.

    Maintains a tabu queue of recently-used QP pairs (size `tabu_size`,
    cleared whenever a ReCom step is taken) to inject diversity in
    the QP local steps.
    """
    tabu = deque(maxlen=max(tabu_size, 1)) if tabu_size > 0 else None

    def step(part):
        if random.random() < p_recom:
            new = recom_valid(part, graph, ideal, eps)
            asn = dict(new.assignment)
            if tabu is not None:
                tabu.clear()       # ReCom moved → fresh state, all pairs available
            kind = "R"; ok = True
        else:
            forbid = list(tabu) if tabu is not None else None
            asn, ok, pair = qp_diffusion_step(
                graph, dict(part.assignment),
                alpha=alpha, beta=beta, epsilon=eps,
                threshold=threshold, normalize=normalize,
                kernel_fn=kernel_fn,
                weight_jitter=weight_jitter,
                forbid_pairs=forbid,
                ideal_pop=ideal)
            new = make_partition(graph, asn)
            if tabu is not None and pair is not None and ok:
                tabu.append(pair)
            kind = "Q"
        c = cut_edges_count(graph, asn)
        return new, asn, c, kind, int(ok)
    return step


def make_qp_mh_step(graph, ideal, eps, lam, alpha, beta, kernel_fn=None):
    """Exactly reversible QP-MH chain (Chain E).

    Every step is a QP proposal with single-shot Bernoulli rounding and
    an exact Metropolis--Hastings correction against
    π(P) ∝ exp(−λ·cut_edges(P)) on the valid space (see
    qp_model.qp_mh_proposal for the detailed-balance argument).
    """
    def step(part):
        asn, acc = qp_mh_proposal(graph, dict(part.assignment),
                                  alpha=alpha, beta=beta, epsilon=eps,
                                  lam=lam, kernel_fn=kernel_fn,
                                  ideal_pop=ideal)
        new = make_partition(graph, asn)
        c = cut_edges_count(graph, asn)
        return new, asn, c, int(acc)
    return step


def make_iqp_hybrid_step(graph, ideal, eps, lam, n_iqp,
                         alpha, beta, kernel_fn, normalize, time_limit,
                         weight_jitter=False, tabu_size=0):
    """Tempered ReCom + n_iqp deterministic IQP local steps.

    Tabu: clear when Tempered ReCom accepts (chain moved); else hold
    the last `tabu_size` IQP pairs and forbid them from being re-picked.
    """
    tabu = deque(maxlen=max(tabu_size, 1)) if tabu_size > 0 else None

    def step(part):
        part2, ok_r, _ = tempered_recom_step(part, graph, ideal, eps, lam)
        asn = dict(part2.assignment)
        if tabu is not None and ok_r:
            tabu.clear()
        n_improved = 0
        for _ in range(n_iqp):
            forbid = list(tabu) if tabu is not None else None
            asn, improved, pair = iqp_local_step(
                graph, asn, alpha=alpha, beta=beta, epsilon=eps,
                normalize=normalize, kernel_fn=kernel_fn,
                time_limit=time_limit,
                weight_jitter=weight_jitter,
                forbid_pairs=forbid,
                ideal_pop=ideal)
            n_improved += int(improved)
            if tabu is not None and pair is not None and improved:
                tabu.append(pair)
        new = make_partition(graph, asn)
        c = cut_edges_count(graph, asn)
        return new, asn, c, int(ok_r), n_improved
    return step


def make_specrecom_mix_step(graph, ideal, eps, p_recom, normalize=False):
    def step(part):
        if random.random() < p_recom:
            new = recom_valid(part, graph, ideal, eps)
            asn = dict(new.assignment)
            kind = "R"; ok = True
        else:
            asn, ok = spec_recom_step(graph, dict(part.assignment),
                                      epsilon=eps, normalize=normalize,
                                      ideal_pop=ideal)
            new = make_partition(graph, asn)
            kind = "S"
        c = cut_edges_count(graph, asn)
        return new, asn, c, kind, int(ok)
    return step
