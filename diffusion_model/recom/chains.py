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
                            iqp_local_step, spec_recom_step)
from recom.problems import cut_edges_count, assignment_canonical


# ─── Tempered ReCom (MH on −λ·Δcuts, symmetric-q approximation) ─────────────

def tempered_recom_step(part, graph, ideal_pop, eps, lam):
    cur = cut_edges_count(graph, dict(part.assignment))
    proposed = recom(part, "population", ideal_pop, eps, node_repeats=1)
    new = cut_edges_count(graph, dict(proposed.assignment))
    if math.log(random.random() + 1e-300) < -lam * (new - cur):
        return proposed, True, new
    return part, False, cur


# ─── run_chain ───────────────────────────────────────────────────────────────

def run_chain(name, init_state, step_fn, n_steps, fiber_lookup, island_of=None):
    """Generic chain runner. step_fn(state) -> (new_state, asn, cuts, *extra)."""
    state = init_state
    cuts, times, extras = [], [], []
    visits = Counter() if fiber_lookup else None
    island_trace = [] if island_of is not None else None
    for s in range(n_steps):
        t0 = time.perf_counter()
        state, asn, c, *extra = step_fn(state)
        times.append(time.perf_counter() - t0)
        cuts.append(c)
        extras.append(extra)
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
    }


# ─── Step-function factories ─────────────────────────────────────────────────

def make_vanilla_step(graph, ideal, eps):
    def step(part):
        new = recom(part, "population", ideal, eps, node_repeats=1)
        c = cut_edges_count(graph, dict(new.assignment))
        return new, dict(new.assignment), c
    return step


def make_tempered_step(graph, ideal, eps, lam):
    def step(part):
        new, ok, c = tempered_recom_step(part, graph, ideal, eps, lam)
        return new, dict(new.assignment), c, int(ok)
    return step


def make_qp_diffusion_mix_step(graph, ideal, eps, lam, p_recom,
                               alpha, beta, threshold, normalize,
                               kernel_fn=None,
                               weight_jitter=False, tabu_size=0):
    """Mixture chain: ReCom w.p. p_recom, QP-diffusion otherwise.

    Maintains a tabu queue of recently-used QP pairs (size `tabu_size`,
    cleared whenever a ReCom step is taken) to inject diversity in
    the QP local steps.
    """
    tabu = deque(maxlen=max(tabu_size, 1)) if tabu_size > 0 else None

    def step(part):
        if random.random() < p_recom:
            new = recom(part, "population", ideal, eps, node_repeats=1)
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
                forbid_pairs=forbid)
            new = make_partition(graph, asn)
            if tabu is not None and pair is not None and ok:
                tabu.append(pair)
            kind = "Q"
        c = cut_edges_count(graph, asn)
        return new, asn, c, kind, int(ok)
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
                forbid_pairs=forbid)
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
            new = recom(part, "population", ideal, eps, node_repeats=1)
            asn = dict(new.assignment)
            kind = "R"; ok = True
        else:
            asn, ok = spec_recom_step(graph, dict(part.assignment),
                                      epsilon=eps, normalize=normalize)
            new = make_partition(graph, asn)
            kind = "S"
        c = cut_edges_count(graph, asn)
        return new, asn, c, kind, int(ok)
    return step
