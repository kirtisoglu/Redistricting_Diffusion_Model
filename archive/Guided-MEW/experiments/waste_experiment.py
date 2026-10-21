"""
waste_experiment.py — Measure within-fiber waste fraction.

Runs two experiments side by side:
    1. Vanilla MEW: e+ chosen uniformly from all non-tree edges.
    2. Boundary MEW: e+ chosen uniformly from boundary non-tree edges only.

For each grid size, reports the fraction of proposals that do NOT change
the partition (within-fiber waste), and the implied speedup factor.

NOTE: We skip MH acceptance (no τ computation) to keep this fast.
The waste fraction is a property of the proposal mechanism, not the
acceptance filter. Computing τ at every step would take hours on
larger grids without changing the waste measurement.

Usage:
    python3 Guided-MEW/experiments/waste_experiment.py
"""

import sys
import os
import time

_GUIDED_MEW_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _GUIDED_MEW_DIR)

from grid import make_grid
from initialize import initialize_from_partition
from mew_step import vanilla_cycle_step, boundary_cycle_step, _marked_edge_step
from state import MEWState


def run_experiment(N, n_steps, boundary_only):
    """
    Run n_steps proposals on an N×N grid with k=4 quadrant districts.

    We only measure whether the proposal changes the partition, without
    computing τ or doing MH acceptance. This isolates the waste fraction
    as a property of the cycle step selection.

    Returns
    -------
    dict with: grid, n, n_steps, valid_proposals, partition_changed,
               waste_frac, f_int, speedup, elapsed
    """
    graph, assignment = make_grid(N, d=4)
    n_nodes = graph.number_of_nodes()

    state = initialize_from_partition(graph, assignment)

    # Compute f_int (fraction of interior non-tree edges) for reference
    n_edges = graph.number_of_edges()
    n_non_tree = n_edges - (n_nodes - 1)
    boundary_nte_count = len(state.boundary_non_tree_edges())
    f_int = 1.0 - boundary_nte_count / n_non_tree if n_non_tree > 0 else 0.0

    cycle_fn = boundary_cycle_step if boundary_only else vanilla_cycle_step

    valid_proposals = 0
    partition_changed = 0

    t0 = time.time()

    for _ in range(n_steps):
        old_asn = state.get_partition()

        # Cycle step
        cycle_result = cycle_fn(state)
        if cycle_result is None:
            continue

        T_new, e_plus, cycle_edges, removable, e_minus = cycle_result
        e_plus_fs = frozenset(e_plus)

        # Marked edge step
        me_result = _marked_edge_step(T_new, state.marked, e_plus_fs)
        if me_result is None:
            continue

        M_new, m, m_prime, u, v_other = me_result

        # Build proposed state and check partition
        new_state = MEWState(graph, T_new, M_new)
        new_asn = new_state.get_partition()

        valid_proposals += 1

        if set(new_asn.items()) != set(old_asn.items()):
            partition_changed += 1

        # Accept unconditionally (we're measuring proposal quality, not running MCMC)
        state = new_state

    elapsed = time.time() - t0

    waste = (valid_proposals - partition_changed) / valid_proposals if valid_proposals > 0 else 0.0
    useful = 1.0 - waste
    speedup = 1.0 / useful if useful > 0 else float("inf")

    return {
        "grid": f"{N}x{N}",
        "n": n_nodes,
        "n_steps": n_steps,
        "valid": valid_proposals,
        "changed": partition_changed,
        "waste_frac": waste,
        "f_int": f_int,
        "speedup": speedup,
        "elapsed": elapsed,
    }


def print_table(title, results):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
    print(f"{'Grid':<10} {'n':>5} {'Valid':>8} {'Changed':>8} "
          f"{'Waste':>8} {'f_int':>8} {'Speedup':>8} {'Time':>8}")
    print("-" * 78)
    for r in results:
        print(f"{r['grid']:<10} {r['n']:>5} {r['valid']:>8} "
              f"{r['changed']:>8} {r['waste_frac']:>7.1%} "
              f"{r['f_int']:>7.1%} {r['speedup']:>7.1f}x "
              f"{r['elapsed']:>7.1f}s")


def main():
    grid_sizes = [6, 10, 14, 20, 26, 30]
    n_steps = 10000

    # Vanilla MEW
    vanilla_results = []
    for N in grid_sizes:
        print(f"  Running vanilla MEW on {N}x{N}...", flush=True)
        r = run_experiment(N, n_steps, boundary_only=False)
        vanilla_results.append(r)

    print_table("VANILLA MEW (e+ from all non-tree edges)", vanilla_results)

    # Boundary MEW
    boundary_results = []
    for N in grid_sizes:
        print(f"  Running boundary MEW on {N}x{N}...", flush=True)
        r = run_experiment(N, n_steps, boundary_only=True)
        boundary_results.append(r)

    print_table("BOUNDARY MEW (e+ from boundary non-tree edges only)", boundary_results)


if __name__ == "__main__":
    main()
