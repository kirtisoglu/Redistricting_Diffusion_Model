"""
Grid graph construction with population attributes.

Builds N×N grid graphs partitioned into d quadrant districts.
Used for controlled experiments where exact spanning tree counts
are known (via Kirchhoff's theorem).
"""

import networkx as nx


def make_grid(N: int, d: int = 4, pop_per_node: int = 1):
    """Build an N×N grid graph with d quadrant districts.

    Parameters
    ----------
    N : int
        Grid side length. Must be even for quadrant partitioning.
    d : int
        Number of districts. Currently only d=4 (quadrants) is supported.
    pop_per_node : int
        Uniform population assigned to each node.

    Returns
    -------
    graph : nx.Graph
        The N×N grid with 'population' attribute on each node.
    assignment : dict
        Mapping {node: district_id} for the quadrant partition.
    """
    if d != 4:
        raise ValueError("Only d=4 (quadrant partition) is currently supported.")
    if N % 2 != 0:
        raise ValueError("N must be even for quadrant partitioning.")

    graph = nx.grid_2d_graph(N, N)

    # Assign uniform population
    for node in graph.nodes:
        graph.nodes[node]["population"] = pop_per_node

    # Quadrant assignment: 4 blocks of (N/2) × (N/2)
    half = N // 2
    assignment = {}
    for r, c in graph.nodes:
        if r < half and c < half:
            assignment[(r, c)] = 0
        elif r < half and c >= half:
            assignment[(r, c)] = 1
        elif r >= half and c < half:
            assignment[(r, c)] = 2
        else:
            assignment[(r, c)] = 3

    return graph, assignment
