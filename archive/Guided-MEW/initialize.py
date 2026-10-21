"""
initialize.py — Build a valid MEW state (T, M) from a partition.

Given a node-to-district assignment, constructs a spanning tree T of G
and a marked edge set M ⊂ E(T) with |M| = k-1, such that the connected
components of T \\ M reproduce the input partition.

Algorithm:
    1. For each district, sample a uniform random spanning tree of its
       induced subgraph.
    2. Build the quotient graph (districts as nodes, edge iff districts
       share a boundary edge in G) and sample its spanning tree.
    3. For each quotient-tree edge, pick one cross-district edge uniformly
       at random and add it to T. These k-1 edges form M.
"""

import random
import networkx as nx
from state import MEWState


def initialize_from_partition(graph, assignment):
    """
    Build MEW state (T, M) from a partition assignment.

    Parameters
    ----------
    graph : nx.Graph
        The full graph G.
    assignment : dict
        {node: district_id} mapping.

    Returns
    -------
    MEWState
    """
    # Group nodes by district
    districts = {}
    for node, d in assignment.items():
        districts.setdefault(d, []).append(node)

    tree = nx.Graph()

    # Step 1: random spanning tree per district
    for d_id, nodes in districts.items():
        sub = graph.subgraph(nodes)
        if sub.number_of_nodes() == 1:
            tree.add_node(nodes[0])
        else:
            st = nx.random_spanning_tree(sub)
            tree.add_edges_from(st.edges())

    # Step 2: quotient graph with cross-district edge lists
    cross_edges = {}
    quotient = nx.Graph()
    for u, v in graph.edges():
        d1, d2 = assignment[u], assignment[v]
        if d1 != d2:
            key = (min(d1, d2), max(d1, d2))
            quotient.add_edge(key[0], key[1])
            cross_edges.setdefault(key, []).append((u, v))

    # Step 3: spanning tree of quotient → one cross edge per pair → marked
    q_tree = nx.random_spanning_tree(quotient)
    marked = set()
    for d1, d2 in q_tree.edges():
        key = (min(d1, d2), max(d1, d2))
        edge = random.choice(cross_edges[key])
        tree.add_edge(*edge)
        marked.add(frozenset(edge))

    return MEWState(graph, tree, marked)
