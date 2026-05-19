"""Problem setup and graph utilities for redistricting experiments.

Public API:
    cut_edges_count(graph, asn)
    assignment_canonical(asn)
    load_fiber_islands(edges_path, n_nodes=4006)
    setup_problem(mode, repo_root)
"""

import json
from pathlib import Path

import networkx as nx
import numpy as np

from gerrychain import Graph
from gerrychain.tree import recursive_tree_part


def cut_edges_count(graph, asn):
    return sum(1 for u, v in graph.edges() if asn[u] != asn[v])


def assignment_canonical(asn):
    """Canonicalise an assignment as frozenset-of-frozensets of node ids."""
    parts = {}
    for n, d in asn.items():
        parts.setdefault(d, set()).add(n)
    return frozenset(frozenset(p) for p in parts.values())


def load_fiber_islands(edges_path, n_nodes=4006, prebuilt_npz=None):
    """Communities on the fiber graph.

    If `prebuilt_npz` is given (e.g. `fiber_5x5_eps02_communities.npz`
    saved by the corridor-analysis script), load the labels directly.
    Otherwise compute greedy-modularity communities — only fast on the
    ε=0 fiber (4006 nodes); not recommended for the ε=0.2 fiber.
    """
    if prebuilt_npz is not None and Path(prebuilt_npz).exists():
        npz = np.load(str(prebuilt_npz))
        labels = npz["node_comm"]
        n_total = len(labels)
        island_of = {int(i): int(labels[i]) for i in range(n_total)}
        n_comm = int(npz["n_communities"])
        # Reconstruct community sets
        communities = [set() for _ in range(n_comm)]
        for i in range(n_total):
            communities[int(labels[i])].add(int(i))
        return island_of, communities
    F = nx.read_edgelist(str(edges_path), nodetype=int)
    F.add_nodes_from(range(n_nodes))
    communities = list(nx.community.greedy_modularity_communities(F))
    island_of = {n: ci for ci, c in enumerate(communities) for n in c}
    return island_of, communities


def _setup_iowa(repo_root, K=4, eps=0.05):
    data = repo_root / "IA_counties" / "IA_counties.json"
    print(f"Loading Iowa from {data} ...")
    graph = Graph.from_json(str(data))
    # Compute population density (TOTPOP / land area), normalised to mean 1
    raw_dens = []
    for n in graph.nodes():
        graph.nodes[n]["population"] = graph.nodes[n]["TOTPOP"]
        area = max(float(graph.nodes[n].get("ALAND10", 1)), 1.0)
        rho = graph.nodes[n]["TOTPOP"] / area
        graph.nodes[n]["density_raw"] = rho
        raw_dens.append(rho)
    import numpy as _np
    mean_rho = float(_np.mean(raw_dens)) or 1.0
    for n in graph.nodes():
        graph.nodes[n]["density"] = graph.nodes[n]["density_raw"] / mean_rho
    total = sum(graph.nodes[n]["TOTPOP"] for n in graph.nodes())
    ideal = total / K
    init = dict(recursive_tree_part(graph, range(K), ideal, "TOTPOP", eps))
    label = f"Iowa, k={K}, ε={eps}"
    return {
        "graph": graph, "K": K, "eps": eps, "ideal": ideal,
        "init": init, "label": label,
        "fiber_lookup": None, "fiber_cuts": None,
    }


def _setup_5x5(repo_root, fiber_dir, K=5, eps=0.05):
    print("Building 5x5 grid graph ...")
    G = nx.grid_graph([5, 5])
    graph = Graph.from_networkx(G)
    for n in graph.nodes():
        graph.nodes[n]["population"] = 50
        graph.nodes[n]["density"] = 0
    ideal = 250.0
    init = dict(recursive_tree_part(graph, range(K), ideal, "population", eps))

    fiber_path = fiber_dir / "fiber_5x5_partitions.json"
    with open(fiber_path) as f:
        data = json.load(f)
    fiber_lookup = {}
    fiber_cuts = {}
    for pid, parts in data.items():
        canon = frozenset(
            frozenset(tuple(s) for s in part) for part in parts
        )
        fiber_lookup[canon] = int(pid)
        asn = {}
        for d_id, part in enumerate(parts):
            for s in part:
                asn[tuple(s)] = d_id
        fiber_cuts[int(pid)] = sum(
            1 for u, v in graph.edges() if asn[u] != asn[v])
    label = f"5×5 grid, k={K}, ε={eps}, |fiber|={len(fiber_lookup)}"
    return {
        "graph": graph, "K": K, "eps": eps, "ideal": ideal,
        "init": init, "label": label,
        "fiber_lookup": fiber_lookup, "fiber_cuts": fiber_cuts,
    }


def _load_simple_dual_graph_json(path):
    """Custom loader for the {nodes, adjacency} JSON dialect used by some
    MGGG-states files (no `multigraph`/`directed` flags, plain integer ids).
    Returns a gerrychain.Graph."""
    with open(path) as f:
        data = json.load(f)
    H = nx.Graph()
    for node in data["nodes"]:
        nid = node["id"]
        attrs = {k: v for k, v in node.items() if k != "id"}
        H.add_node(nid, **attrs)
    for u, neighbours in enumerate(data["adjacency"]):
        for nb in neighbours:
            v = nb["id"]
            if not H.has_edge(u, v):
                H.add_edge(u, v)
    return Graph.from_networkx(H)


def _setup_nc(repo_root, K=4, eps=0.05):
    """North Carolina precinct-level (NC_VTD): 2692 precincts, k=4
    sub-state plan to keep pair sizes comparable to half-state."""
    data = repo_root / "diffusion_model" / "mew" / "data" / "NC_VTD.json"
    print(f"Loading NC from {data} ...")
    graph = _load_simple_dual_graph_json(data)
    for n in graph.nodes():
        graph.nodes[n]["population"] = graph.nodes[n]["TOTPOP"]
        graph.nodes[n]["density"] = 0
    total = sum(graph.nodes[n]["TOTPOP"] for n in graph.nodes())
    ideal = total / K
    print(f"  {len(graph.nodes())} precincts, {len(graph.edges())} edges; "
          f"ideal pop = {ideal:,.0f}")
    init = dict(recursive_tree_part(graph, range(K), ideal, "TOTPOP", eps))
    label = f"NC precinct, k={K}, ε={eps}"
    return {
        "graph": graph, "K": K, "eps": eps, "ideal": ideal,
        "init": init, "label": label,
        "fiber_lookup": None, "fiber_cuts": None,
    }


def _setup_5x5_eps02(repo_root, fiber_dir, K=5, eps=0.20):
    """5×5 grid with ε=0.20 — full 193,128-partition fiber enumerated."""
    print("Building 5x5 grid graph (ε=0.20 fiber) ...")
    G = nx.grid_graph([5, 5])
    graph = Graph.from_networkx(G)
    for n in graph.nodes():
        graph.nodes[n]["population"] = 50
        graph.nodes[n]["density"] = 0
    ideal = 250.0
    init = dict(recursive_tree_part(graph, range(K), ideal, "population", eps))

    fiber_path = fiber_dir / "fiber_5x5_eps02_partitions.json"
    print(f"  Loading 193k fiber from {fiber_path} ...")
    with open(fiber_path) as f:
        data = json.load(f)
    # JSON stores coords as strings like "(0, 0)"; parse to tuples
    def _parse(s):
        # "(x, y)" → (x, y)
        return tuple(int(x.strip()) for x in s.strip("()").split(","))

    fiber_lookup = {}
    fiber_cuts = {}
    for pid, info in data.items():
        parts = info["parts"] if isinstance(info, dict) else info
        canon = frozenset(
            frozenset(_parse(s) for s in part) for part in parts
        )
        fiber_lookup[canon] = int(pid)
        asn = {}
        for d_id, part in enumerate(parts):
            for s in part:
                asn[_parse(s)] = d_id
        fiber_cuts[int(pid)] = sum(
            1 for u, v in graph.edges() if asn[u] != asn[v])
    label = (f"5×5 grid, k={K}, ε={eps}, "
             f"|fiber|={len(fiber_lookup)} (ε=0.20 enumerated)")
    return {
        "graph": graph, "K": K, "eps": eps, "ideal": ideal,
        "init": init, "label": label,
        "fiber_lookup": fiber_lookup, "fiber_cuts": fiber_cuts,
    }


def setup_problem(mode, repo_root, fiber_dir=None):
    """Return a dict {graph, K, eps, ideal, init, label, fiber_lookup, fiber_cuts}.
    fiber_lookup and fiber_cuts are None for modes without ground-truth fiber."""
    if mode == "iowa":
        return _setup_iowa(repo_root)
    if mode == "5x5":
        if fiber_dir is None:
            fiber_dir = Path(__file__).parent / "fiber_5x5"
        return _setup_5x5(repo_root, fiber_dir)
    if mode == "5x5_eps02":
        if fiber_dir is None:
            fiber_dir = Path(__file__).parent / "fiber_5x5"
        return _setup_5x5_eps02(repo_root, fiber_dir)
    if mode == "nc":
        return _setup_nc(repo_root)
    raise ValueError(f"unknown mode {mode!r}")
