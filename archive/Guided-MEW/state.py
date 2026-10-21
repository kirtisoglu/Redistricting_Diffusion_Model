"""
MEW lifted state: (T, M) = (spanning tree, marked edges).

The partition ξ is derived implicitly as the connected components
of the forest T \\ M. Each component is one district.
"""

import networkx as nx


class MEWState:
    """Lifted state for the Marked Edge Walk.

    Attributes
    ----------
    graph : nx.Graph
        The underlying redistricting graph G (read-only reference).
    tree : nx.Graph
        A spanning tree T of G.
    marked : set[frozenset]
        The k-1 marked edges M ⊆ E(T). Each edge is a frozenset({u, v}).
    """

    __slots__ = ("graph", "tree", "marked")

    def __init__(self, graph: nx.Graph, tree: nx.Graph, marked: set):
        self.graph = graph
        self.tree = tree
        self.marked = marked

    def get_partition(self) -> dict:
        """Compute the partition ξ = connected components of T \\ M.

        Returns
        -------
        assignment : dict
            Mapping {node: district_id} where district_id ∈ {0, ..., k-1}.
        """
        forest = self.tree.copy()
        for u, v in self.marked:
            forest.remove_edge(u, v)

        assignment = {}
        for district_id, component in enumerate(nx.connected_components(forest)):
            for node in component:
                assignment[node] = district_id
        return assignment

    def boundary_non_tree_edges(self) -> list:
        """Find non-tree edges whose endpoints are in different districts.

        These are the only edges that can change the partition when swapped
        into the tree via a cycle basis step.

        Returns
        -------
        edges : list[tuple]
            Each element is (u, v) with u, v in different districts.
        """
        assignment = self.get_partition()
        tree_edges = set(frozenset(e) for e in self.tree.edges())

        boundary = []
        for u, v in self.graph.edges():
            if frozenset((u, v)) not in tree_edges:
                if assignment[u] != assignment[v]:
                    boundary.append((u, v))
        return boundary

    def is_feasible(self, pop_target: float, epsilon: float) -> bool:
        """Check whether every district satisfies the population balance constraint.

        Parameters
        ----------
        pop_target : float
            Ideal population per district (total_pop / k).
        epsilon : float
            Fractional tolerance: |pop(ξ_i) - p̄| ≤ ε · p̄.

        Returns
        -------
        bool
        """
        assignment = self.get_partition()
        district_pops = {}
        for node, dist_id in assignment.items():
            pop = self.graph.nodes[node].get("population", 1)
            district_pops[dist_id] = district_pops.get(dist_id, 0) + pop

        for pop in district_pops.values():
            if abs(pop - pop_target) > epsilon * pop_target:
                return False
        return True

    def copy(self) -> "MEWState":
        """Deep copy of the state (graph is shared, tree and marked are copied)."""
        return MEWState(
            graph=self.graph,
            tree=self.tree.copy(),
            marked=set(self.marked),
        )
