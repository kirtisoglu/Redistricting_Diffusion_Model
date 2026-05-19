"""Count the number of partitions of the 5x5 grid into 5 connected pieces of 5 nodes each."""

import networkx as nx


def connected_subsets_containing(graph, seed, size):
    """All connected subsets of `graph` of `size` nodes containing `seed`."""
    results = set()

    def dfs(current, frontier):
        if len(current) == size:
            results.add(frozenset(current))
            return
        for u in list(frontier):
            new_current = current | {u}
            new_frontier = (frontier - {u}) | (set(graph.neighbors(u)) - new_current)
            dfs(new_current, new_frontier)

    dfs(frozenset({seed}), frozenset(graph.neighbors(seed)))
    return results


def count_partitions(graph, k, size):
    """Count partitions of graph into k connected parts each of given size.

    Uses lex-leader canonicalization: the part chosen at each level must
    contain the smallest remaining node, so each unordered partition is
    counted exactly once.
    """
    def helper(remaining):
        if len(remaining) == 0:
            return 1
        sub = graph.subgraph(remaining)
        if not nx.is_connected(sub):
            comps = list(nx.connected_components(sub))
            if any(len(c) % size != 0 for c in comps):
                return 0
        seed = min(remaining)
        sub = graph.subgraph(remaining)
        total = 0
        for piece in connected_subsets_containing(sub, seed, size):
            total += helper(remaining - piece)
        return total

    return helper(frozenset(graph.nodes()))


def main():
    grid_size = 5
    k = 5
    size = (grid_size * grid_size) // k

    G = nx.grid_graph([grid_size, grid_size])
    print(f"Counting partitions of {grid_size}x{grid_size} grid into {k} connected pieces of {size} nodes...")
    n = count_partitions(G, k, size)
    print(f"Number of equal-size connected k-partitions: {n}")


if __name__ == "__main__":
    main()
