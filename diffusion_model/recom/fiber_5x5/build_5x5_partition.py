import json
import random
from pathlib import Path

import numpy as np
import networkx as nx

from gerrychain import Graph
from gerrychain.tree import recursive_tree_part


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

grid_size = 5      # 5 x 5 grid
k = 5              # number of districts
ns = 50            # population per node

graph = nx.grid_graph([grid_size, grid_size])

for n in graph.nodes():
    graph.nodes[n]["population"] = ns
    if 0 in n or grid_size - 1 in n:
        graph.nodes[n]["boundary_node"] = True
        graph.nodes[n]["boundary_perim"] = 1
    else:
        graph.nodes[n]["boundary_node"] = False

total_pop = sum(graph.nodes[v]["population"] for v in graph.nodes())
ideal_pop = total_pop / k

graph = Graph.from_networkx(graph)

random_assignment = recursive_tree_part(
    graph=graph,
    parts=range(k),
    pop_target=ideal_pop,
    pop_col="population",
    epsilon=0.1,
)

serializable_assignment = {
    f"{node[0]},{node[1]}": int(district)
    for node, district in random_assignment.items()
}

output = {
    "grid_size": grid_size,
    "num_districts": k,
    "population_per_node": ns,
    "total_population": total_pop,
    "ideal_pop": ideal_pop,
    "seed": SEED,
    "assignment": serializable_assignment,
}

out_path = Path(__file__).parent / "initial_partition_5x5.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Saved initial partition to {out_path}")
print("\nDistrict sizes:")
for d in range(k):
    nodes = [n for n, p in random_assignment.items() if p == d]
    pop = sum(graph.nodes[n]["population"] for n in nodes)
    print(f"  District {d}: {len(nodes)} nodes, population = {pop}")
