"""
data_loader.py — Load dual graphs for MEW experiments.

Supports any adjacency-format JSON dual graph with TOTPOP attribute.
Initial partitions are generated via GerryChain's recursive_tree_part.
"""

import json
from pathlib import Path
import networkx as nx
from gerrychain import Graph as GCGraph
from gerrychain.tree import recursive_tree_part

_DATA_DIR = Path(__file__).resolve().parent / "data"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_json_graph(path, name=None, k=2):
    """
    Load a dual graph from JSON (adjacency or node-link format).

    The JSON must have 'nodes' with 'TOTPOP' and 'id' fields,
    and either 'adjacency' or 'links' for edges.

    Returns nx.Graph with graph['name'] and graph['districts'] set.
    """
    with open(path) as f:
        data = json.load(f)

    id_to_node = {n["id"]: n for n in data["nodes"]}
    sorted_ids = sorted(id_to_node.keys())
    old_to_new = {old: new for new, old in enumerate(sorted_ids)}

    G = nx.Graph()
    for old_id in sorted_ids:
        n = id_to_node[old_id]
        new = old_to_new[old_id]
        pop = n.get("TOTPOP", n.get("TOT_POP", 0))
        attrs = {"population": pop, "old_id": old_id}
        for key in ("NAME10", "COUNTY", "Precinct", "area", "C_X", "C_Y"):
            if key in n:
                attrs[key] = n[key]
        # Fall back to INTPTLON/LAT for coordinates
        if "C_X" not in attrs and "INTPTLON10" in n:
            attrs["C_X"] = float(n["INTPTLON10"])
        if "C_Y" not in attrs and "INTPTLAT10" in n:
            attrs["C_Y"] = float(n["INTPTLAT10"])
        G.add_node(new, **attrs)

    # Support both adjacency and links format
    if "adjacency" in data:
        for i, adj_list in enumerate(data["adjacency"]):
            for link in adj_list:
                j = link["id"]
                if i < j:
                    G.add_edge(old_to_new[i], old_to_new[j],
                               shared_perim=link.get("shared_perim", 0))
    elif "links" in data:
        for link in data["links"]:
            s, t = link["source"], link["target"]
            if s in old_to_new and t in old_to_new:
                G.add_edge(old_to_new[s], old_to_new[t],
                           shared_perim=link.get("shared_perim", 0))

    G.graph["name"] = name or Path(path).stem
    G.graph["districts"] = k
    return G


def initial_partition(G, k, epsilon=0.10):
    """
    Generate a balanced k-partition via recursive_tree_part.

    Returns dict {node → district_id}.
    """
    gc_graph = GCGraph.from_networkx(G)
    total_pop = sum(G.nodes[v]["population"] for v in G)
    pop_target = total_pop / k
    asn = recursive_tree_part(
        graph=gc_graph,
        parts=range(k),
        pop_target=pop_target,
        pop_col="population",
        epsilon=epsilon,
    )
    return dict(asn)


# ═══════════════════════════════════════════════════════════════════════════════
#  Convenience loaders for specific datasets
# ═══════════════════════════════════════════════════════════════════════════════

def cheshire(k=2):
    data_path = _DATA_DIR / "NH_dual_graph.json"
    with open(data_path) as f:
        data = json.load(f)
    # Filter to Cheshire County
    cheshire_ids = {n["id"] for n in data["nodes"] if n.get("COUNTY") == "Cheshire"}
    id_to_node = {n["id"]: n for n in data["nodes"]}
    sorted_ids = sorted(cheshire_ids)
    old_to_new = {old: new for new, old in enumerate(sorted_ids)}

    G = nx.Graph()
    for old_id in sorted_ids:
        n = id_to_node[old_id]
        new = old_to_new[old_id]
        total_votes = n.get("PRES16D", 0) + n.get("PRES16R", 0)
        G.add_node(new,
                   population=n["TOTPOP"],
                   dem_share=n["PRES16D"] / total_votes if total_votes else 0.5,
                   old_id=old_id)
    for link in data["links"]:
        s, t = link["source"], link["target"]
        if s in cheshire_ids and t in cheshire_ids:
            G.add_edge(old_to_new[s], old_to_new[t])
    G.graph["name"] = "Cheshire County, NH"
    G.graph["districts"] = k
    return G


def new_hampshire(k=2):
    return load_json_graph(_DATA_DIR / "NH_dual_graph.json",
                           name="New Hampshire", k=k)


def iowa_counties(k=5):
    return load_json_graph(_PROJECT_ROOT / "IA_counties" / "IA_counties.json",
                           name="Iowa Counties", k=k)


def pennsylvania(k=18):
    return load_json_graph(_DATA_DIR / "PA_VTDs.json",
                           name="Pennsylvania", k=k)


def north_carolina(k=13):
    return load_json_graph(_DATA_DIR / "NC_VTD.json",
                           name="North Carolina", k=k)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for loader, expected_n in [(cheshire, 27), (new_hampshire, 320),
                                (north_carolina, 2692), (pennsylvania, 8921)]:
        G = loader()
        pop = sum(G.nodes[v]["population"] for v in G)
        print(f"  {G.graph['name']:30s}  n={G.number_of_nodes():>4d}  "
              f"e={G.number_of_edges():>4d}  pop={pop:>8,d}  "
              f"d={G.graph['districts']}")

    # Test initial partition
    G = iowa_counties(k=5)
    asn = initial_partition(G, k=5)
    print(f"\n  Iowa initial partition: {len(set(asn.values()))} districts")
    pops = {}
    for node, d in asn.items():
        pops[d] = pops.get(d, 0) + G.nodes[node]["population"]
    pt = sum(pops.values()) / 5
    for d, p in sorted(pops.items()):
        print(f"    D{d}: pop={p:>8,d}  dev={abs(p-pt)/pt:.4f}")
