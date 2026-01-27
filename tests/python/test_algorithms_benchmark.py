#!/usr/bin/env python3
"""
Test and benchmark Graphos graph algorithms.

This script tests the functionality of all algorithms and benchmarks
their performance comparing:
- Native NetworkX
- Graphos (via NetworkX adapter)
- Native solvOR
- Graphos (via solvOR adapter)
"""

import time
import random
from typing import Callable, Any

# Check if graphos is available
try:
    import graphos
    GRAPHOS_AVAILABLE = True
except ImportError:
    GRAPHOS_AVAILABLE = False
    print("WARNING: graphos not installed")

# Check if networkx is available
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("WARNING: networkx not installed, skipping NetworkX benchmarks")

# Check if solvor is available
try:
    import solvor
    SOLVOR_AVAILABLE = True
except ImportError:
    SOLVOR_AVAILABLE = False
    print("WARNING: solvor not installed, skipping solvOR benchmarks")


def create_test_graph(db: "graphos.GraphosDB", n_nodes: int = 100, n_edges: int = 300) -> dict:
    """Create a random graph for testing."""
    node_ids = []
    for i in range(n_nodes):
        node = db.create_node(["Node"], {"index": i})
        node_ids.append(node.id)

    edges = set()
    while len(edges) < n_edges:
        src = random.choice(node_ids)
        dst = random.choice(node_ids)
        if src != dst and (src, dst) not in edges:
            db.create_edge(src, dst, "EDGE", {"weight": random.uniform(0.1, 10.0)})
            edges.add((src, dst))

    return {"node_ids": node_ids, "edges": list(edges)}


def create_networkx_graph(n_nodes: int = 100, n_edges: int = 300) -> "nx.DiGraph":
    """Create an equivalent NetworkX graph."""
    G = nx.DiGraph()
    for i in range(n_nodes):
        G.add_node(i, index=i)

    edges = set()
    while len(edges) < n_edges:
        src = random.randint(0, n_nodes - 1)
        dst = random.randint(0, n_nodes - 1)
        if src != dst and (src, dst) not in edges:
            G.add_edge(src, dst, weight=random.uniform(0.1, 10.0))
            edges.add((src, dst))

    return G


def create_solvor_graph(n_nodes: int = 100, n_edges: int = 300) -> dict:
    """Create an equivalent solvOR graph representation.

    solvOR uses a functional API with nodes iterable and neighbors callback.
    Returns a dict with nodes, adjacency list, and helper functions.
    """
    adj = [[] for _ in range(n_nodes)]  # List of (dst, weight) tuples
    adj_unweighted = [[] for _ in range(n_nodes)]  # List of dst nodes only
    edges_list = []

    edges = set()
    while len(edges) < n_edges:
        src = random.randint(0, n_nodes - 1)
        dst = random.randint(0, n_nodes - 1)
        if src != dst and (src, dst) not in edges:
            weight = random.uniform(0.1, 10.0)
            adj[src].append((dst, weight))
            adj_unweighted[src].append(dst)
            edges_list.append((src, dst, weight))
            edges.add((src, dst))

    return {
        "nodes": list(range(n_nodes)),
        "adj_weighted": adj,
        "adj_unweighted": adj_unweighted,
        "edges": edges_list,
        "n_nodes": n_nodes,
        # Neighbor functions for solvOR functional API
        "neighbors": lambda n: adj_unweighted[n],
        "neighbors_weighted": lambda n: adj[n],
    }


def benchmark(name: str, func: Callable, iterations: int = 3) -> float:
    """Benchmark a function and return average time in ms."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    avg = sum(times) / len(times)
    return avg


def test_traversal_algorithms(db: "graphos.GraphosDB", node_ids: list):
    """Test traversal algorithms."""
    print("\n=== Traversal Algorithms ===")

    start_node = node_ids[0]

    # BFS
    result = db.algorithms.bfs(start_node)
    print(f"  BFS from node {start_node}: visited {len(result)} nodes")
    assert len(result) > 0, "BFS should visit at least the start node"

    # BFS layers
    layers = db.algorithms.bfs_layers(start_node)
    print(f"  BFS layers: {len(layers)} layers")
    assert len(layers) > 0, "BFS layers should have at least one layer"

    # DFS
    result = db.algorithms.dfs(start_node)
    print(f"  DFS from node {start_node}: visited {len(result)} nodes")

    # DFS all
    result = db.algorithms.dfs_all()
    unique_nodes = len(set(result))
    print(f"  DFS all: visited {len(result)} nodes ({unique_nodes} unique)")
    assert unique_nodes <= len(node_ids), "DFS all should not visit more unique nodes than exist"

    print("  [PASS] Traversal algorithms")


def test_component_algorithms(db: "graphos.GraphosDB", node_ids: list):
    """Test component algorithms."""
    print("\n=== Component Algorithms ===")

    # Connected components
    components = db.algorithms.connected_components()
    print(f"  Connected components: {len(set(components.values()))} components")
    assert len(components) == len(node_ids), "All nodes should have a component"

    # Component count
    count = db.algorithms.connected_component_count()
    print(f"  Component count: {count}")

    # SCC
    scc = db.algorithms.strongly_connected_components()
    print(f"  Strongly connected components: {len(scc)} SCCs")

    # Is DAG
    is_dag = db.algorithms.is_dag()
    print(f"  Is DAG: {is_dag}")

    # Topological sort (may be None if not DAG)
    topo = db.algorithms.topological_sort()
    if topo is not None:
        print(f"  Topological sort: {len(topo)} nodes")
    else:
        print(f"  Topological sort: graph has cycle")

    print("  [PASS] Component algorithms")


def test_shortest_path_algorithms(db: "graphos.GraphosDB", node_ids: list):
    """Test shortest path algorithms."""
    print("\n=== Shortest Path Algorithms ===")

    source = node_ids[0]
    target = node_ids[min(10, len(node_ids) - 1)]

    # Dijkstra - all distances
    distances = db.algorithms.dijkstra(source)
    print(f"  Dijkstra from {source}: found distances to {len(distances)} nodes")

    # Dijkstra - single path
    result = db.algorithms.dijkstra(source, target, "weight")
    if result is not None:
        dist, path = result
        print(f"  Dijkstra path {source} -> {target}: distance={dist:.2f}, path length={len(path)}")
    else:
        print(f"  Dijkstra path {source} -> {target}: no path found")

    # A* (with zero heuristic)
    result = db.algorithms.astar(source, target, weight="weight")
    if result is not None:
        dist, path = result
        print(f"  A* path {source} -> {target}: distance={dist:.2f}, path length={len(path)}")
    else:
        print(f"  A* path {source} -> {target}: no path found")

    # Bellman-Ford
    bf_result = db.algorithms.bellman_ford(source, "weight")
    print(f"  Bellman-Ford: {len(bf_result['distances'])} distances, negative cycle={bf_result['has_negative_cycle']}")

    # Floyd-Warshall (can be slow for large graphs)
    if len(node_ids) <= 200:
        fw_result = db.algorithms.floyd_warshall("weight")
        print(f"  Floyd-Warshall: {len(fw_result)} pairs")
    else:
        print(f"  Floyd-Warshall: skipped (graph too large)")

    print("  [PASS] Shortest path algorithms")


def test_centrality_algorithms(db: "graphos.GraphosDB", node_ids: list):
    """Test centrality algorithms."""
    print("\n=== Centrality Algorithms ===")

    # Degree centrality
    degree = db.algorithms.degree_centrality()
    print(f"  Degree centrality: computed for {len(degree)} nodes")

    # Degree centrality normalized
    degree_norm = db.algorithms.degree_centrality(normalized=True)
    print(f"  Degree centrality (normalized): computed for {len(degree_norm)} nodes")

    # PageRank
    pr = db.algorithms.pagerank()
    pr_sum = sum(pr.values())
    print(f"  PageRank: {len(pr)} nodes, sum={pr_sum:.4f}")
    assert abs(pr_sum - 1.0) < 0.01, "PageRank should sum to ~1.0"

    # Betweenness centrality
    bc = db.algorithms.betweenness_centrality()
    print(f"  Betweenness centrality: {len(bc)} nodes")

    # Closeness centrality
    cc = db.algorithms.closeness_centrality()
    print(f"  Closeness centrality: {len(cc)} nodes")

    print("  [PASS] Centrality algorithms")


def test_community_algorithms(db: "graphos.GraphosDB", node_ids: list):
    """Test community detection algorithms."""
    print("\n=== Community Detection ===")

    # Label propagation
    lp = db.algorithms.label_propagation()
    n_communities_lp = len(set(lp.values()))
    print(f"  Label propagation: {n_communities_lp} communities")

    # Louvain
    louvain = db.algorithms.louvain()
    print(f"  Louvain: {louvain['num_communities']} communities, modularity={louvain['modularity']:.4f}")

    print("  [PASS] Community detection algorithms")


def test_mst_algorithms(db: "graphos.GraphosDB", node_ids: list):
    """Test MST algorithms."""
    print("\n=== Minimum Spanning Tree ===")

    # Kruskal
    kruskal = db.algorithms.kruskal("weight")
    print(f"  Kruskal: {len(kruskal['edges'])} edges, total weight={kruskal['total_weight']:.2f}")

    # Prim
    prim = db.algorithms.prim("weight")
    print(f"  Prim: {len(prim['edges'])} edges, total weight={prim['total_weight']:.2f}")

    print("  [PASS] MST algorithms")


def test_flow_algorithms(db: "graphos.GraphosDB", node_ids: list):
    """Test network flow algorithms."""
    print("\n=== Network Flow ===")

    source = node_ids[0]
    sink = node_ids[-1]

    # Max flow
    flow = db.algorithms.max_flow(source, sink)
    print(f"  Max flow {source} -> {sink}: flow={flow['max_flow']:.2f}, {len(flow['flow_edges'])} flow edges")

    # Min cost max flow
    mcf = db.algorithms.min_cost_max_flow(source, sink)
    print(f"  Min cost flow: flow={mcf['max_flow']:.2f}, cost={mcf['total_cost']:.2f}")

    print("  [PASS] Flow algorithms")


def test_structure_algorithms(db: "graphos.GraphosDB", node_ids: list):
    """Test structure analysis algorithms."""
    print("\n=== Structure Analysis ===")

    # Articulation points
    ap = db.algorithms.articulation_points()
    print(f"  Articulation points: {len(ap)} found")

    # Bridges
    bridges = db.algorithms.bridges()
    print(f"  Bridges: {len(bridges)} found")

    # K-core decomposition
    kcore = db.algorithms.kcore()
    max_core = kcore.get('max_core', 0)
    print(f"  K-core: max core number = {max_core}")

    # Extract specific k-core
    if max_core >= 1:
        k1_core = db.algorithms.kcore(k=1)
        print(f"  1-core: {len(k1_core)} nodes")

    print("  [PASS] Structure analysis algorithms")


def test_networkx_adapter(db: "graphos.GraphosDB"):
    """Test NetworkX adapter."""
    print("\n=== NetworkX Adapter ===")

    nx_adapter = db.as_networkx()
    print(f"  Adapter: {nx_adapter}")
    print(f"  Nodes: {nx_adapter.number_of_nodes}")
    print(f"  Edges: {nx_adapter.number_of_edges}")
    print(f"  Directed: {nx_adapter.is_directed}")

    # Test native algorithms through adapter
    pr = nx_adapter.pagerank()
    print(f"  PageRank via adapter: {len(pr)} nodes")

    # Test conversion to NetworkX (if available)
    if NETWORKX_AVAILABLE:
        G = nx_adapter.to_networkx()
        print(f"  Converted to NetworkX: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("  [PASS] NetworkX adapter")


def test_solvor_adapter(db: "graphos.GraphosDB", node_ids: list):
    """Test solvOR adapter."""
    print("\n=== solvOR Adapter ===")

    solvor_adapter = db.as_solvor()
    print(f"  Adapter: {solvor_adapter}")

    # Graph stats
    stats = solvor_adapter.graph_stats()
    print(f"  Stats: {stats['nodes']} nodes, {stats['edges']} edges, density={stats['density']:.4f}")

    # Shortest path
    source = node_ids[0]
    target = node_ids[min(10, len(node_ids) - 1)]
    result = solvor_adapter.shortest_path(source, target, weight="weight")
    if result is not None:
        dist, path = result
        print(f"  Shortest path: distance={dist:.2f}")

    # MST
    mst = solvor_adapter.minimum_spanning_tree(weight="weight", method="kruskal")
    print(f"  MST: {len(mst['edges'])} edges")

    # Max flow
    flow = solvor_adapter.max_flow(source, node_ids[-1])
    print(f"  Max flow: {flow['max_flow']:.2f}")

    print("  [PASS] solvOR adapter")


def run_benchmarks(n_nodes: int = 1000, n_edges: int = 5000):
    """Run performance benchmarks comparing all 4 implementations."""
    print(f"\n{'='*100}")
    print(f"BENCHMARKS (n={n_nodes}, e={n_edges})")
    print(f"Comparing: NetworkX (native), Graphos-NetworkX bridge, solvOR (native), Graphos-solvOR bridge")
    print('='*100)

    # Seed for reproducibility
    random.seed(42)

    # Create Graphos graph
    print("\nCreating Graphos graph...")
    db = graphos.GraphosDB()
    graph_info = create_test_graph(db, n_nodes, n_edges)
    node_ids = graph_info["node_ids"]
    source = node_ids[0]
    target = node_ids[n_nodes // 2]

    # Get Graphos adapters
    nx_adapter = db.as_networkx()
    sv_adapter = db.as_solvor()

    # Create native NetworkX graph (if available)
    nx_graph = None
    if NETWORKX_AVAILABLE:
        print("Creating native NetworkX graph...")
        random.seed(42)
        nx_graph = create_networkx_graph(n_nodes, n_edges)

    # Create native solvOR graph (if available)
    solvor_graph = None
    if SOLVOR_AVAILABLE:
        print("Creating native solvOR graph...")
        random.seed(42)
        solvor_graph = create_solvor_graph(n_nodes, n_edges)

    algorithms_to_test = [
        "PageRank",
        "Dijkstra",
        "BFS",
        "DFS",
        "SCC",
        "Louvain",
        "Articulation Points",
        "Bridges",
        "K-Core",
        "Betweenness",
    ]

    print("\n" + "-"*100)
    print(f"{'Algorithm':<20} {'NetworkX':<15} {'Graphos-NX':<15} {'solvOR':<15} {'Graphos-solvOR':<15} {'Best':<15}")
    print("-"*100)

    for algo in algorithms_to_test:
        nx_native = None
        nx_graphos = None
        sv_native = None
        sv_graphos = None

        try:
            if algo == "PageRank":
                if NETWORKX_AVAILABLE:
                    nx_native = benchmark(algo, lambda: nx.pagerank(nx_graph))
                nx_graphos = benchmark(algo, lambda: nx_adapter.pagerank())
                if SOLVOR_AVAILABLE and solvor_graph:
                    nodes = solvor_graph["nodes"]
                    neighbors = solvor_graph["neighbors"]
                    sv_native = benchmark(algo, lambda: solvor.pagerank(nodes, neighbors))
                sv_graphos = benchmark(algo, lambda: sv_adapter.pagerank())

            elif algo == "Dijkstra":
                if NETWORKX_AVAILABLE:
                    nx_native = benchmark(algo, lambda: nx.dijkstra_path(nx_graph, 0, n_nodes // 2, weight="weight"))
                nx_graphos = benchmark(algo, lambda: nx_adapter.shortest_path(source, target, "weight"))
                if SOLVOR_AVAILABLE and solvor_graph:
                    neighbors_w = solvor_graph["neighbors_weighted"]
                    sv_native = benchmark(algo, lambda: solvor.dijkstra(0, n_nodes // 2, neighbors_w))
                sv_graphos = benchmark(algo, lambda: sv_adapter.shortest_path(source, target, weight="weight"))

            elif algo == "BFS":
                if NETWORKX_AVAILABLE:
                    nx_native = benchmark(algo, lambda: list(nx.bfs_edges(nx_graph, 0)))
                nx_graphos = benchmark(algo, lambda: db.algorithms.bfs(source))
                if SOLVOR_AVAILABLE and solvor_graph:
                    neighbors = solvor_graph["neighbors"]
                    sv_native = benchmark(algo, lambda: solvor.bfs(0, None, neighbors))
                sv_graphos = benchmark(algo, lambda: db.algorithms.bfs(source))

            elif algo == "DFS":
                if NETWORKX_AVAILABLE:
                    nx_native = benchmark(algo, lambda: list(nx.dfs_edges(nx_graph, 0)))
                nx_graphos = benchmark(algo, lambda: db.algorithms.dfs(source))
                if SOLVOR_AVAILABLE and solvor_graph:
                    neighbors = solvor_graph["neighbors"]
                    sv_native = benchmark(algo, lambda: solvor.dfs(0, None, neighbors))
                sv_graphos = benchmark(algo, lambda: db.algorithms.dfs(source))

            elif algo == "SCC":
                if NETWORKX_AVAILABLE:
                    nx_native = benchmark(algo, lambda: list(nx.strongly_connected_components(nx_graph)))
                nx_graphos = benchmark(algo, lambda: db.algorithms.strongly_connected_components())
                if SOLVOR_AVAILABLE and solvor_graph:
                    nodes = solvor_graph["nodes"]
                    neighbors = solvor_graph["neighbors"]
                    sv_native = benchmark(algo, lambda: solvor.strongly_connected_components(nodes, neighbors))
                sv_graphos = benchmark(algo, lambda: sv_adapter.strongly_connected_components())

            elif algo == "Louvain":
                if NETWORKX_AVAILABLE:
                    nx_native = benchmark(algo, lambda: nx.community.louvain_communities(nx_graph.to_undirected()))
                nx_graphos = benchmark(algo, lambda: db.algorithms.louvain())
                if SOLVOR_AVAILABLE and solvor_graph:
                    nodes = solvor_graph["nodes"]
                    neighbors = solvor_graph["neighbors"]
                    sv_native = benchmark(algo, lambda: solvor.louvain(nodes, neighbors))
                sv_graphos = benchmark(algo, lambda: sv_adapter.louvain())

            elif algo == "Articulation Points":
                if NETWORKX_AVAILABLE:
                    nx_native = benchmark(algo, lambda: list(nx.articulation_points(nx_graph.to_undirected())))
                nx_graphos = benchmark(algo, lambda: db.algorithms.articulation_points())
                if SOLVOR_AVAILABLE and solvor_graph:
                    nodes = solvor_graph["nodes"]
                    neighbors = solvor_graph["neighbors"]
                    sv_native = benchmark(algo, lambda: solvor.articulation_points(nodes, neighbors))
                sv_graphos = benchmark(algo, lambda: sv_adapter.articulation_points())

            elif algo == "Bridges":
                if NETWORKX_AVAILABLE:
                    nx_native = benchmark(algo, lambda: list(nx.bridges(nx_graph.to_undirected())))
                nx_graphos = benchmark(algo, lambda: db.algorithms.bridges())
                if SOLVOR_AVAILABLE and solvor_graph:
                    nodes = solvor_graph["nodes"]
                    neighbors = solvor_graph["neighbors"]
                    sv_native = benchmark(algo, lambda: solvor.bridges(nodes, neighbors))
                sv_graphos = benchmark(algo, lambda: sv_adapter.bridges())

            elif algo == "K-Core":
                if NETWORKX_AVAILABLE:
                    nx_native = benchmark(algo, lambda: nx.core_number(nx_graph.to_undirected()))
                nx_graphos = benchmark(algo, lambda: db.algorithms.kcore())
                if SOLVOR_AVAILABLE and solvor_graph:
                    nodes = solvor_graph["nodes"]
                    neighbors = solvor_graph["neighbors"]
                    sv_native = benchmark(algo, lambda: solvor.kcore_decomposition(nodes, neighbors))
                sv_graphos = benchmark(algo, lambda: db.algorithms.kcore())

            elif algo == "Betweenness":
                if NETWORKX_AVAILABLE:
                    nx_native = benchmark(algo, lambda: nx.betweenness_centrality(nx_graph), iterations=1)
                nx_graphos = benchmark(algo, lambda: db.algorithms.betweenness_centrality(), iterations=1)
                # solvOR doesn't have betweenness
                sv_native = None
                sv_graphos = benchmark(algo, lambda: db.algorithms.betweenness_centrality(), iterations=1)

        except Exception as e:
            print(f"  {algo}: Error - {e}")
            continue

        # Format output
        nx_nat_str = f"{nx_native:.2f}" if nx_native else "N/A"
        nx_gra_str = f"{nx_graphos:.2f}" if nx_graphos else "N/A"
        sv_nat_str = f"{sv_native:.2f}" if sv_native else "N/A"
        sv_gra_str = f"{sv_graphos:.2f}" if sv_graphos else "N/A"

        # Determine fastest
        times = []
        if nx_native: times.append(("NetworkX", nx_native))
        if nx_graphos: times.append(("Graphos-NX", nx_graphos))
        if sv_native: times.append(("solvOR", sv_native))
        if sv_graphos: times.append(("Graphos-solvOR", sv_graphos))

        if times:
            best_name, best_time = min(times, key=lambda x: x[1])
            best_str = f"{best_name}"
        else:
            best_str = "N/A"

        print(f"{algo:<20} {nx_nat_str:<15} {nx_gra_str:<15} {sv_nat_str:<15} {sv_gra_str:<15} {best_str:<15}")

    print("-"*100)
    print("\nNote: Graphos-NX and Graphos-solvOR use native Graphos Rust algorithms via Python bridges")
    print("Lower time is better. 'Best' shows which implementation was fastest.")


def run_scaling_benchmark():
    """Run scaling benchmarks with different graph sizes."""
    print(f"\n{'='*100}")
    print("SCALING BENCHMARKS - PageRank")
    print('='*100)

    sizes = [
        (100, 500),
        (500, 2500),
        (1000, 5000),
        (2000, 10000),
        (5000, 25000),
    ]

    print(f"\n{'Nodes':<8} {'Edges':<8} {'NetworkX':<12} {'Graphos-NX':<12} {'solvOR':<12} {'Graphos-SV':<12} {'Fastest':<15}")
    print("-" * 80)

    for n_nodes, n_edges in sizes:
        random.seed(42)

        # Graphos
        db = graphos.GraphosDB()
        create_test_graph(db, n_nodes, n_edges)
        nx_adapter = db.as_networkx()
        sv_adapter = db.as_solvor()

        g_nx_time = benchmark("PageRank", lambda: nx_adapter.pagerank(), iterations=3)
        g_sv_time = benchmark("PageRank", lambda: sv_adapter.pagerank(), iterations=3)

        # NetworkX native
        if NETWORKX_AVAILABLE:
            random.seed(42)
            nx_graph = create_networkx_graph(n_nodes, n_edges)
            nx_time = benchmark("PageRank", lambda: nx.pagerank(nx_graph), iterations=3)
        else:
            nx_time = None

        # solvOR native
        if SOLVOR_AVAILABLE:
            random.seed(42)
            solvor_graph = create_solvor_graph(n_nodes, n_edges)
            nodes = solvor_graph["nodes"]
            neighbors = solvor_graph["neighbors"]
            sv_time = benchmark("PageRank", lambda: solvor.pagerank(nodes, neighbors), iterations=3)
        else:
            sv_time = None

        # Format
        nx_str = f"{nx_time:.2f}" if nx_time else "N/A"
        g_nx_str = f"{g_nx_time:.2f}"
        sv_str = f"{sv_time:.2f}" if sv_time else "N/A"
        g_sv_str = f"{g_sv_time:.2f}"

        # Find fastest
        times = []
        if nx_time: times.append(("NetworkX", nx_time))
        times.append(("Graphos-NX", g_nx_time))
        if sv_time: times.append(("solvOR", sv_time))
        times.append(("Graphos-SV", g_sv_time))
        best_name, best_time = min(times, key=lambda x: x[1])

        print(f"{n_nodes:<8} {n_edges:<8} {nx_str:<12} {g_nx_str:<12} {sv_str:<12} {g_sv_str:<12} {best_name:<15}")


def main():
    if not GRAPHOS_AVAILABLE:
        print("ERROR: graphos package not available")
        return

    print("="*100)
    print("GRAPHOS ALGORITHM TESTS")
    print("="*100)

    # Create test graph
    random.seed(42)
    db = graphos.GraphosDB()
    graph_info = create_test_graph(db, n_nodes=100, n_edges=300)
    node_ids = graph_info["node_ids"]

    print(f"\nTest graph: {db.node_count} nodes, {db.edge_count} edges")

    # Run all algorithm tests
    test_traversal_algorithms(db, node_ids)
    test_component_algorithms(db, node_ids)
    test_shortest_path_algorithms(db, node_ids)
    test_centrality_algorithms(db, node_ids)
    test_community_algorithms(db, node_ids)
    test_mst_algorithms(db, node_ids)
    test_flow_algorithms(db, node_ids)
    test_structure_algorithms(db, node_ids)
    test_networkx_adapter(db)
    test_solvor_adapter(db, node_ids)

    print("\n" + "="*100)
    print("ALL TESTS PASSED!")
    print("="*100)

    # Run benchmarks
    run_benchmarks(n_nodes=1000, n_edges=5000)

    # Run scaling benchmarks
    run_scaling_benchmark()


if __name__ == "__main__":
    main()
