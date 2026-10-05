---
title: Graph Algorithms
description: Built-in graph algorithms in Grafeo.
---

# Graph Algorithms

Grafeo includes a library of built-in graph algorithms for common analysis tasks, accessible via the `db.algorithms()` API.

## Algorithm Categories

<div class="grid cards" markdown>

-   :material-map-marker-path:{ .lg .middle } **Path Finding**

    ---

    Shortest paths, all paths and path analysis.

    [:octicons-arrow-right-24: Path Algorithms](path-finding.md)

-   :material-chart-bubble:{ .lg .middle } **Centrality**

    ---

    PageRank, betweenness, closeness and degree centrality.

    [:octicons-arrow-right-24: Centrality](centrality.md)

-   :material-group:{ .lg .middle } **Community Detection**

    ---

    Louvain, label propagation and connected components.

    [:octicons-arrow-right-24: Community Detection](community.md)

-   :material-link:{ .lg .middle } **Link Prediction**

    ---

    Predict missing or future relationships.

    [:octicons-arrow-right-24: Link Prediction](link-prediction.md)

-   :material-chart-scatter-plot:{ .lg .middle } **Similarity**

    ---

    Node similarity and graph matching.

    [:octicons-arrow-right-24: Similarity](similarity.md)

-   :material-graph:{ .lg .middle } **Graph Metrics**

    ---

    Density, diameter, clustering coefficient.

    [:octicons-arrow-right-24: Graph Metrics](metrics.md)

</div>

## Quick Reference

| Algorithm | Category | Complexity | Use Case |
|-----------|----------|------------|----------|
| Shortest Path | Path | O(V + E) | Navigation, routing |
| PageRank | Centrality | O(V + E) | Ranking, importance |
| Louvain | Community | O(V log V) | Clustering |
| Stochastic Block Partition | Community | O(V * B^2) | Bayesian community inference |
| Connected Components | Community | O(V + E) | Graph structure |
| Triangle Count | Metrics | O(E^1.5) | Clustering analysis |
| K-Truss | Structure | O(m^1.5) | Dense subgraph discovery |
| Subgraph Isomorphism | Structure | O(n^k) | Pattern matching, motif counting |

## Using Algorithms

### From Python

Algorithms are accessed through the `db.algorithms()` method:

```python
import grafeo

db = grafeo.GrafeoDB()

# Get the algorithms interface
algs = db.algorithms()

# Run PageRank
scores = algs.pagerank()
for node_id, score in scores.items():
    print(f"Node {node_id}: {score:.4f}")

# Find shortest path
path = algs.shortest_path(source=1, target=100)
print(f"Path length: {len(path)}")

# Connected components
components = algs.connected_components()
print(f"Found {len(components)} components")
```

### Available Methods

The `algorithms()` object provides these methods:

**Traversal:**

- `bfs(start)` - Breadth-first search
- `dfs(start)` - Depth-first search

**Shortest Paths:**

- `dijkstra(source, target)` - Dijkstra's algorithm
- `shortest_path(source, target)` - Generic shortest path
- `sssp(source, weight_attr)` - Single-source shortest paths (weighted)
- `all_pairs_shortest_path()` - All-to-all shortest paths

**Centrality:**

- `pagerank()` - PageRank algorithm
- `betweenness_centrality()` - Betweenness centrality
- `closeness_centrality()` - Closeness centrality
- `eigenvector_centrality()` - Eigenvector centrality
- `degree_centrality()` - Degree centrality

**Community Detection:**

- `connected_components()` - Find connected components
- `strongly_connected_components()` - Find SCCs
- `louvain()` - Louvain community detection
- `label_propagation()` - Label propagation

**Structure:**

- `triangles()` - Triangle counting
- `triangles(parallel=True)` - Parallel triangle counting (degree-ordered, merge intersection)
- `transitivity()` - Global clustering coefficient
- `ktruss(k=None)` - K-truss decomposition (dense subgraph discovery)
- `subgraph_isomorphism_count(pattern)` - Count subgraph isomorphisms
- `minimum_spanning_tree()` - MST construction
- `max_flow(source, sink)` - Maximum flow

### From Query Languages (GQL, Cypher, SQL/PGQ)

All algorithms are available via `CALL` statements in any supported query language:

```sql
-- Run PageRank with default parameters
CALL grafeo.pagerank()

-- Run PageRank with custom parameters
CALL grafeo.pagerank({damping: 0.85, max_iterations: 20})

-- Select specific columns with YIELD
CALL grafeo.pagerank() YIELD node_id, score

-- Alias output columns
CALL grafeo.pagerank() YIELD node_id AS id, score AS rank

-- Ignore edge direction: each pair of connected nodes counts once
CALL grafeo.pagerank({directed: false})

-- Run on a projection only (CREATE PROJECTION or create_projection())
CALL grafeo.pagerank({projection: 'people'})

-- List all available procedures
CALL grafeo.procedures()
```

A `CALL` reads the graph the session selects (`USE GRAPH`, or `set_graph()` in
Python), unless its `projection` argument names a projection.

Works the same way across all three languages:

=== "GQL"

    ```python
    result = db.execute("CALL grafeo.pagerank()")
    ```

=== "Cypher"

    ```python
    result = db.execute_cypher("CALL grafeo.pagerank()")
    ```

=== "SQL/PGQ"

    ```python
    result = db.execute_sql("CALL grafeo.pagerank()")
    ```

### Available Procedures

| Procedure | Category | Output Columns |
|-----------|----------|----------------|
| `grafeo.pagerank()` | Centrality | node_id, score |
| `grafeo.betweenness_centrality()` | Centrality | node_id, centrality |
| `grafeo.closeness_centrality()` | Centrality | node_id, centrality |
| `grafeo.degree_centrality()` | Centrality | node_id, in_degree, out_degree, total_degree |
| `grafeo.bfs(start)` | Traversal | node_id, depth |
| `grafeo.dfs(start)` | Traversal | node_id, depth |
| `grafeo.dijkstra(source)` | Shortest Path | node_id, distance |
| `grafeo.bellman_ford(source)` | Shortest Path | node_id, distance, has_negative_cycle |
| `grafeo.sssp(source, weight)` | Shortest Path | node_id, distance |
| `grafeo.floyd_warshall()` | Shortest Path | source, target, distance |
| `grafeo.connected_components()` | Components | node_id, component_id |
| `grafeo.strongly_connected_components()` | Components | node_id, component_id |
| `grafeo.topological_sort()` | Components | node_id, order |
| `grafeo.louvain()` | Community | node_id, community_id, modularity |
| `grafeo.label_propagation()` | Community | node_id, community_id |
| `grafeo.clustering_coefficient()` | Clustering | node_id, coefficient, triangle_count |
| `grafeo.kruskal()` | MST | source, target, weight |
| `grafeo.prim()` | MST | source, target, weight |
| `grafeo.max_flow(source, sink)` | Flow | source, target, flow, max_flow |
| `grafeo.min_cost_max_flow(source, sink)` | Flow | source, target, flow, cost, max_flow |
| `grafeo.articulation_points()` | Structure | node_id |
| `grafeo.bridges()` | Structure | source, target |
| `grafeo.k_core()` | Structure | node_id, core_number, max_core |
| `grafeo.ktruss()` | Structure | source, target, truss_number |
| `grafeo.total_triangles()` | Clustering | total_triangles |
| `grafeo.stochastic_block_partition()` | Community | node_id, block_id, description_length |
| `grafeo.subgraph_isomorphism()` | Structure | count |

## NetworkX Integration

For additional algorithms, use the NetworkX adapter:

```python
# Convert to NetworkX graph
nx_adapter = db.as_networkx(directed=True)
nx_graph = nx_adapter.to_networkx()

# Use any NetworkX algorithm
import networkx as nx
centrality = nx.eigenvector_centrality(nx_graph)
```

## Performance Considerations

- **Graph Size** - Large graphs may require more memory
- **Iterations** - More iterations = better accuracy, longer runtime
- **Parallelism** - Many algorithms support parallel execution
- **Caching** - Results can be cached for repeated queries
