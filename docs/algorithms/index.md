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

    Shortest paths, all paths, and path analysis.

    [:octicons-arrow-right-24: Path Algorithms](path-finding.md)

-   :material-chart-bubble:{ .lg .middle } **Centrality**

    ---

    PageRank, betweenness, closeness, and degree centrality.

    [:octicons-arrow-right-24: Centrality](centrality.md)

-   :material-group:{ .lg .middle } **Community Detection**

    ---

    Louvain, label propagation, and connected components.

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
| Connected Components | Community | O(V + E) | Graph structure |
| Triangle Count | Metrics | O(E^1.5) | Clustering analysis |

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
- `transitivity()` - Global clustering coefficient
- `minimum_spanning_tree()` - MST construction
- `max_flow(source, sink)` - Maximum flow

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
