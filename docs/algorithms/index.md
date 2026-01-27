---
title: Graph Algorithms
description: Built-in graph algorithms in Graphos.
---

# Graph Algorithms

Graphos includes a library of built-in graph algorithms for common analysis tasks.

!!! note "Coming Soon"
    The algorithms module is under active development. This page documents the planned API and available algorithms.

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

### From GQL

```sql
-- PageRank
CALL graphos.pagerank({
    node_label: 'Page',
    relationship_type: 'LINKS',
    damping: 0.85,
    iterations: 20
})
YIELD nodeId, score
MATCH (p:Page) WHERE id(p) = nodeId
RETURN p.url, score
ORDER BY score DESC
LIMIT 10
```

### From Python

```python
import graphos
from graphos.algorithms import pagerank, shortest_path

db = graphos.Database()

# Run PageRank
scores = pagerank(db, damping=0.85, iterations=20)
for node_id, score in scores.items():
    print(f"Node {node_id}: {score:.4f}")

# Find shortest path
path = shortest_path(db, source_id=1, target_id=100)
print(f"Path length: {len(path)}")
```

### From Rust

```rust
use graphos::algorithms::{pagerank, PageRankConfig};

let config = PageRankConfig {
    damping: 0.85,
    iterations: 20,
    tolerance: 1e-6,
};

let scores = pagerank(&db, config)?;

for (node_id, score) in scores.iter() {
    println!("Node {}: {:.4}", node_id, score);
}
```

## Algorithm Configuration

Most algorithms accept configuration parameters:

```python
# PageRank configuration
pagerank(db,
    damping=0.85,           # Damping factor (0-1)
    iterations=20,          # Max iterations
    tolerance=1e-6,         # Convergence threshold
    node_label='Page',      # Filter by label
    relationship_type='LINKS'  # Filter by type
)
```

## Custom Algorithms

See [Extending Graphos](../user-guide/extending/plugins.md) to learn how to add custom algorithms.

## Performance Considerations

- **Graph Size** - Large graphs may require more memory
- **Iterations** - More iterations = better accuracy, longer runtime
- **Parallelism** - Many algorithms support parallel execution
- **Caching** - Results can be cached for repeated queries
