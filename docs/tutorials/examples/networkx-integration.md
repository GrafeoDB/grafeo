---
title: NetworkX Integration
description: Convert Grafeo graphs to NetworkX for algorithms and matplotlib visualization.
tags:
  - example
  - networkx
  - visualization
---

# NetworkX Integration

Convert a Grafeo graph to NetworkX, run centrality and clustering algorithms, compare performance and visualize with matplotlib.

!!! tip "Run it locally"

    ```bash
    marimo run examples/networkx_integration.py
    ```

    **Requirements:** `grafeo`, `networkx`, `matplotlib`, `marimo`

## Build the Graph

```python
from grafeo import GrafeoDB

db = GrafeoDB()

people = {}
for name, age, city in [
    ("Alix", 30, "Utrecht"), ("Gus", 35, "Portland"),
    ("Harm", 28, "Utrecht"), ("Dave", 40, "San Francisco"),
    ("Eve", 32, "Portland"), ("Frank", 45, "Utrecht"),
    ("Grace", 29, "Portland"), ("Henry", 38, "San Francisco"),
]:
    node = db.create_node(["Person"], {"name": name, "age": age, "city": city})
    people[name] = node

connections = [
    ("Alix", "Gus"), ("Alix", "Harm"), ("Gus", "Harm"), ("Gus", "Dave"),
    ("Harm", "Eve"), ("Dave", "Eve"), ("Dave", "Frank"), ("Eve", "Grace"),
    ("Frank", "Grace"), ("Frank", "Henry"), ("Grace", "Henry"),
]

for p1, p2 in connections:
    db.create_edge(people[p1].id, people[p2].id, "KNOWS")

print(f"Created graph with {db.node_count} nodes and {db.edge_count} edges")
```

```title="Output"
Created graph with 8 nodes and 11 edges
```

## Convert to NetworkX

```python
nx_adapter = db.as_networkx(directed=False)

print(f"Nodes: {nx_adapter.number_of_nodes()}")
print(f"Edges: {nx_adapter.number_of_edges()}")

# Full NetworkX graph object
G = nx_adapter.to_networkx()
print(f"Graph type: {type(G).__name__}")
```

```title="Output"
Nodes: 8
Edges: 11
Graph type: Graph
```

## Centrality and Clustering

```python
import networkx as nx

betweenness = nx.betweenness_centrality(G)
clustering = nx.clustering(G)

node_names = {node.id: name for name, node in people.items()}

# Top 5 by betweenness
for node_id in sorted(betweenness, key=betweenness.get, reverse=True)[:5]:
    name = node_names.get(node_id, str(node_id))
    print(f"{name}: betweenness={betweenness[node_id]:.4f}  clustering={clustering[node_id]:.4f}")
```

**Betweenness centrality** measures how often a node lies on shortest paths between other nodes. **Clustering coefficient** measures how interconnected a node's neighbors are.

## Graph Metrics

```python
components = list(nx.connected_components(G))
diameter = nx.diameter(G)
avg_path = nx.average_shortest_path_length(G)
density = nx.density(G)

print(f"Connected components: {len(components)}")
print(f"Diameter: {diameter}")
print(f"Average path length: {avg_path:.2f}")
print(f"Density: {density:.4f}")
```

```title="Output"
Connected components: 1
Diameter: 3
Average path length: 1.89
Density: 0.3929
```

## Matplotlib Visualization

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
pos = nx.spring_layout(G, seed=42)

# Basic graph
nx.draw(
    G, pos, ax=axes[0], with_labels=True,
    labels={n: node_names.get(n, str(n)) for n in G.nodes()},
    node_color="lightblue", node_size=700,
    font_size=10, font_weight="bold", edge_color="gray",
)
axes[0].set_title("Social Network Graph")

# Colored by betweenness
bc = nx.betweenness_centrality(G)
node_colors = [bc[n] for n in G.nodes()]
nx.draw(
    G, pos, ax=axes[1], with_labels=True,
    labels={n: node_names.get(n, str(n)) for n in G.nodes()},
    node_color=node_colors, cmap=plt.cm.Reds, node_size=700,
    font_size=10, font_weight="bold", edge_color="gray",
)
axes[1].set_title("Colored by Betweenness Centrality")

plt.tight_layout()
plt.show()
```

The left plot shows the basic network layout. The right plot colors nodes by betweenness centrality: darker red indicates a more central position in the network.

## Algorithm Comparison

Both Grafeo and NetworkX can compute PageRank:

```python
# Grafeo (Rust-native)
grafeo_pr = db.algorithms.pagerank(damping=0.85)

# NetworkX (Python)
G_new = db.as_networkx().to_networkx()
nx_pr = nx.pagerank(G_new, alpha=0.85)
```

For large graphs, Grafeo's Rust implementation is significantly faster. NetworkX provides more specialized algorithms.

## When to Use Each

| Use Case | Recommendation |
|----------|----------------|
| Large graphs (1M+ nodes) | Grafeo algorithms |
| Visualization | NetworkX + matplotlib |
| Specialized algorithms | NetworkX |
| Graph storage and queries | Grafeo |
| Interactive exploration | Grafeo + anywidget-graph |
| Production applications | Grafeo (embedded, no server) |

The two libraries complement each other well.

## Next Steps

- [Graph Visualization example](graph-visualization.md) for interactive visualization with anywidget-graph
- [Algorithms reference](../../algorithms/index.md) for Grafeo's built-in algorithms
- [Python API](../../user-guide/python/index.md) for the full Python binding
