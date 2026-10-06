---
title: Graph Visualization
description: Interactive graph visualization with Grafeo and anywidget-graph.
tags:
  - example
  - visualization
  - algorithms
---

# Graph Visualization

Build a social network, query it, run graph algorithms and visualize everything interactively.

!!! tip "Run it locally"

    ```bash
    marimo run examples/graph_visualization.py
    ```

    **Requirements:** `grafeo`, `anywidget-graph`, `marimo`

## Create the Graph

```python
from grafeo import GrafeoDB

db = GrafeoDB()

# Create people
alix = db.create_node(["Person"], {"name": "Alix", "age": 30, "city": "Utrecht"})
gus = db.create_node(["Person"], {"name": "Gus", "age": 35, "city": "Portland"})
harm = db.create_node(["Person"], {"name": "Harm", "age": 28, "city": "Utrecht"})
dave = db.create_node(["Person"], {"name": "Dave", "age": 40, "city": "San Francisco"})
eve = db.create_node(["Person"], {"name": "Eve", "age": 32, "city": "Portland"})

# Create companies
acme = db.create_node(["Company"], {"name": "Acme Corp", "industry": "Tech"})
globex = db.create_node(["Company"], {"name": "Globex Inc", "industry": "Finance"})

# Friendships
db.create_edge(alix.id, gus.id, "KNOWS", {"since": 2020})
db.create_edge(alix.id, harm.id, "KNOWS", {"since": 2019})
db.create_edge(gus.id, harm.id, "KNOWS", {"since": 2021})
db.create_edge(gus.id, dave.id, "KNOWS", {"since": 2018})
db.create_edge(harm.id, eve.id, "KNOWS", {"since": 2022})
db.create_edge(dave.id, eve.id, "KNOWS", {"since": 2020})

# Employment
db.create_edge(alix.id, acme.id, "WORKS_AT", {"role": "Engineer"})
db.create_edge(gus.id, acme.id, "WORKS_AT", {"role": "Manager"})
db.create_edge(harm.id, globex.id, "WORKS_AT", {"role": "Analyst"})
db.create_edge(dave.id, globex.id, "WORKS_AT", {"role": "Director"})
db.create_edge(eve.id, acme.id, "WORKS_AT", {"role": "Designer"})

print(f"Created {db.node_count} nodes and {db.edge_count} edges")
```

```title="Output"
Created 7 nodes and 11 edges
```

## Friends of Friends

```python
result = db.execute("""
    MATCH (p:Person)-[:KNOWS]->(friend)-[:KNOWS]->(fof:Person)
    WHERE p.name = 'Alix' AND p <> fof
    RETURN DISTINCT p.name AS person, fof.name AS friend_of_friend
""")

for row in result:
    print(f"{row['person']} -> {row['friend_of_friend']}")
```

```title="Output"
Alix -> Dave
Alix -> Eve
```

## Interactive Visualization

```python
from anywidget_graph import Graph

viz_result = db.execute("MATCH (n)-[r]->(m) RETURN n, r, m")

nodes = viz_result.nodes()
edges = viz_result.edges()

graph_nodes = []
seen_ids = set()
for node in nodes:
    if node.id not in seen_ids:
        seen_ids.add(node.id)
        labels = node.labels
        props = node.properties
        graph_nodes.append({
            "id": str(node.id),
            "label": props.get("name", f"Node {node.id}"),
            "group": labels[0] if labels else "Unknown",
            "properties": props,
        })

graph_edges = []
for edge in edges:
    graph_edges.append({
        "source": str(edge.source_id),
        "target": str(edge.target_id),
        "label": edge.edge_type,
        "properties": edge.properties,
    })

graph_widget = Graph(nodes=graph_nodes, edges=graph_edges, height=500)
graph_widget
```

This renders a force-directed graph with Person nodes (blue) and Company nodes (green), connected by `KNOWS` and `WORKS_AT` edges. Pan, zoom and drag nodes to explore.

## PageRank Analysis

```python
pagerank_scores = db.algorithms.pagerank(damping=0.85)

sorted_scores = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)

for node_id, score in sorted_scores[:5]:
    node = db.get_node(node_id)
    name = node.properties.get("name", f"Node {node_id}") if node else f"Node {node_id}"
    print(f"{name}: {score:.4f}")
```

Higher scores indicate more central nodes. In this network, people with many connections (like Gus and Harm) rank highest.

## Community Detection

```python
communities = db.algorithms.louvain(resolution=1.0)

community_groups = {}
for node_id, community_id in communities.items():
    if community_id not in community_groups:
        community_groups[community_id] = []
    node = db.get_node(node_id)
    if node:
        name = node.properties.get("name", f"Node {node_id}")
        community_groups[community_id].append(name)

for comm_id, members in sorted(community_groups.items()):
    print(f"Community {comm_id}: {', '.join(members)}")
```

The Louvain algorithm partitions the graph into clusters of densely connected nodes.

## Shortest Path

```python
utrecht_result = db.execute("""
    MATCH (p:Person)
    WHERE p.city = 'Utrecht'
    RETURN p.name AS name, id(p) AS id
""")

utrecht_people = [(row["name"], row["id"]) for row in utrecht_result]

if len(utrecht_people) >= 2:
    name1, id1 = utrecht_people[0]
    name2, id2 = utrecht_people[1]
    distances = db.algorithms.dijkstra(id1)
    if id2 in distances:
        print(f"Distance from {name1} to {name2}: {distances[id2]:.0f} hops")
```

## Next Steps

- [anywidget-graph docs](../../ecosystem/anywidget-graph.md) for full widget API
- [Algorithms reference](../../algorithms/index.md) for all built-in algorithms
- [Vector Search example](vector-search.md) for embedding-based search
