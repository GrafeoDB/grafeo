---
title: Migration Guide
description: Migrating to Grafeo from Neo4j, DGraph, and other graph databases.
tags:
  - migration
  - neo4j
  - cypher
---

# Migration Guide

How to migrate your existing graph database to Grafeo.

---

## Migrating from Neo4j

Grafeo supports Cypher queries, making migration from Neo4j straightforward.

### Step 1: Export Data from Neo4j

Use `CALL apoc.export.csv.all` or `neo4j-admin dump`:

```cypher
// Export nodes
CALL apoc.export.csv.query(
  "MATCH (n) RETURN id(n) as id, labels(n) as labels, properties(n) as props",
  "nodes.csv", {}
)

// Export relationships
CALL apoc.export.csv.query(
  "MATCH (a)-[r]->(b) RETURN id(r) as id, type(r) as type, id(a) as source, id(b) as target, properties(r) as props",
  "edges.csv", {}
)
```

### Step 2: Import into Grafeo

```python
import csv
import json
from grafeo import GrafeoDB

db = GrafeoDB("./migrated_db")

# Map old Neo4j IDs to new Grafeo IDs
id_map = {}

# Import nodes
with open("nodes.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        old_id = row["id"]
        labels = json.loads(row["labels"])
        props = json.loads(row["props"])

        node = db.create_node(labels, props)
        id_map[old_id] = node.id

# Import edges
with open("edges.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        edge_type = row["type"]
        source = id_map[row["source"]]
        target = id_map[row["target"]]
        props = json.loads(row["props"])

        db.create_edge(source, target, edge_type, props)

print(f"Imported {db.node_count} nodes and {db.edge_count} edges")
```

### Step 3: Update Your Queries

Most Cypher queries work unchanged:

=== "Neo4j"
    ```cypher
    MATCH (p:Person)-[:KNOWS]->(friend)
    WHERE p.age > 30
    RETURN p.name, friend.name
    ```

=== "Grafeo (Cypher)"
    ```python
    result = db.execute_cypher("""
        MATCH (p:Person)-[:KNOWS]->(friend)
        WHERE p.age > 30
        RETURN p.name, friend.name
    """)
    ```

=== "Grafeo (GQL)"
    ```python
    result = db.execute("""
        MATCH (p:Person)-[:KNOWS]->(friend)
        WHERE p.age > 30
        RETURN p.name, friend.name
    """)
    ```

### Cypher Compatibility Notes

Grafeo supports openCypher 9.0 with these differences:

| Feature | Neo4j | Grafeo |
|---------|-------|--------|
| `MATCH` | Supported | Supported |
| `WHERE` | Supported | Supported |
| `RETURN` | Supported | Supported |
| `CREATE` | Supported | Supported |
| `MERGE` | Supported | Supported |
| `DELETE` | Supported | Supported |
| `SET` | Supported | Supported |
| `WITH` | Supported | Supported |
| `UNWIND` | Supported | Supported |
| `OPTIONAL MATCH` | Supported | Supported |
| `CALL procedures` | Limited | Not supported |
| `APOC functions` | No | Use Python instead |
| `Graph algorithms` | Requires GDS | Built-in via `db.algorithms` |

### Replacing APOC with Python

Neo4j's APOC library provides utility functions. In Grafeo, use Python:

=== "Neo4j APOC"
    ```cypher
    CALL apoc.text.split("a,b,c", ",") YIELD value
    RETURN value
    ```

=== "Grafeo + Python"
    ```python
    text = "a,b,c"
    values = text.split(",")
    for value in values:
        print(value)
    ```

### Replacing GDS with Built-in Algorithms

Neo4j's Graph Data Science library is replaced by Grafeo's built-in algorithms:

=== "Neo4j GDS"
    ```cypher
    CALL gds.pageRank.stream('myGraph')
    YIELD nodeId, score
    RETURN gds.util.asNode(nodeId).name, score
    ORDER BY score DESC
    ```

=== "Grafeo"
    ```python
    # PageRank is built-in
    scores = db.algorithms.pagerank(damping=0.85)

    # Get top 10
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for node_id, score in sorted_scores[:10]:
        node = db.get_node(node_id)
        print(f"{node.properties.get('name')}: {score:.4f}")
    ```

---

## Migrating from DGraph

DGraph uses GraphQL for mutations and queries.

### Step 1: Export from DGraph

```bash
dgraph export --format=json -o ./export
```

### Step 2: Convert and Import

```python
import json
from grafeo import GrafeoDB

db = GrafeoDB("./migrated_db")

with open("export/g01.json") as f:
    data = json.load(f)

id_map = {}

# Import nodes (DGraph nodes have "uid" field)
for item in data:
    uid = item.pop("uid")
    dgraph_type = item.pop("dgraph.type", ["Unknown"])[0]

    # Create node with type as label
    node = db.create_node([dgraph_type], item)
    id_map[uid] = node.id

# Import edges (scan for uid references in properties)
for item in data:
    uid = item.get("uid")
    source_id = id_map.get(uid)
    if not source_id:
        continue

    for key, value in item.items():
        if isinstance(value, dict) and "uid" in value:
            # This is an edge
            target_uid = value["uid"]
            target_id = id_map.get(target_uid)
            if target_id:
                db.create_edge(source_id, target_id, key)
        elif isinstance(value, list):
            for v in value:
                if isinstance(v, dict) and "uid" in v:
                    target_uid = v["uid"]
                    target_id = id_map.get(target_uid)
                    if target_id:
                        db.create_edge(source_id, target_id, key)
```

### Step 3: Update Queries

=== "DGraph GraphQL"
    ```graphql
    {
      people(func: type(Person)) @filter(gt(age, 30)) {
        name
        knows {
          name
        }
      }
    }
    ```

=== "Grafeo GQL"
    ```python
    result = db.execute("""
        MATCH (p:Person)-[:knows]->(friend)
        WHERE p.age > 30
        RETURN p.name, friend.name
    """)
    ```

---

## Migrating from NetworkX

If you have existing NetworkX graphs:

```python
import networkx as nx
from grafeo import GrafeoDB

# Your existing NetworkX graph
G = nx.read_graphml("graph.graphml")

# Create Grafeo database
db = GrafeoDB("./migrated_db")

# Map NetworkX node IDs to Grafeo IDs
id_map = {}

# Import nodes
for nx_id, attrs in G.nodes(data=True):
    labels = attrs.pop("labels", ["Node"]).split(",") if isinstance(attrs.get("labels"), str) else ["Node"]
    node = db.create_node(labels, attrs)
    id_map[nx_id] = node.id

# Import edges
for source, target, attrs in G.edges(data=True):
    edge_type = attrs.pop("type", "CONNECTED")
    db.create_edge(id_map[source], id_map[target], edge_type, attrs)

print(f"Imported {db.node_count} nodes and {db.edge_count} edges")
```

You can also go the other direction:

```python
# Convert Grafeo to NetworkX for visualization
nx_adapter = db.as_networkx()
G = nx_adapter.to_networkx()

import matplotlib.pyplot as plt
nx.draw(G, with_labels=True)
plt.show()
```

---

## Migrating from RDF/SPARQL Stores

Grafeo supports RDF with SPARQL queries.

### From Turtle/N-Triples Files

```python
from grafeo import GrafeoDB

db = GrafeoDB("./rdf_db")

# Parse and insert triples
with open("data.ttl") as f:
    content = f.read()

# Use SPARQL INSERT DATA
# (Assumes simple N-Triples format)
for line in content.strip().split("\n"):
    if line.startswith("#") or not line.strip():
        continue
    parts = line.strip().rstrip(".").split()
    if len(parts) >= 3:
        s, p, o = parts[0], parts[1], " ".join(parts[2:])
        db.execute_sparql(f"INSERT DATA {{ {s} {p} {o} }}")
```

### Query Translation

=== "SPARQL (Original)"
    ```sparql
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    SELECT ?name ?friend
    WHERE {
        ?person foaf:name ?name .
        ?person foaf:knows ?other .
        ?other foaf:name ?friend .
    }
    ```

=== "Grafeo SPARQL"
    ```python
    result = db.execute_sparql("""
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT ?name ?friend
        WHERE {
            ?person foaf:name ?name .
            ?person foaf:knows ?other .
            ?other foaf:name ?friend .
        }
    """)
    ```

---

## Data Type Mapping

| Source Type | Grafeo Type |
|-------------|-------------|
| Integer/Long | `Int64` |
| Float/Double | `Float64` |
| String | `String` |
| Boolean | `Bool` |
| Date/DateTime | `String` (ISO format) |
| List/Array | `List` |
| Map/Object | `Map` |
| Null/None | `Null` |
| Point/Geo | `List` ([lat, lon]) |

---

## Performance Comparison

After migration, you may notice performance differences:

| Operation | Neo4j | Grafeo | Notes |
|-----------|-------|--------|-------|
| Point lookup | Fast | Fast | Both use hash indexes |
| Label scan | Fast | Fast | Both have label indexes |
| Pattern matching | Fast | Fast | Both optimize MATCH |
| Aggregations | Moderate | Fast | Grafeo uses vectorized execution |
| PageRank (1M nodes) | ~5s (GDS) | ~1s | Built-in algorithms |
| Vector similarity | Requires plugin | Native | HNSW built-in |

---

## Checklist

Before going live with your migration:

- [ ] Export and import all data
- [ ] Verify node and edge counts match
- [ ] Test critical queries
- [ ] Create indexes on frequently-queried properties
- [ ] Run any graph algorithms you need
- [ ] Test application integration
- [ ] Monitor performance in staging
- [ ] Plan rollback procedure
