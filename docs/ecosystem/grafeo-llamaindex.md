---
title: grafeo-llamaindex
description: LlamaIndex integration for GrafeoDB with PropertyGraphStore, vector search and knowledge graphs.
---

# grafeo-llamaindex

LlamaIndex integration that implements the `PropertyGraphStore` interface backed by GrafeoDB. Build knowledge graphs from documents and query them with structured and vector search.

[:octicons-mark-github-16: GitHub](https://github.com/GrafeoDB/grafeo-llamaindex){ .md-button }
[:material-package-variant: PyPI](https://pypi.org/project/grafeo-llamaindex/){ .md-button }

## Overview

grafeo-llamaindex provides `GrafeoPropertyGraphStore`, a full implementation of LlamaIndex's PropertyGraphStore interface. It supports:

- Structured queries (GQL, Cypher, Gremlin, GraphQL, SPARQL, SQL/PGQ)
- Vector similarity search via native HNSW indexes
- Knowledge graph construction from documents
- 30+ built-in graph algorithms

All backed by GrafeoDB's embedded database - no servers or Docker required.

## Installation

```bash
uv add grafeo-llamaindex
# or
pip install grafeo-llamaindex
```

Requires Python 3.12+, grafeo >= 0.4.4 and llama-index-core >= 0.14.

## Quick Start

### Basic Usage

```python
from grafeo_llamaindex import GrafeoPropertyGraphStore

store = GrafeoPropertyGraphStore(db_path="./my-graph.db")

# Insert nodes and relationships
from llama_index.core.graph_stores.types import EntityNode, Relation

nodes = [
    EntityNode(name="Alix", label="Person", properties={"age": 30}),
    EntityNode(name="Gus", label="Person", properties={"age": 25}),
]
store.upsert_nodes(nodes)

relations = [
    Relation(source_id="Alix", target_id="Gus", label="KNOWS"),
]
store.upsert_relations(relations)

# Query
result = store.structured_query(
    "MATCH (p:Person)-[:KNOWS]->(f) RETURN p.name, f.name"
)
```

### Vector Search

```python
store = GrafeoPropertyGraphStore(
    db_path="./my-graph.db",
    embedding_dimensions=384,
    distance_metric="cosine",
)

# Nodes with embeddings are automatically indexed
results = store.vector_query(
    query_embedding=[0.1, 0.2, ...],
    similarity_top_k=10,
)
```

### With PropertyGraphIndex

```python
from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()

index = PropertyGraphIndex.from_documents(
    documents,
    property_graph_store=store,
)

query_engine = index.as_query_engine()
response = query_engine.query("What does Alix do?")
```

## PropertyGraphStore Interface

grafeo-llamaindex implements all 8 abstract methods:

| Method | Description |
|--------|-------------|
| `upsert_nodes()` | Insert or update graph nodes |
| `upsert_relations()` | Insert or update relationships |
| `get()` | Retrieve nodes by ID or properties |
| `get_triplets()` | Get relationship triples |
| `get_rel_map()` | Get relationship maps for entities |
| `delete()` | Remove nodes or relationships |
| `structured_query()` | Execute GQL/Cypher/Gremlin queries |
| `vector_query()` | HNSW similarity search over embeddings |

## Vector Search Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_dimensions` | `None` | Vector dimensions (required for vector search) |
| `distance_metric` | `"cosine"` | `cosine`, `euclidean`, `dot_product`, `manhattan` |

## Graph Algorithms

Access 30+ built-in algorithms via `store.client.algorithms`:

- PageRank, betweenness centrality, closeness centrality
- Louvain community detection, connected components
- Shortest path (Dijkstra, BFS)

## Requirements

- Python 3.12+
- grafeo >= 0.4.4
- llama-index-core >= 0.14, < 1

## License

Apache-2.0
