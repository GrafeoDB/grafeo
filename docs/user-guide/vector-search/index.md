---
title: Vector Search
description: Semantic similarity search with vector embeddings in Grafeo.
tags:
  - vector-search
  - embeddings
  - similarity
---

# Vector Search

Grafeo provides first-class support for vector similarity search, enabling semantic search, recommendation systems and AI-powered applications.

## Overview

Vector search finds nodes based on the semantic similarity of their embeddings rather than exact property matches. This is essential for:

- **Semantic search** - Find documents by meaning, not keywords
- **Recommendations** - Suggest similar items based on embeddings
- **RAG applications** - Retrieve relevant context for LLMs
- **Hybrid queries** - Combine graph traversal with vector similarity

## Key Features

| Feature | Description |
| ------- | ----------- |
| **HNSW Index** | O(log n) approximate nearest neighbor search |
| **Distance Metrics** | Cosine, Euclidean, Dot Product, Manhattan |
| **Quantization** | Scalar (4x), Binary (32x), Product (8-192x) compression |
| **Filtered Search** | Property filters: equality, `$gt`, `$gte`, `$lt`, `$lte`, `$ne`, `$in`, `$nin`, `$contains` |
| **MMR Search** | Maximal Marginal Relevance for diverse RAG retrieval |
| **Incremental Indexing** | Indexes auto-sync on `set_node_property()` and batch operations; explicit rebuild is rarely needed |
| **Batch Operations** | `batch_create_nodes()` and `batch_vector_search()` |
| **Hybrid Queries** | Combine graph patterns with vector similarity |
| **BM25 Text Search** | Full-text keyword search with inverted indexes |
| **Hybrid Search** | Combined text + vector search with RRF or weighted fusion |
| **Built-in Embeddings** | In-process ONNX embedding generation (opt-in `embed` feature) |
| **SIMD Acceleration** | AVX2, SSE, NEON optimized distance computation |

## Search Score Conventions

Different search methods return different value types. Understanding these conventions
is critical when post-processing results (e.g. applying temporal decay or thresholds).

| Method | Returns | Sort Order | Interpretation |
| ------ | ------- | ---------- | -------------- |
| `vector_search()` | `(node_id, distance)` | Ascending (lowest first) | Lower = more similar |
| `mmr_search()` | `(node_id, distance)` | MMR selection order | Same distances as `vector_search` |
| `text_search()` | `(node_id, score)` | Descending (highest first) | Higher = more relevant (BM25) |
| `hybrid_search()` | `(node_id, score)` | Descending (highest first) | Higher = more relevant (fusion) |

!!! warning "Common pitfall: score vs distance"
    `hybrid_search()` returns fusion scores (higher = better), while
    `vector_search()` returns distances (lower = better). If you apply a
    temporal decay factor, use **multiplication** for fusion scores and
    **division** for distances:

    ```python
    # Correct for hybrid_search (score, higher = better):
    decayed_score = score * decay_factor

    # Correct for vector_search (distance, lower = better):
    decayed_distance = distance / decay_factor
    ```

## Quick Example

```python
import grafeo

db = grafeo.GrafeoDB()

# Create nodes with embeddings
db.execute("""
    INSERT (:Document {
        title: 'Introduction to Graphs',
        embedding: [0.1, 0.2, 0.3, 0.4, 0.5]
    })
""")

# Find similar documents
query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
result = db.execute("""
    MATCH (d:Document)
    RETURN d.title, cosine_similarity(d.embedding, $query) AS similarity
    ORDER BY similarity DESC
    LIMIT 10
""", {"query": query_embedding})

for row in result:
    print(f"{row['d.title']}: {row['similarity']:.3f}")
```

## Batch Operations

For high-throughput ingestion and multi-query search, use the batch APIs:

### `batch_create_nodes()`

Create many nodes at once, each with a vector property. Returns a list of node IDs.

```python
import grafeo

db = grafeo.GrafeoDB()
db.create_vector_index("Document", "embedding", dimensions=3)

# Create 3 nodes, each with label "Document" and an "embedding" vector
ids = db.batch_create_nodes("Document", "embedding", [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9],
])
print(f"Created node IDs: {ids}")
```

For nodes with additional properties, use `batch_create_nodes_with_props()`:

```python
ids = db.batch_create_nodes_with_props("Document", [
    {"title": "Graph databases", "embedding": [0.1, 0.2, 0.3]},
    {"title": "Vector search",   "embedding": [0.4, 0.5, 0.6]},
])
```

### `batch_vector_search()`

Search for nearest neighbors of multiple query vectors in a single call. Queries run in parallel across all available CPU cores.

```python
results = db.batch_vector_search(
    "Document", "embedding",
    queries=[[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]],
    k=5,
)
for i, matches in enumerate(results):
    print(f"Query {i}:")
    for node_id, distance in matches:
        print(f"  Node {node_id}: distance={distance:.4f}")
```

An optional `ef` parameter controls the search beam width (higher values improve recall at the cost of speed). An optional `filters` dict applies property-based pre-filtering.

## Text Search (BM25)

Create inverted indexes for full-text keyword search with BM25 scoring:

```python
db.create_text_index("Document", "content")
results = db.text_search("Document", "content", "graph database", k=10)
for r in results:
    print(f"Node {r['node_id']}: score {r['score']:.3f}")
```

Text indexes stay in sync automatically as nodes are created, updated or deleted.

## Hybrid Search

Combine BM25 text scores with HNSW vector similarity via Reciprocal Rank Fusion:

```python
results = db.hybrid_search(
    label="Document",
    text_property="content", text_query="graph database",
    vector_property="embedding", vector_query=query_vec,
    k=10,
)
```

## Built-in Embeddings

Generate embeddings in-process with ONNX Runtime (requires the `embed` feature):

```python
from grafeo import load_embedding_model, EmbeddingModelConfig

model = load_embedding_model(EmbeddingModelConfig.MiniLM_L6_v2)
vectors = model.embed(["graph databases are fast", "hello world"])
```

Three presets are available: MiniLM-L6-v2 (22M params), MiniLM-L12-v2 (33M) and BGE-small-en-v1.5 (33M). Models are auto-downloaded from HuggingFace Hub on first use.

## Documentation

- [**Getting Started**](basics.md) - Store and query vector embeddings
- [**HNSW Index**](hnsw-index.md) - Configure the approximate nearest neighbor index
- [**Quantization**](quantization.md) - Compress vectors for memory efficiency
- [**Python API**](python-api.md) - Python bindings for vector operations

## Performance

Benchmark results on 384-dimensional vectors:

| Operation | Performance |
| --------- | ----------- |
| Distance computation (cosine) | 38 ns |
| Brute-force k-NN (10k vectors) | 308 µs |
| HNSW search (5k vectors, k=10) | 108 µs |
| PQ distance with table | 4.5 ns |

## When to Use Vector Search

**Use vector search when:**

- Semantic/meaning-based retrieval is needed
- Working with embeddings from ML models (OpenAI, Sentence Transformers, etc.)
- Building recommendation or similarity features
- Implementing RAG (Retrieval-Augmented Generation)

**Use traditional queries when:**

- Exact property matches are needed
- Working with structured data (names, IDs, dates)
- Relationships matter more than similarity
