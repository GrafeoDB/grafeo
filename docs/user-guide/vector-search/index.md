---
title: Vector Search
description: Semantic similarity search with vector embeddings in Grafeo.
tags:
  - vector-search
  - embeddings
  - similarity
---

# Vector Search

Grafeo provides first-class support for vector similarity search, enabling semantic search, recommendation systems, and AI-powered applications.

## Overview

Vector search allows you to find nodes based on the semantic similarity of their embeddings rather than exact property matches. This is essential for:

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
| **Hybrid Queries** | Combine graph patterns with vector similarity |
| **SIMD Acceleration** | AVX2, SSE, NEON optimized distance computation |

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

- You need semantic/meaning-based retrieval
- Working with embeddings from ML models (OpenAI, Sentence Transformers, etc.)
- Building recommendation or similarity features
- Implementing RAG (Retrieval-Augmented Generation)

**Use traditional queries when:**

- You need exact property matches
- Working with structured data (names, IDs, dates)
- Relationships matter more than similarity
