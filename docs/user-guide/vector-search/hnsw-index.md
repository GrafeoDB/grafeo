---
title: HNSW Index
description: Hierarchical Navigable Small World index for fast approximate nearest neighbor search.
tags:
  - vector-search
  - hnsw
  - index
---

# HNSW Index

The HNSW (Hierarchical Navigable Small World) index provides O(log n) approximate nearest neighbor search, making vector queries fast even with millions of vectors.

## Overview

HNSW builds a multi-layer graph structure where:

- Higher layers have fewer nodes for fast global navigation
- Lower layers have more connections for accurate local search
- Search starts at the top layer and descends to find nearest neighbors

## Creating an HNSW Index

### SQL Syntax

```sql
-- Create a vector index on the embedding property
CREATE VECTOR INDEX movie_embeddings ON :Movie(embedding)
  WITH (dimensions: 384, metric: 'cosine')

-- With custom HNSW parameters
CREATE VECTOR INDEX document_embeddings ON :Document(embedding)
  WITH (
    dimensions: 768,
    metric: 'euclidean',
    m: 16,
    ef_construction: 200
  )
```

### Python API

```python
import grafeo
from grafeo import HnswConfig, DistanceMetric

db = grafeo.GrafeoDB()

# Create index with default settings
db.create_vector_index(
    "movie_embeddings",
    label="Movie",
    property="embedding",
    dimensions=384,
    metric="cosine"
)

# Or use the Rust-level configuration
config = HnswConfig(dimensions=384, metric=DistanceMetric.Cosine)
config.m = 16  # Connections per node
config.ef_construction = 200  # Build-time quality
```

## Configuration Parameters

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `dimensions` | Required | Number of dimensions in vectors |
| `metric` | `cosine` | Distance metric: `cosine`, `euclidean`, `dot_product`, `manhattan` |
| `m` | 16 | Max connections per node (higher = better recall, more memory) |
| `ef_construction` | 200 | Build-time beam width (higher = better index quality, slower build) |
| `ef_search` | 50 | Search-time beam width (higher = better recall, slower search) |

## Tuning Parameters

### M (Connections per Node)

Controls the graph connectivity:

- **Lower (8-12)**: Faster build, less memory, lower recall
- **Default (16)**: Good balance for most use cases
- **Higher (24-48)**: Better recall, more memory, slower build

```sql
-- High-recall configuration
CREATE VECTOR INDEX ... WITH (m: 32, ...)

-- Memory-efficient configuration
CREATE VECTOR INDEX ... WITH (m: 8, ...)
```

### ef_construction (Build Quality)

Controls index build quality:

- **Lower (100)**: Faster build, slightly lower quality
- **Default (200)**: Good balance
- **Higher (400-500)**: Best quality, slower build

```sql
-- Fast build (acceptable for frequently rebuilt indexes)
CREATE VECTOR INDEX ... WITH (ef_construction: 100, ...)

-- High-quality build (for static datasets)
CREATE VECTOR INDEX ... WITH (ef_construction: 400, ...)
```

### ef_search (Search Quality)

Controls search-time recall:

```python
# Adjust at query time
result = db.execute("""
    MATCH (d:Document)
    RETURN d.title, cosine_similarity(d.embedding, $query) AS sim
    ORDER BY sim DESC
    LIMIT 10
""", {"query": embedding}, ef_search=100)  # Higher ef for better recall
```

## Performance Characteristics

### Time Complexity

| Operation | Complexity |
| --------- | ---------- |
| Build | O(n log n) |
| Insert | O(log n) |
| Search | O(log n) |
| Delete | O(log n) |

### Memory Usage

Approximate memory per vector:

```
memory = dimensions * 4 bytes (f32)
       + m * 2 * 8 bytes (connections)
       + overhead (~50 bytes)
```

For 1 million 384-dimensional vectors with m=16:

```
1M * (384 * 4 + 16 * 2 * 8 + 50) ≈ 1.8 GB
```

## Recall vs Speed Tradeoffs

| Configuration | Recall | Search Time | Memory |
| ------------- | ------ | ----------- | ------ |
| m=8, ef=25 | ~85% | Fastest | Lowest |
| m=16, ef=50 (default) | ~95% | Fast | Medium |
| m=32, ef=100 | ~99% | Moderate | Higher |
| m=48, ef=200 | ~99.5% | Slower | Highest |

## Best Practices

### 1. Choose the Right Metric

```sql
-- For text embeddings (usually normalized)
CREATE VECTOR INDEX ... WITH (metric: 'cosine', ...)

-- For image embeddings or spatial data
CREATE VECTOR INDEX ... WITH (metric: 'euclidean', ...)

-- For retrieval models with dot product training
CREATE VECTOR INDEX ... WITH (metric: 'dot_product', ...)
```

### 2. Build Index After Bulk Loading

```python
# Load data first
for doc in documents:
    db.execute("INSERT (:Document {title: $title, embedding: $emb})",
               {"title": doc.title, "emb": doc.embedding})

# Then create index (faster than incremental inserts)
db.execute("""
    CREATE VECTOR INDEX doc_embeddings ON :Document(embedding)
    WITH (dimensions: 384, metric: 'cosine')
""")
```

### 3. Monitor Recall

Periodically verify recall against brute-force search:

```python
# Sample query
query = get_random_query()

# HNSW result
hnsw_result = db.execute("MATCH (d:Document) RETURN d.id ORDER BY cosine_similarity(d.embedding, $q) DESC LIMIT 10",
                          {"q": query})

# Compare with brute-force (on small sample)
# to ensure HNSW recall meets requirements
```

## Combining with Quantization

For memory-constrained environments, combine HNSW with quantization:

```python
from grafeo import QuantizedHnswIndex, HnswConfig, QuantizationType

# Scalar quantization: 4x compression
config = HnswConfig(dimensions=384, metric="cosine")
index = QuantizedHnswIndex(config, QuantizationType.Scalar)

# Product quantization: 48x compression (384/8 subvectors)
index = QuantizedHnswIndex(config, QuantizationType.Product(num_subvectors=8))
```

See [Quantization](quantization.md) for details.

## Next Steps

- [**Quantization**](quantization.md) - Reduce memory with vector compression
- [**Python API**](python-api.md) - Low-level Python bindings
