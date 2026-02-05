---
title: Vector Search Basics
description: Getting started with vector embeddings and similarity search.
tags:
  - vector-search
  - embeddings
  - tutorial
---

# Getting Started with Vector Search

This guide covers the fundamentals of storing and querying vector embeddings in Grafeo.

## Storing Vectors

Vectors are stored as properties on nodes, just like any other property type.

### From Query Language

```sql
-- Store a vector embedding
INSERT (:Document {
    title: 'Machine Learning Basics',
    embedding: [0.12, -0.34, 0.56, 0.78, -0.90]
})

-- Update an existing node's embedding
MATCH (d:Document {title: 'Machine Learning Basics'})
SET d.embedding = [0.15, -0.32, 0.58, 0.75, -0.88]
```

### From Python

```python
import grafeo

db = grafeo.GrafeoDB()

# Using parameters (recommended for real embeddings)
embedding = get_embedding_from_model("Machine Learning Basics")  # Your embedding model
db.execute(
    "INSERT (:Document {title: $title, embedding: $embedding})",
    {"title": "Machine Learning Basics", "embedding": embedding}
)
```

## Distance Functions

Grafeo supports four distance metrics for vector similarity:

### Cosine Similarity

Best for normalized embeddings. Returns values between -1 and 1, where 1 means identical.

```sql
MATCH (d:Document)
WHERE cosine_similarity(d.embedding, $query) > 0.8
RETURN d.title
```

### Cosine Distance

`1 - cosine_similarity`. Lower is more similar.

```sql
MATCH (d:Document)
RETURN d.title, cosine_distance(d.embedding, $query) AS distance
ORDER BY distance
LIMIT 10
```

### Euclidean Distance (L2)

Measures straight-line distance in vector space. Lower is more similar.

```sql
MATCH (d:Document)
RETURN d.title, euclidean_distance(d.embedding, $query) AS distance
ORDER BY distance
LIMIT 10
```

### Dot Product

For unnormalized vectors. Higher is more similar.

```sql
MATCH (d:Document)
RETURN d.title, dot_product(d.embedding, $query) AS score
ORDER BY score DESC
LIMIT 10
```

### Manhattan Distance (L1)

Sum of absolute differences. Lower is more similar.

```sql
MATCH (d:Document)
RETURN d.title, manhattan_distance(d.embedding, $query) AS distance
ORDER BY distance
LIMIT 10
```

## Hybrid Queries

Combine graph traversal with vector similarity for powerful queries.

### Find Similar Documents by Authors You Follow

```sql
MATCH (me:User {id: $user_id})-[:FOLLOWS]->(author:Author)-[:WROTE]->(doc:Document)
WHERE cosine_similarity(doc.embedding, $query) > 0.7
RETURN doc.title, author.name, cosine_similarity(doc.embedding, $query) AS similarity
ORDER BY similarity DESC
LIMIT 20
```

### Recommend Products Based on Purchase History

```sql
-- Find products similar to what the user has purchased
MATCH (u:User {id: $user_id})-[:PURCHASED]->(p:Product)
WITH u, avg(p.embedding) AS user_preference
MATCH (candidate:Product)
WHERE NOT (u)-[:PURCHASED]->(candidate)
RETURN candidate.name, cosine_similarity(candidate.embedding, user_preference) AS score
ORDER BY score DESC
LIMIT 10
```

## Filtering Results

### By Similarity Threshold

```sql
MATCH (d:Document)
WHERE cosine_similarity(d.embedding, $query) > 0.8
RETURN d.title
```

### Combined with Property Filters

```sql
MATCH (d:Document)
WHERE d.category = 'technology'
  AND d.published > date('2024-01-01')
  AND cosine_similarity(d.embedding, $query) > 0.7
RETURN d.title, d.published
```

## Best Practices

### 1. Use Consistent Dimensions

All vectors for a given property should have the same dimensions:

```python
# Good: All embeddings are 384-dimensional
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dims

# Store consistently
for doc in documents:
    embedding = embedding_model.encode(doc.text)
    db.execute(
        "INSERT (:Document {title: $title, embedding: $embedding})",
        {"title": doc.title, "embedding": embedding.tolist()}
    )
```

### 2. Normalize for Cosine Similarity

Normalize vectors when using cosine similarity for consistent results:

```python
import numpy as np

def normalize(vec):
    norm = np.linalg.norm(vec)
    return (vec / norm).tolist() if norm > 0 else vec

embedding = normalize(model.encode(text))
```

### 3. Choose the Right Metric

| Metric | Best For |
| ------ | -------- |
| Cosine | Text embeddings, normalized vectors |
| Euclidean | Image embeddings, spatial data |
| Dot Product | Unnormalized vectors, retrieval models |
| Manhattan | Sparse vectors, outlier-sensitive tasks |

### 4. Use Parameters for Large Vectors

Always pass embeddings as query parameters to avoid parsing overhead:

```python
# Good: Pass as parameter
db.execute("MATCH (d:Document) WHERE cosine_similarity(d.embedding, $q) > 0.8 RETURN d",
           {"q": embedding})

# Avoid: Inline in query string (slow for large vectors)
db.execute(f"MATCH (d:Document) WHERE cosine_similarity(d.embedding, {embedding}) > 0.8 RETURN d")
```

## Next Steps

- [**HNSW Index**](hnsw-index.md) - Speed up search with approximate nearest neighbors
- [**Quantization**](quantization.md) - Reduce memory usage for large datasets
