---
title: Vector Search
description: Vector similarity search with 3D visualization using Grafeo and anywidget-vector.
tags:
  - example
  - vector-search
  - visualization
---

# Vector Search

Store document embeddings, perform similarity search with filters and visualize the embedding space in 3D.

!!! tip "Run it locally"

    ```bash
    marimo run examples/vector_search.py
    ```

    **Requirements:** `grafeo`, `anywidget-vector`, `marimo`, `numpy`

## Generate Embeddings

In production you would use a model (OpenAI, sentence-transformers, etc.). Here we simulate 128-dimensional embeddings where documents in the same category cluster together.

```python
import numpy as np
from grafeo import GrafeoDB

np.random.seed(42)

documents = [
    {"title": "Machine Learning Basics", "category": "AI", "year": 2023},
    {"title": "Deep Learning with Python", "category": "AI", "year": 2022},
    {"title": "Natural Language Processing", "category": "AI", "year": 2023},
    {"title": "Computer Vision Fundamentals", "category": "AI", "year": 2021},
    {"title": "Data Structures and Algorithms", "category": "CS", "year": 2020},
    {"title": "Database Systems", "category": "CS", "year": 2019},
    {"title": "Operating Systems Design", "category": "CS", "year": 2021},
    {"title": "Network Programming", "category": "CS", "year": 2022},
    {"title": "Web Development with React", "category": "Web", "year": 2023},
    {"title": "Backend API Design", "category": "Web", "year": 2022},
    {"title": "Cloud Architecture Patterns", "category": "Cloud", "year": 2023},
    {"title": "Kubernetes in Production", "category": "Cloud", "year": 2022},
]

# Category-level base vectors + per-doc noise
category_bases = {cat: np.random.randn(128) for cat in ["AI", "CS", "Web", "Cloud"]}

embeddings = []
for doc in documents:
    base = category_bases[doc["category"]]
    embedding = base + np.random.randn(128) * 0.3
    embedding = embedding / np.linalg.norm(embedding)  # unit vector
    embeddings.append(embedding.tolist())
```

## Store in Grafeo

```python
db = GrafeoDB()

doc_nodes = []
for doc, embedding in zip(documents, embeddings):
    node = db.create_node(
        ["Document", doc["category"]],
        {
            "title": doc["title"],
            "category": doc["category"],
            "year": doc["year"],
            "embedding": embedding,
        },
    )
    doc_nodes.append(node)

print(f"Created {len(doc_nodes)} document nodes with 128-d embeddings")
```

```title="Output"
Created 12 document nodes with 128-d embeddings
```

## Similarity Search

Find documents similar to "Machine Learning Basics" by cosine similarity:

```python
query_embedding = embeddings[0]  # Machine Learning Basics
query_title = documents[0]["title"]

all_docs = db.execute("""
    MATCH (d:Document)
    RETURN id(d) AS id, d.title AS title, d.embedding AS embedding
""")

similarities = []
for row in all_docs:
    if row["embedding"] and row["title"] != query_title:
        similarity = np.dot(query_embedding, row["embedding"])
        similarities.append((row["title"], similarity))

similarities.sort(key=lambda x: x[1], reverse=True)

print(f"Query: '{query_title}'\n")
for title, sim in similarities[:5]:
    print(f"  {title}: {sim:.4f}")
```

```title="Output"
Query: 'Machine Learning Basics'

  Natural Language Processing: 0.8912
  Deep Learning with Python: 0.8754
  Computer Vision Fundamentals: 0.8601
  Cloud Architecture Patterns: 0.1823
  Web Development with React: 0.1547
```

AI documents score much higher because their embeddings share the same base vector.

## Hybrid Search (Vector + Filters)

Combine property filters with vector ranking:

```python
# Filter: AI docs from 2022+, then rank by similarity
filtered = db.execute("""
    MATCH (d:Document)
    WHERE d.category = 'AI' AND d.year >= 2022
    RETURN id(d) AS id, d.title AS title, d.year AS year, d.embedding AS embedding
""")

hybrid_results = []
for row in filtered:
    if row["embedding"] and row["title"] != "Machine Learning Basics":
        similarity = np.dot(query_embedding, row["embedding"])
        hybrid_results.append((row["title"], row["year"], similarity))

hybrid_results.sort(key=lambda x: x[2], reverse=True)

for title, year, sim in hybrid_results:
    print(f"  {title} ({year}): {sim:.4f}")
```

```title="Output"
  Natural Language Processing (2023): 0.8912
  Deep Learning with Python (2022): 0.8754
```

## 3D Visualization

Project the 128-d embeddings down to 3D with PCA and visualize:

```python
from anywidget_vector import VectorSpace

def simple_pca_3d(vectors):
    X = np.array(vectors)
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return (U[:, :3] * S[:3]).tolist()

coords_3d = simple_pca_3d(embeddings)

points = []
for i, (doc, coord) in enumerate(zip(documents, coords_3d)):
    points.append({
        "id": str(i),
        "label": doc["title"],
        "x": coord[0],
        "y": coord[1],
        "z": coord[2],
        "group": doc["category"],
        "metadata": {"year": doc["year"], "category": doc["category"]},
    })

vector_widget = VectorSpace(points=points, height=500)
vector_widget
```

Colors represent categories (AI, CS, Web, Cloud). Documents in the same category cluster together in the embedding space. Rotate and zoom to explore.

## Next Steps

- [Vector Search guide](../../user-guide/vector-search/index.md) for HNSW indexes and quantization
- [anywidget-vector docs](../../ecosystem/anywidget-vector.md) for full widget API
- [Graph Visualization example](graph-visualization.md) for graph-based analysis
