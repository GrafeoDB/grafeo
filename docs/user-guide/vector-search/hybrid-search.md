---
title: Hybrid Search
description: Combined text (BM25) and vector similarity search with score fusion.
tags:
  - hybrid-search
  - bm25
  - vector-search
  - fusion
---

# Hybrid Search

Hybrid search combines BM25 text relevance with HNSW vector similarity, then fuses the results into a single ranking. This captures both exact keyword matches and semantic meaning.

!!! tip "Filter-expression alternative (0.5.40+)"
    For hybrid predicates that combine with a MATCH pattern, AND/OR with other
    WHERE clauses, or need the score as a projected column, use
    [filter-expression hybrid search](filter-expressions.md) instead.
    `hybrid_search()` is best when you want a single top-K list with RRF or
    weighted fusion.

## Prerequisites

Hybrid search requires the `hybrid-search` feature flag, which is included in the `embedded`, `server`, and `full` profiles. It also depends on `text-index` and `vector-index`.

## Setup

Hybrid search uses **pre-existing** text and vector indexes. You must create both before calling `hybrid_search()`:

```python
import grafeo

db = grafeo.GrafeoDB()

# Create nodes with text and vector properties
db.create_node(["Doc"], {
    "content": "Introduction to graph databases",
    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
})
db.create_node(["Doc"], {
    "content": "Machine learning with neural networks",
    "embedding": [0.5, 0.4, 0.3, 0.2, 0.1]
})

# Create BOTH indexes
db.create_text_index("Doc", "content")
db.create_vector_index("Doc", "embedding", dimensions=5, metric="cosine")
```

## Searching

```python
results = db.hybrid_search(
    label="Doc",
    text_property="content",
    vector_property="embedding",
    query_text="graph databases",
    k=10,
    query_vector=[0.1, 0.2, 0.3, 0.4, 0.5],
)
for node_id, score in results:
    print(f"Node {node_id}: score={score:.4f}")
```

```typescript
const results = await db.hybridSearch(
  "Doc", "content", "embedding",
  "graph databases", 10,
  [0.1, 0.2, 0.3, 0.4, 0.5]
);
for (const [nodeId, score] of results) {
  console.log(`Node ${nodeId}: score ${score}`);
}
```

## Return Value Semantics

!!! warning "Fusion scores, not distances"
    `hybrid_search()` returns `(node_id, score)` tuples sorted by fused score
    **descending** (higher = more relevant). These are **fusion scores**, not
    distances. This is the opposite convention from `vector_search()`, which
    returns distances where lower = better.

    ```python
    # CORRECT: multiply fusion scores for decay (higher = better)
    decayed_score = score * decay_factor

    # WRONG: dividing a fusion score inverts the ranking
    # decayed_score = score / decay_factor  # Don't do this!
    ```

## Fusion Methods

### Reciprocal Rank Fusion (RRF) (default)

RRF is parameter-free and robust. It combines results by rank position, ignoring raw score values:

$$\text{score}(d) = \sum_{\text{source}} \frac{1}{k + \text{rank}_{\text{source}}(d)}$$

where `k` is a smoothing constant (default: 60). Nodes appearing in multiple sources accumulate higher scores.

```python
# Explicit RRF with custom k
results = db.hybrid_search(
    "Doc", "content", "embedding",
    "graph databases", k=10,
    query_vector=query_vec,
    fusion="rrf",
    rrf_k=60,       # smoothing constant (default: 60)
)
```

### Weighted Fusion

Weighted fusion normalizes scores from each source to `[0, 1]` using min-max normalization, then combines them with explicit weights:

```python
results = db.hybrid_search(
    "Doc", "content", "embedding",
    "graph databases", k=10,
    query_vector=query_vec,
    fusion="weighted",
    weights=[0.7, 0.3],  # [text_weight, vector_weight]
)
```

## Graceful Degradation

If either index is missing, `hybrid_search()` silently omits that source from fusion rather than raising an error:

- **No text index**: only vector results are returned
- **No vector index**: only text results are returned
- **Neither index exists**: returns an empty list

This makes it safe to call `hybrid_search()` in code paths where indexes may not yet be configured.

## Text-Only Mode

If you omit `query_vector`, only the text source contributes:

```python
results = db.hybrid_search(
    "Doc", "content", "embedding",
    "graph databases", k=10,
    # no query_vector: text-only
)
```

This is equivalent to `text_search()` when no vector index exists, but uses the fusion pipeline, which means the score format is still a fusion score (not a raw BM25 score).
