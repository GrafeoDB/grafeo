---
title: MMR Search
description: Maximal Marginal Relevance search for diverse, relevant results in RAG pipelines.
tags:
  - mmr
  - vector-search
  - rag
  - diversity
---

# MMR Search

Maximal Marginal Relevance (MMR) search balances **relevance** to your query with **diversity** among results. This avoids returning multiple near-duplicate chunks, which is particularly useful for RAG applications where feeding redundant context to an LLM wastes tokens without adding information.

## How MMR Works

MMR uses a two-phase approach:

1. **Candidate retrieval**: Fetch `fetch_k` nearest neighbors from the HNSW index (same as `vector_search()`)
2. **Iterative selection**: Greedily pick `k` results that balance proximity to the query with dissimilarity to already-selected results

At each step, the next item is chosen by maximizing:

$$\text{MMR}(d) = \lambda \cdot \text{sim}(d, q) - (1 - \lambda) \cdot \max_{s \in S} \text{sim}(d, s)$$

where `q` is the query, `S` is the set of already-selected results, and `lambda` controls the trade-off.

## Parameters

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `k` | (required) | Number of diverse results to return |
| `fetch_k` | `4 * k` | Initial candidates from HNSW (larger = better diversity options, slower) |
| `lambda_mult` | `0.5` | Relevance vs diversity: `1.0` = pure relevance (same as `vector_search`), `0.0` = pure diversity |
| `ef` | index default | HNSW search beam width |
| `filters` | `None` | Property equality filters |

## Usage

### Python

```python
results = db.mmr_search(
    label="Doc",
    property="embedding",
    query=[0.1, 0.2, 0.3, 0.4, 0.5],
    k=5,
    lambda_mult=0.5,  # balanced relevance + diversity
)
for node_id, distance in results:
    print(f"Node {node_id}: distance={distance:.4f}")
```

### Node.js / TypeScript

```typescript
const results = await db.mmrSearch(
  "Doc", "embedding",
  [0.1, 0.2, 0.3, 0.4, 0.5],
  5,       // k
  20,      // fetchK
  0.5,     // lambdaMult
);
for (const [nodeId, distance] of results) {
  console.log(`Node ${nodeId}: distance ${distance}`);
}
```

## Return Value Semantics

`mmr_search()` returns `(node_id, distance)` tuples. Two important details:

1. **The distance values are identical to `vector_search()`** for the same nodes (lower = more similar). They are the original query distances, not MMR scores.
2. **The ordering is MMR selection order**, not distance-sorted. The first result is the most relevant, but subsequent results trade off some relevance for diversity.

This means you can safely compare distances from `mmr_search()` and `vector_search()` for the same node: they will match.

## Tuning Lambda

| Lambda | Behavior | Use case |
| ------ | -------- | -------- |
| `1.0` | Pure relevance (same results as `vector_search`) | When diversity doesn't matter |
| `0.7` | Mostly relevant, some diversity | General RAG retrieval |
| `0.5` | Balanced (default) | Most use cases |
| `0.3` | Mostly diverse, some relevance | Exploration, topic coverage |
| `0.0` | Pure diversity | Maximum coverage of the embedding space |

## When to Use MMR vs vector_search

**Use `mmr_search()`** when:

- Building RAG pipelines where redundant chunks waste LLM context
- You need topic coverage across a corpus
- Your embeddings produce clusters of very similar results

**Use `vector_search()`** when:

- You need the absolute closest matches
- Results will be deduplicated downstream
- Speed is critical (MMR adds a selection pass over candidates)

## Combining with Filters

MMR supports the same property filters as `vector_search()`:

```python
results = db.mmr_search(
    "Doc", "embedding", query_vec, k=5,
    lambda_mult=0.5,
    filters={"category": "science"},
)
```
