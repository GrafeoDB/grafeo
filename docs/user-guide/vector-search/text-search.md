---
title: Text Search (BM25)
description: Full-text keyword search with BM25 scoring and inverted indexes.
tags:
  - text-search
  - bm25
  - full-text
---

# Text Search (BM25)

Grafeo includes a built-in BM25 full-text search engine with Unicode tokenization and stop word removal. Text indexes let you find nodes by keyword relevance without embeddings.

## Prerequisites

Text search requires the `text-index` feature flag, which is included in the `embedded`, `server`, and `full` profiles.

## Creating a Text Index

Create a BM25 inverted index on a node property:

```python
import grafeo

db = grafeo.GrafeoDB()

# Create some nodes with text content
db.create_node(["Article"], {"title": "Introduction to Graph Databases"})
db.create_node(["Article"], {"title": "Machine Learning with Python"})
db.create_node(["Article"], {"title": "Graph Neural Networks for NLP"})

# Create a text index on the title property
db.create_text_index("Article", "title")
```

```typescript
const db = new GrafeoDB();

await db.createNode(["Article"], { title: "Introduction to Graph Databases" });
await db.createNode(["Article"], { title: "Machine Learning with Python" });
await db.createNode(["Article"], { title: "Graph Neural Networks for NLP" });

await db.createTextIndex("Article", "title");
```

## Searching

Use `text_search()` to find nodes by keyword relevance:

```python
results = db.text_search("Article", "title", "graph", k=10)
for node_id, score in results:
    print(f"Node {node_id}: score={score:.4f}")
```

```typescript
const results = await db.textSearch("Article", "title", "graph", 10);
for (const [nodeId, score] of results) {
  console.log(`Node ${nodeId}: score ${score}`);
}
```

### Return value semantics

`text_search()` returns a list of `(node_id, score)` tuples sorted by **descending** relevance (higher score = more relevant). BM25 scores are unbounded positive floats whose magnitude depends on corpus statistics, so compare them only within a single query's results.

## In-Query Text Scoring (0.5.40+)

BM25 is also callable from GQL/Cypher as `text_score()` and `text_match()`,
which the planner pushes down into a `TextScanOperator` when a text index
exists. Two scan modes:

- **Top-K** (`ORDER BY text_score(...) DESC LIMIT k`): returns the `k`
  highest-scoring documents, using the inverted index to avoid scanning
  non-matching docs.
- **Threshold** (`WHERE text_score(...) > t`): returns every document whose
  BM25 score exceeds `t`.

```gql
-- Top-K
MATCH (a:Article)
RETURN a.title, text_score(a.title, 'graph') AS score
ORDER BY text_score(a.title, 'graph') DESC
LIMIT 5

-- Threshold
MATCH (a:Article)
WHERE text_score(a.title, 'graph') > 1.0
RETURN a.title
```

See [filter-expression hybrid search](filter-expressions.md) for the full
syntax, AND/OR composition with vector predicates, and index-missing fallback
behavior.

## Auto-Sync Behavior

Text indexes are **automatically maintained** as nodes change. You do not need to rebuild after normal write operations:

```python
# Index is created once
db.create_text_index("Article", "title")

# New nodes are auto-indexed
db.create_node(["Article"], {"title": "Rust Systems Programming"})

# Updated properties are auto-reindexed
db.set_node_property(node_id, "title", "Updated Title")

# Deleted nodes are auto-removed from the index
db.delete_node(node_id)

# All of the above are reflected in search results immediately,
# no rebuild needed.
```

## When to Rebuild

You only need `rebuild_text_index()` in rare cases:

- Data was loaded through non-standard paths (e.g. persistence restore) before the index existed
- You want to reindex after bulk operations performed through direct store manipulation

```python
db.rebuild_text_index("Article", "title")
```

## BM25 Configuration

Text indexes use the default BM25 configuration (k1=1.2, b=0.75) with Unicode-aware tokenization and English stop word removal. Custom BM25 parameters are not currently configurable through the API.
