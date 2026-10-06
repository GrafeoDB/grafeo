---
title: Filter-Expression Hybrid Search
description: Use text_score(), text_match(), and vector similarity as WHERE/RETURN expressions with planner pushdown.
tags:
  - hybrid-search
  - bm25
  - vector-search
  - planner
---

# Filter-Expression Hybrid Search

Since 0.5.40, BM25 text scoring and vector similarity are callable as ordinary
expressions inside `WHERE` and `RETURN` clauses. The planner rewrites the
matching predicate shapes into dedicated `TextScan` and `VectorScan` operators,
so a text or vector index is used whenever one applies, and a brute-force
per-row fallback kicks in when no index exists.

This is the unified-query companion to [`hybrid_search()`](hybrid-search.md):
instead of calling a fusion API, you express the filter directly in GQL and let
the planner pick the execution strategy.

## When to use each

| Need | Use |
|------|-----|
| Simple top-K fusion across text + vector | [`hybrid_search()`](hybrid-search.md) |
| Text or vector predicate combined with a MATCH pattern | Filter expressions (this page) |
| AND/OR composition with other WHERE predicates | Filter expressions |
| Score column in the result set | Filter expressions (put the same call in `RETURN`) |
| Works without an index | Both: filter expressions fall back to per-row eval |

## Functions

| Function | Returns | Shape |
|----------|---------|-------|
| `text_score(n.prop, "query")` | `Float64` (BM25 score, higher = more relevant) | Use in WHERE with a threshold, or project in RETURN |
| `text_match(n.prop, "query")` | `Boolean` (true if the document matches) | Use directly as a WHERE predicate |
| `cosine_similarity(n.vec, $q)` | `Float64` (higher = more similar) | WHERE threshold or RETURN projection |
| `euclidean_distance(n.vec, $q)` | `Float64` (lower = more similar) | WHERE threshold (use `<`) or RETURN projection |

The same names work in Cypher. SPARQL and SQL/PGQ follow the same shape where
supported.

## `text_score` with a threshold

```gql
MATCH (doc:Article)
WHERE text_score(doc.body, 'attention mechanisms') > 0.0
RETURN doc.title
```

With a text index on `Article.body`, the planner rewrites this into a
`TextScanOperator` in threshold mode, pulling only matching documents from the
inverted index. Without an index, the same query falls through to per-row BM25
evaluation (slow but correct).

## `text_match` as a boolean

```gql
MATCH (doc:Article)
WHERE text_match(doc.body, 'rust')
RETURN doc.title
```

`text_match` is the index-friendly way to ask "does this document match the
query at all?" and maps to the same `TextScan` operator without needing a
threshold.

## Top-K by score

Pair `ORDER BY ... DESC LIMIT k` with a score function and the planner
recognizes it as top-K, pushing `k` into the underlying scan:

```gql
MATCH (doc:Article)
RETURN doc.title, text_score(doc.body, 'attention mechanisms') AS score
ORDER BY text_score(doc.body, 'attention mechanisms') DESC
LIMIT 10
```

The same pattern works for vector similarity:

```gql
MATCH (doc:Article)
RETURN doc.title
ORDER BY cosine_similarity(doc.embedding, [0.85, 0.15, 0.05]) DESC
LIMIT 5
```

## Vector similarity thresholds

```gql
MATCH (doc:Article)
WHERE cosine_similarity(doc.embedding, [0.85, 0.15, 0.05]) > 0.5
RETURN doc.title
```

With a vector index, this pushes down into a `VectorScanOperator`. Without one,
the planner falls back to brute-force per-row evaluation so the query still
returns the correct rows.

Use `euclidean_distance(...) < threshold` for the distance formulation:

```gql
MATCH (doc:Article)
WHERE euclidean_distance(doc.embedding, [0.9, 0.1, 0.0]) < 0.5
RETURN doc.title
```

!!! note "Operator direction matters for pushdown"
    Natural directions push down to an index scan: `cosine_similarity(prop, q) > t`, `euclidean_distance(prop, q) < t`, `manhattan_distance(prop, q) < t`, and `text_score(prop, q) > t` / `text_score(prop, q) >= t`. Inverted comparisons (e.g. `cosine_similarity < t`), `dot_product` (not currently pushdown-supported), and queries whose vector is not resolvable at plan time (property reference, unresolved parameter) also fall through to brute-force per-row evaluation.

## Compound predicates (AND / OR)

Filter expressions compose with other WHERE predicates. AND narrows, OR unions:

```gql
MATCH (doc:Article)
WHERE cosine_similarity(doc.embedding, [0.85, 0.15, 0.05]) > 0.3
  AND text_match(doc.body, 'attention mechanisms')
RETURN doc.title
```

```gql
MATCH (doc:Article)
WHERE cosine_similarity(doc.embedding, [0.1, 0.9, 0.0]) > 0.9
   OR text_match(doc.body, 'attention mechanisms')
RETURN doc.title
```

## Combining with graph patterns

Filter expressions run after pattern matching, so you can gate a traversal on
similarity:

```gql
MATCH (u:User {name: 'Alix'})-[:FOLLOWS]->(friend)-[:WROTE]->(doc:Article)
WHERE cosine_similarity(doc.embedding, [0.85, 0.15, 0.05]) > 0.3
RETURN doc.title
```

Here the user → friend → article traversal produces candidate rows, and the
vector similarity predicate filters them per-row.

## Projecting the score

Reusing the same call in `WHERE` and `RETURN` does not recompute: the planner
keeps the score column from the scan and projects it through.

```gql
MATCH (doc:Article)
WHERE text_score(doc.body, 'attention mechanisms') > 0.0
RETURN doc.title, text_score(doc.body, 'attention mechanisms') AS score
```

If you only need the score (no threshold), put it in `RETURN` without a
WHERE clause. The planner falls back to per-row scoring, returning one row per
matched node with a Float64 (or 0.0 for non-matches):

```gql
MATCH (doc:Article)
RETURN doc.title, text_score(doc.body, 'rust database') AS score
```

## Graceful degradation without indexes

| Missing index | Behavior |
|---------------|----------|
| No text index | `text_score` returns 0.0 per row, `text_match` returns false per row, query still runs |
| No vector index | `cosine_similarity` / `euclidean_distance` evaluate per-row over all candidates |

Queries still return correct results in every case, but with an index the
planner executes them through dedicated scan operators instead of a full scan.
