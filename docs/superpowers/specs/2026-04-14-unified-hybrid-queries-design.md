# Unified Hybrid Queries: Graph + Vector + Text in One Query

**Date:** 2026-04-14
**Status:** Draft
**Scope:** Shape B — composable WHERE/ORDER BY predicates with index acceleration

## Problem

Grafeo has three search engines — graph traversal, HNSW vector similarity, and BM25 full-text — but they can't compose in a single query. Users must make three separate API calls and join results in application code. This is the same "duct tape" problem CoordiNode markets against, except Grafeo already has all three engines built. The missing piece is planner integration.

**Today (three API calls, application-side glue):**

```python
related = db.query("""
    MATCH (me:User {id: $uid})-[:FOLLOWS*1..2]->(friend)
    MATCH (friend)-[:WROTE]->(doc:Article)
    RETURN doc
""", uid=user_id)

similar = db.vector_search("Article", "embedding", query_vec, k=100)
text_hits = db.text_search("Article", "body", "graph databases", k=100)

# Application code: intersect, rank, handle ID mismatches,
# lose transactional consistency between the three calls
```

**After this feature (one query, one transaction):**

```cypher
MATCH (me:User {id: $uid})-[:FOLLOWS*1..2]->(friend)
MATCH (friend)-[:WROTE]->(doc:Article)
WHERE cosine_similarity(doc.embedding, $query_vec) > 0.7
  AND text_score(doc.body, "graph databases") > 2.0
RETURN doc.title,
       cosine_similarity(doc.embedding, $query_vec) AS relevance,
       text_score(doc.body, "graph databases") AS text_rank
ORDER BY relevance DESC LIMIT 20
```

## Competitive context

Researched from [structured-world/coordinode](https://github.com/structured-world/coordinode) (v0.3-alpha, ~53K LOC, Rust, AGPL-3.0). CoordiNode's entire pitch is "graph + vector + text in one query." They have it. But they're server-only — no embedded mode, no WASM. Grafeo already has embedded + WASM + all three engines. This feature makes Grafeo the only embeddable database where graph, vector, and text compose in a single query.

| | Grafeo after this | CoordiNode | Neo4j | SQLite+vec+FTS5 |
|---|---|---|---|---|
| Graph + vector + text in one query | Yes | Yes | No | No graph |
| Embedded / WASM | Yes | No | No | Yes (no graph) |
| Index-accelerated (not post-filter) | Yes | Yes | N/A | Partial |

## Design decisions

### D1: No new syntax

The four existing vector functions (`cosine_similarity`, `euclidean_distance`, `dot_product`, `manhattan_distance`) already parse and evaluate in WHERE clauses. They keep working everywhere. This feature makes them faster by teaching the planner to use HNSW indexes when available.

### D2: Two new text functions, require index

| Function | Returns | Requires text index |
|---|---|---|
| `text_score(n.prop, "query")` | `Float64` (BM25 score) | Yes — errors without one |
| `text_match(n.prop, "query")` | `Bool` (`text_score > 0.0`) | Yes — errors without one |

**Why text functions require an index but vector functions don't:** Vector distance is a pure function of two vectors — give it inputs, get a number. BM25 is corpus-relative — it needs document frequency (IDF), average document length, and total document count. These statistics live in the `InvertedIndex` (`inverted_index.rs` lines 163-175). Without the index, there is no correct BM25 score. A "fallback" that returns different results than the indexed path is a bug.

Vector functions keep their current behavior: work anywhere as pure functions, index just makes them faster (sublinear HNSW instead of linear brute-force).

### D3: Score projection — compute once, use everywhere

When the planner pushes a predicate into an index scan, it assigns a synthetic column name (e.g., `_vscore_0`, `_tscore_0`) for the score. It then scans downstream expressions (RETURN, ORDER BY, further WHERE) for function calls matching the same function + variable + property + query argument, and rewrites them to column references.

**Why mandatory:** VectorScanOperator already outputs `(NodeId, f32 score)` in its DataChunk (`scan_vector.rs` lines 307-335). Without projection, the RETURN clause hits `eval_vector_fn`, which loads the embedding from the store and recomputes cosine distance. For 10K results with 1536-dim embeddings, that's ~60MB of unnecessary memory reads and redundant dot products for a number we already have.

### D4: Top-K recognition — ORDER BY + LIMIT eliminates Sort

The planner recognizes:

```
Sort(key = vector_fn(n.prop, $vec)) → Limit(k) → NodeScan(label)
```

and rewrites to:

```
VectorScan(prop, $vec, k, metric)  // Sort and Limit eliminated
```

Same for text:

```
Sort(key = text_score(n.prop, "q"), DESC) → Limit(k) → NodeScan(label)
```

rewrites to:

```
TextScan(prop, "q", k)  // Sort and Limit eliminated
```

**Why mandatory, not deferred:** `ORDER BY cosine_similarity(...) LIMIT 20` is the most natural vector query pattern. It's how every vector database tutorial works. If this does a full scan, the feature is half-useful.

Both HNSW and BM25 already return results in score order. Eliminating Sort is free.

### D5: Compound predicates — run both indexes, join

For `WHERE vector_pred AND text_pred`:
1. Run VectorScan with threshold → candidate set A with scores
2. Run TextScan with threshold → candidate set B with scores
3. Hash-join intersect on NodeId → only nodes passing both, both scores attached

For `WHERE vector_pred OR text_pred`:
- Same two scans, hash-join union instead of intersect.

**Why not "pick the more selective index":** We evaluated a selectivity heuristic and rejected it. At plan time, available statistics are:
- HNSW: `len()` only. No distribution info, no way to estimate how many vectors pass a distance threshold.
- InvertedIndex: `len()`, `term_count()`. Posting lists are private. Can't get df for a term without running the search.
- Cardinality estimator: hard-codes `0.7` selectivity for vector thresholds (`cardinality.rs` line 940-956).

Any heuristic built on this will guess. A wrong guess picks the less selective index, producing a bigger candidate set and more residual work than the other choice. Since both indexes are in-memory and both lookups are fast (HNSW: O(log n); BM25: O(df) per term), running both and joining is simpler, more predictable, and eliminates a class of plan quality bugs.

### D6: Shape C eliminated — B subsumes it

We considered `CALL hybrid_search("Article", "embedding", "body", $vec, "query", 20) YIELD node, score` as Shape C. It was eliminated because:

1. It doesn't compose with graph traversal — the CALL results join awkwardly with MATCH patterns.
2. Shape B's compound WHERE predicates with both scores projected cover the same queries.
3. Users who want combined ranking write it in the query: `ORDER BY 0.7 * cosine_similarity(...) + 0.3 * text_score(...)`.
4. The standalone search case (`ORDER BY ... LIMIT k` with no MATCH) works naturally with top-K recognition.

## Architecture

### New components

**`TextScanOperator`** — New physical operator in `crates/grafeo-core/src/execution/operators/scan_text.rs`. Parallel to `VectorScanOperator`. Takes: store ref, text index ref, query string, optional threshold, optional k. Outputs: `DataChunk` with `(NodeId, Float64)` schema.

- For WHERE threshold: calls `InvertedIndex.search_with_threshold(query, threshold)`
- For ORDER BY + LIMIT top-k: calls `InvertedIndex.search(query, k)`

**`TextScanOp`** — New logical operator variant in `plan.rs`, parallel to existing `VectorScanOp`.

### InvertedIndex API additions

Two new methods on `InvertedIndex` (`crates/grafeo-core/src/index/text/inverted_index.rs`):

```rust
/// Score a single document against a query using BM25.
/// Looks up each query term in its posting list, finds the entry for this node_id,
/// computes BM25 with corpus statistics already in the index.
/// O(query_terms) per call. Returns 0.0 if document has no matching terms.
pub fn score_document(&self, id: NodeId, query: &str) -> f64
```

Needed for `eval_text_fn` — per-row text scoring when text is a residual filter (e.g., vector was the primary scan, text is evaluated on candidates).

```rust
/// Return all documents scoring above threshold.
/// Same internal loop as search(), but filters by score instead of taking top-k.
pub fn search_with_threshold(&self, query: &str, threshold: f64) -> Vec<(NodeId, f64)>
```

Needed for `TextScanOperator` when the predicate is `text_score(...) > threshold`.

### eval_text_fn in filter.rs

New function evaluation category in the `eval_function` cascade (`crates/grafeo-core/src/execution/operators/filter.rs`):

```rust
fn eval_text_fn(&self, name: &str, args: &[FilterExpression], chunk: &DataChunk, row: usize) -> Option<Value> {
    match name {
        "text_score" => {
            // Extract property access → get node_id from chunk
            // Get node's label from store
            // Look up text index via store.get_text_index(label, property)
            // Call index.score_document(node_id, query_string)
            // Return Value::Float64(score)
        }
        "text_match" => {
            // Same as text_score, return Value::Bool(score > 0.0)
        }
        _ => None,
    }
}
```

Added to the cascade after `eval_vector_fn`:
```rust
.or_else(|| self.eval_vector_fn(name, args, chunk, row))
.or_else(|| self.eval_text_fn(name, args, chunk, row))  // NEW
.or_else(|| self.eval_session_fn(name, args, chunk, row))
```

### Planner changes

All in `crates/grafeo-engine/src/query/planner/lpg/filter.rs`, following the established pattern of `try_plan_filter_with_property_index()` (line 819) and `try_plan_filter_with_range_index()` (line 1066).

**`try_plan_filter_with_vector_index()`** — Pattern matches on:
- `cosine_similarity(n.prop, $vec) > threshold` → VectorScan with `min_similarity=threshold`, metric=Cosine
- `euclidean_distance(n.prop, $vec) < threshold` → VectorScan with `max_distance=threshold`, metric=Euclidean
- `manhattan_distance(n.prop, $vec) < threshold` → VectorScan with `max_distance=threshold`, metric=Manhattan
- `dot_product(n.prop, $vec) > threshold` → VectorScan with `min_similarity=threshold`, metric=DotProduct
- Note the operator inversion: similarity functions use `>`, distance functions use `<`. If the user writes `cosine_similarity(...) < 0.3` (unusual but valid), no pushdown — falls through to per-row eval.
- Validates: one arg is property access on a bound variable with known label, other arg resolves to vector literal/parameter
- **Only fires when input is a NodeScan** (full label scan). If the input is already narrowed by graph traversal (Expand, Join), per-row `eval_vector_fn` is faster for the typically small candidate set.
- Checks: `store.get_vector_index(label, property)` exists
- Produces: `VectorScanOperator` with threshold, score projected as `_vscore_N`
- Falls through if: no label, both args are property accesses (node-to-node), no index exists, input is not NodeScan

**`try_plan_filter_with_text_index()`** — Pattern matches on:
- `text_score(n.prop, "query") > threshold` or `text_match(n.prop, "query")`
- Validates: first arg is property access with known label, second arg resolves to string
- **Only fires when input is a NodeScan** (same rationale as vector: if graph already narrowed, per-row `eval_text_fn` with `score_document()` is faster)
- Checks: `store.get_text_index(label, property)` exists
- Produces: `TextScanOperator` with threshold, score projected as `_tscore_N`
- Errors if: no text index exists (D2)

**`try_plan_filter_compound_hybrid()`** — When both vector and text predicates are extracted:
- Runs both scan operators
- Hash-join intersect (AND) or union (OR) on NodeId
- Both scores available as projected columns
- Remaining scalar predicates become residual FilterOperator on top

**Top-K recognition** — In `crates/grafeo-engine/src/query/planner/lpg/mod.rs`:
- After building the initial plan, inspect for `Sort → Limit → NodeScan` where sort key is a vector or text function
- Rewrite to VectorScan(k) or TextScan(k), eliminate Sort and Limit operators

**Score projection** — When creating a scan operator from a predicate:
1. Assign synthetic column name (`_vscore_0`, `_tscore_0`)
2. Walk downstream LogicalOperators (Project, Sort, Filter)
3. Pattern-match FunctionCall expressions with identical function name + args
4. Rewrite matching expressions to column references

### Call order in plan_filter()

```
plan_filter(filter):
    1. extract_complex_exists()        // existing
    2. extract_exists_from_or()        // existing
    3. extract_count_comparison()      // existing
    4. check_zone_map_for_predicate()  // existing
    5. try_plan_filter_with_property_index()   // existing
    6. try_plan_filter_with_range_index()      // existing
    7. try_plan_filter_with_vector_index()     // NEW
    8. try_plan_filter_with_text_index()       // NEW
    9. try_plan_filter_compound_hybrid()       // NEW (if both 7 and 8 found predicates)
   10. generic FilterOperator fallback         // existing
```

### Optimizer and cost model additions

**`cardinality.rs`:** Add `estimate_text_scan()` — returns `k` for top-k queries, or `index.len() * selectivity` for threshold queries (use 0.1 default selectivity for text — most queries match a small fraction of the corpus).

**`cost.rs`:** Add TextScan cost formula — `cpu = doc_frequency_sum * cpu_tuple_cost * 5` (BM25 scoring is ~5x a simple tuple comparison). For threshold queries where df is unknown at plan time, use `index.len() * 0.1 * cpu_tuple_cost * 5`.

## What doesn't get pushdown

| Case | Why | What happens |
|---|---|---|
| No label on the variable | Can't look up index by `(label, property)` | Vector: brute-force `eval_vector_fn`. Text: error (needs index). |
| Node-to-node similarity `cosine_similarity(a.emb, b.emb)` | Neither arg is a literal/parameter vector | `eval_vector_fn` per-row |
| Function in WITH alias `WITH doc, cosine_similarity(...) AS sim WHERE sim > 0.7` | Would need to trace through pipeline barrier | `eval_vector_fn` per-row |
| No index exists for the property | Nothing to push into | Vector: brute-force per-row. Text: error. |
| Input narrowed by graph traversal (Expand/Join before Filter) | Per-row eval on small candidate set is faster than full index scan + join | `eval_vector_fn` + `eval_text_fn` (via `score_document`) per-row |
| Inverted comparison (`cosine_similarity(...) < 0.3`, `euclidean_distance(...) > 5.0`) | Unusual semantics — "find dissimilar" — not worth optimizing | `eval_vector_fn` per-row |

## Example queries — full execution trace

### Simple vector with pushdown

```cypher
MATCH (doc:Article)
WHERE cosine_similarity(doc.embedding, $vec) > 0.7
RETURN doc.title, cosine_similarity(doc.embedding, $vec) AS sim
```

Execution:
1. Planner sees `cosine_similarity(doc.embedding, $vec) > 0.7` in WHERE
2. `try_plan_filter_with_vector_index()`: label=Article, property=embedding, finds HNSW index
3. Creates `VectorScanOperator` with `min_similarity=0.7`, projects score as `_vscore_0`
4. Scans RETURN: `cosine_similarity(doc.embedding, $vec)` matches → rewritten to `_vscore_0`
5. Physical plan: `VectorScan(HNSW) → Project(doc.title, _vscore_0 AS sim)`
6. No redundant computation.

### Top-K without WHERE

```cypher
MATCH (doc:Article)
RETURN doc.title, cosine_similarity(doc.embedding, $vec) AS sim
ORDER BY sim DESC LIMIT 20
```

Execution:
1. No WHERE to push down
2. Top-K recognizer sees: `Sort(cosine_similarity(doc.embedding, $vec) DESC) → Limit(20) → NodeScan(Article)`
3. HNSW index exists on (Article, embedding)
4. Rewrite to: `VectorScan(embedding, $vec, k=20, metric=Cosine)` → score projected as `_vscore_0`
5. Sort and Limit eliminated
6. Physical plan: `VectorScan(HNSW, k=20) → Project(doc.title, _vscore_0 AS sim)`

### Compound graph + vector + text

```cypher
MATCH (me:User {id: $uid})-[:FOLLOWS*1..2]->(friend)
MATCH (friend)-[:WROTE]->(doc:Article)
WHERE cosine_similarity(doc.embedding, $vec) > 0.6
  AND text_score(doc.body, "attention mechanisms") > 1.5
RETURN doc.title,
       cosine_similarity(doc.embedding, $vec) AS relevance,
       text_score(doc.body, "attention mechanisms") AS text_rank
ORDER BY relevance DESC LIMIT 10
```

Execution:
1. Graph traversal: User→FOLLOWS*1..2→friend→WROTE→doc produces candidate Articles
2. Input to filter is Expand (graph traversal), NOT a NodeScan → **no index pushdown**
3. Per-row evaluation on each candidate:
   - `eval_vector_fn("cosine_similarity", doc.embedding, $vec)` → loads vector, computes cosine, O(dims) per doc
   - `eval_text_fn("text_score", doc.body, "attention mechanisms")` → calls `index.score_document(doc_id, query)`, O(query_terms) per doc
4. Both scores are available in the DataChunk for RETURN and ORDER BY
5. Filter by thresholds (> 0.6 AND > 1.5), sort by cosine DESC, limit 10
6. Physical plan: `Expand(User→FOLLOWS→friend→WROTE→doc) → Filter(cosine_similarity > 0.6 AND text_score > 1.5) → Sort(cosine DESC) → Limit(10) → Project`

**Why no pushdown here:** The graph traversal typically narrows to a small candidate set (say 50-500 articles reachable from the user). Running two full index scans over all Articles and then joining back would do more work, not less. Per-row eval with `score_document()` is O(candidates * query_terms) — fast for reasonable candidate sets.

**When pushdown DOES fire:** When the MATCH is a bare label scan with no prior graph narrowing:

```cypher
MATCH (doc:Article)
WHERE cosine_similarity(doc.embedding, $vec) > 0.6
  AND text_score(doc.body, "attention mechanisms") > 1.5
RETURN doc.title
```

Here the input is NodeScan(Article) → `try_plan_filter_compound_hybrid()` fires:
1. VectorScan on (Article, embedding) with threshold 0.6 → set A with `_vscore_0`
2. TextScan on (Article, body) with threshold 1.5 → set B with `_tscore_0`
3. Hash-join intersect A ∩ B on NodeId → both scores projected
4. Physical plan: `HybridScan(VectorScan ∩ TextScan) → Project(doc.title)`

### OR predicate

```cypher
MATCH (doc:Article)
WHERE cosine_similarity(doc.embedding, $vec) > 0.8
   OR text_match(doc.body, "graph neural networks")
RETURN doc.title
```

Execution:
1. Compound OR detected
2. VectorScan with threshold 0.8 → set A
3. TextScan with threshold 0.0 (text_match = score > 0.0) → set B
4. Hash-join union A ∪ B
5. Physical plan: `HybridScan(VectorScan ∪ TextScan) → Project(doc.title)`

## Files changed

| File | Change | Size estimate |
|---|---|---|
| `crates/grafeo-core/src/index/text/inverted_index.rs` | Add `score_document()`, `search_with_threshold()` | ~50 lines |
| `crates/grafeo-core/src/execution/operators/scan_text.rs` | New `TextScanOperator` (modeled on `scan_vector.rs`) | ~300 lines |
| `crates/grafeo-core/src/execution/operators/mod.rs` | Export `TextScanOperator` | ~2 lines |
| `crates/grafeo-core/src/execution/operators/filter.rs` | Add `eval_text_fn` to function cascade | ~40 lines |
| `crates/grafeo-engine/src/query/plan.rs` | Add `TextScan` to `LogicalOperator`, `TextScanOp` struct | ~40 lines |
| `crates/grafeo-engine/src/query/planner/lpg/filter.rs` | `try_plan_filter_with_vector_index()`, `try_plan_filter_with_text_index()`, `try_plan_filter_compound_hybrid()` | ~400 lines |
| `crates/grafeo-engine/src/query/planner/lpg/mod.rs` | Top-K recognition, score projection + expression rewriting | ~200 lines |
| `crates/grafeo-engine/src/query/optimizer/cardinality.rs` | `estimate_text_scan()` | ~20 lines |
| `crates/grafeo-engine/src/query/optimizer/cost.rs` | TextScan cost formula | ~30 lines |
| **Tests** | Unit + integration for each component | ~600 lines |
| **Total** | | **~1,680 lines** |

## Test plan

### Unit tests

- `InvertedIndex.score_document()`: known corpus, verify BM25 score matches `search()` for same document
- `InvertedIndex.search_with_threshold()`: verify returns all docs above threshold, none below
- `TextScanOperator`: threshold mode, top-k mode, empty index, no matches
- `eval_text_fn`: text_score returns correct BM25, text_match returns bool, errors without index, non-string property returns error
- Score projection: verify downstream expressions rewritten correctly, verify non-matching expressions left alone
- Top-K recognition: Sort+Limit on vector fn → VectorScan, Sort+Limit on text fn → TextScan, Sort+Limit on non-index fn → no rewrite

### Integration tests (Cypher queries via grafeo-engine)

- `WHERE cosine_similarity(...) > threshold` → uses HNSW, correct results
- `WHERE text_score(...) > threshold` → uses inverted index, correct results
- `WHERE text_match(...)` → correct boolean semantics
- `WHERE vector AND text` → intersect, both scores in RETURN
- `WHERE vector OR text` → union, correct results
- `ORDER BY cosine_similarity(...) LIMIT k` → top-K, no full scan
- `ORDER BY text_score(...) LIMIT k` → top-K from BM25
- Compound: `MATCH (doc:Article) WHERE vector AND text` (bare label scan) → both index scans, intersect
- Compound: `MATCH (graph traversal)->(doc:Article) WHERE vector AND text` → per-row eval on candidates (no pushdown)
- Score in RETURN matches score from WHERE (no double computation)
- `euclidean_distance(...) < 2.0` → pushes down with `max_distance=2.0` (operator inversion)
- `cosine_similarity(...) < 0.3` (inverted) → no pushdown, per-row eval
- No index on property → vector falls back to brute-force, text errors
- No label in MATCH → no pushdown, correct results via eval fallback
- Node-to-node similarity → no pushdown, correct via eval_vector_fn
- Null/missing vector property → excluded from results
- Dimension mismatch → excluded from results
- Empty query string → text_score returns 0.0, text_match returns false
- text_score on non-string property → error

## Non-goals

- **Shape C (`CALL hybrid_search`)**: Eliminated. B's compound WHERE predicates subsume it.
- **RRF/weighted fusion in query language**: Stays in the `hybrid_search()` API. Users write their own ranking formulas in ORDER BY.
- **WITH clause alias tracing**: Functions behind a WITH pipeline barrier fall back to per-row eval. Optimizing through WITH is a general planner problem, not specific to this feature.
- **Selectivity-based index selection**: Both indexes run for compound predicates. No heuristic, no guessing.
- **HNSW ef parameter in queries**: Uses index default. Power users tune via API or future session parameter.
