---
title: WebAssembly API
description: API reference for the @grafeo-db/wasm package.
---

# WebAssembly API

Run Grafeo in the browser, Deno or Cloudflare Workers. ~513 KB gzipped.

```bash
npm install @grafeo-db/wasm
```

## Quick Start

```javascript
import init, { Database } from '@grafeo-db/wasm';

await init();
const db = new Database();

db.execute("INSERT (:Person {name: 'Alix', age: 30})");
db.execute("INSERT (:Person {name: 'Gus', age: 25})");

const results = db.execute("MATCH (p:Person) RETURN p.name, p.age");
console.log(results); // [{name: "Alix", age: 30}, {name: "Gus", age: 25}]
```

## Database

```javascript
const db = new Database();   // in-memory (all WASM databases are in-memory)
```

## Query Methods

```javascript
db.execute(gql);                              // GQL: returns array of row objects
db.executeRaw(gql);                          // GQL: returns {columns, rows, executionTimeMs, counters}, and profile for PROFILE
db.executeWithParams(gql, params);           // GQL with parameter binding
db.executeWithLanguage(query, language);     // "gql", "cypher", "graphql", etc.
db.executeWithLanguageAndParams(query, language, params);  // language + params
db.executeCypher(query);                     // Cypher shorthand
db.executeGremlin(query);                    // Gremlin shorthand
db.executeGraphql(query);                    // GraphQL shorthand
db.executeSparql(query);                     // SPARQL shorthand (requires rdf feature)
db.executeSql(query);                        // SQL/PGQ shorthand
db.executeRawWithLanguage(query, language);  // raw result with language selection
db.executeRawWithParams(gql, params);        // raw result with parameter binding
db.executeRawWithLanguageAndParams(query, language, params);  // raw result, language + params
```

A raw result's `counters` tell what a write changed: `nodesCreated`, `nodesDeleted`, `edgesCreated`,
`edgesDeleted`, `propertiesSet`, `labelsAdded` and `labelsRemoved` (all zero for a read).

## Properties

```javascript
db.getNodeEdges(id, { direction: "out", edgeTypes: ["KNOWS"] });  // [[neighborId, edgeId], ...]
db.nodeCount();   // number of nodes
db.edgeCount();   // number of edges
db.schema();      // database schema as JSON
Database.version(); // Grafeo version string
```

## Decimals

An exact decimal (GQL `1.50`, `CAST(x AS DECIMAL(10, 2))`, SPARQL
`xsd:decimal`) comes back as `{ $decimal: '1.50' }`, its text with its scale,
and such an object passed as a parameter is a decimal. With other keys beside
`$decimal`, the object is a map.

```javascript
db.execute("RETURN 0.1 + 0.2 AS x");      // [{ x: { $decimal: '0.3' } }]
db.executeWithParams("RETURN $p * 3 AS x", { p: { $decimal: '0.10' } });  // { $decimal: '0.30' }
```

## Text Search

Create BM25 text indexes and run full-text queries:

```javascript
db.createTextIndex("Document", "content");
const results = db.textSearch("Document", "content", "graph database", 10);
// [{id, score}, ...]

// Options: BM25 k1 (>= 0, default 1.2) and b (0 to 1, default 0.75), the
// tokenizer ("simple", the default, "standard" or "cjk_bigram") and stop words
db.createTextIndex("Note", "body", { tokenizer: "cjk_bigram", k1: 1.5, stopWords: ["住在"] });

// Only the nodes the filters match, as in vectorSearch()
const berlin = db.textSearch("Document", "content", "graph", 10, { filters: { city: "Berlin" } });

db.rebuildTextIndex("Document", "content");
db.dropTextIndex("Document", "content");
```

## Hybrid Search

Combine BM25 text scores with HNSW vector similarity:

```javascript
// Create indexes via GQL queries
db.execute("CREATE TEXT INDEX ON Document(content)");
db.execute("CREATE VECTOR INDEX ON Document(embedding) OPTIONS {dimensions: 384}");

const results = db.hybridSearch(
    "Document",
    "content",                        // text field
    "embedding",                      // vector field
    "graph database",                 // text query
    10,                               // top-k
    { filters: { city: "Berlin" } }   // optional: property filters, as in vectorSearch()
);
```

With `filters`, the search keeps only the matching nodes before fusion, so up to `k` matching nodes come back.

!!! tip "Vector Index Creation"
    Both `createVectorIndex()` and GQL `CREATE VECTOR INDEX` queries via `db.execute()` are supported in WASM.

## Batch Import

Load structured data in a single call, avoiding per-row query overhead.

### LPG Import

```javascript
const result = db.importLpg({
    nodes: [
        { labels: ["Person"], properties: { name: "Alix", age: 30 } },
        { labels: ["Person"], properties: { name: "Gus", age: 25 } },
    ],
    edges: [
        { source: 0, target: 1, type: "KNOWS", properties: { since: 2020 } }
    ]
});
console.log(result); // { nodes: 2, edges: 1 }
```

Edge `source` and `target` are zero-based indexes into the `nodes` array from the same call.

### Batches and Upserts

The batches, upserts and graph handles (below) need the `direct-api` feature, which the `lpg` and
`full` builds include (`@grafeo-db/wasm`); the lite `edge` build leaves them out to keep its size.

```javascript
// One transaction each; IDs in input order
const ids = db.batchCreateNodesWithProps(["Person"], [{ name: "Alix" }, { name: "Gus" }]);
db.batchCreateEdges([{ src: ids[0], dst: ids[1], type: "KNOWS", properties: { since: 2020 } }]);

// Create or update by a key property: {created, updated, skipped, skippedRows}
db.upsertNodes(["File"], [{ path: "a.rs", size: 3 }], { key: "path" });
db.upsertEdges("IMPORTS", [{ id: "a-b", src: "a.rs", dst: "b.rs" }],
    { endpointKey: "path", endpointLabels: ["File"] });
```

An edge to a missing node fails the edge batch and writes nothing. A node row without the key is
skipped, and so is an edge row whose endpoint matches no node or several; endpoints are never
created. The upsert options are `key` and `replace`, and for edges also `endpointKey`,
`endpointLabels`, `srcField` and `dstField`. Upserts run in the open transaction when there is one;
the batches commit on their own, so they throw while a transaction is open.

With the `vector-index` feature, `db.batchVectorSearch(label, property, queries, k, { ef, filters })`
searches several query vectors at once and returns one `[{id, distance}]` array per query.

### Named Graphs

With the `direct-api` feature, `db.graph(name)` returns a `GraphHandle` that works in one named graph without switching the current
graph: the `execute*` and `executeRaw*` methods, `batchCreateNodesWithProps`, `batchCreateEdges`,
`upsertNodes`, `upsertEdges`, and its `name`. Each call runs on its own, so a call throws while the
database has a transaction open, once the graph is dropped, and after `db.close()`.

```javascript
db.execute("CREATE GRAPH trips");
const trips = db.graph("trips");
trips.execute("INSERT (:City {name: 'Prague'})");
trips.execute("MATCH (c:City) RETURN c.name");  // the default graph stays empty
```

### RDF Import

Requires the `rdf` feature flag.

```javascript
const result = db.importRdf({
    triples: [
        {
            subject: "http://example.org/Alix",
            predicate: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            object: "http://example.org/Person"
        },
        {
            subject: "http://example.org/Alix",
            predicate: "http://example.org/name",
            object: { value: "Alix" }
        },
        {
            subject: "http://example.org/Alix",
            predicate: "http://example.org/age",
            object: { value: "30", datatype: "http://www.w3.org/2001/XMLSchema#integer" }
        }
    ]
});
console.log(result); // { triples: 3 }
```

Objects can be a plain string (treated as IRI), or a structured literal with `value`, optional `datatype` and optional `language` fields.

## Compact Store

`compact()` drops the old versions no open transaction can see any more and returns what it did: `{ checkpointed, versions_collected, duration_ms }` (a database in memory, as every WASM database is, writes no checkpoint). Since 0.6.0 it no longer builds a separate columnar store. See the [Compact Store guide](../../user-guide/compact-store.md).

```javascript
const report = db.compact();
```

## Snapshots (Persistence)

Export/import the entire database as a binary snapshot for IndexedDB persistence:

```javascript
// Export
const snapshot = db.exportSnapshot();
// Store in IndexedDB...

// Import
const db2 = Database.importSnapshot(snapshot);
```

## Supported Query Languages

The WASM build supports query languages based on compile-time features:

| Feature | Language | Default |
|---------|----------|---------|
| `gql` | GQL | Yes |
| `cypher` | Cypher | No |
| `sparql` | SPARQL | No |
| `gremlin` | Gremlin | No |
| `graphql` | GraphQL | No |
| `sql-pgq` | SQL/PGQ | No |

The `full` feature enables all languages. The default npm package includes only GQL to minimize bundle size.

## Bundle Size

| Build | Size |
|-------|------|
| Lite (GQL only) | ~513 KB gzipped |
| AI variant (GQL + vector/text/hybrid search) | ~531 KB gzipped |

## Links

- [npm package](https://www.npmjs.com/package/@grafeo-db/wasm)
- [GitHub](https://github.com/GrafeoDB/grafeo/tree/main/crates/bindings/wasm)
