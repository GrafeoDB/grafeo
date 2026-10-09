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
db.executeRaw(gql);                          // GQL: returns {columns, rows, executionTimeMs}
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

## Properties

```javascript
db.nodeCount();   // number of nodes
db.edgeCount();   // number of edges
db.schema();      // database schema as JSON
Database.version(); // Grafeo version string
```

## Text Search

Create BM25 text indexes and run full-text queries:

```javascript
db.createTextIndex("Document", "content");
const results = db.textSearch("Document", "content", "graph database", 10);
// [{nodeId, score}, ...]

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
    "content", "graph database",     // text field + query
    "embedding", queryVector,         // vector field + query
    10                                // top-k
);
```

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
