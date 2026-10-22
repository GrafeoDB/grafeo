# @grafeo-db/js

Node.js/TypeScript bindings for [Grafeo](https://grafeo.dev), a high-performance, embeddable graph database with a Rust core.

## Installation

```bash
npm install @grafeo-db/js
```

## Quick Start

```typescript
import { GrafeoDB } from '@grafeo-db/js';

// In-memory database
const db = GrafeoDB.create();

// Or persistent
// const db = GrafeoDB.create('./my-graph');

// Create nodes
db.createNode(['Person'], { name: 'Alix', age: 30 });
db.createNode(['Person'], { name: 'Gus', age: 25 });
db.createEdge(0, 1, 'KNOWS', { since: 2024 });

// Query with GQL
const result = await db.execute('MATCH (p:Person) WHERE p.age > 20 RETURN p.name, p.age');
for (const row of result.toArray()) {
  console.log(row);
}

db.close();
```

## API Reference

### Database

```typescript
// Create / open
const db = GrafeoDB.create();           // in-memory
const db = GrafeoDB.create('./path');    // persistent
const db = GrafeoDB.open('./path');      // open existing

// Counts
db.nodeCount();   // number of nodes
db.edgeCount();   // number of edges
```

### Query Languages

All query methods return `Promise<QueryResult>` and accept optional parameters:

```typescript
await db.execute(gql, params?);         // GQL (ISO standard)
await db.executeCypher(query, params?);  // Cypher
await db.executeGremlin(query, params?); // Gremlin
await db.executeGraphql(query, params?); // GraphQL
await db.executeSparql(query, params?);  // SPARQL
await db.executeSql(query, params?);     // SQL/PGQ (SQL:2023)
```

### Node & Edge CRUD

```typescript
const node = db.createNode(['Label'], { key: 'value' });
const edge = db.createEdge(sourceId, targetId, 'TYPE', { key: 'value' });

const n = db.getNode(id);     // JsNode | null
const e = db.getEdge(id);     // JsEdge | null

db.setNodeProperty(id, 'key', 'value');
db.setEdgeProperty(id, 'key', 'value');

db.deleteNode(id);  // returns boolean
db.deleteEdge(id);  // returns boolean
```

### Transactions

```typescript
const tx = db.beginTransaction();
try {
  await tx.execute("INSERT (:Person {name: 'Harm'})");
  tx.commit();
} catch (e) {
  tx.rollback();
}

// Node.js 22+ with explicit resource management:
using tx = db.beginTransaction();
await tx.execute("INSERT (:Person {name: 'Harm'})");
tx.commit(); // auto-rollback if not committed
```

### QueryResult

```typescript
result.columns;          // column names
result.length;           // row count
result.executionTimeMs;  // execution time (ms)
result.get(0);           // single row as object
result.toArray();        // all rows as objects
result.scalar();         // first column of first row
result.nodes();          // extracted nodes
result.edges();          // extracted edges
```

### Graph Projections

```typescript
db.createProjection('people', ['Person'], ['KNOWS']); // returns boolean
const projections = db.listProjections();  // ['people']
db.dropProjection('people');
```

### Data Import

```typescript
const count = await db.importCsv('./users.csv', { label: 'Person', headers: true });
const count2 = await db.importJsonl('./events.jsonl', { label: 'Event' });
```

### Backup and Restore

```typescript
db.backupFull('/backups/full');
db.backupIncremental('/backups/incr');
GrafeoDB.restoreToEpoch('/backups/full', 100, './restored');
```

### Vector Search

```typescript
// Create an HNSW index
await db.createVectorIndex('Document', 'embedding', 384);

// Bulk insert
const ids = await db.batchCreateNodes('Document', 'embedding', vectors);

// Search
const results = await db.vectorSearch('Document', 'embedding', queryVector, 10);
```

## Features

- GQL, Cypher, SPARQL, Gremlin, GraphQL and SQL/PGQ query languages
- Full node/edge CRUD with property management
- ACID transactions with automatic rollback
- HNSW vector similarity search with batch operations
- Graph projections (filtered virtual views)
- CSV and JSON Lines import
- Incremental backup and restore
- Async/await API backed by Rust + Tokio
- TypeScript definitions included

## Links

- [Documentation](https://grafeo.dev)
- [GitHub](https://github.com/GrafeoDB/grafeo)
- [Python Package](https://pypi.org/project/grafeo/)
- [WASM Package](https://www.npmjs.com/package/@grafeo-db/wasm)

## License

Apache-2.0
