---
title: Node.js / TypeScript API
description: API reference for the @grafeo-db/js package.
---

# Node.js / TypeScript API Reference

Complete reference for the `@grafeo-db/js` package, native bindings for Grafeo via [napi-rs](https://napi.rs).

## Installation

```bash
npm install @grafeo-db/js
```

## Quick Start

```typescript
import { GrafeoDB } from '@grafeo-db/js';

const db = GrafeoDB.create();

db.createNode(['Person'], { name: 'Alix', age: 30 });
db.createNode(['Person'], { name: 'Gus', age: 25 });
db.createEdge(0, 1, 'KNOWS', { since: 2024 });

const result = await db.execute('MATCH (p:Person) RETURN p.name, p.age');
console.log(result.toArray());

db.close();
```

## Classes

| Class | Description |
|-------|-------------|
| [GrafeoDB](database.md) | Database connection, queries, CRUD, vector/text search |
| [QueryResult](query.md) | Query result with rows, columns, nodes, edges |
| [Transaction](transaction.md) | ACID transaction management |
| [JsNode](node.md) | Graph node with labels and properties |
| [JsEdge](edge.md) | Graph edge with type, endpoints and properties |

## GrafeoDB

### openReadOnly(path)

Open an existing database in read-only mode. Multiple processes can read concurrently.

```typescript
const db = GrafeoDB.openReadOnly('./my_graph.db');
```

### clearPlanCache()

Clear the query plan cache. Called automatically after DDL.

```typescript
db.clearPlanCache();
```

### setSchema(name) / resetSchema() / currentSchema()

Set, clear, or read the schema context for subsequent queries.

```typescript
db.setSchema('reporting');
const name = db.currentSchema();  // 'reporting'
db.resetSchema();
```

## QueryResult

### toString()

Format the result as a Unicode table string.

```typescript
const result = await db.execute('MATCH (n:Person) RETURN n.name, n.age');
console.log(result.toString());
```

## Query Languages

All query methods return `Promise<QueryResult>` and accept an optional `params` object:

```typescript
await db.execute(gql, params?);          // GQL (ISO standard)
await db.executeCypher(query, params?);   // Cypher
await db.executeGremlin(query, params?);  // Gremlin
await db.executeGraphql(query, params?);  // GraphQL
await db.executeSparql(query, params?);   // SPARQL
await db.executeSql(query, params?);      // SQL/PGQ
```

## Type Mapping

| JavaScript | Grafeo | Notes |
|-----------|--------|-------|
| `null` / `undefined` | Null | |
| `boolean` | Bool | |
| `number` (integer) | Int64 | No fractional part and within safe integer range |
| `number` (float) | Float64 | |
| `string` | String | |
| `BigInt` | Int64 | |
| `Array` | List | Elements converted recursively |
| `Object` | Map | Keys must be strings |
| `Buffer` | Bytes | |
| `Date` | Timestamp | Millisecond precision |
| `Float32Array` | Vector | For embeddings and similarity search |

## Links

- [npm package](https://www.npmjs.com/package/@grafeo-db/js)
- [GitHub](https://github.com/GrafeoDB/grafeo/tree/main/crates/bindings/node)
