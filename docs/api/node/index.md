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
| [GraphHandle](database.md#named-graphs) | Queries and direct calls in one named graph, from `db.graph(name)` |
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

`kind` says what the result holds: `'rows'`, `'boolean'` for a SPARQL ASK
(its answer is `boolean`, one row with the column `boolean`) or `'graph'` for
SPARQL CONSTRUCT and DESCRIBE (rows of `subject`, `predicate` and `object`).

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
| `{ $decimal: '1.50' }` | Decimal | An exact decimal as its text with its scale, both ways: GQL `1.50` comes back as `{ $decimal: '1.50' }`. Only as an object's only key; with other keys it is a map |
| `string` | String | |
| `{ termType: 'NamedNode', value }` | RDF term | An IRI, both ways (see below); also `BlankNode` and `Literal`, and the terms of other RDF/JS libraries |
| `BigInt` | Int64, or Decimal past 64 bits | Exact up to 38 digits (scale 0); past that an error naming the parameter or property |
| `Array` | List | Elements converted recursively |
| `Object` | Map | Keys must be strings |
| `Buffer` | Bytes | |
| `Date` | Timestamp | Millisecond precision |
| `Float32Array` | Vector | For embeddings and similarity search |

## RDF Terms

A SPARQL result holds each IRI, blank node and literal as an RDF/JS term
object: `{ termType: 'NamedNode', value }`, `{ termType: 'BlankNode', value }`
(the label, without `_:`) or `{ termType: 'Literal', value, language,
datatype }`, where `language` is the tag in lowercase, `''` without a tag,
and `datatype` is a `NamedNode` (`xsd:string` for a simple literal,
`rdf:langString` for a tagged one). Each term, and its datatype, has the RDF/JS
`equals(other)` method: true for a term of the same `termType` and `value`,
and for a literal the same language (tags compare without case) and datatype;
false for `null` and `undefined`. The method is not enumerable, so a term's
keys and its JSON hold the fields only.

A parameter or property takes any object of that shape, the terms of other
RDF/JS libraries (N3.js, rdf-data-factory) too: it is read by `termType`
(`NamedNode`, `BlankNode` or `Literal`) and `value`, and for a literal
`language` and the `value` of `datatype`, through getters as well, and other
keys are ignored. A language tag is kept in lowercase. An object with another
`termType`, or none, is a map. The rows of `executeStream` and the
properties of change events (`nodeHistory`, `changesBetween`) hold terms the
same way. The types (`NamedNode`, `BlankNode`, `Literal`, `RdfTerm` and the
row value union `GrafeoValue`) are exported from the package.

```typescript
const rows = await db.executeSparql('SELECT ?o WHERE { <http://ex.org/apple> <http://ex.org/label> ?o }');
// [{ o: { termType: 'Literal', value: 'Pomme', language: 'fr',
//         datatype: { termType: 'NamedNode', value: 'http://www.w3.org/1999/02/22-rdf-syntax-ns#langString' } } }]
const label = rows.toArray()[0].o;
label.equals({ termType: 'Literal', value: 'Pomme', language: 'FR',
               datatype: { termType: 'NamedNode', value: 'http://www.w3.org/1999/02/22-rdf-syntax-ns#langString' } });  // true
```

## Links

- [npm package](https://www.npmjs.com/package/@grafeo-db/js)
- [GitHub](https://github.com/GrafeoDB/grafeo/tree/main/crates/bindings/node)
