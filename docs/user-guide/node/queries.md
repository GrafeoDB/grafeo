---
title: Queries
description: Running queries in Node.js with GQL, Cypher, and other languages.
tags:
  - node
  - queries
  - gql
  - cypher
---

# Queries

All query methods are async and return `Promise<QueryResult>`.

## GQL (Default)

```typescript
const result = await db.execute('MATCH (p:Person) RETURN p.name, p.age');
```

## Parameterized Queries

Use `$param` syntax to safely pass values without string interpolation:

```typescript
const result = await db.execute(
  'MATCH (p:Person) WHERE p.age > $minAge RETURN p.name',
  { minAge: 25 }
);
```

Parameters prevent injection and allow the engine to cache query plans.

## Cypher

If you are coming from Neo4j, you can use Cypher syntax:

```typescript
const result = await db.executeCypher(
  'MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name'
);
```

Cypher queries also support parameters:

```typescript
const result = await db.executeCypher(
  'MATCH (p:Person) WHERE p.name = $name RETURN p',
  { name: 'Alix' }
);
```

Cypher is available on transactions too:

```typescript
const tx = db.beginTransaction();
await tx.executeCypher("CREATE (p:Person {name: $name})", { name: 'Gus' });
tx.commit();
```

## Other Languages

```typescript
// Gremlin
await db.executeGremlin("g.V().hasLabel('Person').values('name')");

// GraphQL
await db.executeGraphql('{ Person { name age } }');

// SPARQL (for RDF data)
await db.executeSparql('SELECT ?name WHERE { ?p a :Person ; :name ?name }');

// SQL/PGQ (SQL:2023 GRAPH_TABLE)
await db.executeSql("SELECT name FROM GRAPH_TABLE(g MATCH (p:Person) COLUMNS (p.name))");
```

## Query Languages on Transactions

The `Transaction` class supports the same query language methods as `GrafeoDB`:

```typescript
const tx = db.beginTransaction();
await tx.execute('...');           // GQL
await tx.executeCypher('...');     // Cypher
await tx.executeSql('...');        // SQL/PGQ
await tx.executeGremlin('...');    // Gremlin
await tx.executeGraphql('...');    // GraphQL
await tx.executeSparql('...');     // SPARQL
tx.commit();
```

Each language method requires its corresponding feature flag (e.g. `cypher`, `sql-pgq`).

## Sync vs Async

Query execution is always async (returns a `Promise`), because the Rust engine runs queries on a background thread via `tokio::task::spawn_blocking` to avoid blocking the Node.js event loop.

CRUD operations (`createNode`, `deleteNode`, etc.) and transaction control (`commit`, `rollback`) are synchronous.

| Operation | Sync/Async |
| --------- | ---------- |
| `GrafeoDB.create()`, `.open()` | Sync |
| `createNode`, `createEdge`, `deleteNode`, `deleteEdge` | Sync |
| `setNodeProperty`, `setEdgeProperty` | Sync |
| `nodeCount`, `edgeCount`, `info`, `schema` | Sync |
| `beginTransaction`, `commit`, `rollback` | Sync |
| `save`, `walCheckpoint`, `backupFull`, `backupIncremental` | Sync |
| `execute`, `executeCypher`, `executeSql`, etc. | **Async** |
| `createVectorIndex`, `vectorSearch` | **Async** |
| `batchCreateNodes`, `batchVectorSearch` | **Async** |
| `importCsv`, `importJsonl` | **Async** |
