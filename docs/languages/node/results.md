---
title: Query Results
description: Working with query results in Node.js.
tags:
  - node
  - results
---

# Query Results

Every `execute` call returns a `Promise<QueryResult>`.

## Accessing Rows

```typescript
const result = await db.execute('MATCH (p:Person) RETURN p.name, p.age');

// All rows as objects
const rows = result.toArray();
// [{ 'p.name': 'Alix', 'p.age': 30 }, { 'p.name': 'Gus', 'p.age': 25 }]

// Single row by index
const first = result.get(0);
// { 'p.name': 'Alix', 'p.age': 30 }

// Row count
console.log(result.length); // 2

// Column names
console.log(result.columns); // ['p.name', 'p.age']
```

## Single Values

For queries that return a single value:

```typescript
const result = await db.execute('MATCH (p:Person) RETURN count(p)');
const count = result.scalar(); // 2
```

## Extracting Entities

When queries return full nodes or edges, extract them as typed objects:

```typescript
const result = await db.execute('MATCH (p:Person) RETURN p');

for (const node of result.nodes()) {
  console.log(node.id);
  console.log(node.labels);
  console.log(node.get('name'));
  console.log(node.properties());
}
```

```typescript
const result = await db.execute('MATCH ()-[r:KNOWS]->() RETURN r');

for (const edge of result.edges()) {
  console.log(edge.edgeType);
  console.log(edge.sourceId, '->', edge.targetId);
}
```

## Raw Rows

For performance-sensitive code, access rows as arrays (no column name mapping). The TypeScript return type is `object`, but the runtime value is an array of arrays:

```typescript
const raw = result.rows() as any[][];
// [[value, value], [value, value], ...]
```

## Metadata

```typescript
result.executionTimeMs; // query time in milliseconds (or null)
result.rowsScanned;     // rows examined by the engine (or null)
```

## Table Display

For debugging, `toString()` formats the result as a Unicode table:

```typescript
const result = await db.execute('MATCH (p:Person) RETURN p.name, p.age');
console.log(result.toString());
// +--------+-------+
// | p.name | p.age |
// +--------+-------+
// | Alix   | 30    |
// | Gus    | 25    |
// +--------+-------+
```
