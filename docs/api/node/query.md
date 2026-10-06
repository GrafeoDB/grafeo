---
title: QueryResult
description: QueryResult class reference for Node.js.
tags:
  - api
  - nodejs
---

# QueryResult

Returned by all query methods (`execute`, `executeCypher`, etc.) and `Transaction.execute`.

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `columns` | `string[]` | Column names from the query |
| `length` | `number` | Number of rows in the result |
| `executionTimeMs` | `number \| null` | Query execution time in milliseconds |
| `rowsScanned` | `number \| null` | Number of rows scanned during execution |

## Methods

### get()

Get a single row by index as a plain object with column names as keys.

```typescript
get(index: number): object
```

```typescript
const row = result.get(0);
console.log(row['p.name']);  // 'Alix'
```

Throws if `index` is out of range.

### toArray()

Get all rows as an array of objects.

```typescript
toArray(): object[]
```

```typescript
const rows = result.toArray();
for (const row of rows) {
  console.log(row['p.name'], row['p.age']);
}
```

### rows()

Get all rows as an array of arrays (no column names). The TypeScript return type is `object`, but the runtime value is an array of arrays.

```typescript
rows(): object  // runtime: any[][]
```

```typescript
const raw = result.rows() as any[][];
// [[value1, value2], [value3, value4], ...]
```

### scalar()

Get the first column of the first row as a single value.

```typescript
scalar(): any
```

```typescript
const count = (await db.execute('MATCH (n) RETURN count(n)')).scalar();
console.log(count);  // 42
```

Throws if the result has no rows or no columns.

### nodes()

Get all nodes found in the result.

```typescript
nodes(): JsNode[]
```

```typescript
const result = await db.execute('MATCH (p:Person) RETURN p');
for (const node of result.nodes()) {
  console.log(node.id, node.labels);
}
```

### edges()

Get all edges found in the result.

```typescript
edges(): JsEdge[]
```

## Example

```typescript
const result = await db.execute('MATCH (p:Person) RETURN p.name, p.age');

// Column names
console.log(result.columns);  // ['p.name', 'p.age']

// Row count
console.log(result.length);  // 3

// Single row
const first = result.get(0);
console.log(first['p.name']);

// All rows as objects
for (const row of result.toArray()) {
  console.log(`${row['p.name']} is ${row['p.age']} years old`);
}

// Scalar value
const count = (await db.execute('MATCH (n) RETURN count(n)')).scalar();

// Metrics
console.log(`Query took ${result.executionTimeMs}ms`);
console.log(`Scanned ${result.rowsScanned} rows`);
```
