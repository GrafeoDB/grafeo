---
title: Transactions
description: Transaction management in Node.js.
tags:
  - node
  - transactions
---

# Transactions

Transactions group multiple operations into an atomic unit: either all succeed (commit) or all are rolled back.

## Basic Usage

```typescript
const tx = db.beginTransaction();
try {
  await tx.execute("INSERT (:Person {name: 'Alix'})");
  await tx.execute("INSERT (:Person {name: 'Gus'})");
  await tx.execute("MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) INSERT (a)-[:KNOWS]->(b)");
  tx.commit();
} catch (err) {
  tx.rollback();
  throw err;
}
```

## Using `using` (Node.js 22+)

With explicit resource management, the transaction auto-rolls back if `commit()` is not called:

```typescript
{
  using tx = db.beginTransaction();
  await tx.execute("INSERT (:Person {name: 'Harm'})");
  tx.commit();
  // If an exception occurs before commit, tx is automatically rolled back
}
```

## Isolation Levels

```typescript
// Default: snapshot isolation
const tx = db.beginTransaction();

// Explicit isolation level
const tx = db.beginTransaction('serializable');
```

Available levels:

| Level | Description |
| ----- | ----------- |
| `read_committed` | See committed data from other transactions |
| `snapshot` (default) | See a consistent snapshot from transaction start |
| `serializable` | Full serializability, detects write conflicts |

## Query Languages in Transactions

All query languages work inside transactions:

```typescript
const tx = db.beginTransaction();
await tx.execute("INSERT (:Person {name: 'Alix'})");           // GQL
await tx.executeCypher("CREATE (:Person {name: 'Gus'})");      // Cypher
await tx.executeSparql("INSERT DATA { :s :p :o }");            // SPARQL
await tx.executeGremlin("g.addV('Person').property('name','Harm')"); // Gremlin
tx.commit();
```

## Checking Transaction State

```typescript
const tx = db.beginTransaction();
console.log(tx.isActive); // true

tx.commit();
console.log(tx.isActive); // false
```

## Error Handling

If a query fails inside a transaction, the transaction remains active. You can retry or rollback:

```typescript
const tx = db.beginTransaction();
try {
  await tx.execute("INSERT (:Person {name: 'Alix'})");
  await tx.execute("INVALID SYNTAX"); // throws
} catch (err) {
  console.error('Query failed:', err.message);
  // Transaction is still active, you can retry or rollback
  tx.rollback();
}
```

Errors from the Grafeo engine are thrown as standard JavaScript `Error` objects. The message is prefixed with the error category:

- `"Query error: ..."` for syntax or execution errors
- `"Transaction error: ..."` for conflicts or state errors
- `"Database error: ..."` for storage or configuration errors
