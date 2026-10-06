---
title: Transaction
description: Transaction class reference for Node.js.
tags:
  - api
  - nodejs
---

# Transaction

A database transaction with explicit commit/rollback. Transactions provide ACID guarantees for groups of operations.

## Creating a Transaction

```typescript
const tx = db.beginTransaction();
const tx = db.beginTransaction('serializable');
```

| Isolation Level | Description |
|----------------|-------------|
| `"snapshot"` | Default. Reads see a consistent snapshot at transaction start |
| `"read_committed"` | Reads see the latest committed data |
| `"serializable"` | Full serialization of concurrent transactions |

## Methods

### execute()

Execute a query within this transaction.

```typescript
async execute(query: string, params?: object): Promise<QueryResult>
```

```typescript
const tx = db.beginTransaction();
await tx.execute("INSERT (:Person {name: 'Alix'})");
await tx.execute(
  "INSERT (:Person {name: $name})",
  { name: 'Gus' }
);
```

Throws if the transaction is no longer active (already committed or rolled back).

### commit()

Commit the transaction, making all changes permanent.

```typescript
commit(): void
```

Throws if already committed or rolled back.

### rollback()

Roll back the transaction, discarding all changes.

```typescript
rollback(): void
```

Throws if already committed or rolled back.

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `isActive` | `boolean` | Whether the transaction is still active |

## Auto-rollback

If a transaction is dropped (garbage collected) without being committed or rolled back, it automatically rolls back. In Node.js 22+, you can use `using` for deterministic cleanup:

```typescript
using tx = db.beginTransaction();
await tx.execute("INSERT (:Person {name: 'Alix'})");
tx.commit();
// auto-rollback if commit not called
```

## Example

```typescript
// Basic transaction
const tx = db.beginTransaction();
try {
  await tx.execute("INSERT (:Person {name: 'Alix'})");
  await tx.execute("INSERT (:Person {name: 'Gus'})");
  tx.commit();  // Both inserts committed atomically
} catch (e) {
  tx.rollback();  // Discard all changes on error
}

// Serializable transaction
const tx = db.beginTransaction('serializable');
await tx.execute("MATCH (p:Person) SET p.verified = true");
tx.commit();

// Check active state
console.log(tx.isActive);  // false after commit/rollback
```
