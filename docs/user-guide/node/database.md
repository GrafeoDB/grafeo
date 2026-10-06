---
title: Database Operations
description: Creating and managing databases in Node.js.
tags:
  - node
  - database
---

# Database Operations

## Creating a Database

```typescript
import { GrafeoDB } from '@grafeo-db/js';

// In-memory (no persistence)
const db = GrafeoDB.create();

// Persistent (creates or opens)
const db = GrafeoDB.create('./my-graph');

// Open existing (fails if not found)
const db = GrafeoDB.open('./my-graph');

// Read-only (multiple processes can read concurrently)
const db = GrafeoDB.openReadOnly('./my-graph');
```

`GrafeoDB.create()` is synchronous and returns immediately. All factory methods throw on failure.

## Closing

Always close the database when done to flush pending writes:

```typescript
db.close();
```

For long-running servers, close on shutdown:

```typescript
process.on('SIGTERM', () => {
  db.close();
  process.exit(0);
});
```

## Database Info

```typescript
db.nodeCount();   // number of nodes
db.edgeCount();   // number of edges
db.version();     // Grafeo engine version
db.info();        // full database info as JSON object
db.schema();      // labels, edge types, property keys
```

## Schema Context

Scope queries to a named schema (namespace):

```typescript
db.setSchema('production');
// All queries now operate within the 'production' schema
await db.execute("INSERT (:User {name: 'Alix'})");

console.log(db.currentSchema()); // 'production'

db.resetSchema();
// Back to default namespace
```

## Concurrency

`GrafeoDB` is safe to share across async handlers in Express, Fastify, or similar frameworks. The Rust engine uses MVCC (multi-version concurrency control), so readers never block writers. For write-heavy workloads, use explicit transactions to batch operations.

```typescript
import express from 'express';

const app = express();
const db = GrafeoDB.create('./app.db');

app.get('/users', async (req, res) => {
  const result = await db.execute('MATCH (u:User) RETURN u.name');
  res.json(result.toArray());
});

app.post('/users', async (req, res) => {
  await db.execute(
    "INSERT (:User {name: $name})",
    { name: req.body.name }
  );
  res.sendStatus(201);
});
```

## Cache Management

```typescript
// Clear the query plan cache (automatic after DDL, manual if needed)
db.clearPlanCache();
```
