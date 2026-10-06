---
title: grafeo-web
description: Grafeo graph database in the browser via WebAssembly with IndexedDB persistence and framework integrations.
---

# grafeo-web

Grafeo graph database running entirely in the browser via WebAssembly. Zero backend, data stays on the client, with optional IndexedDB persistence across sessions.

[:octicons-mark-github-16: GitHub](https://github.com/GrafeoDB/grafeo-web){ .md-button }
[:material-package-variant: npm](https://www.npmjs.com/package/@grafeo-db/web){ .md-button }

## Overview

`@grafeo-db/web` wraps the Grafeo WASM binary in a TypeScript SDK with persistence, Web Worker support and framework-specific hooks. It supports all major query languages (GQL, Cypher, SPARQL, Gremlin, GraphQL, SQL/PGQ) and provides a consistent API across React, Vue and Svelte.

## Installation

```bash
npm install @grafeo-db/web
```

## Quick Start

```typescript
import { GrafeoDB } from '@grafeo-db/web';

// In-memory database
const db = await GrafeoDB.create();

// Or persist to IndexedDB
const db = await GrafeoDB.create({ persist: 'my-database' });

// Create data
await db.execute(`INSERT (:Person {name: 'Alix', age: 30})`);
await db.execute(`INSERT (:Person {name: 'Gus', age: 25})`);
await db.execute(`
  MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
  INSERT (a)-[:KNOWS {since: 2020}]->(b)
`);

// Query
const result = await db.execute(`
  MATCH (p:Person)-[:KNOWS]->(friend)
  RETURN p.name, friend.name
`);
console.log(result);

await db.close();
```

## Multi-Language Queries

All six query languages supported by the Grafeo engine are available:

```typescript
// GQL (default)
await db.execute(`MATCH (p:Person) RETURN p.name`);

// Cypher
await db.execute(`MATCH (p:Person) RETURN p.name`, { language: 'cypher' });

// SPARQL
await db.execute(`SELECT ?name WHERE { ?p a :Person ; :name ?name }`, { language: 'sparql' });
```

Supported values: `gql`, `cypher`, `sparql`, `gremlin`, `graphql`, `sql`.

## API

### `GrafeoDB`

| Method | Description |
|--------|-------------|
| `GrafeoDB.create(options?)` | Create a database instance |
| `GrafeoDB.version()` | Get the WASM engine version |
| `db.execute(query, options?)` | Execute a query, returns `Record<string, unknown>[]` |
| `db.executeRaw(query)` | Execute a query, returns columns + rows + timing |
| `db.nodeCount()` | Number of nodes |
| `db.edgeCount()` | Number of edges |
| `db.schema()` | Schema info: labels, edge types, property keys |
| `db.export()` | Export full database as a snapshot |
| `db.import(snapshot)` | Restore from a snapshot |
| `db.clear()` | Delete all data |
| `db.storageStats()` | IndexedDB usage and quota |
| `db.close()` | Release WASM memory and cleanup |

### Create Options

```typescript
{
  persist?: string;          // IndexedDB key for persistence
  worker?: boolean;          // Run WASM in a Web Worker (default: false)
  persistInterval?: number;  // Debounce interval in ms (default: 1000)
}
```

## Persistence

Data persists to IndexedDB automatically when `persist` is set. Persistence only triggers on mutating queries (INSERT, CREATE, DELETE), not on reads.

```typescript
// First visit
const db = await GrafeoDB.create({ persist: 'my-app' });
await db.execute(`INSERT (:User {name: 'Alix'})`);

// Later visit, data is still there
const db = await GrafeoDB.create({ persist: 'my-app' });
const result = await db.execute(`MATCH (u:User) RETURN u.name`);
// -> [{ 'u.name': 'Alix' }]
```

### Storage Management

```typescript
const stats = await db.storageStats();
console.log(`Using ${stats.bytesUsed} of ${stats.quota} bytes`);

// Export / import
const snapshot = await db.export();
const db2 = await GrafeoDB.create();
await db2.import(snapshot);
```

## Web Worker Mode

For large databases or complex queries, run in a Web Worker to keep the UI thread responsive:

```typescript
const db = await GrafeoDB.create({
  worker: true,
  persist: 'large-database',
});
```

## Framework Integrations

### React

```tsx
import { useGrafeo, useQuery } from '@grafeo-db/web/react';

function App() {
  const { db, loading, error } = useGrafeo({ persist: 'my-app' });
  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;
  return <PersonList db={db} />;
}

function PersonList({ db }) {
  const { data, loading, refetch } = useQuery(
    db,
    `MATCH (p:Person) RETURN p.name`,
  );
  if (loading) return <div>Loading...</div>;
  return (
    <ul>
      {data.map((row, i) => <li key={i}>{row['p.name']}</li>)}
    </ul>
  );
}
```

### Vue

```vue
<script setup>
import { useGrafeo, useQuery } from '@grafeo-db/web/vue';

const { db, loading, error } = useGrafeo({ persist: 'my-app' });
const { data } = useQuery(db, `MATCH (p:Person) RETURN p.name`);
</script>
```

### Svelte

```svelte
<script>
  import { createGrafeo } from '@grafeo-db/web/svelte';
  const { db, loading, error } = createGrafeo({ persist: 'my-app' });
</script>

{#if $loading}Loading...{/if}
{#if $error}Error: {$error.message}{/if}
```

## Lite Build

A smaller build with GQL support only:

```typescript
import { GrafeoDB } from '@grafeo-db/web/lite';

const db = await GrafeoDB.create();
await db.execute(`MATCH (n) RETURN n`);
```

## Browser Support

| Browser | Version |
|---------|---------|
| Chrome  | 89+     |
| Firefox | 89+     |
| Safari  | 15+     |
| Edge    | 89+     |

Requires WebAssembly, IndexedDB and Web Workers.

## Limitations

| Constraint   | Limit                          |
|--------------|--------------------------------|
| Database size | ~500 MB (IndexedDB quota)     |
| Memory       | ~256 MB (WASM heap)            |
| Concurrency  | Single writer, multiple readers |

For larger datasets, use [Grafeo](https://github.com/GrafeoDB/grafeo) server-side or via [grafeo-server](grafeo-server.md).

## When to Use

| Scenario | Recommendation |
|----------|----------------|
| Offline-first web apps | grafeo-web |
| Prototyping without a backend | grafeo-web |
| Multi-client access over HTTP | [grafeo-server](grafeo-server.md) |
| Embedded in Python / Rust | [grafeo](https://github.com/GrafeoDB/grafeo) (library) |

## License

Apache-2.0
