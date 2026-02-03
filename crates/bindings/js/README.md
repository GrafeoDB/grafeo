# @grafeo-db/js

> **Status: Coming Soon** - This package is under active development.

Native Node.js bindings for [Grafeo](https://github.com/GrafeoDB/grafeo), a high-performance graph database.

## Which Package Do You Need?

| Package | Use Case |
|---------|----------|
| [`@grafeo-db/js`](https://www.npmjs.com/package/@grafeo-db/js) | Node.js, Bun, server-side (native performance) |
| [`@grafeo-db/web`](https://www.npmjs.com/package/@grafeo-db/web) | Browser apps with IndexedDB, Web Workers, React/Vue/Svelte |
| [`@grafeo-db/wasm`](https://www.npmjs.com/package/@grafeo-db/wasm) | Deno, Cloudflare Workers, edge runtimes (WASM) |

**This package uses native bindings (napi-rs)** for maximum performance in Node.js. For browser or edge runtimes, use `@grafeo-db/web` or `@grafeo-db/wasm`.

## Installation

```bash
npm install @grafeo-db/js
```

## Usage

```typescript
import { GrafeoDB } from '@grafeo-db/js';

// In-memory database
const db = await GrafeoDB.create();

// Or with file persistence
const db = await GrafeoDB.create({ path: './my-graph.db' });

// Create data
await db.execute(`INSERT (:Person {name: 'Alice', age: 30})`);
await db.execute(`INSERT (:Person {name: 'Bob', age: 25})`);
await db.execute(`
  MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
  INSERT (a)-[:KNOWS]->(b)
`);

// Query
const result = await db.execute(`
  MATCH (p:Person)-[:KNOWS]->(friend)
  RETURN p.name, friend.name
`);

console.log(result.rows);
// [{ 'p.name': 'Alice', 'friend.name': 'Bob' }]

// Clean up
await db.close();
```

## Progress

- [ ] Native bindings via napi-rs
- [ ] In-memory and file-based storage
- [ ] All query languages (GQL, Cypher, SPARQL, GraphQL, Gremlin)
- [ ] Full TypeScript type definitions
- [ ] Async/await API
- [ ] Transactions
- [ ] Prebuilt binaries (linux-x64, darwin-x64, darwin-arm64, win32-x64)

## Runtime Support

| Runtime | Status |
|---------|--------|
| Node.js 18+ | Planned |
| Bun | Planned |
| Deno | Use `@grafeo-db/wasm` instead |
| Browser | Use `@grafeo-db/web` instead |

## Current Alternatives

While Node.js bindings are in development, you can use:

- **Python**: [`grafeo`](https://pypi.org/project/grafeo/) - fully functional
- **Rust**: [`grafeo`](https://crates.io/crates/grafeo) - fully functional

## Links

- [Documentation](https://grafeo.dev)
- [GitHub](https://github.com/GrafeoDB/grafeo)
- [Roadmap](https://github.com/GrafeoDB/grafeo/blob/main/docs/roadmap.md)

## License

Apache-2.0
