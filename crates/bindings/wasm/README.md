# @grafeo-db/wasm

Low-level WebAssembly binary for [Grafeo](https://github.com/GrafeoDB/grafeo), a high-performance graph database.

## Which Package Do You Need?

| Package | Use Case |
|---------|----------|
| [`@grafeo-db/web`](https://www.npmjs.com/package/@grafeo-db/web) | Browser apps with IndexedDB, Web Workers, React/Vue/Svelte (recommended) |
| [`@grafeo-db/wasm`](https://www.npmjs.com/package/@grafeo-db/wasm) | Raw WASM binary for custom loaders or non-standard runtimes |
| [`@grafeo-db/js`](https://www.npmjs.com/package/@grafeo-db/js) | Node.js native bindings (faster than WASM for server-side) |

**Most users should use `@grafeo-db/web`** - it wraps this package and adds browser-specific features.

## Installation

```bash
npm install @grafeo-db/wasm
```

## Usage

```typescript
import init, { Database } from '@grafeo-db/wasm';

// Initialize the WASM module
await init();

// Create a database and query
const db = new Database();
const result = db.execute(`MATCH (n:Person) RETURN n.name`);
```

## Status

- [x] Core WASM bindings via wasm-bindgen
- [x] In-memory database support
- [x] GQL query language (default via `browser` profile)
- [x] TypeScript type definitions
- [x] Built for speed at a small size (about 1.1 MB gzipped lite, 1.7 MB full)
- [x] Vector search bindings (k-NN, MMR)
- [x] Snapshot export/import for IndexedDB persistence
- [x] Batch import (importLpg, importRdf, importRows)
- [x] Memory introspection (memoryUsage)

## Package Contents

```
@grafeo-db/wasm/
├── grafeo_wasm_bg.wasm    # WebAssembly binary
├── grafeo_wasm.js         # JavaScript loader
├── grafeo_wasm.d.ts       # TypeScript definitions
└── package.json
```

## Bundle Size

| Package | Contents | Size (gzip) |
|---------|----------|-------------|
| `@grafeo-db/wasm-lite` | property graphs, GQL | about 1.1 MB |
| `@grafeo-db/wasm` | every query language, RDF, vector and text search, graph algorithms | about 1.7 MB |

Both are built for speed (opt-level `s`): about 1.6 times as fast as the smallest build, and 4.8 times on inserts. Regular expressions use the lightweight `regex-lite` engine in both.

## Runtime Support

| Runtime | Status |
|---------|--------|
| Browser (Chrome, Firefox, Safari, Edge) | Supported |
| Deno | Supported |
| Cloudflare Workers | Untested |
| Node.js | Use `@grafeo-db/js` instead |

## Links

- [Documentation](https://grafeo.dev)
- [API Stability](https://grafeo.dev/stability/): which calls are stable and which are experimental; a call here has the level of the Rust call it wraps
- [GitHub](https://github.com/GrafeoDB/grafeo)
- [Roadmap](https://github.com/GrafeoDB/grafeo/blob/main/docs/roadmap.md)

## License

Apache-2.0
