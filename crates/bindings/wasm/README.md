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
- [x] Size optimization (513 KB gzipped lite, 531 KB AI variant)
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

| Build | Size (gzip) |
|-------|-------------|
| AI variant (GQL + vector/text/hybrid search) | 531 KB |
| Lite variant (GQL only) | 513 KB |

## Runtime Support

| Runtime | Status |
|---------|--------|
| Browser (Chrome, Firefox, Safari, Edge) | Supported |
| Deno | Supported |
| Cloudflare Workers | Untested |
| Node.js | Use `@grafeo-db/js` instead |

## Links

- [Documentation](https://grafeo.dev)
- [GitHub](https://github.com/GrafeoDB/grafeo)
- [Roadmap](https://github.com/GrafeoDB/grafeo/blob/main/docs/roadmap.md)

## License

Apache-2.0
