# @grafeo-db/wasm

> **Status: Coming Soon** - This package is under active development.

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
import init, { GrafeoCore } from '@grafeo-db/wasm';

// Initialize the WASM module
await init();

// Low-level API
const core = new GrafeoCore();
const result = core.execute(`MATCH (n:Person) RETURN n.name`);
```

## Progress

- [ ] Core WASM bindings via wasm-bindgen
- [ ] In-memory database support
- [ ] All query languages (GQL, Cypher, SPARQL, GraphQL, Gremlin)
- [ ] TypeScript type definitions
- [ ] Size optimization (target: <1MB gzipped)

## Package Contents

```
@grafeo-db/wasm/
├── grafeo_wasm_bg.wasm    # WebAssembly binary
├── grafeo_wasm.js         # JavaScript loader
├── grafeo_wasm.d.ts       # TypeScript definitions
└── package.json
```

## Target Bundle Size

| Build | Size (gzip) |
|-------|-------------|
| Full (all languages) | ~600 KB |
| Minimal (GQL only) | ~300 KB |

## Runtime Support

| Runtime | Status |
|---------|--------|
| Browser (Chrome, Firefox, Safari, Edge) | Planned |
| Deno | Planned |
| Cloudflare Workers | Planned |
| Node.js | Use `@grafeo-db/js` instead |

## Current Alternatives

While WASM bindings are in development, you can use:

- **Python**: [`grafeo`](https://pypi.org/project/grafeo/) - fully functional
- **Rust**: [`grafeo`](https://crates.io/crates/grafeo) - fully functional

## Links

- [Documentation](https://grafeo.dev)
- [GitHub](https://github.com/GrafeoDB/grafeo)
- [Roadmap](https://github.com/GrafeoDB/grafeo/blob/main/docs/roadmap.md)

## License

Apache-2.0
