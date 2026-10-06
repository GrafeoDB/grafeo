---
title: API Reference
description: Complete API reference for Grafeo.
---

# API Reference

Complete API documentation for all supported languages.

## Python API

The Python API provides a Pythonic interface to Grafeo.

- [Python Overview](python/index.md)
- [grafeo.GrafeoDB](python/database.md)
- [grafeo.Node](python/node.md)
- [grafeo.Edge](python/edge.md)
- [grafeo.QueryResult](python/result.md)
- [grafeo.Transaction](python/transaction.md)

## Node.js / TypeScript API

Native bindings via napi-rs with full TypeScript definitions.

- [Node.js Overview](node/index.md)
- [GrafeoDB](node/database.md)
- [QueryResult](node/query.md)
- [Transaction](node/transaction.md)
- [JsNode](node/node.md)
- [JsEdge](node/edge.md)

## Go API

CGO bindings for cloud-native applications.

- **Package**: [`github.com/GrafeoDB/grafeo/crates/bindings/go`](https://pkg.go.dev/github.com/GrafeoDB/grafeo/crates/bindings/go)
- Node/edge CRUD, property management, label operations
- ACID transactions, vector search, batch operations

## C# / .NET API

Full-featured .NET 8 bindings via source-generated P/Invoke.

- **Package**: `GrafeoDB` (NuGet)
- GQL + multi-language queries (sync and async), ACID transactions
- Typed node/edge CRUD, vector search (k-NN + MMR), temporal type support
- `SafeHandle`-based resource management

## Dart API

Dart FFI bindings wrapping the C layer.

- **Package**: [`grafeo`](https://pub.dev/packages/grafeo)
- GQL query execution with parameterized queries, ACID transactions
- Typed node/edge CRUD, vector search, `NativeFinalizer` resource management

## WebAssembly API

Run Grafeo in the browser, Deno or Cloudflare Workers.

- **Package**: [`@grafeo-db/wasm`](https://www.npmjs.com/package/@grafeo-db/wasm)
- In-memory only, all query languages supported
- 660 KB gzipped binary

## Rust API

The Rust API provides direct access to Grafeo internals.

- [Rust Overview](rust/index.md)
- [grafeo-common](rust/common.md)
- [grafeo-core](rust/core.md)
- [grafeo-adapters](rust/adapters.md)
- [grafeo-engine](rust/engine.md)

## API Stability

| API | Stability |
|-----|-----------|
| Python | Stable |
| Node.js / TypeScript | Stable |
| Go | Stable |
| C# / .NET | Stable |
| Dart | Stable |
| WebAssembly | Stable |
| Rust (grafeo-engine) | Stable |
| Rust (internal crates) | Unstable |
