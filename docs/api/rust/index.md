---
title: Rust API
description: Rust API reference.
---

# Rust API Reference

Graphos is written in Rust and provides a native Rust API.

## Crates

| Crate | docs.rs |
|-------|---------|
| graphos | [docs.rs/graphos](https://docs.rs/graphos) |
| graphos-engine | [docs.rs/graphos-engine](https://docs.rs/graphos-engine) |

## Quick Start

```rust
use graphos::Database;

fn main() -> Result<(), graphos::Error> {
    let db = Database::open_in_memory()?;
    let session = db.session()?;

    session.execute("INSERT (:Person {name: 'Alice'})")?;

    Ok(())
}
```

## Crate Documentation

- [graphos-common](common.md) - Foundation types
- [graphos-core](core.md) - Core data structures
- [graphos-adapters](adapters.md) - Parsers and storage
- [graphos-engine](engine.md) - Database facade

## API Stability

The public API (`graphos` and `graphos-engine`) follows semver.

Internal crates may have breaking changes in minor versions.
