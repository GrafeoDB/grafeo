---
title: Rust API
description: Rust API reference.
---

# Rust API Reference

Grafeo is written in Rust and provides a native Rust API.

## Crates

| Crate | docs.rs |
|-------|---------|
| grafeo | [docs.rs/grafeo](https://docs.rs/grafeo) |
| grafeo-engine | [docs.rs/grafeo-engine](https://docs.rs/grafeo-engine) |

## Quick Start

```rust
use grafeo::GrafeoDB;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session.execute("INSERT (:Person {name: 'Alix'})")?;

    Ok(())
}
```

## Crate Documentation

- [grafeo-common](common.md) - Foundation types
- [grafeo-core](core.md) - Core data structures
- [grafeo-adapters](adapters.md) - Parsers and storage
- [grafeo-engine](engine.md) - Database facade

## API Stability

The public API (`grafeo` and `grafeo-engine`) follows semver.

Internal crates may have breaking changes in minor versions.
