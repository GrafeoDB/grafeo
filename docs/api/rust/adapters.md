---
title: graphos-adapters
description: Adapters crate.
tags:
  - api
  - rust
---

# graphos-adapters

Parsers, storage backends, and plugins.

## GQL Parser

```rust
use graphos_adapters::query::gql::GqlParser;

let ast = GqlParser::parse("MATCH (n:Person) RETURN n")?;
```

## Storage

```rust
use graphos_adapters::storage::{MemoryBackend, WalBackend};

let backend = MemoryBackend::new();
let backend = WalBackend::open("path/to/wal")?;
```

## Plugins

```rust
use graphos_adapters::plugins::{Plugin, PluginRegistry};

let registry = PluginRegistry::new();
registry.register(MyPlugin::new())?;
```

## Note

This is an internal crate. The API may change between minor versions.
