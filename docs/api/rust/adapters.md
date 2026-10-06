---
title: grafeo-adapters
description: Adapters crate.
tags:
  - api
  - rust
---

# grafeo-adapters

Parsers, storage backends and plugins.

## GQL Parser

```rust
use grafeo_adapters::query::gql;

let ast = gql::parse("MATCH (n:Person) RETURN n")?;
```

## Storage

```rust
use grafeo_adapters::storage::MemoryBackend;
use grafeo_storage::wal::WalManager;

let backend = MemoryBackend::new();
let wal = WalManager::open("path/to/wal")?;
```

## Plugins

```rust
use std::sync::Arc;
use grafeo_adapters::plugins::{Plugin, PluginRegistry};

let registry = PluginRegistry::new();
registry.register_plugin(Arc::new(MyPlugin::new()))?;
```

## Note

This is an internal crate. The API may change between minor versions.
