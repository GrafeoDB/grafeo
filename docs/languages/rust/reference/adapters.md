---
title: grafeo-adapters
description: Adapters crate.
tags:
  - api
  - rust
---

# grafeo-adapters

Query parsers and plugins. Storage lives in `grafeo-storage` (the WAL and the `.grafeo` container) and `grafeo-core` (the stores).

## GQL Parser

```rust
use grafeo_adapters::query::gql;

let ast = gql::parse("MATCH (n:Person) RETURN n")?;
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
