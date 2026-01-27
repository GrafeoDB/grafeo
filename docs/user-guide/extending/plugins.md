---
title: Plugins
description: Creating and using plugins in Graphos.
tags:
  - extending
  - plugins
---

# Plugins

Plugins extend Graphos with custom functionality.

## Plugin Architecture

Graphos uses a trait-based plugin system:

```rust
pub trait Plugin: Send + Sync {
    fn name(&self) -> &str;
    fn version(&self) -> &str;
    fn on_load(&mut self, context: &PluginContext) -> Result<()>;
    fn on_unload(&mut self) -> Result<()>;
}
```

## Creating a Plugin (Rust)

```rust
use graphos_adapters::plugins::{Plugin, PluginContext};

pub struct MyPlugin {
    name: String,
}

impl Plugin for MyPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn on_load(&mut self, context: &PluginContext) -> Result<()> {
        // Register custom functions
        context.register_function("my_function", my_function_impl)?;
        Ok(())
    }

    fn on_unload(&mut self) -> Result<()> {
        Ok(())
    }
}

fn my_function_impl(args: &[Value]) -> Result<Value> {
    // Implementation
    Ok(Value::String("Hello from plugin!".into()))
}
```

## Loading Plugins

```rust
use graphos::{Database, Config};

let config = Config::builder()
    .plugin_dir("./plugins")
    .build()?;

let db = Database::open_with_config("my_graph.db", config)?;
```

## Built-in Plugin Types

| Type | Purpose |
|------|---------|
| Function Plugin | Add custom GQL functions |
| Algorithm Plugin | Add graph algorithms |
| Storage Plugin | Add storage backends |
| Import/Export Plugin | Add data format support |

## Plugin Lifecycle

1. **Discovery** - Plugins found in plugin directory
2. **Loading** - Plugin loaded and initialized
3. **Registration** - Plugin registers its functionality
4. **Active** - Plugin functionality available
5. **Unloading** - Plugin cleanup on database close

## Example: Algorithm Plugin

```rust
pub struct PageRankPlugin;

impl Plugin for PageRankPlugin {
    fn name(&self) -> &str { "pagerank" }
    fn version(&self) -> &str { "1.0.0" }

    fn on_load(&mut self, ctx: &PluginContext) -> Result<()> {
        ctx.register_algorithm("pagerank", |graph, params| {
            let damping = params.get("damping").unwrap_or(0.85);
            let iterations = params.get("iterations").unwrap_or(20);
            compute_pagerank(graph, damping, iterations)
        })?;
        Ok(())
    }

    fn on_unload(&mut self) -> Result<()> { Ok(()) }
}
```

## Using Plugin Functions

```sql
-- Use a custom function from a plugin
MATCH (p:Person)
RETURN p.name, my_custom_function(p.data) AS result
```
