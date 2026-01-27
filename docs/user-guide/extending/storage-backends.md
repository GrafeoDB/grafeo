---
title: Storage Backends
description: Implementing custom storage backends.
tags:
  - extending
  - storage
---

# Storage Backends

Implement custom storage backends for specialized use cases.

## Storage Backend Trait

```rust
pub trait StorageBackend: Send + Sync {
    fn name(&self) -> &str;

    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>>;
    fn put(&mut self, key: &[u8], value: &[u8]) -> Result<()>;
    fn delete(&mut self, key: &[u8]) -> Result<()>;

    fn scan(&self, start: &[u8], end: &[u8]) -> Result<Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)>>>;

    fn flush(&mut self) -> Result<()>;
    fn close(&mut self) -> Result<()>;
}
```

## Built-in Backends

| Backend | Description |
|---------|-------------|
| `MemoryBackend` | In-memory storage |
| `FileBackend` | File-based persistent storage |
| `WalBackend` | Write-ahead log for durability |

## Custom Backend Example

```rust
use graphos_adapters::storage::StorageBackend;

pub struct RedisBackend {
    client: redis::Client,
}

impl StorageBackend for RedisBackend {
    fn name(&self) -> &str {
        "redis"
    }

    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let mut conn = self.client.get_connection()?;
        let result: Option<Vec<u8>> = redis::cmd("GET")
            .arg(key)
            .query(&mut conn)?;
        Ok(result)
    }

    fn put(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        let mut conn = self.client.get_connection()?;
        redis::cmd("SET")
            .arg(key)
            .arg(value)
            .execute(&mut conn);
        Ok(())
    }

    fn delete(&mut self, key: &[u8]) -> Result<()> {
        let mut conn = self.client.get_connection()?;
        redis::cmd("DEL")
            .arg(key)
            .execute(&mut conn);
        Ok(())
    }

    // ... implement remaining methods
}
```

## Using Custom Backend

```rust
use graphos::{Database, Config};

let backend = Box::new(RedisBackend::new("redis://localhost")?);

let config = Config::builder()
    .storage_backend(backend)
    .build()?;

let db = Database::open_with_config("my_graph", config)?;
```

## Backend Requirements

1. **Thread Safety** - Must be `Send + Sync`
2. **Durability** - Implement `flush()` for persistence
3. **Atomicity** - Ensure atomic operations where needed
4. **Error Handling** - Use `Result` for all fallible operations
