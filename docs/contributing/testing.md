---
title: Testing
description: Test strategy and running tests.
tags:
  - contributing
---

# Testing

## Running Tests

```bash
# All tests
cargo test --workspace

# Specific crate
cargo test -p graphos-core

# Single test
cargo test test_name -- --nocapture

# With output
cargo test -- --nocapture
```

## Coverage

```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Generate report
cargo tarpaulin --workspace --out Html
```

## Coverage Targets

| Crate | Target |
|-------|--------|
| graphos-common | 95% |
| graphos-core | 90% |
| graphos-adapters | 85% |
| graphos-engine | 85% |
| graphos-python | 80% |

## Test Categories

- **Unit tests** - Same file, `#[cfg(test)]` module
- **Integration tests** - `tests/` directory
- **Property tests** - Using `proptest` crate

## Writing Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let store = LpgStore::new();
        let id = store.create_node(&["Person"], Default::default());
        assert!(store.get_node(id).is_some());
    }
}
```
