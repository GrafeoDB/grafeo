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
cargo test -p grafeo-core

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
| grafeo-common | 95% |
| grafeo-core | 90% |
| grafeo-adapters | 85% |
| grafeo-engine | 85% |
| grafeo-python (`crates/bindings/python`) | 80% |

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
