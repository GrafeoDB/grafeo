---
title: Code Style
description: Coding standards for Graphos.
tags:
  - contributing
---

# Code Style

## Rust Guidelines

### Formatting

Use `rustfmt` with default settings:

```bash
cargo fmt --all
```

### Linting

Use `clippy`:

```bash
cargo clippy --workspace -- -D warnings
```

### Naming

| Item | Convention |
|------|------------|
| Types | `PascalCase` |
| Functions | `snake_case` |
| Constants | `SCREAMING_SNAKE_CASE` |

### Documentation

All public items must have doc comments:

```rust
/// Creates a new database session.
///
/// # Errors
///
/// Returns an error if the database is shutting down.
pub fn session(&self) -> Result<Session, Error> {
    // ...
}
```

### Error Handling

- Use `Result` for fallible operations
- Use `thiserror` for error types
- Never panic in library code

## Python Guidelines

- Follow PEP 8
- Use type hints
- Document public APIs
