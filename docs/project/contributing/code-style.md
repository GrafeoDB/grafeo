---
title: Code Style
description: Coding standards for Grafeo.
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

The workspace clippy configuration lives in the root `Cargo.toml` under `[workspace.lints.clippy]`.
Key settings:

- `clippy::all` and `clippy::pedantic` are enabled as warnings
- `wildcard_imports` is set to `warn` (`use super::*` in `#[cfg(test)]` modules is an accepted exception)
- Several pedantic lints are allowed for database-specific patterns (casting, similar names, etc.)

Run clippy the same way CI does:

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

The workspace also sets `unsafe_code = "warn"` in `[workspace.lints.rust]`, so unsafe code
compiles but produces warnings visible in CI.

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
