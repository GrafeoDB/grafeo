# Contributing to Graphos

Thank you for your interest in contributing to Graphos! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites

- Rust 1.91.1 or later
- Python 3.9+ (for Python bindings)
- Git

### Setup

```bash
git clone https://github.com/StevenBtw/graphos.git
cd graphos
cargo build --workspace
```

## Architecture

For detailed architecture documentation, see [.claude/ARCHITECTURE.md](.claude/ARCHITECTURE.md).

### Crate Overview

| Crate              | Purpose                                        |
| ------------------ | ---------------------------------------------- |
| `graphos-common`   | Foundation types, memory allocators, utilities |
| `graphos-core`     | LPG storage, indexes, execution engine         |
| `graphos-adapters` | GQL parser, storage backends, plugins          |
| `graphos-engine`   | Database facade, sessions, transactions        |
| `graphos-python`   | Python bindings via PyO3                       |

### Query Language Architecture

Graphos supports multiple query languages through a translator pattern:

```
Query String → Parser → AST → Translator → LogicalPlan → Optimizer → Executor
```

| Component | LPG Path | RDF Path |
|-----------|----------|----------|
| **Parser** | `graphos-adapters/query/gql/` | `graphos-adapters/query/sparql/` |
| | `graphos-adapters/query/cypher/` | |
| **Translator** | `graphos-engine/query/gql_translator.rs` | `graphos-engine/query/sparql_translator.rs` |
| | `graphos-engine/query/cypher_translator.rs` | |
| **Storage** | `graphos-core/graph/lpg/` | `graphos-core/graph/rdf/` |
| **Operators** | NodeScan, Expand, CreateNode | TripleScan, LeftJoin, AntiJoin |

### Data Model Compatibility

| Query Language | LPG | RDF | Notes |
|----------------|-----|-----|-------|
| GQL | ✅ | ❌ | Primary language, ISO standard |
| Cypher | ✅ | ❌ | Feature-gated, openCypher compatible |
| SPARQL | ❌ | 🚧 | Parser ready, RDF store exists, integration planned |
| Gremlin | 🚧 | ❌ | Planned - imperative traversal language |
| GraphQL | 🚧 | 🚧 | Planned - schema-driven, maps to both models |

### Implementation Plan

See [.claude/IMPLEMENTATION_PLAN.md](.claude/IMPLEMENTATION_PLAN.md) for the detailed implementation and test plan.

## Coding Standards

### Rust Style

- Follow standard Rust conventions (rustfmt, clippy)
- Use `#[must_use]` for pure functions that return values
- Use `#[inline]` for small, frequently-called functions
- Prefer `parking_lot` locks over `std::sync` (faster, no poisoning)
- Use `FxHashMap`/`FxHashSet` from `graphos_common::utils::hash` for internal hash tables

### Documentation

- All public items must have doc comments
- Use `#![warn(missing_docs)]` in lib.rs files
- Include examples in doc comments for complex APIs

### Error Handling

- Use `graphos_common::utils::error::Result` for fallible operations
- Provide meaningful error messages with context
- Use `thiserror` for error types

### Testing

- Write tests in the same file using `#[cfg(test)]` module
- Use descriptive test names: `test_<function>_<scenario>`
- Aim for 85% overall coverage (see implementation plan for per-crate targets)

## Running Tests

```bash
# Run all tests
cargo test --workspace

# Run tests for a specific crate
cargo test -p graphos-core

# Run with output visible
cargo test -- --nocapture

# Run a specific test
cargo test test_name -- --nocapture

# Run tests with coverage (requires cargo-tarpaulin)
cargo tarpaulin --workspace --out Html
```

## Running Benchmarks

```bash
# Run all benchmarks
cargo bench --workspace

# Run specific benchmark
cargo bench -p graphos-common arena
```

## Building Python Bindings

```bash
cd crates/graphos-python

# Development build
maturin develop

# Release build
maturin build --release
```

## Pull Request Process

1. Fork the repository and create a feature branch
2. Write tests for new functionality
3. Ensure all tests pass: `cargo test --workspace`
4. Run clippy: `cargo clippy --workspace -- -D warnings`
5. Format code: `cargo fmt --all`
6. Update documentation if needed
7. Submit PR with clear description of changes

### Commit Messages

Use conventional commit format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `ci:` CI/CD changes

### PR Checklist

- [ ] Tests pass locally
- [ ] New code has tests
- [ ] Documentation updated
- [ ] No new clippy warnings
- [ ] Code formatted with rustfmt

## Project Links

- **Repository**: <https://github.com/StevenBtw/graphos>
- **Issues**: <https://github.com/StevenBtw/graphos/issues>
- **Documentation**: <https://graphos.tech>

## Code of Conduct

Be respectful and constructive. We're all here to build something great together.

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 license.
