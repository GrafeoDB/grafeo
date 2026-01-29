# Contributing to Grafeo

Thank you for your interest in contributing to Grafeo! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites

- Rust 1.91.0+
- Python 3.12+ (for Python bindings)
- Git

### Setup

```bash
git clone https://github.com/StevenBtw/grafeo.git
cd grafeo
cargo build --workspace
```

## Architecture

For detailed architecture documentation, see [.claude/ARCHITECTURE.md](.claude/ARCHITECTURE.md).

### Crate Overview

| Crate              | Purpose                                        |
| ------------------ | ---------------------------------------------- |
| `grafeo`          | Top-level facade, re-exports public API        |
| `grafeo-common`   | Foundation types, memory allocators, utilities |
| `grafeo-core`     | LPG storage, indexes, execution engine         |
| `grafeo-adapters` | GQL parser, storage backends, plugins          |
| `grafeo-engine`   | Database facade, sessions, transactions        |
| `grafeo-python`   | Python bindings via PyO3                       |

### Query Language Architecture

Grafeo supports multiple query languages through a translator pattern:

```
Query String → Parser → AST → Translator → LogicalPlan → Optimizer → Executor
```

| Component | LPG Path | RDF Path |
|-----------|----------|----------|
| **Parser** | `grafeo-adapters/query/gql/` | `grafeo-adapters/query/sparql/` |
| | `grafeo-adapters/query/cypher/` | |
| | `grafeo-adapters/query/gremlin/` | |
| | `grafeo-adapters/query/graphql/` | `grafeo-adapters/query/graphql/` |
| **Translator** | `grafeo-engine/query/gql_translator.rs` | `grafeo-engine/query/sparql_translator.rs` |
| | `grafeo-engine/query/cypher_translator.rs` | `grafeo-engine/query/graphql_rdf_translator.rs` |
| | `grafeo-engine/query/gremlin_translator.rs` | |
| | `grafeo-engine/query/graphql_translator.rs` | |
| **Storage** | `grafeo-core/graph/lpg/` | `grafeo-core/graph/rdf/` |
| **Operators** | NodeScan, Expand, CreateNode | TripleScan, LeftJoin, AntiJoin |

### Data Model Compatibility

| Query Language | LPG | RDF | Notes |
|----------------|-----|-----|-------|
| GQL | ✅ | — | Primary language, ISO standard |
| Cypher | ✅ | — | Feature-gated, openCypher compatible |
| Gremlin | ✅ | — | Feature-gated, Apache TinkerPop traversal language |
| GraphQL | ✅ | ✅ | Feature-gated, schema-driven, maps to both models |
| SPARQL | — | ✅ | Feature-gated, W3C standard for RDF queries |

### Implementation Plan

See [.claude/IMPLEMENTATION_PLAN.md](.claude/IMPLEMENTATION_PLAN.md) for the detailed implementation and test plan.

## Coding Standards

### Rust Style

- Follow standard Rust conventions (rustfmt, clippy)
- Use `#[must_use]` for pure functions that return values
- Use `#[inline]` for small, frequently-called functions
- Prefer `parking_lot` locks over `std::sync` (faster, no poisoning)
- Use `FxHashMap`/`FxHashSet` from `grafeo_common::utils::hash` for internal hash tables

### Documentation

- All public items should have doc comments
- Include examples in doc comments for complex APIs

### Error Handling

- Use `grafeo_common::utils::error::Result` for fallible operations
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
cargo test -p grafeo-core

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
cargo bench -p grafeo-common arena
```

## Building Python Bindings

```bash
cd crates/grafeo-python

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

- **Repository**: <https://github.com/StevenBtw/grafeo>
- **Issues**: <https://github.com/StevenBtw/grafeo/issues>
- **Documentation**: <https://grafeo.dev>

## Code of Conduct

Be respectful and constructive. We're all here to build something great together.

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 license.
