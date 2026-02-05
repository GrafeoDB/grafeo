# Contributing to Grafeo

Thank you for your interest in contributing to Grafeo! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites

- Rust 1.91.1+
- Python 3.12+ (for Python bindings)
- Git

### Setup

```bash
git clone https://github.com/GrafeoDB/grafeo.git
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
| `grafeo-python`   | Python bindings via PyO3 (`crates/bindings/python`) |
| `grafeo-cli`      | Command-line interface for admin operations    |

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
| Cypher | ✅ | — | openCypher compatible |
| Gremlin | ✅ | — | Apache TinkerPop traversal language |
| GraphQL | ✅ | ✅ | Schema-driven, maps to both models |
| SPARQL | — | ✅ | W3C standard for RDF queries |

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

# Run with all features (matches CI)
cargo test --all-features --workspace

# Run release mode tests (also run in CI)
cargo test --all-features --workspace --release

# Run tests for a specific crate
cargo test -p grafeo-core

# Run with output visible
cargo test -- --nocapture

# Run a specific test
cargo test test_name -- --nocapture
```

## Code Coverage

We use [cargo-llvm-cov](https://github.com/taiki-e/cargo-llvm-cov) for code coverage.

### Installation

```bash
cargo install cargo-llvm-cov
```

### Running Coverage

```bash
# Full workspace coverage with summary
cargo llvm-cov --all-features --workspace

# Coverage for a specific crate
cargo llvm-cov -p grafeo-core --all-features

# Generate HTML report
cargo llvm-cov --all-features --workspace --html
# Open target/llvm-cov/html/index.html

# Generate LCOV report (for CI/Codecov)
cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info

# Show only uncovered lines
cargo llvm-cov --all-features --workspace --show-missing-lines
```

### Coverage Targets

| Crate           | Target |
| --------------- | ------ |
| grafeo-common   | 85%    |
| grafeo-core     | 80%    |
| grafeo-adapters | 85%    |
| grafeo-engine   | 80%    |
| Overall         | 80%    |

### Python Tests

```bash
# Install test dependencies
uv pip install numpy scipy networkx solvor

# Run Python tests
pytest tests/python/ -v --ignore=tests/python/benchmark_grafeo.py
```

## CI Checks

### Local CI Script (Recommended)

Run all CI checks locally before making a PR:

```bash
# Linux/macOS
./scripts/ci-local.sh

# Windows PowerShell
.\scripts\ci-local.ps1

# Quick mode (skip release tests)
./scripts/ci-local.sh --quick
.\scripts\ci-local.ps1 -Quick
```

This runs: format check, clippy, docs, Rust tests, and Python tests.

### Clean CI Test

For catching dependency issues, use the clean CI script which creates an isolated environment:

```bash
# Linux/macOS - test all Python versions (3.12, 3.13, 3.14)
./scripts/ci-clean.sh

# Windows PowerShell
.\scripts\ci-clean.ps1

# Skip Rust checks (Python only)
./scripts/ci-clean.sh --skip-rust
.\scripts\ci-clean.ps1 -SkipRust

# Test specific Python version only
./scripts/ci-clean.sh --python 3.12
.\scripts\ci-clean.ps1 -Python 3.12
```

### Automated Pre-commit Hooks

Install prek hooks to automatically check formatting and linting:

```bash
# Install prek (Rust-native pre-commit alternative)
cargo install prek

# Install git hooks
prek install

# Run manually on all files
prek run --all-files

# Run coverage check (manual hook, not run on every commit)
prek run cargo-llvm-cov
```

This runs `cargo fmt`, `cargo clippy`, `cargo deny`, and file checks automatically before each commit. The coverage check is a manual hook since it's slow.

## Running Benchmarks

```bash
# Run all benchmarks
cargo bench --workspace

# Run specific benchmark
cargo bench -p grafeo-common arena
```

## Building Python Bindings

```bash
cd crates/bindings/python

# Development build
maturin develop

# Release build
maturin build --release
```

## Pull Request Process

1. Fork the repository and create a feature branch
2. Write tests for new functionality
3. Run local CI checks: `./scripts/ci-local.sh` (or `.\scripts\ci-local.ps1` on Windows)
4. Update documentation if needed
5. Submit PR with clear description of changes

Or run individual checks manually:

```bash
cargo fmt --all              # Format code
cargo clippy --workspace -- -D warnings  # Lint
cargo test --all-features --workspace    # Test
```

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

- **Repository**: <https://github.com/GrafeoDB/grafeo>
- **Issues**: <https://github.com/GrafeoDB/grafeo/issues>
- **Documentation**: <https://grafeo.dev>

## Code of Conduct

Be respectful and constructive. We're all here to build something great together.

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 license.
