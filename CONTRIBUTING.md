# Contributing to Grafeo

Thanks for wanting to help out! Here's what you need to know.

## Setup

```bash
git clone https://github.com/GrafeoDB/grafeo.git
cd grafeo
cargo build --workspace
```

You'll need **Rust 1.91.1+** and optionally **Python 3.12+** / **Node.js 20+** for the bindings.

## Branching

We use feature branches off `main`:

- `feature/<description>` for new functionality
- `fix/<description>` for bug fixes
- `release/<version>` for release stabilization

Create your branch from `main`, open a PR back to `main` when ready.

## Making Changes

1. Create a branch: `git checkout -b feature/my-thing`
2. Write code and tests
3. Run checks: `./scripts/ci-local.sh` (or `.\scripts\ci-local.ps1` on Windows)
4. Push and open a PR

You can also run checks individually:

```bash
cargo fmt --all              # Format
cargo clippy --workspace --all-features -- -D warnings  # Lint
cargo test --all-features --workspace     # Test
```

### Commit Messages

We use conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`, `ci:`.

## Architecture

See [ARCHITECTURE.md](.claude/ARCHITECTURE.md) for the full picture. The short version:

| Crate | What it does |
| ----- | ------------ |
| `grafeo-common` | Foundation types, memory, utilities |
| `grafeo-core` | Graph storage, indexes, execution |
| `grafeo-adapters` | Query parsers (GQL, Cypher, SPARQL, etc.) |
| `grafeo-engine` | Database facade, sessions, transactions |
| `grafeo-python` | Python bindings (PyO3) |
| `grafeo-node` | Node.js bindings (napi-rs) |
| `grafeo-cli` | CLI tool |

## Code Style

- Standard Rust conventions — `rustfmt` and `clippy` are enforced in CI
- Use `thiserror` for error types
- Tests go in the same file under `#[cfg(test)]`
- Descriptive test names: `test_<function>_<scenario>`

See [CODE_STYLE.md](.claude/CODE_STYLE.md) for the full guide.

## Python Bindings

```bash
cd crates/bindings/python
maturin develop
pytest tests/python/ -v --ignore=tests/python/benchmark_grafeo.py
```

## Pre-commit Hooks (Optional)

```bash
cargo install prek
prek install
```

This runs format, lint, and license checks automatically before each commit.

## Links

- [Repository](https://github.com/GrafeoDB/grafeo)
- [Issues](https://github.com/GrafeoDB/grafeo/issues)
- [Documentation](https://grafeo.dev)

## License

By contributing, you agree that your contributions will be licensed under Apache-2.0.
