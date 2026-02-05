---
title: Development Setup
description: Setting up your development environment.
tags:
  - contributing
---

# Development Setup

## Prerequisites

- Rust 1.80.0+
- Python 3.9+ (for Python bindings)
- Git

## Clone Repository

```bash
git clone https://github.com/GrafeoDB/grafeo.git
cd grafeo
```

## Build

```bash
# Build all crates
cargo build --workspace

# Build in release mode
cargo build --workspace --release
```

## Run Tests

```bash
cargo test --workspace
```

## Build Python Package

```bash
cd crates/bindings/python
uv add maturin
maturin develop
```

## IDE Setup

### VS Code

Recommended extensions:

- rust-analyzer
- Python
- TOML

### IntelliJ/CLion

- Install Rust plugin
- Open as Cargo project
