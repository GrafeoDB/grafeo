---
title: Development Setup
description: Setting up the development environment.
tags:
  - contributing
---

# Development Setup

## Prerequisites

- Rust 1.99.0+
- Python 3.12+ (for Python bindings, CI tests 3.12, 3.13, 3.14 and 3.15)
- Node.js 22+ (for Node.js bindings, CI tests 22 and 24)
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

## Build Node.js Package

```bash
cd crates/bindings/node
npm install
npm run build
npm test
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
