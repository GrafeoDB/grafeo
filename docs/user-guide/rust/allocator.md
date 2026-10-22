---
title: Choosing an Allocator
description: Run Grafeo on mimalloc, jemalloc or another global allocator from your own Rust application.
tags:
  - rust
  - performance
---

# Choosing an Allocator

Grafeo allocates through Rust's global allocator and does not pick one for you: by default that is the system
allocator, and Grafeo's tests and benchmarks run on it. A Rust application chooses its allocator once, in its own
binary, with `#[global_allocator]`. Every allocation Grafeo makes in that process then goes through it, with no
change to Grafeo or its features.

The bindings (Python, Node.js, WebAssembly, C and the languages built on it) ship a fixed build with the system
allocator.

## mimalloc

```toml
[dependencies]
grafeo = "0.6"
mimalloc = "0.1"
```

```rust
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
```

## jemalloc

jemalloc builds on Linux and macOS, not with the MSVC toolchain on Windows, so depend on it only where it builds.

```toml
[dependencies]
grafeo = "0.6"

[target.'cfg(not(target_env = "msvc"))'.dependencies]
tikv-jemallocator = "0.7"
```

```rust
#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
```

## rusty_alloc

[rusty_alloc](https://github.com/remade-with-rust/rusty_alloc) is a Rust port of mimalloc without C code. Its global
allocator is in the `rusty_alloc-api` crate. The crate is young and releases often: pin the exact version you have
tested, and read its changelog before you move to a newer one.

```toml
[dependencies]
grafeo = "0.6"
rusty_alloc-api = "=2.2.6"
```

```rust
#[global_allocator]
static GLOBAL: rusty_alloc_api::RustyAlloc = rusty_alloc_api::RustyAlloc;
```

## One Allocator Per Process

A process has one global allocator, and declaring a second one fails to link. The `grafeo` crate's `jemalloc` and
`mimalloc-allocator` features declare one for you: use either a feature or your own `#[global_allocator]`, not both.
The same holds for any other dependency that installs an allocator when you depend on it.

Measure with your own workload before you switch: an allocator that is faster on one workload can use more memory,
or be slower, on another.
