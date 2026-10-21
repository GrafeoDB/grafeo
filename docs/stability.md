---
title: API Stability
description: Which Grafeo APIs are stable, which are experimental, and what each level promises.
tags:
  - reference
---

# API Stability

Every public API of Grafeo is at one of two levels: **Stable** or **Experimental**. Stable is the default. An API
is experimental only when its documentation says so.

| Level | What may change | When | How you hear about it |
| --- | --- | --- | --- |
| **Stable** | Nothing that breaks code written against it: a signature, a behavior, a result shape | Only in a minor release (`0.6` to `0.7`), and a removal or rename only after a deprecation in an earlier release | A **Breaking** entry in the [CHANGELOG](changelog.md), the upgrade guide of that release, and `#[deprecated]` (or the binding's own deprecation warning) from the release that deprecates it |
| **Experimental** | Anything, including removal | In any release, patch releases included | An entry under **Changed** in the [CHANGELOG](changelog.md) |

A stable API still grows within a minor version: a patch release can add a setting, a field to a result type, or a
key to a dict or object a binding returns (see [Versioning](versioning.md#the-stable-surface)).

If you depend on an experimental API, pin the exact version (`grafeo = "=0.6.2"`, `grafeo==0.6.2`) and read the
CHANGELOG before you upgrade.

## Where the Levels Are Marked

- **Rust**: the documentation of the `grafeo` crate and of each module it re-exports says *Stable unless marked*.
  An experimental item has a `# Stability: Experimental` section in its documentation. A deprecated item carries
  `#[deprecated]`, which names its replacement, so the compiler warns where you use it.
- **Bindings** (Python, Node.js, WebAssembly, C, C#, Dart, Go): a binding call has the level of the Rust call it
  wraps. A call that only a binding has is stable once released. An experimental call says so in its docstring,
  JSDoc or header comment.
- **The command line tool and the database file format** are stable. The file format changes only in a minor
  release, which migrates a database on its first read-write open (see
  [Versioning](versioning.md#database-files)).

The implementation crates (`grafeo-common`, `grafeo-core`, `grafeo-storage`, `grafeo-adapters` and
`grafeo-engine`) have no levels: their APIs may change in any release. Depend on `grafeo` instead.

## Experimental APIs

| API | Why it is experimental |
| --- | --- |
| Streaming results: `Session::execute_streaming` and `GrafeoDB::execute_streaming` with `ResultStream`, `OwnedResultStream`, `RowIterator` and `OwnedRowIterator`; Python `execute_lazy()` and `ResultStream`; Node.js `executeStream()` and `ResultStream` | New in 0.5.40; the iterator and its lifetime rules may still change |
| The numbers `memory_usage()` reports: the fields of `MemoryUsage` and its parts, and what each one counts | Memory accounting follows the storage engine, which changes within 0.6. The types are `#[non_exhaustive]`, so a field can be added; what a field counts can change too. The call `memory_usage()` itself is stable |
| `Session::statement_writes`: whether a statement writes, without running it | New in 0.6.0; which statements count as writes may be refined |
| File import from Rust: `import::import_nodes` and `ImportFormat` (the Python `import_csv` and `import_jsonl`, the Node.js `importCsv` and `importJsonl` and the CLI `grafeo import csv` and `grafeo import jsonl` that run it are stable) | New in 0.6.0; formats and options may be added |
| The operator tree of a `PROFILE` statement: `QueryResult::profile` and `QueryProfile`, the Python and Node.js `profile` and the WASM `profile` key | New in 0.6.0; its operators and fields may still change |

## Stable APIs

Everything else on the public surface is stable, among others:

- `GrafeoDB` and `Session`: opening and creating databases, queries in every supported language, parameters,
  transactions, the direct read and write API, batches, upserts, graph handles, schemas and named graphs
- `Config` with its constructors and `with_*` methods, and the option types it takes
- `QueryResult`, `Value`, `NodeId` and `EdgeId`
- `Error` and its codes (`GRAFEO-...` codes and GQLSTATUS values)
- indexes and search (property, vector, text and hybrid), backup and restore, checkpoints, `compact()`
- the `auth`, `admin` and `cdc` modules: identities and roles, statistics and validation, change data capture
  events
- the store contract: the `GraphStore` and `GraphStoreMut` traits, `GrafeoDB::with_store`, `store()`,
  `graph_store()` and `GraphHandle::graph_store()`

## The Promise for 0.7 and 1.0

**0.7** is the next release that may break stable APIs. It removes what 0.6.x deprecated (each deprecation names its
replacement and says "removed in 0.7.0"), and it no longer opens 0.5.x databases: open each one for writing with
0.6 first, which migrates it. Its upgrade guide lists every breaking change with the code to change.

**1.0** keeps these two levels and moves the rules up one level: stable APIs then break only in a major release
(`1.x` to `2.0`), and a minor release (`1.1`) only adds and deprecates. Experimental APIs stay exempt until they are
promoted, which the CHANGELOG announces.
