# Changelog

All notable changes to Grafeo, for future reference (and enjoyment).

## [Unreleased]

## [0.2.0] - Unreleased

_Pre-workout_



### Added

- **Benchmarks**: Multi-hop traversal and fan-out pattern benchmarks for performance validation
- **Factorized Execution**: Factorized vector types to avoid Cartesian product materialization, inspired by [Kùzu](https://kuzudb.com/)'s approach to worst-case optimal joins
- **Factorized Expand**: Expand operator produces factorized output for multi-hop queries
- **Factorized Aggregation**: COUNT, SUM, AVG, MIN, MAX on multi-hop queries without flattening
- **ChunkState Abstraction**: Unified state tracking with cached multiplicities for O(1) access
- **Factorized Selection Vectors**: Lazy filtering without data copying (O(n_physical) instead of O(n_logical))
- **FactorizedFilterOperator**: Filter factorized data using selection vectors
- **Improved Iteration**: PrecomputedIter and StreamingIter with SmallVec for cache-friendly traversal
- **FactorizedOperator Trait**: Composable operators that produce/consume factorized data natively

### Changed

- Version bump to 0.2.0, foundation complete, focusing on performance for 0.2.x
- **Pre-commit Hooks**: Switched from Python-based pre-commit to [prek](https://github.com/j178/prek) (Rust-native, faster)

---

## [0.1.4] - 2026-01-31

_Foundation Complete_

### Added

- **REMOVE Clause**: GQL parser now supports `REMOVE n:Label` for label removal and `REMOVE n.property` for property removal
- **Label APIs**: Python methods for direct label manipulation - `add_node_label()`, `remove_node_label()`, `get_node_labels()`
- **WAL Support**: Label operations now logged to write-ahead log for durability
- **RDF Transaction Support**: SPARQL operations now support proper commit/rollback semantics with buffered writes

### Changed

- **Default Features**: All query languages (GQL, Cypher, Gremlin, GraphQL, SPARQL) now enabled by default - no feature flags needed
- **Better Out-of-Box Experience**: Users get full functionality without any configuration

### Fixed

- RDF store transaction rollback now properly discards uncommitted changes
- npm publishing paths corrected for @grafeo-db/js and @grafeo-db/wasm packages
- Go module path corrected to match directory structure

### Documentation

- README updated with new default feature status and label API examples

## [0.1.3] - 2026-01-30

_Admin Tools & Performance_

### Added

- **CLI** (`grafeo-cli`): Full command-line interface for database administration - inspect, backup, export, manage WAL, and compact databases with table or JSON output
- **Admin APIs**: Python bindings for introspection - `info()`, `detailed_stats()`, `schema()`, `validate()`, WAL management, and persistence utilities
- **Adaptive Execution**: Runtime re-optimization when cardinality estimates deviate 3x+ from actual values
- **Run-Length Encoding**: Compression for repetitive data with zigzag encoding and efficient iterators
- **Property Compression**: Type-specific codecs (Dictionary, Delta, RLE) with hot buffer pattern
- **Pre-commit Hooks**: Automated `cargo fmt` and `clippy` checks before commits

### Improved

- **Query Optimizer**: Projection pushdown and improved join reordering
- **Cardinality Estimation**: Histogram-based estimation with adaptive feedback loop
- **Parsers**: Better edge patterns (Cypher), more traversal steps (Gremlin), improved pattern matching (GQL)
- **Aggregate Operator**: Parallel hash aggregation with improved memory efficiency
- **Adjacency Index**: Bloom filters for faster edge membership tests
- **RDF Planner**: Major improvements to triple pattern handling and optimization

### Documentation

- CLI guide at `docs/getting-started/cli.md`
- README expanded with admin API examples and CLI usage

## [0.1.2] - 2026-01-29

_Testing & Documentation_

### Added

- **Python Test Suite**: Comprehensive tests covering LPG and RDF graph operations
- **Query Language Tests**: Coverage for all five languages; GQL, Cypher, Gremlin, GraphQL and SPARQL
- **Test Infrastructure**: Fixtures, base classes, and shared utilities for consistent testing
- **Plugin Tests**: NetworkX and solvOR integration tests across all query languages

### Changed

- **Database Implementation**: Core functionality now fully operational end-to-end
- **Query Pipeline**: Complete execution path from parsing through result materialization

### Documentation

- Docstring pass across all crates - added tables, examples, and practical guidance
- Python bindings documentation with NetworkX and solvOR library references

## [0.1.1] - Unreleased

_Query Languages & Python Bindings_

### Added

- **GQL Parser**: Full ISO/IEC 39075 standard query language support
- **Multi-Language Support**: Cypher, Gremlin, GraphQL and SPARQL translators
- **MVCC Transactions**: Snapshot isolation with multi-version concurrency control
- **Index Types**: Hash indexes for equality, B-tree for range queries, trie for prefix matching, adjacency lists for traversals
- **Storage Backends**: In-memory for speed, write-ahead log for durability
- **Python Bindings**: PyO3-based API exposing full database functionality

### Changed

- **Breaking**: Renamed project from Graphos to Grafeo

## [0.1.0] - Unreleased

_Foundation_

### Added

- **Core Architecture**: Modular crate structure designed for extensibility
- **Crate Layout**:
  - `grafeo-common`: Foundation types, memory allocators, hashing utilities
  - `grafeo-core`: LPG storage engine, index structures, execution operators
  - `grafeo-adapters`: Query parsers, storage backends, plugin system
  - `grafeo-engine`: Database facade, session management, transaction coordination
  - `grafeo-python`: Python bindings via PyO3
- **Graph Models**: Labeled Property Graph (LPG) and RDF triple store support
- **In-Memory Storage**: Fast graph operations without persistence overhead

---

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
