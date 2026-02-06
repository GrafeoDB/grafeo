# Changelog

All notable changes to Grafeo, for future reference (and enjoyment).

## [0.4.0] - 2026-02-06

_Node.js/TypeScript Bindings_

### Added

- **Node.js/TypeScript bindings** (`@grafeo-db/js`): Native Rust bindings via napi-rs
  - `GrafeoDB.create()` / `GrafeoDB.open(path)` for in-memory and persistent databases
  - Async `execute()` with GQL, Cypher, Gremlin, GraphQL, and SPARQL support
  - Full CRUD: `createNode`, `createEdge`, `getNode`, `getEdge`, `deleteNode`, `deleteEdge`
  - Property management: `setNodeProperty`, `setEdgeProperty`
  - `QueryResult` with `toArray()`, `scalar()`, `rows()`, `nodes()`, `edges()`, column access
  - `Transaction` with `commit()`, `rollback()`, `isActive`, and auto-rollback on drop
  - Type mapping: JS primitives, BigInt, Date, Buffer, Float32Array all map to Grafeo types
  - Auto-generated TypeScript definitions with full JSDoc documentation
  - Parameterized queries via JSON objects
  - Feature-gated query languages matching Rust crate features

### Changed

- Bumped all crate versions from 0.3.4 to 0.4.0

## [0.3.4] - 2026-02-06

_Quality of Life Improvements_

### Added

- **Query Performance Metrics**: Query results now include execution timing and row counts
  - `QueryResult.execution_time_ms` - execution time in milliseconds
  - `QueryResult.rows_scanned` - number of rows scanned during execution
  - Python: `result.execution_time_ms` and `result.rows_scanned` properties

- **Error Message Suggestions**: Fuzzy matching for helpful "Did you mean X?" hints
  - Levenshtein distance-based matching for undefined variables and labels
  - Case-insensitive comparison with configurable edit distance thresholds
  - `find_similar()`, `format_suggestion()`, `format_suggestions()` utilities

- **Python Pagination**: `get_nodes_by_label()` now supports `offset` parameter
  - Enables efficient pagination: `db.get_nodes_by_label("Person", limit=10, offset=20)`

### Documentation

- **Troubleshooting Guide**: Common errors, debugging tips, and solutions
- **Glossary**: Terminology reference for graph database concepts
- **Migration Guide**: Moving from Neo4j, NetworkX, and other databases
- **Security Guide**: Authentication patterns and secure deployment
- **Performance Baselines**: Benchmark results and optimization guidance
- **Example Notebooks**: Interactive anywidget visualizations for graphs and vectors

## [0.3.3] - Unreleased

_Hybrid Query Support & Vector Optimization_

### Added

- **VectorJoin Operator**: Combines graph patterns with vector similarity search
  - `VectorJoinOp` logical operator in query plans
  - `VectorJoinOperator` physical operator with two modes:
    - `with_static_query()` for constant query vectors
    - `entity_to_entity()` for comparing embeddings between nodes
  - Supports HNSW index and brute-force search
  - Configurable k, similarity thresholds, and label filtering

- **Vector Zone Maps**: Block-level pruning for vector search
  - `VectorZoneMap` tracks magnitude bounds, centroid, and bounding box
  - Centroid-based pruning with max_radius for block skipping
  - Per-dimension min/max for hyperrectangle pruning
  - `might_contain_within_distance()` for Euclidean and Cosine metrics

- **Vector Cost & Cardinality Estimation**:
  - `vector_scan_cost()` in optimizer with HNSW O(ef * log N) and brute-force O(N) models
  - `vector_join_cost()` for hybrid query planning
  - `estimate_vector_scan()` and `estimate_vector_join()` cardinality estimators
  - Accounts for k parameter and similarity threshold selectivity

- **Product Quantization (PQ)**: High-compression vector quantization
  - `ProductQuantizer`: Splits vectors into M subvectors, quantizes each to K centroids
  - K-means training with configurable iterations
  - Asymmetric distance computation (ADC) using precomputed tables
  - 8-32x compression with ~90% recall retention
  - `QuantizationType::Product { num_subvectors }` variant
  - Integrated with `QuantizedHnswIndex` for memory-efficient search

- **Memory-Mapped Vector Storage**: Disk-backed storage for large datasets
  - `VectorStorage` trait for storage backend abstraction
  - `RamStorage`: In-memory HashMap storage (fastest access)
  - `MmapStorage`: Memory-mapped file storage (low memory footprint)
  - LRU cache for frequently accessed vectors
  - Automatic persistence with custom file format

- **Python Quantization API**: Vector quantization accessible from Python
  - `grafeo.QuantizationType`: None, Scalar, Binary, Product variants
  - `grafeo.ScalarQuantizer`: Train, quantize, dequantize, distance methods
  - `grafeo.ProductQuantizer`: Train, quantize, reconstruct, asymmetric distance
  - `grafeo.BinaryQuantizer`: Static quantize and hamming distance methods

## [0.3.2] - Unreleased

_Batch Read Optimizations & Code Quality_

### Added

- **Selective Property Loading** (Projection Pushdown):
  - `PropertyStorage::get_selective_batch()` - O(N×K) vs O(N×C) for subset of properties
  - `LpgStore::get_nodes_properties_selective_batch()` wrapper method
  - `LpgStore::get_edges_properties_selective_batch()` wrapper method
  - Significant speedup when queries only need a few properties from many-column nodes

- **Parallel Node Scan Source**:
  - `ParallelNodeScanSource` for morsel-driven parallel execution
  - Implements `ParallelSource` trait with partitioning support
  - `with_label()` for filtered scans, `from_node_ids()` for pre-computed lists
  - Enables 3-8x speedup on large scans (10K+ nodes) by saturating CPU cores

### Changed

- **MVCC Hot Path Optimizations**:
  - Added `#[inline]` to `VersionChain::visible_at`, `visible_to`
  - Added `#[inline]` to `ColdVersionRef::is_visible_at`, `is_visible_to`
  - Added `#[inline]` to `VersionIndex::visible_at`, `visible_to`
  - Reduces function call overhead during full table scans

- **Delta Encoding Safety**:
  - Added `debug_assert!` to verify input is sorted in `DeltaEncoding::encode()`
  - Improved documentation clarifying sorted input requirement
  - Prevents silent data corruption from unsorted input in debug builds

- **Batch Property Reading**:
  - Pre-allocated result vectors in `PropertyStorage::get_all_batch()`
  - HashMap capacity hints based on column count (NebulaGraph MultiGet pattern)
  - Reduces allocation overhead for bulk property retrieval

## [0.3.1] - Unreleased

_Vector Optimization & Cache Improvements_

### Added

- **Vector Quantization**: Memory-efficient vector storage for large-scale similarity search
  - `ScalarQuantizer`: f32 → u8 compression
    - `train()` learns min/max per dimension from sample vectors
    - Asymmetric distance computation
  - `BinaryQuantizer`: f32 → 1-bit compression
    - Sign-bit extraction with packed u64 storage
    - SIMD-accelerated hamming distance (popcnt instruction)
  - `QuantizationType` enum for configuration (None, Scalar, Binary)

- **QuantizedHnswIndex**: HNSW with quantization and rescoring
  - Two-phase search: approximate quantized → exact rescore
  - Configurable rescore factor (default 2x candidates)
  - Automatic quantizer training from insertion samples

- **SIMD Vector Acceleration**: 4-8x faster distance computations
  - AVX2 + FMA for modern x86_64 CPUs
  - SSE fallback for older x86_64
  - NEON support for ARM (aarch64)
  - `simd_support()` function for runtime CPU detection
  - Python `grafeo.simd_support()` diagnostic function

- **Vector Batch Operations**:
  - `batch_insert()` for HNSW index bulk loading
  - `batch_search()` / `batch_search_with_ef()` for parallel multi-query search (rayon)
  - `batch_search_slices()` for slice-based query batches

- **VectorScan Operators**: Query execution for vector similarity search
  - `VectorScanOp` logical operator in query plans
  - `VectorScanOperator` physical operator (HNSW or brute-force)
  - Distance/similarity threshold filtering
  - Label-filtered brute-force search fallback

- **Adaptive WAL Flusher**: `AdaptiveFlusher` with self-tuning timing based on actual flush duration
  - Background thread adjusts wait time: `timeout = target_interval - last_flush_duration`
  - Maintains consistent flush cadence regardless of disk speed
  - `FlusherStats` for observability (flush count, avg/max times, exceeded target count)
  - Graceful shutdown with final flush guarantee

- **DurabilityMode::Adaptive**: New WAL durability mode for variable disk latency workloads
  - Unlike `Batch` which checks thresholds inline, `Adaptive` uses dedicated flusher thread
  - Prevents thundering herd problems when disk is slow

- **FingerprintedHashIndex**: Sharded hash index with fingerprint-based fast rejection
  - 64-shard concurrent access (similar to DashMap)
  - 48-bit fingerprints for ~99.99% rejection rate without full key comparison
  - `AtomicFingerprintStats` for thread-safe observability (lookups, rejections, comparisons)
  - Useful for expensive key comparisons (strings) or future disk-backed indices

## [0.3.0] - Unreleased

_AI Compatibility Release_

Major release introducing **Vector as a first-class type** for AI/ML workloads. Graph + vector hybrid queries are the unique differentiator - no pure vector database can efficiently combine graph traversal with vector similarity.

### Added

- **Vector Type Foundation**:
  - `Value::Vector(Arc<[f32]>)` - First-class vector type with 4x compression vs f64
  - `LogicalType::Vector(dim)` - Dimension-aware vector type for schema validation
  - `as_vector()`, `is_vector()`, `vector_dimensions()` accessor methods
  - `HashableValue` support for vectors (bit-level equality and hashing)
  - Serialization support via serde

- **Vector Distance Functions** (`grafeo-core/index/vector`):
  - `cosine_distance()` / `cosine_similarity()` - For normalized embeddings
  - `euclidean_distance()` / `euclidean_distance_squared()` - L2 distance
  - `dot_product()` - Maximum inner product search
  - `manhattan_distance()` - L1 distance, outlier-resistant
  - `DistanceMetric` enum with `FromStr` for runtime configuration
  - `normalize()` and `l2_norm()` utilities

- **Brute-Force k-NN Search**:
  - `brute_force_knn()` - O(n) exact nearest neighbor search
  - `brute_force_knn_filtered()` - k-NN with predicate filtering
  - `batch_distances()` - Efficient batch distance computation
  - `VectorConfig` for dimensions and metric configuration

- **HNSW Index** (feature: `vector-index`):
  - `HnswIndex` - O(log n) approximate nearest neighbor search
  - `HnswConfig` with tunable parameters (M, ef_construction, ef)
  - Presets: `high_recall()`, `fast()` for common use cases
  - Thread-safe concurrent reads with exclusive writes
  - `insert()`, `search()`, `search_with_ef()`, `remove()` operations
  - Seeded RNG option for reproducible benchmarks

- **GQL Vector Syntax**:
  - `VECTOR` keyword and `vector([...])` literal function
  - `cosine_similarity()`, `euclidean_distance()`, `dot_product()`, `manhattan_distance()` functions
  - `CREATE VECTOR INDEX name ON :Label(property) WITH (dimensions: N, metric: 'cosine')` statement
  - `CreateVectorIndexStatement` AST node

- **SPARQL Vector Functions**:
  - `VECTOR(...)`, `COSINE_SIMILARITY(vec1, vec2)`, `EUCLIDEAN_DISTANCE(vec1, vec2)`
  - `DOT_PRODUCT(vec1, vec2)`, `MANHATTAN_DISTANCE(vec1, vec2)`
  - Enables hybrid queries: `SELECT ?doc WHERE { ?doc :embedding ?vec FILTER(COSINE_SIMILARITY(?vec, ?query) > 0.8) }`

- **Serializable Snapshot Isolation (SSI)**:
  - `IsolationLevel` enum: `ReadCommitted`, `SnapshotIsolation` (default), `Serializable`
  - `begin_with_isolation()` for explicit isolation level selection
  - SSI validation detects read-write conflicts to prevent write skew anomaly
  - `TransactionError::SerializationFailure` for SSI violations

### Fixed

- **RDF Pending Delete Filtering**: `find_with_pending()` now correctly excludes pending deletes from query results within a transaction

---

## [0.2.7] - 2026-02-05

_Parallel Execution & Cache Patterns_

### Added

- **Second-Chance LRU Cache**: `SecondChanceLru<K, V>` with lock-free access marking via atomic bools
- **Hash Fingerprinting**: `FingerprintBucket` for fast miss detection using upper hash bits
- **Parallel Fold-Reduce Utilities**: `parallel_count`, `parallel_sum`, `parallel_stats`, `parallel_partition`
- **Generic Collector Trait**: `Collector` and `PartitionCollector` for composable parallel aggregation
  - Built-in collectors: `CountCollector`, `MaterializeCollector`, `LimitCollector`, `StatsCollector`

### Improved

- **Code Cleanup**: Removed unused CLI output functions, fixed incorrect dead_code allows

---

## [0.2.6] - 2026-02-04

_Filter Performance & Batch Read Optimizations_

### Added

- **Local Clustering Coefficient Algorithm**: Triangle counting and clustering coefficients with parallel rayon execution
- **Chunk-Level Zone Map Filtering**: `ChunkZoneHints` struct for attaching zone map metadata to `DataChunk`
  - Enables filter operators to skip entire chunks when predicates can't match
  - `might_match_chunk()` method on `Predicate` trait for chunk-level pruning
- **ComparisonPredicate Zone Map Support**: Implements `might_match_chunk()` for equality and range operators
  - Uses `might_contain_equal()`, `might_contain_less_than()`, `might_contain_greater_than()`

### Improved

- **Batch Property Retrieval**: `PropertyStorage::get_batch()` and `get_all_batch()` now acquire lock once
  - Previously acquired lock per-entity, now single lock for entire batch
  - Reduces lock contention for bulk property reads
- **FilterOperator**: Changed recursive `next()` call to loop (prevents stack overflow on many empty chunks)
  - Added zone map check before row-by-row predicate evaluation

### Documentation

- **Added CONTRIBUTORS.md**: Feel free to join me and contribute to Grafeo's future!
- Updated docs with references to other GrafeoDB projects

---

## [0.2.5] - 2026-02-03

_SPARQL Completeness & more performance Optimizations_

### Added

- **SPARQL String Functions**: CONCAT, REPLACE, STRLEN, UCASE/UPPER, LCASE/LOWER, SUBSTR, STRSTARTS, STRENDS, CONTAINS, STRBEFORE, STRAFTER, ENCODE_FOR_URI
- **SPARQL Type Functions**: COALESCE, IF, BOUND, STR, ISIRI/ISURI, ISBLANK, ISLITERAL, ISNUMERIC
- **SPARQL Math Functions**: ABS, CEIL, FLOOR, ROUND
- **SPARQL REGEX**: Pattern matching using the `regex` crate
- **EXISTS/NOT EXISTS**: Proper subquery support with semi-joins (EXISTS) and anti-joins (NOT EXISTS)
- **Platform Allocators**: Optional jemalloc (Linux/macOS) and mimalloc (Windows) for 10-20% faster allocations
  - Enable via `jemalloc` or `mimalloc-allocator` feature flags
- **Collection Type Aliases**: `GrafeoMap`, `GrafeoSet`, `GrafeoConcurrentMap` for consistent FxHash usage
- **Batch Property APIs**: `get_node_property_batch()`, `get_nodes_properties_batch()` for efficient bulk reads
- **Compound Predicate Optimization**: Filter pushdown now handles `n.a = 1 AND n.b = 2` compound predicates
- **Range Query Support**: `find_nodes_in_range()` with zone map pruning for range predicates
- **Python Batch APIs**: `get_nodes_by_label(label, limit)` and `get_property_batch(ids, prop)` for efficient bulk access

### Improved

- **Community Detection**: Label propagation algorithm now O(E) instead of O(V²E) using backward adjacency
  - Expected 100-500x improvement for large graphs
- **Zone Map Integration**: Filter planning now uses zone maps for predicate pushdown
  - Scans short-circuit with `EmptyOperator` when zone maps prove no matches possible
- **Filter Equality**: Property index optimization extended to compound AND predicates
- **Filter Range**: Zone map checks for range predicates (`<`, `>`, `<=`, `>=`)
- **Pre-commit Hooks**: Added typos checker, cargo-deny (security/license), and ruff (Python linting)
- **Profiling Profile**: Added `cargo build --profile profiling` for flamegraph analysis

### Changed

- Updated CONTRIBUTING.md with CI scripts and expanded testing documentation

---

## [0.2.4b] - 2026-02-02

Fixed release workflow `--exclude` flag (requires `--workspace`)

## [0.2.4] - 2026-02-02

_Benchmark-Driven Performance Optimizations_

Targeted improvements based on comparative benchmarks against LadybugDB, DuckDB, Neo4j, and Memgraph.

### Improved

- **Single Read Performance**:
  - **Lock-Free Concurrent Reads**: Hash indexes now use DashMap for lock-free reads
    - 4-6x improvement under concurrent read workloads
    - Sharded design eliminates global lock contention
  - **Direct Lookup APIs**: New bypass methods for O(1) point reads without query planning
    - `get_node(id)`, `get_node_property(id, key)`, `get_edge(id)`
    - `get_neighbors_outgoing(node)`, `get_neighbors_incoming(node)`
    - `get_nodes_batch(ids)` for efficient batch lookups
    - 10-20x faster than equivalent MATCH queries
  - **Expanded Hot Buffer**: Property storage hot buffer increased from 256 to 4096 entries
    - Keeps more recent data uncompressed for faster reads
  - **Adjacency Delta Buffer**: Increased inline SmallVec from 8 to 16 entries
    - Better cache locality for common node degrees

- **Filter Performance**:
  - **Direct Property Access**: `LpgStore::get_node_property()` and `get_edge_property()`
    - O(1) single-property lookup instead of loading all properties
  - **PropertyPredicate Optimization**: Uses direct access instead of `get_node().get_property()`
    - Avoids loading entire property map when only one value is needed
  - **Batch Evaluation**: `evaluate_batch()` for better cache efficiency on large datasets
  - Expected 20-50x improvement for equality and range filters

---

## [0.2.3] - Unreleased

### Added

- **Succinct Data Structures** (feature: `succinct-indexes`):
  - `SuccinctBitVector`: O(1) rank/select with superblock/block ranks and select sampling (<5% overhead)
  - `EliasFano`: Quasi-succinct encoding for sparse monotonic sequences (near-optimal space)
  - `WaveletTree`: Sequence rank/select/access in O(log σ) time for Ring Index foundation
- **Block-STM Parallel Execution** (feature: `block-stm`):
  - `ParallelExecutor`: 4-phase execution (optimistic → validate → re-execute → commit)
  - `ExecutionResult`: Read/write set tracking for conflict detection
  - Batch transaction API for ETL workloads (3-4x speedup on 4 cores with 0% conflicts)
- **Ring Index for RDF** (feature: `ring-index`):
  - `TripleRing`: Compact triple storage using wavelet trees (~3x space reduction vs HashMaps)
  - `SuccinctPermutation`: O(1) navigation between SPO, POS, OSP orderings
  - `RingIterator`: Efficient filtered iteration via wavelet tree select
  - `LeapfrogRing`: Foundation for worst-case optimal joins over RDF patterns

### Improved

- **Query Plan Caching**: Optimized logical plans are now cached per-database, shared across sessions
  - Repeated identical queries skip parsing, translation, binding, and optimization
  - LRU cache with 1000 query capacity (500 parsed + 500 optimized)
  - Expected 5-10x speedup for workloads with repeated queries (benchmarks, read-heavy apps)

---

## [0.2.2] - Unreleased

_Performance Tuning_

### Added

- **Bidirectional Edge Indexing**: `edges_to()`, `in_degree()`, `out_degree()` methods for efficient incoming edge queries
- **NUMA-Aware Scheduling**: Work-stealing scheduler now prefers same-node stealing to minimize cross-node memory access
- **Leapfrog TrieJoin**: Worst-case optimal join (WCOJ) for cyclic patterns like triangles - O(N^1.5) vs O(N²)
- **Cyclic Pattern Detection**: Planner detects triangle/clique patterns using DFS coloring
- **WCOJ Cost Model**: `leapfrog_join_cost()` and `prefer_leapfrog_join()` for optimizer decisions
- **Factorized Benefit Estimation**: `factorized_benefit()` for compression ratio predictions

### Improved

- **Direction::Incoming queries**: Now use `backward_adj` index for O(1) neighbor lookup
- **NumaConfig**: Auto-detection heuristic for NUMA topology (2 nodes for >8 cores)
- **Cost Model**: Extended with memory cost tracking via `Cost::with_memory()`

---

## [0.2.1] - Unreleased

_Tiered Storage_

### Added

- **Tiered Version Index**: `VersionIndex` with `HotVersionRef`/`ColdVersionRef` separation using SmallVec (no heap for typical 1-2 versions)
- **Compressed Epoch Store**: `CompressedEpochBlock` with zone maps for predicate pushdown
- **Arena Extensions**: `alloc_value_with_offset()`, `read_at()` for tiered access patterns
- **Epoch Freeze**: `freeze_epoch()` method to compress and archive old epochs
- **Zone Maps**: Min/max tracking for node/edge IDs enables block skipping

### Improved

- **LpgStore Integration**: All CRUD operations now support hot/cold version reads (feature-gated)
- **Memory Efficiency**: `OptionalEpochId` saves 4 bytes vs `Option<EpochId>` using u32 sentinel
- **GC Performance**: Arena-based batch deallocation instead of per-version heap frees

### Internal

- Feature flag `tiered-storage` for opt-in (default off for backwards compatibility)
- 635 tests pass with feature enabled, all tests pass with feature disabled

---

## [0.2.0] - 2026-02-01

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
