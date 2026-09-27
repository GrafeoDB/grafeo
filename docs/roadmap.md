# Roadmap

Grafeo is a high-performance, embeddable graph database written in Rust. This roadmap shows where the project has been, where it is now and where it's going. Priorities may shift based on community feedback and real-world usage.

For detailed release notes, see the [CHANGELOG](changelog.md).

---

## Completed Releases

### 0.1: Foundation

Established the core graph engine: labeled property graph (LPG) storage with MVCC transactions, WAL persistence and multiple index types (hash, B-tree, trie, adjacency). Shipped the GQL (ISO standard) parser as the primary query language, with experimental support for Cypher, SPARQL, Gremlin and GraphQL. Python bindings via PyO3 from day one.

### 0.2: Performance

Made the engine competitive on query throughput. Factorized query processing eliminates Cartesian products in multi-hop traversals. Worst-case optimal joins (Leapfrog TrieJoin) handle cyclic patterns efficiently. Lock-free concurrent reads, query plan caching and direct lookup APIs brought large speedups on common access patterns. First graph algorithms (community detection, clustering coefficient, BFS shortest path). Added RDF Ring Index, Block-STM parallel transactions, tiered storage and succinct data structures as opt-in features.

### 0.3: AI Compatibility

Added first-class vector support: `Value::Vector` type, HNSW approximate nearest neighbor index, four distance metrics (cosine, euclidean, dot product, manhattan) with SIMD acceleration (AVX2, SSE, NEON). Vector quantization (scalar, binary, product) for memory-constrained deployments. Hybrid graph + vector queries across all supported query languages. Serializable snapshot isolation for write-heavy workloads.

### 0.4: Developer Accessibility

Expanded the binding surface: Node.js/TypeScript (napi-rs), Go (C FFI), WebAssembly (wasm-bindgen, 660 KB gzipped), SQL/PGQ (SQL:2023 GRAPH_TABLE). Shipped grafeo-cli with interactive shell, filtered/MMR vector search with incremental indexing. All public API items documented.

---

## Current: 0.5, Beta

*Preparing for production readiness.*

The beta series focuses on correctness, completeness and real-world durability. Key areas:

**Search and retrieval**: BM25 text search, hybrid search (BM25 + vector via RRF/weighted fusion), optional in-process ONNX embeddings, MMR for diverse RAG results.

**Graph algorithms**: CALL procedure interface exposing all 22 algorithms (PageRank, Dijkstra, Louvain, SSSP, etc.) from GQL, Cypher and SQL/PGQ. Algorithms themselves were introduced in 0.2, the query-callable interface is new in 0.5.

**Data management**: Temporal types (Date, Time, Duration, DateTime), graph type enforcement with schema validation and constraints, LOAD DATA for CSV/JSONL/Parquet, named graph persistence, cross-graph transactions.

**Transaction correctness**: MVCC dirty-read prevention, DELETE rollback with full undo log, write-write conflict detection, session auto-rollback, savepoints.

**Persistence**: Single-file `.grafeo` format with dual-header crash safety, index metadata persistence (snapshot v4), read-only open mode with shared file lock for concurrent reader processes.

**Bindings**: C#/.NET 8, Dart/Flutter (community contribution), C FFI layer shared by Go, and C#.

**Ecosystem**: [grafeo-server](https://github.com/GrafeoDB/grafeo-server), [grafeo-web](https://github.com/GrafeoDB/grafeo-web), [grafeo-mcp](https://github.com/GrafeoDB/grafeo-mcp), [grafeo-memory](https://github.com/GrafeoDB/grafeo-memory), [grafeo-langchain](https://github.com/GrafeoDB/grafeo-langchain), [grafeo-llamaindex](https://github.com/GrafeoDB/grafeo-llamaindex).

### Delivered in 0.5.30-0.5.32

- **Async storage backend** (0.5.30): `AsyncStorageBackend` trait, `AsyncTypedWal` with async WAL operations, `AsyncLocalBackend` filesystem implementation
- **CompactStore** (0.5.31): read-only columnar store with per-label tables, double-indexed CSR adjacency, zone-map skip optimization, `CompactStoreBuilder` API. Integrates via `GrafeoDB::with_read_store()`
- **`compact()` method** (0.5.32): one-call conversion from live database to CompactStore. Available in Python, Node.js, WASM, C, Rust. ~60x memory reduction, 100x+ traversal speedup
- **Hybrid Logical Clock** (0.5.32): monotonic HLC timestamps in CDC events for causal ordering
- **Session CDC** (0.5.32): mutations via query sessions (`INSERT`, `SET`, `DELETE`) now generate CDC events, buffered per-transaction
- **Correctness hardening** (0.5.32): epoch monotonicity guarantees, concurrent stress tests, write-write conflict detection improvements
- **grafeo-server replication** (0.5.32): atomic sync apply (all-or-nothing transaction visibility on replicas), persistent replica epoch tracking (replicas resume from last position after restart), CDC auto-activation on replication primaries, relaxed replica guard for sync endpoints

### Delivered in 0.5.33

- **GraphChallenge benchmark suite**: k-truss decomposition, parallel triangle counting, subgraph isomorphism (VF2), stochastic block partitioning
- **TSV/MMIO bulk import**: `import_tsv()`, `import_mmio()`, `import_tsv_rdf()` for GraphChallenge datasets
- **RDF streaming Turtle**: TripleSink-based streaming parser for large RDF datasets
- **`RdfGraphStoreAdapter`**: bridges `RdfStore` to `GraphStore`, giving RDF graphs access to all 25+ graph algorithms

### Delivered in 0.5.34

- **GQL schema hierarchy** (ISO/IEC 39075 Section 4.2.5): `CREATE SCHEMA`/`DROP SCHEMA`, `SESSION SET SCHEMA`, full data isolation
- **Streaming RDF triple sink**: `TripleSink` trait with `BatchInsertSink` and `CountSink`
- **Golden fixture tests**: snapshot v4, `.grafeo` file format and WAL frame byte-equality checks
- **Feature matrix CI**: per-profile build+test jobs (gql-only, gql+vector, gql+rdf, embedded, browser)

### Delivered in 0.5.35

- **Persona-based feature profiles**: `lpg`, `rdf`, `analytics`, `ai`, `edge`, `enterprise` replace deployment-target profiles. Old names (`embedded`, `browser`, `server`, `full`) kept as deprecated aliases, scheduled for removal in 0.7.0
- **Section-based container format**: `.grafeo` files use a section directory with checksummed, independently addressable sections
- **`grafeo-storage` crate**: persistence I/O extracted from `grafeo-adapters` as a sibling to `grafeo-core`
- **Block-based LPG/RDF section format (v2)**: structured layout with string tables, packed arrays, columnar property blocks, per-block CRC
- **Arrow IPC export**: zero-copy export for DuckDB, Polars, pandas, DataFusion interop
- **GEXF + GraphML export**: graph interchange for Gephi, Cytoscape, NetworkX, yEd, igraph
- **Incremental backup**: `backup_full()`, `backup_incremental()`, `restore_to_epoch()` with CLI commands
- **CDC retention and eviction**: `CdcRetentionConfig` with epoch-based and count-based limits
- **Python named graph management**: `create_graph()`, `drop_graph()`, `list_graphs()`, `set_graph()`, `set_schema()`
- **Python per-transaction CDC override**: `begin_transaction_with_cdc(True|False)`
- **Breaking changes**: `QueryResult.rows` now private (use `rows()`/`into_rows()`), 95 public enums are `#[non_exhaustive]`, storage format changed (databases from 0.5.34 or earlier must be re-created)

### Delivered in 0.5.36

- **Role-based access control (Auth M1)**: `Identity`, `Role` (`Admin`, `ReadWrite`, `ReadOnly`), `StatementKind` for session-level permission scoping
- **Per-graph access grants**: `Grant` type restricts identity access to specific named graphs
- **Graph projections**: `CREATE PROJECTION`, `DROP PROJECTION`, `SHOW PROJECTIONS` in GQL, plus API in Python, Node.js, WASM and C
- **CSV/JSON Lines import**: CLI `grafeo import csv`/`grafeo import jsonl`, Python `import_csv()`/`import_jsonl()`, Node.js `importCsv()`/`importJsonl()`
- **Gremlin `repeat().times()`/`.emit()`**: fixed-depth and all-depths traversal
- **Unified aggregate accumulator**: push-based aggregate operator gains all 30+ aggregate functions

### Delivered in 0.5.37

- **SPARQL compliance pass**: 18 spec gaps closed: `CONSTRUCT`, `BIND`, `OPTIONAL`, `MINUS`, `UNION`, `FILTER`, `EXISTS`/`NOT EXISTS`, named graph CRUD, SPARQL UPDATE. Composite indexes (SP, PO, OS) for O(1) multi-bound lookups. 109 new W3C tests
- **Ring Index planner**: wavelet-tree compact triple index wired into the SPARQL planner with Leapfrog WCOJ for multi-way star joins and hash join fallback for LANG/DATATYPE columns. Ring Index persistence via bincode serialization to `.grafeo` container
- **SHACL validation**: W3C Shapes Constraint Language with all 28 Core constraint types, SHACL-SPARQL (`sh:sparql`), 7 property path types with cycle detection, `ValidationReport` with `to_triples()` RDF materialization. Python: `db.validate_shacl("shapes_graph")`
- **EXPLAIN ANALYZE**: physical plan tree without executing (`EXPLAIN`), or profiled execution with per-operator timing (`EXPLAIN ANALYZE`). Python `explain_sparql()` binding
- **Arrow bulk export**: `nodes_to_arrow()`/`edges_to_arrow()` (pyarrow Table), `nodes_to_polars()`/`edges_to_polars()` (Polars DataFrame), `nodes_to_pandas()`/`edges_to_pandas()` (pandas DataFrame via Arrow). ~10-100x faster than per-element `nodes_df()`/`edges_df()` at scale
- **RDF query optimizer**: per-predicate cardinality estimates, cached statistics, cost-based join reordering
- **Dictionary encoding**: `TermDictionary` maps RDF terms to u32 IDs with lazy construction, `DictResolveOperator` resolves at result boundaries
- **COUNT(\*) fast paths**: O(1) for unbound scans via `store.len()`, O(log sigma) for bound patterns via Ring Index

### Delivered in 0.5.38

- **Quantized vector indexes**: `"scalar"`, `"binary"`, `"product"` quantization for 4x memory reduction on large vector datasets
- **EXPLAIN/PROFILE for all 6 query languages**: Gremlin, GraphQL, and SQL/PGQ now support `EXPLAIN` and `EXPLAIN ANALYZE`, matching GQL, Cypher, and SPARQL
- **Unicode identifiers**: GQL, Cypher, and SQL/PGQ parsers accept Unicode letters per ISO GQL 39075
- **Parser recursion depth limits**: 128-level nesting limit across all 6 parsers, preventing stack overflow on malicious input
- **Incremental backup fix** (#267): `backup_incremental` always failed after a full backup because the WAL cursor was not rotated
- **Edge variable resolution** (#268): multi-hop queries returned edge variables as raw IDs instead of maps
- **Arrow/DataFrame structural column rename** (#272): underscore-prefixed columns (`_id`, `_type`, `_source`, `_target`) prevent collision with user properties

### Delivered in 0.5.39

- **Push-based pipeline execution**: queries with filter, sort, aggregate, limit, or distinct now execute through a push-based pipeline instead of the Volcano pull loop
- **Encryption at rest** (`encryption` feature): AES-256-GCM for WAL records and `.grafeo` sections with password-based (Argon2id) or raw-key setup
- **Block-STM conflict partitioning**: union-find clustering of conflicting transactions for parallel re-execution of disjoint conflict sets
- **Runtime metrics**: query, transaction, session, cache, and GC counters with Prometheus text export
- **WASM binary size**: reduced to 650 KB gzipped (competitive with sql.js)
- **C# enterprise APIs**: schema management, backup/restore, compact, projections, CDC toggle, `IGrafeoDB`/`ITransaction` interfaces
- **Layered compact store**: `compact()` now produces a writable two-layer store (columnar base + overlay) instead of a read-only snapshot. `recompact()` merges the overlay back into the base.

### Delivered in 0.5.40

- **Unified hybrid queries**: `text_score()` and `text_match()` are now evaluable as per-row filter expressions. The planner pushes `text_score(n.prop, "query") > threshold` and vector score predicates into dedicated `TextScan` / `VectorScan` operators, supports compound AND/OR hybrid joins, recognizes `ORDER BY ... LIMIT` as top-K, and projects the score column to avoid recompute. See [Filter-expression hybrid search](user-guide/vector-search/filter-expressions.md).
- **BM25 text scan operator**: `TextScanOperator` supports both top-K and threshold modes. `InvertedIndex` gains `score_document`, `search_with_threshold`, and `bm25_term_score` helpers.
- **Float64 and Float32Vector column codecs**: `CompactStore` stores `Value::Float64` and `Value::Vector` properties natively instead of falling back to dictionary encoding. Mixed `Int64 + Float64` columns coalesce to `Float64`.
- **MERGE uses property indexes**: `MERGE (n:Label {prop: value})` now goes through the property index when one exists, matching `MATCH` performance. Previously O(n) on large graphs.
- **Index and search after `compact()`**: `create_vector_index`, `vector_search`, `create_text_index`, `text_search`, and the other ~26 index/search methods now work correctly on layered stores (previously panicked or silently returned empty).
- **`LayeredStore` new-node visibility**: `get_node` and `get_node_property` fall back to the overlay for nodes added after `compact()`, fixing `recompact()` dropping those nodes from the merged base.
- **Named graphs across `compact()` / `recompact()`**: `list_graphs`, `drop_graph`, `create_graph`, and `set_current_graph` now see graphs that existed before compaction.

### Delivered in 0.5.41

- **Compact-store correctness**: post-`compact()` read path and signed integer round-trip fixes
- **Search procedures**: `CALL grafeo.search.*` for vector, text and MMR search from queries
- **Disk-backed compact base**: the columnar base of a compacted store can live on disk under memory pressure
- **Memory introspection** for RDF and CDC components

### Delivered in 0.5.42

- **Tiered storage**: per-section RAM/disk tier configuration, spill under memory pressure, introspection and reload
- **Paged HNSW topology** and **packed RDF Ring**: neighbor lookups and ring data served directly from mmap
- **Streaming top-K operator**: `ORDER BY ... LIMIT k` in a single bounded-heap pass (~12x faster at 1M rows)
- **`WHERE prop IN [...]` index fast path** and filter pushdown through `OPTIONAL MATCH` and subqueries

### Delivered in 0.5.43

- **Stabilization**: green CI, dependency and security updates, Rust toolchain pinned
- **Query correctness** from community reports: `ORDER BY ... LIMIT` over whole nodes, `UNION` column checks, aggregates next to aliased items, duplicate column names, SPARQL named-graph updates and `path+`
- **Statements are no longer truncated**: consecutive `INSERT`s and `INSERT ... RETURN` work in GQL, and trailing input is a syntax error in GQL, Cypher and Gremlin
- **Rollbacks that undo everything**: property and label changes on persistent databases, and SPARQL updates inside transactions
- **Refusing to write databases beyond the storage format limits** instead of producing unreadable files
- **HNSW connectivity** when indexed vectors are updated

### Planned Releases

| Version    | Focus |
|------------|-------|
| **0.5.44** | Durability and persistence completeness: transaction-grouped WAL records, durable writes without `close()`, safe WAL checkpoints, a database directory lock, vector and text indexes that survive reopen in WAL mode, storage format support for databases over 4 GiB; index seeks for keys that come from earlier rows (`UNWIND ... MATCH (n {id: row.id})`), property equi-joins as hash joins, temporal properties (point-in-time queries), streaming results, shortest-path elements (`nodes(p)`, `edges(p)`), `VECTOR(n)` schema type and filtered hybrid search, next phase of tiered storage |
| **0.5.45** | Driver and protocol compatibility: ADBC driver, W3C SPARQL HTTP Protocol and Graph Store Protocol, offline-first sync, cross-language query translation, Jupyter cell magic, final audit. Last 0.5.x release |

---

## Next: 0.6, Release Candidate

*No new major features. Bug fixes, community integrations, and quality of life.*

The scope is intentionally narrow:

- **Bug fixes** from real-world 0.5/6 usage
- **Performance tuning** informed by actual workloads, not synthetic benchmarks
- **API ergonomics** and documentation polish
- **Binary size and compile time** optimization
- **C FFI parity refactor**: expand grafeo-c to match Python/Node.js API surface, update downstream bindings

The goal is confidence: if something works in 0.6, it works in 1.0.

---

## 0.7: Pluggable Auth + Profile Cleanup (Auth M2)

- **Pluggable auth providers**: `AuthProvider` trait with JWT and OIDC backends
- **Audit logging**: structured request-level audit trail per identity
- **Deprecated feature profile removal**: old names (`embedded`, `browser`, `server`, `full`) removed

---

## 1.0: Stable

Semantic versioning commitment. Public API frozen. No breaking changes without a major version bump.

- **Enterprise auth (Auth M3)**: row-level security, property masking, GRANT/REVOKE, LDAP/SAML, per-tenant encryption, compliance reporting

---

## Future Considerations

Not scheduled, but on the radar:

- Distributed/clustered deployment
- Additional language bindings (Java/Kotlin, Swift)
- Cloud-native integrations

---

## Contributing

Interested in contributing? Check the [GitHub Issues](https://github.com/GrafeoDB/grafeo/issues), join the [Discussions](https://github.com/orgs/GrafeoDB/discussions) or hop into the [Discord server](https://discord.gg/jrgMD2Zj3).

---

Last updated: September 2026
