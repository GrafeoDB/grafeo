# Roadmap

This roadmap outlines the planned development of Grafeo. Priorities may shift based on community feedback and real-world usage.

---

## 0.1.x - Foundation

*Building a fully functional graph database*

### Core Database
- Labeled Property Graph (LPG) storage model
- MVCC transactions with snapshot isolation
- Multiple index types (hash, B-tree, trie, adjacency)
- Write-ahead logging (WAL) for durability
- In-memory and persistent storage modes

### Query Languages
- **GQL** (ISO standard) - full support
- **Cypher** - experimental
- **Gremlin** - experimental
- **GraphQL** - experimental
- **SPARQL** - experimental

### Data Models
- **LPG** - full support
- **RDF** - experimental

### Bindings
- **Python** - full support via PyO3
- NetworkX integration - experimental
- solvOR graph algorithms - experimental

---

## 0.2.x - Performance

*Competitive with the fastest graph databases*

### Performance Improvements 

- **Factorized query processing** for multi-hop traversals (avoid Cartesian products)
- **Worst-case optimal joins** via Leapfrog TrieJoin for cyclic patterns (O(N^1.5) triangles)
- **Lock-free concurrent reads** using DashMap-backed hash indexes (4-6x improvement)
- **Direct lookup APIs** bypassing query planning for O(1) point reads (10-20x faster)
- **Query plan caching** with LRU cache for repeated queries (5-10x speedup)
- **NUMA-aware scheduling** with same-node work-stealing preference

### New Features 

- **Ring Index for RDF** (`ring-index` feature) - 3x space reduction using wavelet trees
- **Block-STM parallel execution** (`block-stm` feature) - optimistic parallel transactions
- **Tiered hot/cold storage** (`tiered-storage` feature) - compressed epoch archival
- **Succinct data structures** (`succinct-indexes` feature) - rank/select bitvectors, Elias-Fano

### Expanded Support

- **RDF** - full support with Ring Index and SPARQL optimization
- All 5 query languages promoted to full support
- NetworkX and solvOR integrations promoted to full support

---

## 0.3.x - AI Compatibility 

*Ready for modern AI/ML workloads*

### Vector Features

- **Vector Type** - First-class `Vector` type with f32 storage (4x compression vs f64)
- **Distance Functions** - Cosine, Euclidean, Dot Product, Manhattan metrics
- **HNSW Index** - O(log n) approximate nearest neighbor search with batch insert/search
- **Hybrid Queries** - Combine graph traversal with vector similarity in GQL/Cypher/SPARQL
- **Serializable Isolation** - SSI for write skew prevention and strong consistency

### Vector Quantization

- **Scalar Quantization** - f32 → u8, 4x compression with ~97% recall
- **Binary Quantization** - f32 → 1 bit, 32x compression with SIMD-accelerated hamming distance
- **Product Quantization** - Codebook-based 8-32x compression with asymmetric distance computation
- **QuantizedHnswIndex** - Two-phase search with rescoring support
- **SIMD Acceleration** - AVX2+FMA, SSE, and NEON support for 4-8x faster distance computations

### Vector Storage & Search

- **Memory-Mapped Vector Storage** - Disk-backed storage with LRU cache for large datasets
- **VectorScan Operators** - HNSW and brute-force search in query execution plans
- **VectorJoin Operator** - Hybrid graph pattern + vector similarity search
- **Vector Zone Maps** - Centroid and bounding box pruning for block skipping
- **Vector Cost Estimation** - HNSW O(ef * log N) and brute-force O(N) cost models
- **Python Quantization API** - Full quantization support from Python

### Execution & Quality

- **Selective Property Loading** - Projection pushdown for O(N*K) vs O(N*C) reads
- **Parallel Node Scan** - Morsel-driven parallel execution for 3-8x speedup on large scans
- **Query Performance Metrics** - Execution timing and row counts on query results
- **Error Message Suggestions** - Fuzzy "Did you mean X?" hints for undefined variables and labels
- **Adaptive WAL Flusher** - Self-tuning background flush with consistent cadence

### Syntax Support

```gql
-- Vector literals and similarity functions
MATCH (m:Movie)
WHERE cosine_similarity(m.embedding, $query) > 0.8
RETURN m.title

-- Create vector index
CREATE VECTOR INDEX movie_embeddings ON :Movie(embedding)
  WITH (dimensions: 384, metric: 'cosine')
```

---

## 0.4.x - Developer Accessibility

*Reach more developers*

### New Bindings

- **Node.js / TypeScript** (`@grafeo-db/js`) - native bindings via napi-rs with full type definitions (0.4.0)
- **Go** (`github.com/GrafeoDB/grafeo/crates/bindings/go`) - CGO bindings with C FFI layer for cloud-native applications (0.4.1)
- **WebAssembly** (`@grafeo-db/wasm`) - feature-gated platform code, wasm-bindgen bindings, 660 KB gzipped (0.4.2)

### Ecosystem Integration

- **[grafeo-server](https://github.com/GrafeoDB/grafeo-server)** - standalone HTTP server with web UI, Docker image
- **[grafeo-web](https://github.com/GrafeoDB/grafeo-web)** - Grafeo in the browser via IndexedDB, Web Workers, React/Vue/Svelte

### Query Languages

- **SQL/PGQ** (SQL:2023) - GRAPH_TABLE function for SQL-native graph queries

### Quality

- Continued bug fixes
- Stricter linting rules
- Performance tuning based on real-world usage

---

## 0.5.x - Beta

*Preparing for production readiness*

### Focus Areas
- Performance optimizations across all components
- Lots of bug hunting and fixing
- Documentation improvements, user guides and examples
- API stabilization

### Goal
- Ready for production evaluation
- Clear path to 1.0

---

## Future Considerations

These features are under consideration for future releases:

- Additional language bindings (Java/Kotlin, Swift)
- Distributed/clustered deployment
- Cloud-native integrations

---

## Contributing

Interested in contributing to a specific feature? Check our [GitHub Issues](https://github.com/GrafeoDB/grafeo/issues) or join the discussion.

---

*Last updated: February 2026*
