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

## 0.2.x - Performance (Complete)

*Competitive with the fastest graph databases*

### Performance Improvements (Delivered)

- **Factorized query processing** for multi-hop traversals (avoid Cartesian products)
- **Worst-case optimal joins** via Leapfrog TrieJoin for cyclic patterns (O(N^1.5) triangles)
- **Lock-free concurrent reads** using DashMap-backed hash indexes (4-6x improvement)
- **Direct lookup APIs** bypassing query planning for O(1) point reads (10-20x faster)
- **Query plan caching** with LRU cache for repeated queries (5-10x speedup)
- **NUMA-aware scheduling** with same-node work-stealing preference

### New Features (Delivered)

- **Ring Index for RDF** (`ring-index` feature) - 3x space reduction using wavelet trees
- **Block-STM parallel execution** (`block-stm` feature) - optimistic parallel transactions
- **Tiered hot/cold storage** (`tiered-storage` feature) - compressed epoch archival
- **Succinct data structures** (`succinct-indexes` feature) - rank/select bitvectors, Elias-Fano

### Expanded Support (Delivered)

- **RDF** - full support with Ring Index and SPARQL optimization
- All 5 query languages promoted to full support
- NetworkX and solvOR integrations promoted to full support

---

## 0.3.x - AI Compatibility (Complete)

*Ready for modern AI/ML workloads*

### Vector Features (Delivered)

- **Vector Type** - First-class `Vector` type with f32 storage (4x compression vs f64)
- **Distance Functions** - Cosine, Euclidean, Dot Product, Manhattan metrics
- **HNSW Index** - O(log n) approximate nearest neighbor search with batch insert/search
- **Hybrid Queries** - Combine graph traversal with vector similarity in GQL/Cypher/SPARQL
- **Serializable Isolation** - SSI for write skew prevention and strong consistency

### Vector Quantization (Delivered)

- **Scalar Quantization** - f32 → u8, 4x compression with ~97% recall
- **Binary Quantization** - f32 → 1 bit, 32x compression with SIMD-accelerated hamming distance
- **Product Quantization** - Codebook-based 8-32x compression with asymmetric distance computation
- **QuantizedHnswIndex** - Two-phase search with rescoring support
- **SIMD Acceleration** - AVX2+FMA, SSE, and NEON support for 4-8x faster distance computations

### Vector Storage & Search (Delivered)

- **Memory-Mapped Vector Storage** - Disk-backed storage with LRU cache for large datasets
- **VectorScan Operators** - HNSW and brute-force search in query execution plans
- **VectorJoin Operator** - Hybrid graph pattern + vector similarity search
- **Vector Zone Maps** - Centroid and bounding box pruning for block skipping
- **Vector Cost Estimation** - HNSW O(ef * log N) and brute-force O(N) cost models
- **Python Quantization API** - Full quantization support from Python

### Execution & Quality (Delivered)

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

### New Bindings (Experimental)

- **Node.js / TypeScript** (`@grafeo-db/js`) - native bindings with full type definitions
- **WebAssembly** (`@grafeo-db/wasm`) - raw WASM binary for edge runtimes
- **Browser** (`@grafeo-db/web`) - IndexedDB persistence, Web Workers, React/Vue/Svelte integrations
- **Go** - CGO bindings for cloud-native applications

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
