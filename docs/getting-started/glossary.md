---
title: Glossary
description: Definitions of key terms used in Grafeo documentation.
tags:
  - glossary
  - terminology
  - reference
---

# Glossary

Key terms and concepts used throughout Grafeo documentation.

---

## Graph Concepts

### Node
A vertex in the graph representing an entity (person, product, concept). Nodes have:

- **ID**: Unique 64-bit identifier
- **Labels**: Categories like "Person", "Product" (can have multiple)
- **Properties**: Key-value attributes like `{name: "Alice", age: 30}`

### Edge
A relationship connecting two nodes. Edges have:

- **ID**: Unique 64-bit identifier
- **Type**: Relationship name like "KNOWS", "PURCHASED"
- **Source/Target**: The nodes it connects
- **Properties**: Key-value attributes like `{since: "2024-01-01"}`

### Label
A tag categorizing nodes. Nodes can have multiple labels (e.g., a node can be both "Person" and "Employee").

### Property
A key-value attribute on a node or edge. Keys are strings; values can be null, boolean, integer, float, string, list, or map.

### LPG (Labeled Property Graph)
The primary graph model in Grafeo where nodes have labels and both nodes and edges have properties. This is the model used by Neo4j, TigerGraph, and most modern graph databases.

### RDF (Resource Description Framework)
An alternative graph model using subject-predicate-object triples. Used for semantic web and knowledge graphs. Grafeo supports RDF with the `rdf` feature flag.

---

## Query Language Terms

### GQL
ISO/IEC 39075:2024 - the international standard for graph query languages. Grafeo uses GQL as its primary query language.

### Cypher
A graph query language originally developed by Neo4j. Grafeo supports Cypher via the `cypher` feature flag.

### SPARQL
W3C standard query language for RDF data. Grafeo supports SPARQL via the `sparql` feature flag.

### Gremlin
Apache TinkerPop's graph traversal language. Grafeo supports Gremlin via the `gremlin` feature flag.

### Pattern Matching
Finding subgraphs that match a specified pattern:
```
MATCH (a:Person)-[:KNOWS]->(b:Person)
```
This finds all pairs of people where one knows the other.

---

## Execution Engine Terms

### DataChunk
A batch of rows (typically 2048) processed together for efficiency. Grafeo processes data in chunks rather than row-by-row to maximize CPU cache utilization.

### Morsel
A unit of work in parallel query execution. The scheduler divides work into morsels that workers can steal from each other for load balancing.

### Vectorized Execution
Processing multiple values at once using SIMD (Single Instruction Multiple Data) CPU instructions. Grafeo uses vectorized execution for operations like filtering and aggregation.

### Push-Based Execution
An execution model where data flows from producers to consumers. Operators "push" results downstream rather than being "pulled" by upstream operators.

### Factorized Execution
An optimization that avoids materializing Cartesian products in multi-hop traversals. Instead of expanding all paths, Grafeo represents them compactly and expands lazily.

### Pipeline
A sequence of operators processing data. For example: Scan -> Filter -> Project -> Aggregate.

### Zone Map
Metadata storing min/max values for chunks. Enables skipping entire chunks when a filter can't possibly match (e.g., filtering for age > 100 when max age in chunk is 80).

---

## Transaction Terms

### MVCC (Multi-Version Concurrency Control)
Grafeo's approach to handling concurrent transactions. Each transaction sees a consistent snapshot; readers never block writers and vice versa.

### Epoch
A logical timestamp representing a point in database history. Each committed transaction increments the epoch.

### Snapshot Isolation
The default isolation level where transactions see a consistent snapshot of the database from their start time. Changes made by other transactions aren't visible until commit.

### Write-Write Conflict
When two transactions try to modify the same entity. The second transaction to commit will fail and must retry.

### SSI (Serializable Snapshot Isolation)
The strongest isolation level, detecting potential anomalies like write skew. Enabled with `Serializable` isolation.

### Block-STM
An optional parallel transaction execution strategy for batch workloads. Executes transactions optimistically and re-executes on conflicts.

---

## Index Terms

### Hash Index
Index structure for O(1) equality lookups. Best for unique identifiers like email addresses.

### B-Tree Index
Index structure for O(log n) range queries. Best for sortable data like timestamps or ages.

### Adjacency Index
Specialized index for graph traversal. Stores outgoing/incoming edges per node for fast neighbor lookups.

### Trie Index
Tree structure for prefix matching and worst-case optimal joins (WCOJ).

### HNSW (Hierarchical Navigable Small World)
Algorithm for approximate nearest neighbor search on vectors. Used by Grafeo's vector index.

### Ring Index
Space-efficient RDF triple index using wavelet trees. Provides 3x compression over separate indexes.

---

## Storage Terms

### WAL (Write-Ahead Log)
Durability mechanism that logs changes before applying them. Enables crash recovery by replaying the log.

### Checkpoint
Process of flushing WAL changes to main storage, allowing the WAL to be truncated.

### Dictionary Encoding
Compression technique storing unique values once and referencing them by ID. Reduces storage for repeated strings like labels.

### Delta Encoding
Compression storing differences between consecutive values. Efficient for sorted integers like IDs.

### Bit-Packing
Compression using minimum bits needed per value. Efficient for small integers like ages.

### Tiered Storage
Separating "hot" (recent, mutable) data from "cold" (historical, compressed) data for better performance and space efficiency.

---

## Memory Terms

### Arena Allocator
Memory allocator that allocates sequentially and frees all at once. Used for version chains in MVCC.

### Bump Allocator
Fast allocator that just increments a pointer. Used for temporary per-query allocations.

### Buffer Manager
Component managing memory allocation with pressure awareness. Can trigger spill-to-disk when memory is tight.

### Memory Pressure
When allocated memory approaches limits. Grafeo responds by evicting cached data or spilling to disk.

---

## API Terms

### Session
Lightweight handle for executing queries. Cheap to create; typically one per thread or request.

### Query Builder
Fluent API for constructing parameterized queries safely.

### Adapter
Bridge to external ecosystems like NetworkX or solvOR.

---

## Architecture Terms

### Crate
Rust's term for a package/library. Grafeo is organized into multiple crates:

- **grafeo-common**: Foundation types
- **grafeo-core**: Storage and execution
- **grafeo-adapters**: Parsers and plugins
- **grafeo-engine**: Database facade
- **grafeo**: Public API

### Feature Flag
Compile-time option enabling/disabling functionality:

- `gql`, `cypher`, `sparql`, `gremlin`, `graphql`: Query languages
- `rdf`: RDF triple store support
- `vector-index`: HNSW similarity search
- `block-stm`: Parallel batch transactions

---

## Abbreviations

| Abbr | Meaning |
|------|---------|
| ACID | Atomicity, Consistency, Isolation, Durability |
| API | Application Programming Interface |
| CLI | Command-Line Interface |
| CPU | Central Processing Unit |
| GQL | Graph Query Language |
| ID | Identifier |
| I/O | Input/Output |
| LPG | Labeled Property Graph |
| MVCC | Multi-Version Concurrency Control |
| NUMA | Non-Uniform Memory Access |
| O(n) | Big-O notation for complexity |
| PyO3 | Python-Rust interop library |
| RAM | Random Access Memory |
| RDF | Resource Description Framework |
| SIMD | Single Instruction Multiple Data |
| SQL | Structured Query Language |
| SSI | Serializable Snapshot Isolation |
| WAL | Write-Ahead Log |
| WCOJ | Worst-Case Optimal Join |
