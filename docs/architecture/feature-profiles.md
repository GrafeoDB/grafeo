# Feature Profiles

## Motivation

Grafeo aims to be a universal graph database: property graphs, RDF, analytics, AI memory, browser, production server. But no single user needs all of that. Feature profiles let every user get exactly what they need, with zero overhead from what they don't.

There are two layers:

- **Layer 1: Profiles**, named groups consistent across the entire ecosystem. This is what most users interact with.
- **Layer 2: Atoms**, individual feature flags for power users who want precise control. Profiles are composed from these.

## Profiles

Since 0.5.35, profiles are named after *what you are building*, not *where it runs*. They are defined in the `grafeo` facade crate and the binding crates.

| Profile | Persona | What it enables |
| --- | --- | --- |
| `lpg` | Graph App Developer | Labeled property graph model, GQL, Cypher, Gremlin, SQL/PGQ, storage, regex |
| `rdf` | Knowledge Engineer | RDF triple store, GQL, SPARQL, GraphQL, SHACL validation, storage (with the LPG store it needs), regex |
| `analytics` | Data Scientist | Graph algorithms, vector, text and hybrid search, JSON Lines and Parquet import |
| `ai` | AI Memory / Agent Developer | Vector, text and hybrid search, change data capture |
| `edge` | Frontend / Edge Developer | LPG model, GQL, lightweight regex (minimal, WASM-friendly) |
| `enterprise` | Platform Operator | Metrics, tracing, async storage; grafeo-server adds auth, TLS, sync, replication and transports |

### Composition Rules

- **Model profiles** (`lpg`, `rdf`): pick one or both. These are the foundation.
- **Capability profiles** (`analytics`, `ai`, `enterprise`): stack on top of a model profile.
- **Constrained profile** (`edge`): minimal by default. Add atoms such as `algos` if you accept the size increase.

Examples:

```toml
# AI memory developer
grafeo = { version = "0.5", default-features = false, features = ["lpg", "ai"] }

# Semantic data scientist
grafeo = { version = "0.5", default-features = false, features = ["rdf", "analytics"] }

# Browser app
grafeo = { version = "0.5", default-features = false, features = ["edge"] }

# Power user: just Cypher and vector search with persistence
grafeo = { version = "0.5", default-features = false, features = ["cypher", "vector-index", "wal"] }

# Full production stack (grafeo-server)
# grafeo-server = { features = ["lpg", "rdf", "ai", "enterprise"] }
```

## Profile Definitions

As defined in the `grafeo` facade crate:

### LPG

```toml
lpg = ["grafeo-engine/lpg", "gql", "cypher", "gremlin", "sql-pgq", "storage", "regex"]
```

All labeled property graph query languages plus persistence. The default choice for application developers working with nodes, edges, labels and properties.

### RDF

```toml
rdf = ["triple-store", "grafeo-engine/lpg", "gql", "sparql", "graphql", "storage", "regex", "shacl"]
```

RDF triple store with SPARQL, GraphQL, SHACL validation and persistence, for knowledge engineers working with ontologies and linked data. Persistence needs the LPG store for now, so the profile includes it ([#544](https://github.com/GrafeoDB/grafeo/issues/544)): without it a database lost its triples on reopen. Add the `ring-index` atom for compact RDF indexing (it pulls in `succinct-indexes`).

> **Note:** in the lower-level crates (`grafeo-core`, `grafeo-adapters`, `grafeo-engine`), `rdf` is a deprecated alias for the `triple-store` atom only. The profile above applies to the facade and binding crates.

### Analytics

```toml
analytics = ["algos", "vector-index", "text-index", "hybrid-search", "jsonl-import", "parquet-import"]
```

25+ graph algorithms (PageRank, Louvain, SSSP, Dijkstra, BFS/DFS, centrality, community detection, MST, flow, isomorphism, clustering), search indexes and bulk import. Combine with `lpg` or `rdf` depending on the dataset.

### AI

```toml
ai = ["vector-index", "text-index", "hybrid-search", "cdc"]
```

Structured memory for LLMs, agents and RAG pipelines: vector and text retrieval plus change feeds. Point-in-time queries (`temporal`) are an opt-in atom today.

> **Note:** `embed` (in-process ONNX embedding generation, about 17 MB) is deliberately not part of this profile. Most AI memory use cases bring embeddings through an API. Opt in explicitly with `features = ["ai", "embed"]`.

### Edge

```toml
edge = ["grafeo-engine/lpg", "gql", "regex-lite"]
```

Minimal profile for browsers, mobile and other constrained environments, with the smallest possible binary. Add `compact-store` for pre-built read-only datasets.

### Enterprise

```toml
enterprise = ["metrics", "tracing", "async-storage"]
```

Production operations. In the engine this enables observability and the async storage backend. On **grafeo-server**, `enterprise` additionally enables authentication, TLS, sync, replication, the push changefeed and all transports (HTTP, GWP, Bolt, Studio), which exist only in the server workspace.

## Deprecated Profile Names

!!! warning "Deprecated, removed in 0.7.0"
    The deployment-based names `embedded`, `browser`, `server` and `full` still work as aliases, but they are deprecated and will be removed in 0.7.0 ([#468](https://github.com/GrafeoDB/grafeo/issues/468)). Use the persona profiles in new projects.

| Deprecated name | Use instead | Notes |
| --- | --- | --- |
| `embedded` | `lpg` + `ai` + `algos` + `parallel` + `arrow-export` | Currently still the default of the facade and the Python, Node.js and C bindings. `lpg` adds Cypher, Gremlin, SQL/PGQ and the rest of `storage`; the bindings' `embedded` also includes `compact-store` |
| `browser` | `edge` | Currently still the default of the WASM binding |
| `server` | `lpg` + `rdf` + `ai` + `algos` + `parallel` + `arrow-export` + `async-storage` + `tracing` | `enterprise` without `metrics`; no bulk import |
| `full` | same as `server` | In the facade, `full` is an alias of `server`. The bindings' `full` is all languages, `ai`, `algos` and the RDF triple store |

The binding defaults move to persona names before the aliases are removed, so depending on `grafeo` without features keeps working.

## Ecosystem Matrix

The profile names are consistent across every project. The table below shows which profiles are available in each project, either as configurable feature flags or as the project's inherent profile alignment.

### Core Engine

| Project | LPG | RDF | Analytics | AI | Edge | Enterprise | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **grafeo** (engine) | flag | flag | flag | flag | flag | flag | Engine-level `enterprise` is observability and async storage |
| **grafeo-server** | flag | flag | flag | flag | n/a | flag | Server `enterprise` adds auth, TLS, sync, replication, transports |
| **grafeo-cli** | flag | flag | flag | flag | n/a | n/a | Interactive REPL and CLI tooling |

### Language Bindings

| Project | LPG | RDF | Analytics | AI | Edge | Enterprise | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Python** (grafeo-py) | flag | flag | flag | flag | flag | n/a | Default: `embedded` (deprecated alias) |
| **Node.js** (grafeo-node) | flag | flag | flag | flag | flag | n/a | Default: `embedded` (deprecated alias) |
| **WASM** (grafeo-wasm) | flag | flag | flag | flag | flag (default) | n/a | Default: `browser` (deprecated alias for `edge`) |
| **C** (grafeo-c) | flag | flag | flag | flag | flag | n/a | Bridge for C#, Dart, Go. Default: `embedded` (deprecated alias) |
| **C#** | via C | via C | via C | via C | via C | n/a | Feature selection at C build time |
| **Dart** | via C | via C | via C | via C | via C | n/a | Feature selection at C build time |
| **Go** | via C | via C | via C | via C | via C | n/a | Feature selection at C build time |

The bindings use the same profile names, with a few differences from the facade:

- **C** (and C#, Dart, Go): `rdf` has no SHACL validation.
- **WASM**: `lpg` and `rdf` have no storage, `rdf` has no SHACL and uses the lightweight regex engine, `ai` has no change data capture, `analytics` is `ai` plus `algos` (no bulk import), and `edge` includes `compact-store`.
- **Python, Node.js, C**: `embedded` also includes `compact-store`.

### AI / Agent Ecosystem

| Project | LPG | RDF | Analytics | AI | Edge | Enterprise | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **grafeo-memory** | inherent | - | - | inherent | - | - | AI memory layer |
| **grafeo-langchain** | inherent | - | - | inherent | - | - | LangChain integration |
| **grafeo-llamaindex** | inherent | - | - | inherent | - | - | LlamaIndex integration |
| **grafeo-mcp** | inherent | - | - | inherent | - | - | MCP server for AI agents |

### Web / Visualization

| Project | LPG | RDF | Analytics | AI | Edge | Enterprise | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **grafeo-web** | - | - | - | - | inherent | - | WASM in browser |
| **playground** | - | - | - | - | inherent | - | Interactive graph playground |
| **anywidget-graph** | inherent | - | - | - | - | - | Notebook graph visualization |
| **anywidget-vector** | - | - | - | inherent | - | - | Notebook vector visualization |

### Protocol Libraries

| Project | LPG | RDF | Analytics | AI | Edge | Enterprise | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **boltr** | - | - | - | - | - | inherent | Bolt v5 protocol |
| **gwp** | - | - | - | - | - | inherent | GQL Wire Protocol (gRPC) |

### Accelerators and Tooling

| Project | LPG | RDF | Analytics | AI | Edge | Enterprise | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **grafeo-cuda** | - | - | inherent | - | - | - | GPU-accelerated algorithms |
| **graph-bench** | all | all | all | all | - | - | Benchmark suite |

### Legend

- **flag**: Profile is available as a configurable feature flag. User opts in.
- **inherent**: The project is inherently aligned with this profile. No flag needed.
- **via C**: Feature selection happens at C binding compile time, propagates to higher-level bindings.
- **n/a**: Profile does not apply to this project.
- **-**: Not applicable or not supported.

## Atom Reference

The individual feature flags (Layer 2) that profiles are composed from. "(standalone)" means the atom is not part of any persona profile and is enabled on its own; some of those are part of the deprecated `embedded` default, as noted.

### Query Languages

| Atom | Profile | Description | Status |
| --- | --- | --- | --- |
| `gql` | LPG, RDF, Edge | ISO/IEC GQL standard | Implemented |
| `cypher` | LPG | openCypher 9.0 | Implemented |
| `sparql` | RDF | W3C SPARQL 1.1 | Implemented |
| `gremlin` | LPG | Apache TinkerPop | Implemented |
| `graphql` | RDF | GraphQL over RDF | Implemented |
| `sql-pgq` | LPG | SQL:2023 GRAPH_TABLE | Implemented |

### Storage

| Atom | Profile | Description | Status |
| --- | --- | --- | --- |
| `storage` | LPG, RDF | Umbrella: WAL + grafeo-file + spill + mmap | Implemented |
| `wal` | (storage) | Write-ahead log persistence | Implemented |
| `grafeo-file` | (storage) | Single-file .grafeo format | Implemented |
| `spill` | (storage) | Out-of-core disk spilling | Implemented |
| `mmap` | (storage) | Memory-mapped file storage | Implemented |
| `async-storage` | Enterprise | Async WAL backend (tokio) | Implemented |
| `compact-store` | (standalone); in the bindings' `embedded` and in WASM `edge` | Columnar store for read-mostly datasets | Implemented |

### Graph Model

| Atom | Profile | Description | Status |
| --- | --- | --- | --- |
| `triple-store` | RDF | RDF triple store with 6-way indexing | Implemented |
| `shacl` | RDF | SHACL validation (core and SPARQL-based constraints) | Implemented |
| `ring-index` | (standalone) | Space-efficient RDF index (pulls in succinct-indexes) | Implemented |
| `succinct-indexes` | (pulled in by ring-index) | Rank/select bitvectors, Elias-Fano, wavelet trees | Implemented |
| `owl-schema` | RDF (server only) | OWL schema loading | Server only |
| `rdfs-schema` | RDF (server only) | RDFS schema support | Server only |

### Search and AI

| Atom | Profile | Description | Status |
| --- | --- | --- | --- |
| `vector-index` | Analytics, AI | HNSW approximate nearest neighbor | Implemented |
| `text-index` | Analytics, AI | BM25 inverted index | Implemented |
| `hybrid-search` | Analytics, AI | Combined vector + text search | Implemented |
| `embed` | (standalone) | ONNX embedding generation (~17 MB overhead) | Implemented |
| `algos` | Analytics | 25+ graph algorithms | Implemented |

### Temporal and Change Tracking

| Atom | Profile | Description | Status |
| --- | --- | --- | --- |
| `temporal` | (standalone) | Append-only versioned properties, point-in-time queries | Implemented |
| `cdc` | AI | Change data capture with history API | Implemented |

### Import and Export

| Atom | Profile | Description | Status |
| --- | --- | --- | --- |
| `jsonl-import` | Analytics | JSON Lines file import | Implemented |
| `parquet-import` | Analytics | Apache Parquet import | Implemented |
| `arrow-export` | (standalone); in `embedded` (not in C or WASM) | Arrow IPC export for DuckDB, Polars, pandas | Implemented |

### Execution

| Atom | Profile | Description | Status |
| --- | --- | --- | --- |
| `parallel` | (standalone); in `embedded` | Parallel execution (rayon) | Implemented |
| `tiered-storage` | (standalone) | Hot/cold version storage with epochs | Implemented |

> **Note:** Block-STM parallel transaction execution is compiled unconditionally. It is not gated behind a feature flag.

### Operations

| Atom | Profile | Description | Status |
| --- | --- | --- | --- |
| `metrics` | Enterprise | Lock-free query and transaction metrics, Prometheus export | Implemented |
| `tracing` | Enterprise | Tracing spans | Implemented |
| `auth` | Enterprise (server) | Authentication provider | Server only |
| `tls` | Enterprise (server) | TLS/HTTPS encryption | Server only |
| `sync` | Enterprise (server) | Pull-based changefeed for offline-first | Server only |
| `push-changefeed` | Enterprise (server) | Push-based SSE/WebSocket changefeed | Server only |
| `replication` | Enterprise (server) | Primary-replica replication | Server only |

### Transports (grafeo-server only)

| Atom | Profile | Description | Status |
| --- | --- | --- | --- |
| `http` | Enterprise | HTTP/REST + OpenAPI + WebSocket | Server only |
| `gwp` | Enterprise | GQL Wire Protocol (gRPC) | Server only |
| `bolt` | Enterprise | Bolt v5 (Neo4j driver compat) | Server only |
| `studio` | Enterprise | Embedded web UI | Server only |

### Regex

| Atom | Profile | Description | Status |
| --- | --- | --- | --- |
| `regex` | LPG, RDF | Full regex engine | Implemented |
| `regex-lite` | Edge | Lightweight regex for WASM | Implemented |
