# Roadmap

Grafeo is a high-performance, embeddable graph database written in Rust. This page gives the high-level direction. The detailed, always-current plan is on GitHub: every planned release is a [milestone](https://github.com/GrafeoDB/grafeo/milestones) with its issues, and larger themes are tracked as epics on the [project board](https://github.com/orgs/GrafeoDB/projects/1). Priorities may shift based on community feedback and real-world usage.

For what has shipped, see the [CHANGELOG](changelog.md).

---

## Completed

| Version | Focus |
| --- | --- |
| **0.1** | Foundation: LPG storage with MVCC transactions, WAL persistence, indexes, GQL parser, Python bindings |
| **0.2** | Performance: factorized execution, worst-case optimal joins, lock-free reads, plan caching |
| **0.3** | AI compatibility: vector type, HNSW index, SIMD distance functions, quantization, hybrid graph and vector queries |
| **0.4** | Developer accessibility: Node.js, Go and WASM bindings, SQL/PGQ, CLI, filtered vector search |
| **0.5.0 to 0.5.44** | Beta: text and hybrid search, 25+ graph algorithms callable from queries, schema and constraints, RDF with SPARQL and SHACL, single-file storage, encryption at rest, backup, access control, compact and tiered storage, C#, Dart and C bindings; 0.5.43 was a stabilization release and 0.5.44 a durability and consistency release |
| [**0.6.0**](https://github.com/GrafeoDB/grafeo/milestone/10) | Dependable by design: one storage format with WAL v2, chunked sections past 4 GiB and automatic migration; transactions that own their changes, with snapshot isolation on every read path; a row-group store with incremental checkpoints and online backups; time travel; exact decimals; RDF terms through SPARQL |

---

## Planned

First we fix what 0.6.0 got wrong and prove that its storage holds up. Then comes a faster query engine, the remaining driver, protocol and language work, and binding parity. 0.7.0 is the one remaining release that changes the file format, followed by the features that build on it, and then 1.0.

| Release | Focus |
| --- | --- |
| [**0.6.1**](https://github.com/GrafeoDB/grafeo/milestone/4) | Fixes for problems found in 0.6.0: backups, very long queries, decimal arithmetic, time travel and transaction edge cases, and docs that promise too much |
| [**0.6.2**](https://github.com/GrafeoDB/grafeo/milestone/5) | Stability: test storage and transactions against power loss, damaged files and heavy concurrency, and keep the store within its memory limit |
| [**0.6.3**](https://github.com/GrafeoDB/grafeo/milestone/6) | A new query engine, query memory kept within the limit, and a plan cache that replans when the data changes |
| [**0.6.4**](https://github.com/GrafeoDB/grafeo/milestone/16) | Queries that use several cores where benchmarks show a gain, and real `Serializable` isolation |
| [**0.6.5**](https://github.com/GrafeoDB/grafeo/milestone/7) | Drivers, protocols and languages that need no file format change: the ADBC driver, the SPARQL 1.1 Protocol and Graph Store Protocol, missing GQL syntax, and more complete GQL, Cypher, SQL/PGQ, GraphQL and Gremlin support |
| [**0.6.6**](https://github.com/GrafeoDB/grafeo/milestone/11) | The same API in every binding, simpler feature flags, deeper tests, and a check that every documented guarantee holds |
| [**0.7.0**](https://github.com/GrafeoDB/grafeo/milestone/8) | The data model release: type inheritance, the RDF 1.2 data model and GQL vector types in one file format change, and the removal of everything 0.6 deprecated |
| [**0.7.1**](https://github.com/GrafeoDB/grafeo/milestone/12) | Features on the new data model: SPARQL 1.2, and queries that match a type together with the types that extend it |
| [**0.7.2**](https://github.com/GrafeoDB/grafeo/milestone/13), [**0.7.3**](https://github.com/GrafeoDB/grafeo/milestone/15) | Room for work not planned yet, with no file format changes |
| [**1.0**](https://github.com/GrafeoDB/grafeo/milestone/14) | Stable: no new features, the public API and file format frozen, semantic versioning from here on |

0.7.0 is the last planned file format change before 1.0. A 0.6 database is converted automatically the first time 0.7 opens it for writing, and the old file is kept next to the new one (see [Versioning](versioning.md)). 0.7 no longer opens 0.5.x databases: open each one for writing with 0.6 first.

---

## After 1.0

| Release | Focus |
| --- | --- |
| [**1.1**](https://github.com/GrafeoDB/grafeo/milestone/9) | Enterprise features: per-graph access control with pluggable authentication (JWT, OIDC), enterprise authorization (row-level security, property masking, `GRANT` and `REVOKE`), a reactive event bus, and inbound connectors starting with Kafka |

**Later, not scheduled**: distributed deployment, more language bindings.

---

## Contributing

Issues in the upcoming milestones are a good place to start, especially the ones without an assignee. Check the [GitHub Issues](https://github.com/GrafeoDB/grafeo/issues), join the [Discussions](https://github.com/orgs/GrafeoDB/discussions) or hop into the [Discord server](https://discord.gg/nqU6RUVaxW).

---

Last updated: October 2026
