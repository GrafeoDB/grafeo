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

## Planned: the rest of 0.6

Fixes first, then proof that the storage format holds, then the new execution engine, one last format revision, the language features on top of both, and a feature freeze before the release candidate.

| Release | Focus |
| --- | --- |
| [**0.6.1**](https://github.com/GrafeoDB/grafeo/milestone/4) | Fixes only, for problems found in 0.6.0: backups, long statements, decimal arithmetic, time travel and transaction edge cases, docs that overstate guarantees |
| [**0.6.2**](https://github.com/GrafeoDB/grafeo/milestone/5) | Stability: power-loss, fuzz and concurrency testing of the storage format and transactions, and store memory bounded by the memory limit |
| [**0.6.3**](https://github.com/GrafeoDB/grafeo/milestone/6) | Push-only execution engine with benchmark-gated parallelism, one memory-accounting model for operators, a plan cache bounded in bytes, Serializable that records reads |
| [**0.6.4**](https://github.com/GrafeoDB/grafeo/milestone/7) | The last format revision before 1.0: type inheritance, the RDF 1.2 data model, GQL vector types |
| [**0.6.5**](https://github.com/GrafeoDB/grafeo/milestone/11) | Languages and protocols: SPARQL 1.2, SPARQL 1.1 Protocol and Graph Store Protocol, polymorphic matching in GQL and Cypher, language completeness |
| [**0.6.6**](https://github.com/GrafeoDB/grafeo/milestone/12) | Feature freeze: API parity across bindings, the ADBC driver, feature flag cleanup, test depth, an audit of every documented guarantee, the stability levels for 1.0 |

0.6.4 is the last planned format change before 1.0: files from 0.5.x and earlier 0.6 releases are converted automatically on their first read-write open, and the old file is kept next to the new one (see [Versioning](versioning.md)). After 0.6.4 the format is frozen.

---

## Next

| Release | Focus |
| --- | --- |
| [**0.7.0**](https://github.com/GrafeoDB/grafeo/milestone/8) | Release candidate: no new features, removal of what 0.6.x deprecated, final audit and blocker review. If it works in 0.7, it works in 1.0 |
| [**0.8.0**](https://github.com/GrafeoDB/grafeo/milestone/9) | Per-graph access control with pluggable authentication (JWT, OIDC), reactive event bus |
| **1.0** | Stable: semantic versioning commitment, public API frozen |

**Later, not scheduled**: enterprise authorization (row-level security, property masking, `GRANT`/`REVOKE`, LDAP and SAML), inbound connectors starting with Kafka, distributed deployment, more language bindings.

---

## Contributing

Issues in the upcoming milestones are a good place to start, especially the ones without an assignee. Check the [GitHub Issues](https://github.com/GrafeoDB/grafeo/issues), join the [Discussions](https://github.com/orgs/GrafeoDB/discussions) or hop into the [Discord server](https://discord.gg/nqU6RUVaxW).

---

Last updated: October 2026
