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

---

## Planned: finishing the 0.5 beta

The rest of the 0.5 series makes Grafeo dependable by design: durable and crash-safe persistence, real snapshot isolation, and memory use close to what a dense layout needs, followed by driver, protocol and language completeness.

| Release | Focus |
| --- | --- |
| [**0.5.45**](https://github.com/GrafeoDB/grafeo/milestone/3) | One storage format: WAL v2 and chunked sections without the 4 GiB limit, with automatic migration |
| [**0.5.46**](https://github.com/GrafeoDB/grafeo/milestone/4) | Transactions own their changes: snapshot isolation for properties and labels, one write path; ADBC driver, SPARQL 1.1 Protocol and Graph Store Protocol, GQL vector types |
| [**0.5.47**](https://github.com/GrafeoDB/grafeo/milestone/5) | Compact-core store: a dense memory layout, enforced memory limits, incremental checkpoints |
| [**0.5.48**](https://github.com/GrafeoDB/grafeo/milestone/6) | Push-only execution engine and benchmark-gated parallelism |
| [**0.5.49**](https://github.com/GrafeoDB/grafeo/milestone/7) | API parity across bindings, feature flag cleanup, query language completeness, test depth |

The 0.5.45 storage format change is the only planned migration: older files are converted automatically on open.

---

## Next

| Release | Focus |
| --- | --- |
| [**0.6.0**](https://github.com/GrafeoDB/grafeo/milestone/8) | Release candidate: no new features, blocker review and final audit. If it works in 0.6, it works in 1.0 |
| [**0.7.0**](https://github.com/GrafeoDB/grafeo/milestone/9) | Per-graph access control with pluggable authentication (JWT, OIDC), reactive event bus, removal of the deprecated feature profile names |
| **1.0** | Stable: semantic versioning commitment, public API frozen |

**Later, not scheduled**: enterprise authorization (row-level security, property masking, `GRANT`/`REVOKE`, LDAP and SAML), inbound connectors starting with Kafka, distributed deployment, more language bindings.

---

## Contributing

Issues in the upcoming milestones are a good place to start, especially the ones without an assignee. Check the [GitHub Issues](https://github.com/GrafeoDB/grafeo/issues), join the [Discussions](https://github.com/orgs/GrafeoDB/discussions) or hop into the [Discord server](https://discord.gg/nqU6RUVaxW).

---

Last updated: October 2026
