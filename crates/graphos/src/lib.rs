//! # Graphos
//!
//! A high-performance, pure-Rust, embeddable graph database.
//!
//! Graphos supports multiple query languages through feature flags:
//!
//! - **GQL** (ISO standard, default) — `gql` feature
//! - **Cypher** — `cypher` feature
//! - **SPARQL** — `sparql` feature
//! - **Gremlin** — `gremlin` feature
//! - **GraphQL** — `graphql` feature
//!
//! Enable all with the `full` feature.
//!
//! ## Quick Start
//!
//! ```rust
//! use graphos::GraphosDB;
//!
//! let db = GraphosDB::new_in_memory();
//! let session = db.session();
//! let result = session.execute("INSERT (:Person {name: 'Alice'})");
//! ```

// Re-export the main database API
pub use graphos_engine::{Catalog, CatalogError, Config, GraphosDB, IndexDefinition, IndexType, Session};

// Re-export core types for advanced usage
pub use graphos_common::types::{EdgeId, NodeId, Value};
