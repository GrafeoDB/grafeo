//! # Grafeo
//!
//! A high-performance, embeddable graph database with a Rust core and no required
//! C dependencies. Optional allocators (jemalloc/mimalloc) and TLS use C libraries
//! for performance.
//!
//! If you're new here, start with [`GrafeoDB`] - that's your entry point for
//! creating databases and running queries. Grafeo uses GQL (the ISO standard)
//! by default, but you can enable other query languages through feature flags.
//!
//! ## Query Languages
//!
//! | Feature | Language | Notes |
//! | ------- | -------- | ----- |
//! | `gql` | GQL | ISO standard, enabled by default |
//! | `cypher` | Cypher | Neo4j-style queries |
//! | `sparql` | SPARQL | For RDF triple stores |
//! | `gremlin` | Gremlin | Apache TinkerPop traversals |
//! | `graphql` | GraphQL | Schema-based queries |
//! | `sql-pgq` | SQL/PGQ | SQL:2023 GRAPH_TABLE |
//!
//! Use the `full` feature to enable everything.
//!
//! ## Quick Start
//!
//! ```rust
//! use grafeo::GrafeoDB;
//!
//! // Create an in-memory database
//! let db = GrafeoDB::new_in_memory();
//! let mut session = db.session();
//!
//! // Add a person
//! session.execute("INSERT (:Person {name: 'Alix', age: 30})")?;
//!
//! // Find them
//! let result = session.execute("MATCH (p:Person) RETURN p.name")?;
//! # Ok::<(), grafeo::Error>(())
//! ```
//!
#![forbid(unsafe_code)]

//! ## Performance Features
//!
//! Enable platform-optimized memory allocators for 10-20% faster allocations:
//!
//! - `jemalloc` - Linux/macOS (x86_64, aarch64)
//! - `mimalloc-allocator` - Windows

// Platform-optimized memory allocators (enabled via features)
// jemalloc: Linux/macOS x86_64/aarch64 - better multi-threaded performance
// mimalloc: Windows - optimized for Windows, better than MSVC allocator
#[cfg(all(
    feature = "jemalloc",
    not(target_os = "windows"),
    not(target_os = "openbsd"),
    not(target_env = "musl"),
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(all(feature = "mimalloc-allocator", target_os = "windows"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// Re-export the main database API
#[cfg(feature = "vector-index")]
pub use grafeo_engine::IndexedVectorRead;
pub use grafeo_engine::{
    AccessMode, Catalog, CatalogError, Config, ConfigError, DurabilityMode, GrafeoDB, Grant,
    GraphModel, GraphStore, GraphStoreMut, Identity, IndexDefinition, IndexType, Role, Session,
    StatementKind, VERSION,
};

// Re-export submodules for qualified access (e.g. grafeo::auth::Identity)
pub use grafeo_engine::admin;
pub use grafeo_engine::auth;
#[cfg(feature = "cdc")]
pub use grafeo_engine::cdc;
pub use grafeo_engine::database;
pub use grafeo_engine::memory_usage;
pub use grafeo_engine::session;

// Re-export query results
pub use grafeo_engine::database::QueryResult;

// Re-export MemoryUsage, ProjectionSpec at top level for convenience
pub use grafeo_engine::MemoryUsage;
pub use grafeo_engine::ProjectionSpec;

// Re-export core types - you'll need these for working with IDs and values
pub use grafeo_common::types::{EdgeId, NodeId, Value};

// Re-export error types so users don't need to depend on grafeo-common directly
pub use grafeo_common::utils::error::{Error, Result};
