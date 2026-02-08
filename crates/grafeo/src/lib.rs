//! # Grafeo
//!
//! A high-performance, pure-Rust, embeddable graph database.
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
//! session.execute("INSERT (:Person {name: 'Alice', age: 30})")?;
//!
//! // Find them
//! let result = session.execute("MATCH (p:Person) RETURN p.name")?;
//! # Ok::<(), grafeo_common::utils::error::Error>(())
//! ```
//!
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
pub use grafeo_engine::{
    Catalog, CatalogError, Config, ConfigError, DurabilityMode, GrafeoDB, GraphModel,
    IndexDefinition, IndexType, Session,
};

// Re-export core types - you'll need these for working with IDs and values
pub use grafeo_common::types::{EdgeId, NodeId, Value};
