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
//! ## Stability
//!
//! Everything this crate exports is stable unless its documentation has a
//! `# Stability: Experimental` section: a stable item breaks only in a minor
//! release (`0.6` to `0.7`), an experimental one may change in any release.
//! See <https://grafeo.dev/stability/> and <https://grafeo.dev/versioning/>.
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
pub use grafeo_engine::{
    AccessMode, Catalog, CatalogError, Config, ConfigError, Direction, DurabilityMode, GrafeoDB,
    Grant, GraphModel, GraphStore, Identity, IndexDefinition, IndexType, Role, Session, Snapshot,
    StatementKind, VERSION,
};

// Re-export what the `Config` methods take, so a configuration needs no other
// crate
#[cfg(feature = "encryption")]
pub use grafeo_common::encryption::KeyChain;
// Derives the master key a `KeyChain` takes from a passphrase
#[cfg(feature = "encryption")]
pub use grafeo_common::encryption::PasswordKeyProvider;
#[cfg(feature = "cdc")]
pub use grafeo_engine::cdc::CdcRetentionConfig;
#[cfg(feature = "encryption")]
pub use grafeo_engine::config::EncryptionConfig;
pub use grafeo_engine::config::StorageFormat;

// Re-export time travel: the retention setting `Config` takes, and the view and
// points in time `GrafeoDB::as_of` works with, in every build.
pub use grafeo_engine::HistoryRetention;
pub use grafeo_engine::{AsOfView, TimePoint};

// Re-export what `GrafeoDB::create_text_index_with` takes
#[cfg(feature = "text-index")]
pub use grafeo_engine::{TextIndexOptions, TokenizerKind};

// Re-export submodules for qualified access (e.g. grafeo::auth::Identity)
pub use grafeo_engine::admin;
pub use grafeo_engine::auth;
#[cfg(feature = "cdc")]
pub use grafeo_engine::cdc;
pub use grafeo_engine::database;
#[cfg(feature = "gql")]
pub use grafeo_engine::import;
pub use grafeo_engine::loader;
// Re-export the syntax `GrafeoDB::load_rdf` reads
#[cfg(feature = "triple-store")]
pub use grafeo_engine::RdfFormat;
pub use grafeo_engine::memory_usage;
pub use grafeo_engine::session;

// Re-export query results
pub use grafeo_engine::database::QueryResult;
pub use grafeo_engine::database::ResultKind;

// Re-export MemoryUsage, ProjectionSpec at top level for convenience
pub use grafeo_engine::MemoryUsage;
pub use grafeo_engine::ProjectionSpec;

// Re-export core types - you'll need these for working with IDs and values
pub use grafeo_common::types::{EdgeId, NodeId, Value};

// Re-export error types so users don't need to depend on grafeo-common directly
pub use grafeo_common::utils::error::{Error, Result};

/// The store contract: what a store of one's own implements to hold a
/// database's graph, for [`GrafeoDB::with_store`] (the read side
/// [`GraphStore`] and the write side
/// [`GraphBackend`](store::GraphBackend)) or [`GrafeoDB::with_read_store`]
/// (the read side only), with the types their methods take. The built-in
/// [`RowGroupStore`](store::RowGroupStore) is a store to delegate to.
///
/// With the `conformance` feature, [`conformance_suite!`](store::conformance_suite)
/// runs the suite the built-in store passes against a store of one's own,
/// one test per case. See <https://grafeo.dev/user-guide/extending/storage-backends/>.
///
/// # Stability: Experimental
///
/// Calling the traits is stable. Implementing them, and the conformance
/// suite, are experimental through 0.6.x: a release may add a required
/// method or a case.
pub mod store {
    pub use grafeo_common::change::{
        Before, Change, ChangeMark, DataOp, Entity, Labels, PendingVersion, Properties,
    };
    pub use grafeo_common::types::{
        ArcStr, EdgeId, EpochId, NodeId, PropertyKey, ReadView, TransactionId, Value,
    };
    pub use grafeo_common::utils::error::{Error, Result};
    pub use grafeo_core::graph::apply::{Applied, ApplyError, Writer};
    pub use grafeo_core::graph::lpg::{Edge, Node};
    pub use grafeo_core::graph::rowgroup::RowGroupStore;
    pub use grafeo_core::graph::{BackendTransaction, Direction, GraphBackend, GraphStore};
    pub use grafeo_core::statistics::Statistics;

    #[cfg(feature = "conformance")]
    pub use grafeo_core::conformance_suite;
    #[cfg(feature = "conformance")]
    pub use grafeo_core::graph::conformance::{KnownGap, Store, run_named};
}
