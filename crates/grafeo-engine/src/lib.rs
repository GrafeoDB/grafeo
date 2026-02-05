//! # grafeo-engine
//!
//! The engine behind Grafeo. You'll find everything here for creating databases,
//! running queries, and managing transactions.
//!
//! Most users should start with the main `grafeo` crate, which re-exports the
//! key types. If you're here directly, [`GrafeoDB`] is your entry point.
//!
//! ## Modules
//!
//! - [`database`] - Create and manage databases with [`GrafeoDB`]
//! - [`session`] - Lightweight handles for concurrent access
//! - [`config`] - Tune memory, threads, and durability settings
//! - [`transaction`] - MVCC transaction management (snapshot isolation)
//! - [`query`] - The full query pipeline: parsing, planning, optimization, execution
//! - [`catalog`] - Schema metadata: labels, property keys, indexes
//! - [`admin`] - Admin API types for inspection, backup, and maintenance

pub mod admin;
pub mod catalog;
pub mod config;
pub mod database;
pub mod query;
pub mod session;
pub mod transaction;

pub use admin::{
    CompactionStats, DatabaseInfo, DatabaseMode, DatabaseStats, DumpFormat, DumpMetadata,
    IndexInfo, LpgSchemaInfo, RdfSchemaInfo, SchemaInfo, ValidationError, ValidationResult,
    ValidationWarning, WalStatus,
};
pub use catalog::{Catalog, CatalogError, IndexDefinition, IndexType};
pub use config::Config;
pub use database::GrafeoDB;
pub use session::Session;
