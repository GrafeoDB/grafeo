//! # grafeo-adapters
//!
//! The integration layer - parsers, storage backends, and plugins.
//!
//! This is where external formats meet Grafeo's internal representation.
//! You probably don't need this crate directly unless you're extending Grafeo.
//!
//! ## Modules
//!
//! - [`query`] - Parsers for GQL, Cypher, SPARQL, Gremlin, GraphQL
//! - [`storage`] - Persistence: write-ahead log, memory-mapped files
//! - [`plugins`] - Extension points for custom functions and algorithms

pub mod plugins;
pub mod query;
pub mod storage;
