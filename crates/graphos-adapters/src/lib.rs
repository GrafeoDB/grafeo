//! # graphos-adapters
//!
//! Adapters layer for Graphos: storage backends, query language parsers,
//! and plugin interfaces.
//!
//! ## Modules
//!
//! - [`storage`] - Storage backends (memory, mmap, WAL)
//! - [`query`] - Query language parsers (GQL, Cypher)
//! - [`plugins`] - Plugin system and bridges

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]

pub mod plugins;
pub mod query;
pub mod storage;
