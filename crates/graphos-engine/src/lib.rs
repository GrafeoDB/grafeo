//! # graphos-engine
//!
//! The main entry point for Graphos: database management, transactions,
//! query processing, and optimization.
//!
//! ## Modules
//!
//! - [`database`] - GraphosDB struct and lifecycle management
//! - [`session`] - Session/Connection management
//! - [`config`] - Configuration options
//! - [`transaction`] - Transaction management and MVCC
//! - [`query`] - Query processing, binding, planning, optimization, execution
//! - [`catalog`] - Schema and index catalog

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]

pub mod catalog;
pub mod config;
pub mod database;
pub mod query;
pub mod session;
pub mod transaction;

pub use config::Config;
pub use database::GraphosDB;
pub use session::Session;
