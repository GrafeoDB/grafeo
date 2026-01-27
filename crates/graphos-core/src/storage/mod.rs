//! Storage utilities for graph data.
//!
//! This module provides compression and encoding utilities:
//!
//! - [`dictionary`] - Dictionary encoding for strings

pub mod dictionary;

pub use dictionary::{DictionaryBuilder, DictionaryEncoding};
