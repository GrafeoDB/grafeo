#![forbid(unsafe_code)]
//! Shared abstractions for Grafeo language bindings.
//!
//! This crate provides language-agnostic implementations of common binding
//! logic: entity extraction from query results, error classification, and
//! JSON-to-Value conversion. Each language binding (Python, Node.js, C, WASM)
//! depends on this crate and maps the generic types to its FFI layer.

#[cfg(feature = "cdc")]
pub mod cdc;
pub mod entity;
pub mod error;
pub mod json;
pub mod warning;

/// The features a binding reports as built in (`features()` in Python and
/// Node.js), in this order: each Cargo feature of the calling crate that is
/// on. A macro, so each `cfg!` reads the binding's own features, not this
/// crate's: a binding that uses it defines a feature of every name here.
/// Groups and profiles (`full`, `ai`) are not listed, only what they enable.
#[macro_export]
macro_rules! reported_features {
    () => {
        [
            ("gql", cfg!(feature = "gql")),
            ("cypher", cfg!(feature = "cypher")),
            ("sparql", cfg!(feature = "sparql")),
            ("gremlin", cfg!(feature = "gremlin")),
            ("graphql", cfg!(feature = "graphql")),
            ("sql-pgq", cfg!(feature = "sql-pgq")),
            ("algos", cfg!(feature = "algos")),
            ("vector-index", cfg!(feature = "vector-index")),
            ("text-index", cfg!(feature = "text-index")),
            ("hybrid-search", cfg!(feature = "hybrid-search")),
            ("cdc", cfg!(feature = "cdc")),
            ("triple-store", cfg!(feature = "triple-store")),
            ("shacl", cfg!(feature = "shacl")),
            ("embed", cfg!(feature = "embed")),
            ("arrow-export", cfg!(feature = "arrow-export")),
            ("jsonl-import", cfg!(feature = "jsonl-import")),
            ("parquet-import", cfg!(feature = "parquet-import")),
            ("metrics", cfg!(feature = "metrics")),
        ]
        .into_iter()
        .filter_map(|(name, enabled)| enabled.then_some(name))
        .collect::<::std::vec::Vec<&'static str>>()
    };
}
