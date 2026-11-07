//! Node.js/TypeScript bindings for Grafeo graph database.
//!
//! Use Grafeo from JavaScript with native Rust performance.
//!
//! ```js
//! const { GrafeoDB } = require('@grafeo-db/js');
//!
//! const db = GrafeoDB.create();
//! await db.execute("INSERT (:Person {name: 'Alix'})");
//! const result = await db.execute("MATCH (p:Person) RETURN p.name");
//! console.log(result.toArray());
//! ```

// napi FFI requires unsafe casts between JS value types
#![allow(unsafe_code)]

#[macro_use]
extern crate napi_derive;

mod database;
mod error;
mod graph;
mod query;
mod stream;
mod time_travel;
mod transaction;
mod types;

/// Returns the active SIMD instruction set for vector operations.
#[napi]
pub fn simd_support() -> String {
    grafeo_core::index::vector::simd_support().to_string()
}

/// Returns the Grafeo version.
#[napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Returns the features this package was built with, as Python's
/// `grafeo.features()` lists them: the query languages (`gql`, `cypher`,
/// `sparql`, `gremlin`, `graphql`, `sql-pgq`) and the optional capabilities
/// (`algos`, `vector-index`, `text-index`, `hybrid-search`, `cdc`,
/// `triple-store`, `shacl`, `embed`, `arrow-export`, `jsonl-import`,
/// `parquet-import`, `metrics`). Profiles such as `full` are not listed, only
/// what they enable.
///
/// ```ts
/// if (!features().includes('cypher')) throw new Error('this build has no Cypher')
/// ```
#[napi]
pub fn features() -> Vec<String> {
    grafeo_bindings_common::reported_features!()
        .into_iter()
        .map(str::to_string)
        .collect()
}
