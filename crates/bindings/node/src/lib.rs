//! Node.js/TypeScript bindings for Grafeo graph database.
//!
//! Use Grafeo from JavaScript with native Rust performance.
//!
//! ```js
//! const { GrafeoDB } = require('@grafeo-db/js');
//!
//! const db = GrafeoDB.create();
//! await db.execute("INSERT (:Person {name: 'Alice'})");
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
