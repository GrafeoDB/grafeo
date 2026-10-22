//! Lazy, cursor-based query results for Node.js.
//!
//! `ResultStream` exposes `GrafeoDB::execute_streaming()` to JavaScript as an
//! async cursor. Each `next()` call pulls one row off the Rust pipeline inside
//! a `spawn_blocking` task so the Node.js event loop stays responsive. Rows
//! arrive as plain JS objects keyed by column name.
//!
//! Iterate manually:
//!
//! ```js
//! const stream = await db.executeStream("MATCH (p:Person) RETURN p.name");
//! let row;
//! while ((row = await stream.next()) !== null) {
//!     console.log(row["p.name"]);
//! }
//! ```
//!
//! Or wrap with an async generator:
//!
//! ```js
//! async function* iterate(stream) {
//!     let row;
//!     while ((row = await stream.next()) !== null) yield row;
//! }
//! for await (const row of iterate(stream)) { ... }
//! ```
//!
//! # Stability: Experimental
//!
//! New in 0.5.40. API may change before being promoted to Beta.

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use parking_lot::{Mutex, RwLock};
use serde_json::{Map as JsonMap, Value as JsonValue};

use grafeo_bindings_common::json::value_to_json;
use grafeo_engine::database::GrafeoDB;
use grafeo_engine::{OwnedResultStream, OwnedRowIterator};

use crate::error::NodeGrafeoError;

/// Async cursor over the rows of a lazy GQL query.
///
/// Obtained from [`GrafeoDB.executeStream()`](super::database::JsGrafeoDB::execute_stream).
/// Holds bounded memory regardless of result size (one chunk at a time).
///
/// # Stability: Experimental
#[napi(js_name = "ResultStream")]
pub struct JsResultStream {
    /// Keepalive: the database Arc must outlive the stream so the operator
    /// tree's shared stores do not get dropped mid-iteration.
    _database: Arc<RwLock<GrafeoDB>>,
    columns: Arc<Vec<String>>,
    /// Behind a `Mutex` because napi's async runtime can park tasks across
    /// threads, and `OwnedRowIterator` is not `Sync`.
    iter: Arc<Mutex<Option<OwnedRowIterator>>>,
}

impl JsResultStream {
    pub(crate) fn new(database: Arc<RwLock<GrafeoDB>>, stream: OwnedResultStream) -> Self {
        let columns = Arc::new(stream.columns().to_vec());
        Self {
            _database: database,
            columns,
            iter: Arc::new(Mutex::new(Some(stream.into_row_iter()))),
        }
    }
}

#[napi]
impl JsResultStream {
    /// Column names in the order they appear in each row object.
    #[napi(getter)]
    pub fn columns(&self) -> Vec<String> {
        (*self.columns).clone()
    }

    /// Pulls the next row as a plain object, or `null` when the stream is
    /// exhausted. Runs the Rust work on a blocking thread so the Node.js
    /// event loop stays responsive.
    ///
    /// Values follow the Grafeo JSON convention: strings, numbers, booleans,
    /// and `null` map directly; temporal and complex types use the tagged
    /// form documented in `grafeo-bindings-common::json` (e.g. `{ "$date": ... }`).
    #[napi]
    pub async fn next(&self) -> Result<Option<JsonValue>> {
        let iter = Arc::clone(&self.iter);
        let columns = Arc::clone(&self.columns);

        tokio::task::spawn_blocking(move || -> Result<Option<JsonValue>> {
            let mut guard = iter.lock();
            let Some(iter_mut) = guard.as_mut() else {
                return Ok(None);
            };
            match iter_mut.next() {
                Some(Ok(values)) => {
                    let mut obj = JsonMap::with_capacity(columns.len());
                    for (col, val) in columns.iter().zip(values.iter()) {
                        obj.insert(col.clone(), value_to_json(val));
                    }
                    Ok(Some(JsonValue::Object(obj)))
                }
                Some(Err(err)) => {
                    *guard = None;
                    Err(NodeGrafeoError::from(err).into())
                }
                None => {
                    *guard = None;
                    Ok(None)
                }
            }
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Explicitly releases the underlying operator tree. After calling this,
    /// `next()` returns `null`. Useful when you want to break out of iteration
    /// early without waiting for GC.
    ///
    /// Async because acquiring the iterator mutex (and dropping the operator
    /// tree it holds) can block while an in-flight `next()` completes; doing
    /// that on the Node.js main thread would stall the event loop.
    #[napi]
    pub async fn close(&self) -> Result<()> {
        let iter = Arc::clone(&self.iter);
        tokio::task::spawn_blocking(move || {
            *iter.lock() = None;
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))
    }
}
