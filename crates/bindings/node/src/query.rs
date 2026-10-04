//! Query results for the Node.js API.

use napi::bindgen_prelude::*;
use napi::sys;
use napi_derive::napi;

use grafeo_common::types::Value;

use crate::graph::{JsEdge, JsNode};
use crate::types;

/// What a query's writes changed.
#[napi(object)]
pub struct WriteCounters {
    /// Nodes created, by `INSERT`, `CREATE` or `MERGE`.
    pub nodes_created: i64,
    /// Nodes deleted.
    pub nodes_deleted: i64,
    /// Edges created.
    pub edges_created: i64,
    /// Edges deleted, also those `DETACH DELETE` removes.
    pub edges_deleted: i64,
    /// Property values written or removed, also those of created entities.
    pub properties_set: i64,
    /// Labels added, also those of created nodes.
    pub labels_added: i64,
    /// Labels removed.
    pub labels_removed: i64,
}

/// Results from a query - access rows, nodes, and edges.
#[napi]
pub struct QueryResult {
    pub(crate) columns: Vec<String>,
    pub(crate) rows: Vec<Vec<Value>>,
    pub(crate) nodes: Vec<JsNode>,
    pub(crate) edges: Vec<JsEdge>,
    pub(crate) execution_time_ms: Option<f64>,
    pub(crate) rows_scanned: Option<u64>,
    pub(crate) counters: grafeo_engine::database::WriteCounters,
}

#[napi]
impl QueryResult {
    /// Get column names.
    #[napi(getter)]
    pub fn columns(&self) -> Vec<String> {
        self.columns.clone()
    }

    /// Get number of rows.
    #[napi(getter)]
    pub fn length(&self) -> u32 {
        // reason: Result sets are bounded by graph size, well within u32::MAX
        #[allow(clippy::cast_possible_truncation)]
        let len = self.rows.len() as u32;
        len
    }

    /// Query execution time in milliseconds (if available).
    #[napi(getter, js_name = "executionTimeMs")]
    pub fn execution_time_ms(&self) -> Option<f64> {
        self.execution_time_ms
    }

    /// Number of rows scanned during execution (if available).
    #[napi(getter, js_name = "rowsScanned")]
    pub fn rows_scanned(&self) -> Option<f64> {
        self.rows_scanned.map(|r| r as f64)
    }

    /// What the query's writes changed: nodes and edges created and deleted,
    /// properties set, labels added and removed.
    #[napi(getter)]
    pub fn counters(&self) -> WriteCounters {
        let c = &self.counters;
        let count = |n: u64| i64::try_from(n).unwrap_or(i64::MAX);
        WriteCounters {
            nodes_created: count(c.nodes_created),
            nodes_deleted: count(c.nodes_deleted),
            edges_created: count(c.edges_created),
            edges_deleted: count(c.edges_deleted),
            properties_set: count(c.properties_set),
            labels_added: count(c.labels_added),
            labels_removed: count(c.labels_removed),
        }
    }

    /// Get a single row by index as a plain object.
    #[napi]
    pub fn get(&self, env: Env, index: u32) -> Result<Object<'_>> {
        let idx = index as usize;
        if idx >= self.rows.len() {
            return Err(napi::Error::new(
                napi::Status::InvalidArg,
                "Row index out of range",
            ));
        }
        self.row_to_object(env.raw(), idx)
    }

    /// Get all rows as an array of objects.
    #[napi(js_name = "toArray")]
    pub fn to_array(&self, env: Env) -> Result<Vec<Object<'_>>> {
        let mut result = Vec::with_capacity(self.rows.len());
        for i in 0..self.rows.len() {
            result.push(self.row_to_object(env.raw(), i)?);
        }
        Ok(result)
    }

    /// Get first column of first row (single value).
    #[napi]
    pub fn scalar(&self, env: Env) -> Result<Unknown<'_>> {
        if self.rows.is_empty() {
            return Err(napi::Error::new(
                napi::Status::GenericFailure,
                "No rows in result",
            ));
        }
        if self.columns.is_empty() {
            return Err(napi::Error::new(
                napi::Status::GenericFailure,
                "No columns in result",
            ));
        }
        types::value_to_js(env.raw(), &self.rows[0][0])
    }

    /// Get nodes found in the result.
    #[napi]
    pub fn nodes(&self) -> Vec<JsNode> {
        self.nodes.clone()
    }

    /// Get edges found in the result.
    #[napi]
    pub fn edges(&self) -> Vec<JsEdge> {
        self.edges.clone()
    }

    /// Returns the result formatted as a Unicode table.
    #[napi(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        grafeo_common::fmt::format_result_table(
            &self.columns,
            &self.rows,
            self.execution_time_ms,
            None,
        )
    }

    /// Get all rows as an array of arrays (no column names).
    #[napi]
    pub fn rows(&self, env: Env) -> Result<Object<'_>> {
        let env_raw = env.raw();
        let mut arr = std::ptr::null_mut();
        // SAFETY: env_raw is valid; napi_create_array_with_length writes to our out-pointer
        types::check_napi(unsafe {
            sys::napi_create_array_with_length(env_raw, self.rows.len(), &raw mut arr)
        })?;
        for (i, row) in self.rows.iter().enumerate() {
            let mut row_arr = std::ptr::null_mut();
            // SAFETY: env_raw is valid; napi_create_array_with_length writes to our out-pointer
            types::check_napi(unsafe {
                sys::napi_create_array_with_length(env_raw, row.len(), &raw mut row_arr)
            })?;
            for (j, val) in row.iter().enumerate() {
                let napi_val = types::value_to_napi(env_raw, val)?;
                // SAFETY: env_raw, row_arr, and napi_val are valid napi values
                // reason: JS arrays are limited to 2^32-1 elements
                #[allow(clippy::cast_possible_truncation)]
                types::check_napi(unsafe {
                    sys::napi_set_element(env_raw, row_arr, j as u32, napi_val)
                })?;
            }
            // SAFETY: env_raw, arr, and row_arr are valid napi values
            // reason: JS arrays are limited to 2^32-1 elements
            #[allow(clippy::cast_possible_truncation)]
            types::check_napi(unsafe { sys::napi_set_element(env_raw, arr, i as u32, row_arr) })?;
        }
        Ok(Object::from_raw(env_raw, arr))
    }
}

// A separate block: napi-derive registers every method of a `#[napi]`
// impl, so a method behind a cfg needs a block behind that cfg.
#[cfg(feature = "arrow-export")]
#[napi]
impl QueryResult {
    /// Returns the result as Arrow IPC stream bytes (Buffer).
    ///
    /// Use with the `apache-arrow` npm package:
    /// ```js
    /// import { tableFromIPC } from 'apache-arrow';
    /// const table = tableFromIPC(result.toArrowIPC());
    /// ```
    #[napi(js_name = "toArrowIPC")]
    pub fn to_arrow_ipc(&self) -> Result<napi::bindgen_prelude::Buffer> {
        let col_types = vec![grafeo_common::LogicalType::Any; self.columns.len()];
        let batch = grafeo_engine::database::arrow::query_result_to_record_batch(
            &self.columns,
            &col_types,
            &self.rows,
        )
        .map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("Arrow export failed: {e}"),
            )
        })?;
        let ipc_bytes = grafeo_engine::database::arrow::record_batch_to_ipc_stream(&batch)
            .map_err(|e| {
                napi::Error::new(
                    napi::Status::GenericFailure,
                    format!("Arrow IPC failed: {e}"),
                )
            })?;
        Ok(ipc_bytes.into())
    }
}

impl QueryResult {
    /// Convert a row to a JS object with column names as keys.
    fn row_to_object(&self, env: sys::napi_env, idx: usize) -> Result<Object<'_>> {
        let row = &self.rows[idx];
        let mut raw_obj = std::ptr::null_mut();
        // SAFETY: env is valid; napi_create_object writes to our out-pointer
        types::check_napi(unsafe { sys::napi_create_object(env, &raw mut raw_obj) })?;
        let mut obj = Object::from_raw(env, raw_obj);
        for (col, val) in self.columns.iter().zip(row.iter()) {
            let val_raw = types::value_to_napi(env, val)?;
            // SAFETY: env and val_raw are valid napi values produced by value_to_napi
            let val_unknown = unsafe { Unknown::from_raw_unchecked(env, val_raw) };
            obj.set_named_property(col, val_unknown)?;
        }
        Ok(obj)
    }

    pub fn new(
        columns: Vec<String>,
        rows: Vec<Vec<Value>>,
        nodes: Vec<JsNode>,
        edges: Vec<JsEdge>,
    ) -> Self {
        Self {
            columns,
            rows,
            nodes,
            edges,
            execution_time_ms: None,
            rows_scanned: None,
            counters: grafeo_engine::database::WriteCounters::default(),
        }
    }

    pub fn with_metrics(
        columns: Vec<String>,
        rows: Vec<Vec<Value>>,
        nodes: Vec<JsNode>,
        edges: Vec<JsEdge>,
        execution_time_ms: Option<f64>,
        rows_scanned: Option<u64>,
    ) -> Self {
        Self {
            columns,
            rows,
            nodes,
            edges,
            execution_time_ms,
            rows_scanned,
            counters: grafeo_engine::database::WriteCounters::default(),
        }
    }

    /// Sets the counters of the query's writes.
    #[must_use]
    pub(crate) fn with_counters(
        mut self,
        counters: grafeo_engine::database::WriteCounters,
    ) -> Self {
        self.counters = counters;
        self
    }

    pub fn empty() -> Self {
        Self {
            columns: Vec::new(),
            rows: Vec::new(),
            nodes: Vec::new(),
            edges: Vec::new(),
            execution_time_ms: None,
            rows_scanned: None,
            counters: grafeo_engine::database::WriteCounters::default(),
        }
    }
}
