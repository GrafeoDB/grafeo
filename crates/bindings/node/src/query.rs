//! Query results for the Node.js API.

use napi::bindgen_prelude::*;
use napi::{JsObject, JsUnknown};
use napi_derive::napi;

use grafeo_common::types::Value;

use crate::graph::{JsEdge, JsNode};
use crate::types::value_to_js;

/// Results from a query - access rows, nodes, and edges.
#[napi]
pub struct QueryResult {
    pub(crate) columns: Vec<String>,
    pub(crate) rows: Vec<Vec<Value>>,
    pub(crate) nodes: Vec<JsNode>,
    pub(crate) edges: Vec<JsEdge>,
    pub(crate) execution_time_ms: Option<f64>,
    pub(crate) rows_scanned: Option<u64>,
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
        self.rows.len() as u32
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

    /// Get a single row by index as a plain object.
    #[napi]
    pub fn get(&self, env: Env, index: u32) -> Result<JsObject> {
        let idx = index as usize;
        if idx >= self.rows.len() {
            return Err(napi::Error::new(
                napi::Status::InvalidArg,
                "Row index out of range",
            ));
        }
        self.row_to_object(&env, idx)
    }

    /// Get all rows as an array of objects.
    #[napi(js_name = "toArray")]
    pub fn to_array(&self, env: Env) -> Result<Vec<JsObject>> {
        let mut result = Vec::with_capacity(self.rows.len());
        for i in 0..self.rows.len() {
            result.push(self.row_to_object(&env, i)?);
        }
        Ok(result)
    }

    /// Get first column of first row (single value).
    #[napi]
    pub fn scalar(&self, env: Env) -> Result<JsUnknown> {
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
        value_to_js(&env, &self.rows[0][0])
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

    /// Get all rows as an array of arrays (no column names).
    #[napi]
    pub fn rows(&self, env: Env) -> Result<JsObject> {
        let mut arr = env.create_array_with_length(self.rows.len())?;
        for (i, row) in self.rows.iter().enumerate() {
            let mut row_arr = env.create_array_with_length(row.len())?;
            for (j, val) in row.iter().enumerate() {
                row_arr.set_element(j as u32, value_to_js(&env, val)?)?;
            }
            arr.set_element(i as u32, row_arr)?;
        }
        Ok(arr)
    }
}

impl QueryResult {
    /// Convert a row to a JS object with column names as keys.
    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn row_to_object(&self, env: &Env, idx: usize) -> Result<JsObject> {
        let row = &self.rows[idx];
        let mut obj = env.create_object()?;
        for (col, val) in self.columns.iter().zip(row.iter()) {
            obj.set_named_property(col, value_to_js(env, val)?)?;
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
        }
    }

    pub fn empty() -> Self {
        Self {
            columns: Vec::new(),
            rows: Vec::new(),
            nodes: Vec::new(),
            edges: Vec::new(),
            execution_time_ms: None,
            rows_scanned: None,
        }
    }
}
