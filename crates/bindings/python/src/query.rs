//! Query results and builders for the Python API.

use std::collections::HashMap;

use pyo3::prelude::*;

use grafeo_common::types::Value;

use crate::graph::{PyEdge, PyNode};
use crate::types::PyValue;

/// Results from a GQL query - iterate rows or access nodes and edges directly.
///
/// Iterate with `for row in result:` where each row is a dict. Or use
/// `result.nodes()` and `result.edges()` to get graph elements. For single
/// values, `result.scalar()` grabs the first column of the first row.
///
/// Query performance metrics are available via `execution_time_ms` and
/// `rows_scanned` properties when timing is enabled.
#[pyclass(name = "QueryResult")]
pub struct PyQueryResult {
    pub(crate) columns: Vec<String>,
    pub(crate) rows: Vec<Vec<Value>>,
    pub(crate) nodes: Vec<PyNode>,
    pub(crate) edges: Vec<PyEdge>,
    current_row: usize,
    /// Query execution time in milliseconds.
    pub(crate) execution_time_ms: Option<f64>,
    /// Number of rows scanned during execution.
    pub(crate) rows_scanned: Option<u64>,
}

#[pymethods]
impl PyQueryResult {
    /// Get column names.
    #[getter]
    fn columns(&self) -> Vec<String> {
        self.columns.clone()
    }

    /// Get number of rows.
    fn __len__(&self) -> usize {
        self.rows.len()
    }

    /// Get a row by index.
    fn __getitem__(&self, idx: isize, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let idx = if idx < 0 {
            (self.rows.len() as isize + idx) as usize
        } else {
            idx as usize
        };

        if idx >= self.rows.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "Row index out of range",
            ));
        }

        let row = &self.rows[idx];
        let dict = pyo3::types::PyDict::new(py);
        for (col, val) in self.columns.iter().zip(row.iter()) {
            dict.set_item(col, PyValue::to_py(val, py))?;
        }
        Ok(dict.unbind().into_any())
    }

    /// Iterate over rows.
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Get next row.
    fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> Option<Py<PyAny>> {
        if slf.current_row >= slf.rows.len() {
            return None;
        }

        let idx = slf.current_row;
        slf.current_row += 1;

        let row = slf.rows[idx].clone();
        let columns = slf.columns.clone();

        let dict = pyo3::types::PyDict::new(py);
        for (col, val) in columns.iter().zip(row.iter()) {
            dict.set_item(col, PyValue::to_py(val, py)).ok()?;
        }
        Some(dict.unbind().into_any())
    }

    /// Get all nodes from the result.
    fn nodes(&self) -> Vec<PyNode> {
        self.nodes.clone()
    }

    /// Get all edges from the result.
    fn edges(&self) -> Vec<PyEdge> {
        self.edges.clone()
    }

    /// Convert to list of dictionaries.
    ///
    /// # Panics
    ///
    /// Panics on memory exhaustion during Python list/dict allocation.
    fn to_list(&self, py: Python<'_>) -> Py<PyAny> {
        let list = pyo3::types::PyList::empty(py);
        for row in &self.rows {
            let dict = pyo3::types::PyDict::new(py);
            for (col, val) in self.columns.iter().zip(row.iter()) {
                dict.set_item(col, PyValue::to_py(val, py))
                    .expect("dict.set_item only fails on memory exhaustion");
            }
            list.append(dict)
                .expect("list.append only fails on memory exhaustion");
        }
        list.unbind().into_any()
    }

    /// Get single value (first column of first row).
    fn scalar(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.rows.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("No rows in result"));
        }
        if self.columns.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "No columns in result",
            ));
        }
        Ok(PyValue::to_py(&self.rows[0][0], py))
    }

    /// Query execution time in milliseconds (if available).
    ///
    /// Example:
    /// ```python
    /// result = db.execute("MATCH (n:Person) RETURN n")
    /// if result.execution_time_ms:
    ///     print(f"Query took {result.execution_time_ms:.2f}ms")
    /// ```
    #[getter]
    fn execution_time_ms(&self) -> Option<f64> {
        self.execution_time_ms
    }

    /// Number of rows scanned during query execution (if available).
    ///
    /// Example:
    /// ```python
    /// result = db.execute("MATCH (n:Person) RETURN n")
    /// if result.rows_scanned:
    ///     print(f"Scanned {result.rows_scanned} rows")
    /// ```
    #[getter]
    fn rows_scanned(&self) -> Option<u64> {
        self.rows_scanned
    }

    fn __repr__(&self) -> String {
        let time_str = self
            .execution_time_ms
            .map(|t| format!(", time={:.2}ms", t))
            .unwrap_or_default();
        format!(
            "QueryResult(columns={:?}, rows={}{})",
            self.columns,
            self.rows.len(),
            time_str
        )
    }
}

impl PyQueryResult {
    /// Creates a new query result (used internally).
    pub fn new(
        columns: Vec<String>,
        rows: Vec<Vec<Value>>,
        nodes: Vec<PyNode>,
        edges: Vec<PyEdge>,
    ) -> Self {
        Self {
            columns,
            rows,
            nodes,
            edges,
            current_row: 0,
            execution_time_ms: None,
            rows_scanned: None,
        }
    }

    /// Creates a new query result with execution metrics (used internally).
    pub fn with_metrics(
        columns: Vec<String>,
        rows: Vec<Vec<Value>>,
        nodes: Vec<PyNode>,
        edges: Vec<PyEdge>,
        execution_time_ms: Option<f64>,
        rows_scanned: Option<u64>,
    ) -> Self {
        Self {
            columns,
            rows,
            nodes,
            edges,
            current_row: 0,
            execution_time_ms,
            rows_scanned,
        }
    }

    /// Creates an empty result (used internally).
    pub fn empty() -> Self {
        Self {
            columns: Vec::new(),
            rows: Vec::new(),
            nodes: Vec::new(),
            edges: Vec::new(),
            current_row: 0,
            execution_time_ms: None,
            rows_scanned: None,
        }
    }
}

/// Builds parameterized queries with a fluent API.
///
/// Add parameters with `.param("name", value)` to safely inject values
/// without string concatenation (prevents injection).
#[pyclass(name = "QueryBuilder")]
pub struct PyQueryBuilder {
    pub(crate) query: String,
    pub(crate) params: HashMap<String, Value>,
}

impl PyQueryBuilder {
    /// Creates a new query builder (Rust API).
    pub fn create(query: String) -> Self {
        Self {
            query,
            params: HashMap::new(),
        }
    }
}

#[pymethods]
impl PyQueryBuilder {
    /// Create a new query builder.
    #[new]
    fn new(query: String) -> Self {
        Self::create(query)
    }

    /// Set a parameter.
    fn param(&mut self, name: String, value: &Bound<'_, PyAny>) {
        if let Ok(v) = PyValue::from_py(value) {
            self.params.insert(name, v);
        }
    }

    /// Get the query string.
    #[getter]
    fn query(&self) -> &str {
        &self.query
    }
}
