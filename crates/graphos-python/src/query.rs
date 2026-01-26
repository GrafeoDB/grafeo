//! Python query interface.

use pyo3::prelude::*;
use std::collections::HashMap;

use graphos_common::types::Value;

use crate::graph::{PyEdge, PyNode};
use crate::types::PyValue;

/// Query result wrapper.
#[pyclass(name = "QueryResult")]
pub struct PyQueryResult {
    pub(crate) columns: Vec<String>,
    pub(crate) rows: Vec<Vec<Value>>,
    pub(crate) nodes: Vec<PyNode>,
    pub(crate) edges: Vec<PyEdge>,
    current_row: usize,
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
    fn __getitem__(&self, idx: isize, py: Python<'_>) -> PyResult<PyObject> {
        let idx = if idx < 0 {
            (self.rows.len() as isize + idx) as usize
        } else {
            idx as usize
        };

        if idx >= self.rows.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err("Row index out of range"));
        }

        let row = &self.rows[idx];
        let dict = pyo3::types::PyDict::new(py);
        for (col, val) in self.columns.iter().zip(row.iter()) {
            dict.set_item(col, PyValue::to_py(val, py))?;
        }
        Ok(dict.into())
    }

    /// Iterate over rows.
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Get next row.
    fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> Option<PyObject> {
        if slf.current_row >= slf.rows.len() {
            return None;
        }

        let row = &slf.rows[slf.current_row];
        slf.current_row += 1;

        let dict = pyo3::types::PyDict::new(py);
        for (col, val) in slf.columns.iter().zip(row.iter()) {
            dict.set_item(col, PyValue::to_py(val, py)).ok()?;
        }
        Some(dict.into())
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
    fn to_list(&self, py: Python<'_>) -> PyObject {
        let list = pyo3::types::PyList::empty(py);
        for row in &self.rows {
            let dict = pyo3::types::PyDict::new(py);
            for (col, val) in self.columns.iter().zip(row.iter()) {
                dict.set_item(col, PyValue::to_py(val, py)).unwrap();
            }
            list.append(dict).unwrap();
        }
        list.into()
    }

    /// Get single value (first column of first row).
    fn scalar(&self, py: Python<'_>) -> PyResult<PyObject> {
        if self.rows.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("No rows in result"));
        }
        if self.columns.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("No columns in result"));
        }
        Ok(PyValue::to_py(&self.rows[0][0], py))
    }

    fn __repr__(&self) -> String {
        format!(
            "QueryResult(columns={:?}, rows={})",
            self.columns,
            self.rows.len()
        )
    }
}

impl PyQueryResult {
    /// Create a new query result.
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
        }
    }

    /// Create an empty result.
    pub fn empty() -> Self {
        Self {
            columns: Vec::new(),
            rows: Vec::new(),
            nodes: Vec::new(),
            edges: Vec::new(),
            current_row: 0,
        }
    }
}

/// Query builder for fluent API.
#[pyclass(name = "QueryBuilder")]
pub struct PyQueryBuilder {
    pub(crate) query: String,
    pub(crate) params: HashMap<String, Value>,
}

#[pymethods]
impl PyQueryBuilder {
    /// Create a new query builder.
    #[new]
    fn new(query: String) -> Self {
        Self {
            query,
            params: HashMap::new(),
        }
    }

    /// Set a parameter.
    fn param(mut slf: PyRefMut<'_, Self>, name: String, value: &Bound<'_, PyAny>) -> PyRefMut<'_, Self> {
        if let Ok(v) = PyValue::from_py(value) {
            slf.params.insert(name, v);
        }
        slf
    }

    /// Get the query string.
    #[getter]
    fn query(&self) -> &str {
        &self.query
    }
}
