//! Lazy, cursor-based query results for Python.
//!
//! `PyResultStream` exposes `GrafeoDB::execute_streaming()` to Python as an
//! iterator that yields row dicts on demand. Pulling a chunk releases the
//! GIL so other Python threads make progress while Rust does the work.
//!
//! # Stability: Experimental
//!
//! New in 0.5.40. API may change before being promoted to Beta.

use std::sync::Arc;

use parking_lot::RwLock;
use pyo3::prelude::*;

use grafeo_engine::database::GrafeoDB;
use grafeo_engine::{OwnedResultStream, OwnedRowIterator};

use crate::error::PyGrafeoError;
use crate::types::PyValue;

/// Iterator over the rows of a lazy query.
///
/// Returned by [`GrafeoDB.execute_lazy()`](super::database::PyGrafeoDB::execute_lazy).
/// Iterate with `for row in stream:`; each row is a dict keyed by column name.
/// Memory usage is bounded by one chunk (~2048 rows) at a time regardless of
/// the total result size.
///
/// # Stability: Experimental
#[pyclass(name = "ResultStream")]
pub struct PyResultStream {
    /// Keepalive for the database: underlying stores must outlive the stream.
    _database: Arc<RwLock<GrafeoDB>>,
    columns: Vec<String>,
    iter: Option<OwnedRowIterator>,
}

impl PyResultStream {
    pub(crate) fn new(database: Arc<RwLock<GrafeoDB>>, stream: OwnedResultStream) -> Self {
        let columns = stream.columns().to_vec();
        Self {
            _database: database,
            columns,
            iter: Some(stream.into_row_iter()),
        }
    }
}

#[pymethods]
impl PyResultStream {
    /// Column names in the order they appear in each row dict.
    #[getter]
    fn columns(&self) -> Vec<String> {
        self.columns.clone()
    }

    /// Returns self for `iter()`.
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Pulls the next row from the pipeline.
    ///
    /// Releases the GIL while Rust walks the operator tree, then reacquires
    /// it to construct the row dict. Returns `None` when the stream is
    /// exhausted (which Python's iterator protocol turns into `StopIteration`).
    fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let Some(iter) = slf.iter.as_mut() else {
            return Ok(None);
        };

        // Pull the next row with the GIL released. The row materializes as a
        // `Vec<Value>` (pure Rust data), which we convert to a dict under the
        // GIL below.
        let row = py.detach(|| iter.next());

        match row {
            Some(Ok(values)) => {
                let dict = pyo3::types::PyDict::new(py);
                for (col, val) in slf.columns.iter().zip(values.iter()) {
                    dict.set_item(col, PyValue::to_py(val, py))?;
                }
                Ok(Some(dict.unbind().into_any()))
            }
            Some(Err(err)) => {
                // Drop the iterator so subsequent __next__ returns cleanly.
                slf.iter = None;
                Err(PyGrafeoError::from(err).into())
            }
            None => {
                slf.iter = None;
                Ok(None)
            }
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ResultStream(columns={:?}, exhausted={})",
            self.columns,
            self.iter.is_none()
        )
    }
}
