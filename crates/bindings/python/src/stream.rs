//! Lazy, cursor-based query results for Python.
//!
//! `PyResultStream` exposes `GrafeoDB::execute_streaming()` to Python as an
//! iterator that yields row dicts on demand. Pulling a chunk releases the
//! GIL so other Python threads make progress while Rust does the work.
//!
//! # Stability: Experimental
//!
//! New in 0.5.40. It may change or be removed in any release, with a
//! CHANGELOG entry: see <https://grafeo.dev/stability/>.

use std::sync::Arc;

use parking_lot::RwLock;
use pyo3::prelude::*;

use grafeo_engine::OwnedRowIterator;
use grafeo_engine::database::GrafeoDB;

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
    /// The warnings raised through `warnings.warn` so far: at the open, and
    /// once the stream is exhausted; what the stream had when it ended.
    warned: Vec<grafeo_bindings_common::warning::Warning>,
}

impl PyResultStream {
    /// The stream `execute_lazy()` returns, a GQL query's (only a build with
    /// GQL has it).
    #[cfg(feature = "gql")]
    pub(crate) fn new(
        database: Arc<RwLock<GrafeoDB>>,
        stream: grafeo_engine::OwnedResultStream,
    ) -> Self {
        let columns = stream.columns().to_vec();
        Self {
            _database: database,
            columns,
            iter: Some(stream.into_row_iter()),
            warned: Vec::new(),
        }
    }

    /// The warnings the statement raised so far.
    fn current(&self) -> Vec<grafeo_bindings_common::warning::Warning> {
        match &self.iter {
            Some(iter) => grafeo_bindings_common::warning::of_warnings(&iter.warnings()),
            None => self.warned.clone(),
        }
    }

    /// Raises through `warnings.warn` each warning raised since the last
    /// call, once.
    ///
    /// # Errors
    ///
    /// A warning a filter turned into an exception.
    pub(crate) fn warn_new(&mut self, py: Python<'_>) -> PyResult<()> {
        let current = self.current();
        let new: Vec<_> = current
            .iter()
            .filter(|warning| !self.warned.contains(warning))
            .cloned()
            .collect();
        self.warned = current;
        crate::warning::raise(py, &new)
    }
}

#[pymethods]
impl PyResultStream {
    /// Column names in the order they appear in each row dict.
    #[getter]
    fn columns(&self) -> Vec<String> {
        self.columns.clone()
    }

    /// The warnings the statement raised so far, ordered by code: its
    /// translation's from the start, its run's as the rows that raise them
    /// are pulled, all of them once the stream is exhausted. The ones there
    /// at the open are raised through `warnings.warn` by `execute_lazy`,
    /// the rest when the stream is exhausted.
    #[getter]
    fn warnings(&self) -> Vec<crate::warning::PyQueryWarning> {
        crate::warning::to_py(&self.current())
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
                    dict.set_item(col, PyValue::to_py(val, py)?)?;
                }
                Ok(Some(dict.unbind().into_any()))
            }
            Some(Err(err)) => {
                // Keep what was raised, then drop the iterator so subsequent
                // __next__ returns cleanly.
                let current = slf.current();
                slf.warned = current;
                slf.iter = None;
                Err(PyGrafeoError::from(err).into())
            }
            None => {
                slf.warn_new(py)?;
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
