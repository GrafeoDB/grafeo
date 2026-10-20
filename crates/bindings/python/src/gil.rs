//! Releasing the GIL around long-running engine calls.
//!
//! Queries, commits, algorithms, searches, batches, imports and persistence
//! run with the GIL released, so other Python threads make progress while
//! Rust works. Cheap single-entity calls (get a node, set a property) keep
//! it: releasing and taking it back costs more than they do.
//!
//! The rule that keeps this free of deadlocks: every blocking lock (the
//! database lock, a transaction's session) is taken inside the closure,
//! Python arguments are converted before it and errors become Python
//! exceptions after it. No lock is held while the GIL is taken again.

use pyo3::prelude::*;

use crate::error::PyGrafeoError;

/// Runs `work` with the GIL released and turns its error into a Python
/// exception once the GIL is back.
///
/// `work` must not touch Python objects; it returns a `PyGrafeoError`
/// rather than a `PyErr` because building a Python exception needs the GIL.
pub(crate) fn without_gil<T, F>(py: Python<'_>, work: F) -> PyResult<T>
where
    T: Send,
    F: FnOnce() -> Result<T, PyGrafeoError> + Send,
{
    py.detach(work).map_err(PyErr::from)
}
