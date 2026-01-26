//! Python bindings for Graphos graph database.
//!
//! This crate provides Python bindings via PyO3, exposing the core
//! graph database functionality to Python users.

use pyo3::prelude::*;

mod database;
mod error;
mod graph;
mod query;
mod types;

use database::PyGraphosDB;
use graph::{PyEdge, PyNode};
use query::PyQueryResult;
use types::PyValue;

/// Graphos Python module.
#[pymodule]
fn graphos(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraphosDB>()?;
    m.add_class::<PyNode>()?;
    m.add_class::<PyEdge>()?;
    m.add_class::<PyQueryResult>()?;
    m.add_class::<PyValue>()?;

    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
