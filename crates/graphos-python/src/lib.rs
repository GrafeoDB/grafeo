//! Python bindings for Graphos graph database.
//!
//! This crate provides Python bindings via PyO3, exposing the core
//! graph database functionality to Python users.

#![warn(missing_docs)]

use pyo3::prelude::*;

mod bridges;
mod database;
mod error;
mod graph;
mod query;
mod types;

use bridges::{PyAlgorithms, PyNetworkXAdapter, PySolvORAdapter};
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
    m.add_class::<PyAlgorithms>()?;
    m.add_class::<PyNetworkXAdapter>()?;
    m.add_class::<PySolvORAdapter>()?;

    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
