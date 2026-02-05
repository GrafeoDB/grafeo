//! Use Grafeo from Python with native Rust performance.
//!
//! You get full access to the graph database through a Pythonic API - same
//! query speed, same durability, with the convenience of Python's ecosystem.
//!
//! ## Quick Start
//!
//! ```python
//! from grafeo import GrafeoDB
//!
//! # Create an in-memory database (or pass a path for persistence)
//! db = GrafeoDB()
//!
//! # Create some people
//! db.execute("INSERT (:Person {name: 'Alice', role: 'Engineer'})")
//! db.execute("INSERT (:Person {name: 'Bob', role: 'Manager'})")
//! db.execute("""
//!     MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
//!     INSERT (a)-[:REPORTS_TO]->(b)
//! """)
//!
//! # Query the graph
//! result = db.execute("MATCH (p:Person)-[:REPORTS_TO]->(m) RETURN p.name, m.name")
//! for row in result:
//!     print(f"{row['p.name']} reports to {row['m.name']}")
//! ```
//!
//! ## Data Science Integration
//!
//! | Library | How to use | Best for |
//! | ------- | ---------- | -------- |
//! | NetworkX | `db.as_networkx().to_networkx()` | Graph visualization, analysis |
//! | pandas | `result.to_list()` then `pd.DataFrame()` | Tabular operations |
//! | solvOR | `db.as_solvor()` | Operations research algorithms |

#![warn(missing_docs)]

use pyo3::prelude::*;

mod bridges;
mod database;
mod error;
mod graph;
mod query;
mod types;

use bridges::{PyAlgorithms, PyNetworkXAdapter, PySolvORAdapter};
use database::{AsyncQueryResult, AsyncQueryResultIter, PyGrafeoDB};
use graph::{PyEdge, PyNode};
use query::PyQueryResult;
use types::PyValue;

/// Returns the active SIMD instruction set for vector operations.
///
/// Useful for debugging and verifying that SIMD acceleration is being used.
///
/// Returns one of: "avx2", "sse", "neon", or "scalar"
///
/// Example:
///     import grafeo
///     print(f"SIMD support: {grafeo.simd_support()}")  # e.g., "avx2"
#[pyfunction]
fn simd_support() -> &'static str {
    grafeo_core::index::vector::simd_support()
}

/// Grafeo Python module.
#[pymodule]
fn grafeo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGrafeoDB>()?;
    m.add_class::<PyNode>()?;
    m.add_class::<PyEdge>()?;
    m.add_class::<PyQueryResult>()?;
    m.add_class::<AsyncQueryResult>()?;
    m.add_class::<AsyncQueryResultIter>()?;
    m.add_class::<PyValue>()?;
    m.add_class::<PyAlgorithms>()?;
    m.add_class::<PyNetworkXAdapter>()?;
    m.add_class::<PySolvORAdapter>()?;

    // Add module-level functions
    m.add_function(wrap_pyfunction!(simd_support, m)?)?;

    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
