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
//! db.execute("INSERT (:Person {name: 'Alix', role: 'Engineer'})")
//! db.execute("INSERT (:Person {name: 'Gus', role: 'Manager'})")
//! db.execute("""
//!     MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
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
//! | pandas | `result.to_pandas()` or `db.nodes_df()` | Tabular operations |
//! | polars | `result.to_polars()` | Fast columnar analytics |
//! | NetworkX | `db.as_networkx().to_networkx()` | Graph visualization, analysis |
//! | solvOR | `db.as_solvor()` | Operations research algorithms |

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use pyo3::prelude::*;
use pyo3::types::PyDict;

mod bridges;
mod database;
mod direct;
mod error;
mod graph;
mod graph_handle;
mod quantization;
mod query;
mod stream;
mod types;

#[cfg(feature = "algos")]
use bridges::{PyAlgorithms, PyNetworkXAdapter, PySolvORAdapter};
use database::{AsyncQueryResult, AsyncQueryResultIter, PyGrafeoDB, PyIsolationLevel};
use graph::{PyEdge, PyNode};
use query::PyQueryResult;
use stream::PyResultStream;
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

/// Returns the optional features compiled into this build.
///
/// Names are the Cargo feature names: the query languages (`gql`, `cypher`,
/// `sparql`, `gremlin`, `graphql`, `sql-pgq`) and the optional capabilities
/// (`algos`, `vector-index`, `text-index`, `hybrid-search`, `cdc`,
/// `triple-store`, `shacl`, `temporal`, `embed`, `arrow-export`,
/// `jsonl-import`, `parquet-import`, `metrics`). Groups and profiles such as
/// `full` or `ai` are not listed, only the features they enable. Persistence
/// (WAL and `.grafeo` files) is part of every build, and so is reading a
/// database file compacted by 0.5.x.
///
/// Example:
///     import grafeo
///     if "cypher" not in grafeo.features():
///         raise RuntimeError("this grafeo build has no Cypher support")
#[pyfunction]
fn features() -> Vec<&'static str> {
    [
        ("gql", cfg!(feature = "gql")),
        ("cypher", cfg!(feature = "cypher")),
        ("sparql", cfg!(feature = "sparql")),
        ("gremlin", cfg!(feature = "gremlin")),
        ("graphql", cfg!(feature = "graphql")),
        ("sql-pgq", cfg!(feature = "sql-pgq")),
        ("algos", cfg!(feature = "algos")),
        ("vector-index", cfg!(feature = "vector-index")),
        ("text-index", cfg!(feature = "text-index")),
        ("hybrid-search", cfg!(feature = "hybrid-search")),
        ("cdc", cfg!(feature = "cdc")),
        ("triple-store", cfg!(feature = "triple-store")),
        ("shacl", cfg!(feature = "shacl")),
        ("temporal", cfg!(feature = "temporal")),
        ("embed", cfg!(feature = "embed")),
        ("arrow-export", cfg!(feature = "arrow-export")),
        ("jsonl-import", cfg!(feature = "jsonl-import")),
        ("parquet-import", cfg!(feature = "parquet-import")),
        ("metrics", cfg!(feature = "metrics")),
    ]
    .into_iter()
    .filter_map(|(name, enabled)| enabled.then_some(name))
    .collect()
}

/// Describes this build, for run metadata and for checking an installed wheel.
///
/// Returns a dict with:
///
/// - `version`: the package version.
/// - `commit`: the full git commit hash the module was built from, or `None`
///   when it was built outside a git checkout (for example from an sdist).
/// - `dirty`: whether tracked files had uncommitted changes at build time,
///   or `None` when `commit` is `None`.
/// - `features`: the same list as `grafeo.features()`.
/// - `profile`: `"release"` or `"debug"`.
///
/// Example:
///     import grafeo
///     info = grafeo.build_info()
///     print(info["version"], info["commit"], info["profile"])
#[pyfunction]
fn build_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let info = PyDict::new(py);
    info.set_item("version", env!("CARGO_PKG_VERSION"))?;
    info.set_item("commit", option_env!("GRAFEO_BUILD_COMMIT"))?;
    info.set_item(
        "dirty",
        option_env!("GRAFEO_BUILD_DIRTY").map(|dirty| dirty == "true"),
    )?;
    info.set_item("features", features())?;
    info.set_item(
        "profile",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
    )?;
    Ok(info)
}

/// Grafeo Python module.
#[pymodule]
fn grafeo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "GrafeoError",
        m.py().get_type::<crate::error::GrafeoError>(),
    )?;
    m.add(
        "DatabaseClosedError",
        m.py().get_type::<crate::error::DatabaseClosedError>(),
    )?;
    m.add(
        "GrafeoCorruptionError",
        m.py().get_type::<crate::error::GrafeoCorruptionError>(),
    )?;
    m.add_class::<PyGrafeoDB>()?;
    m.add_class::<graph_handle::PyGraphHandle>()?;
    m.add_class::<PyNode>()?;
    m.add_class::<PyEdge>()?;
    m.add_class::<PyQueryResult>()?;
    m.add_class::<PyResultStream>()?;
    m.add_class::<AsyncQueryResult>()?;
    m.add_class::<AsyncQueryResultIter>()?;
    m.add_class::<PyValue>()?;
    m.add_class::<PyIsolationLevel>()?;
    #[cfg(feature = "algos")]
    {
        m.add_class::<PyAlgorithms>()?;
        m.add_class::<PyNetworkXAdapter>()?;
        m.add_class::<PySolvORAdapter>()?;
    }

    // Register quantization types
    quantization::register(m)?;

    // Add module-level functions
    m.add_function(wrap_pyfunction!(simd_support, m)?)?;
    m.add_function(wrap_pyfunction!(features, m)?)?;
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    m.add_function(wrap_pyfunction!(types::vector, m)?)?;

    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
