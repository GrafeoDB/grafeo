//! Your main entry point for using Grafeo from Python.
//!
//! [`PyGrafeoDB`] wraps the Rust database engine and gives you a Pythonic API.
//! Start here - create a database, run queries, and manage transactions.

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use parking_lot::RwLock;
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;

use grafeo_common::types::{EdgeId, LogicalType, NodeId, Value};
use grafeo_engine::config::Config;
use grafeo_engine::database::{GrafeoDB, QueryResult};

use crate::bridges::{PyAlgorithms, PyNetworkXAdapter, PySolvORAdapter};
use crate::error::PyGrafeoError;
use crate::graph::{PyEdge, PyNode};
use crate::query::{PyQueryBuilder, PyQueryResult};
use crate::types::PyValue;

/// Holds results from async query execution.
///
/// Works like [`PyQueryResult`] but without node/edge extraction (async context
/// limitations). Iterate directly or call [`rows()`](Self::rows) to get all data.
#[pyclass(name = "AsyncQueryResult")]
pub struct AsyncQueryResult {
    #[pyo3(get)]
    columns: Vec<String>,
    rows: Vec<Vec<Value>>,
    #[allow(dead_code)]
    column_types: Vec<LogicalType>,
}

#[pymethods]
impl AsyncQueryResult {
    /// Get all rows as a list of lists.
    fn rows(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let list = pyo3::types::PyList::empty(py);
        for row in &self.rows {
            let py_row = pyo3::types::PyList::empty(py);
            for val in row {
                let py_val = PyValue::to_py(val, py);
                py_row.append(py_val)?;
            }
            list.append(py_row)?;
        }
        Ok(list.into())
    }

    /// Get the number of rows.
    fn __len__(&self) -> usize {
        self.rows.len()
    }

    /// Iterate over rows.
    fn __iter__(slf: PyRef<'_, Self>) -> AsyncQueryResultIter {
        AsyncQueryResultIter {
            rows: slf.rows.clone(),
            index: 0,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "AsyncQueryResult(columns={:?}, rows={})",
            self.columns,
            self.rows.len()
        )
    }
}

/// Iterates through async query result rows one at a time.
#[pyclass]
pub struct AsyncQueryResultIter {
    rows: Vec<Vec<Value>>,
    index: usize,
}

#[pymethods]
impl AsyncQueryResultIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> Option<Py<PyAny>> {
        if slf.index >= slf.rows.len() {
            return None;
        }
        let row = slf.rows[slf.index].clone();
        slf.index += 1;

        let py_row = pyo3::types::PyList::empty(py);
        for val in &row {
            let py_val = PyValue::to_py(val, py);
            let _ = py_row.append(py_val);
        }
        Some(py_row.into())
    }
}

/// Your connection to a Grafeo database.
///
/// Create one with `GrafeoDB()` for in-memory storage (fast, temporary) or
/// `GrafeoDB("path/to/db")` for persistent storage (survives restarts).
/// Then use [`execute()`](Self::execute) to run GQL queries.
#[pyclass(name = "GrafeoDB")]
pub struct PyGrafeoDB {
    inner: Arc<RwLock<GrafeoDB>>,
}

#[pymethods]
impl PyGrafeoDB {
    /// Creates a database. Pass a path for persistence, or omit for in-memory.
    ///
    /// Examples:
    ///     db = GrafeoDB()           # In-memory (fast, temporary)
    ///     db = GrafeoDB("./mydb")   # Persistent (survives restarts)
    #[new]
    #[pyo3(signature = (path=None))]
    fn new(path: Option<String>) -> PyResult<Self> {
        let config = if let Some(p) = path {
            Config::persistent(p)
        } else {
            Config::in_memory()
        };

        let db = GrafeoDB::with_config(config).map_err(PyGrafeoError::from)?;

        Ok(Self {
            inner: Arc::new(RwLock::new(db)),
        })
    }

    /// Open an existing database.
    #[staticmethod]
    fn open(path: String) -> PyResult<Self> {
        let config = Config::persistent(path);
        let db = GrafeoDB::with_config(config).map_err(PyGrafeoError::from)?;

        Ok(Self {
            inner: Arc::new(RwLock::new(db)),
        })
    }

    /// Runs a GQL query and returns the results.
    ///
    /// Use params for parameterized queries to avoid injection:
    ///     result = db.execute("MATCH (p:Person {name: $name}) RETURN p", {"name": "Alice"})
    #[pyo3(signature = (query, params=None))]
    fn execute(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        _py: Python<'_>,
    ) -> PyResult<PyQueryResult> {
        let db = self.inner.read();

        let result = if let Some(p) = params {
            // Convert Python params to Rust HashMap
            let mut param_map = HashMap::new();
            for (key, value) in p.iter() {
                let key_str: String = key.extract()?;
                let val = PyValue::from_py(&value).map_err(PyGrafeoError::from)?;
                param_map.insert(key_str, val);
            }
            db.execute_with_params(query, param_map)
                .map_err(PyGrafeoError::from)?
        } else {
            db.execute(query).map_err(PyGrafeoError::from)?
        };

        // Extract nodes and edges based on column types
        let (nodes, edges) = extract_entities(&result, &db);

        Ok(PyQueryResult::new(
            result.columns,
            result.rows,
            nodes,
            edges,
        ))
    }

    /// Execute a query and return a query builder.
    fn query(&self, query: String) -> PyQueryBuilder {
        PyQueryBuilder::create(query)
    }

    /// Execute a Cypher query.
    #[cfg(feature = "cypher")]
    #[pyo3(signature = (query, params=None))]
    fn execute_cypher(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        _py: Python<'_>,
    ) -> PyResult<PyQueryResult> {
        let db = self.inner.read();

        let result = if let Some(p) = params {
            // Convert Python params to Rust HashMap
            let mut param_map = HashMap::new();
            for (key, value) in p.iter() {
                let key_str: String = key.extract()?;
                let val = PyValue::from_py(&value).map_err(PyGrafeoError::from)?;
                param_map.insert(key_str, val);
            }
            db.execute_cypher_with_params(query, param_map)
                .map_err(PyGrafeoError::from)?
        } else {
            db.execute_cypher(query).map_err(PyGrafeoError::from)?
        };

        // Extract nodes and edges based on column types
        let (nodes, edges) = extract_entities(&result, &db);

        Ok(PyQueryResult::new(
            result.columns,
            result.rows,
            nodes,
            edges,
        ))
    }

    /// Execute a GQL query asynchronously.
    ///
    /// This method returns a Python awaitable that can be used with asyncio.
    ///
    /// Example:
    /// ```python
    /// async def main():
    ///     db = GrafeoDB()
    ///     result = await db.execute_async("MATCH (n:Person) RETURN n")
    ///     for row in result:
    ///         print(row)
    ///
    /// asyncio.run(main())
    /// ```
    #[pyo3(signature = (query, params=None))]
    fn execute_async<'py>(
        &self,
        py: Python<'py>,
        query: String,
        params: Option<&Bound<'py, pyo3::types::PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Convert params before the async block since they contain Python references
        let param_map: Option<HashMap<String, Value>> = if let Some(p) = params {
            let mut map = HashMap::new();
            for (key, value) in p.iter() {
                let key_str: String = key.extract()?;
                let val = PyValue::from_py(&value).map_err(PyGrafeoError::from)?;
                map.insert(key_str, val);
            }
            Some(map)
        } else {
            None
        };

        let db = self.inner.clone();

        future_into_py(py, async move {
            // Perform the query execution in the async context
            // We use spawn_blocking since the actual db.execute is synchronous
            let result = tokio::task::spawn_blocking(move || {
                let db = db.read();
                if let Some(params) = param_map {
                    db.execute_with_params(&query, params)
                } else {
                    db.execute(&query)
                }
            })
            .await
            .map_err(|e| PyGrafeoError::Database(e.to_string()))?
            .map_err(PyGrafeoError::from)?;

            // Create PyQueryResult from the result
            // Note: We can't call extract_entities here because we don't have
            // Python references in the async context. We return raw data.
            Ok(AsyncQueryResult {
                columns: result.columns,
                rows: result.rows,
                column_types: result.column_types,
            })
        })
    }

    /// Execute a Gremlin query.
    #[cfg(feature = "gremlin")]
    #[pyo3(signature = (query, params=None))]
    fn execute_gremlin(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        _py: Python<'_>,
    ) -> PyResult<PyQueryResult> {
        let db = self.inner.read();

        let result = if let Some(p) = params {
            // Convert Python params to Rust HashMap
            let mut param_map = HashMap::new();
            for (key, value) in p.iter() {
                let key_str: String = key.extract()?;
                let val = PyValue::from_py(&value).map_err(PyGrafeoError::from)?;
                param_map.insert(key_str, val);
            }
            db.execute_gremlin_with_params(query, param_map)
                .map_err(PyGrafeoError::from)?
        } else {
            db.execute_gremlin(query).map_err(PyGrafeoError::from)?
        };

        // Extract nodes and edges based on column types
        let (nodes, edges) = extract_entities(&result, &db);

        Ok(PyQueryResult::new(
            result.columns,
            result.rows,
            nodes,
            edges,
        ))
    }

    /// Execute a GraphQL query.
    #[cfg(feature = "graphql")]
    #[pyo3(signature = (query, params=None))]
    fn execute_graphql(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        _py: Python<'_>,
    ) -> PyResult<PyQueryResult> {
        let db = self.inner.read();

        let result = if let Some(p) = params {
            // Convert Python params to Rust HashMap
            let mut param_map = HashMap::new();
            for (key, value) in p.iter() {
                let key_str: String = key.extract()?;
                let val = PyValue::from_py(&value).map_err(PyGrafeoError::from)?;
                param_map.insert(key_str, val);
            }
            db.execute_graphql_with_params(query, param_map)
                .map_err(PyGrafeoError::from)?
        } else {
            db.execute_graphql(query).map_err(PyGrafeoError::from)?
        };

        // Extract nodes and edges based on column types
        let (nodes, edges) = extract_entities(&result, &db);

        Ok(PyQueryResult::new(
            result.columns,
            result.rows,
            nodes,
            edges,
        ))
    }

    /// Execute a SPARQL query against the RDF triple store.
    ///
    /// SPARQL is the W3C standard query language for RDF data.
    ///
    /// Example:
    ///     result = db.execute_sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")
    #[cfg(feature = "sparql")]
    #[pyo3(signature = (query, params=None))]
    fn execute_sparql(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        _py: Python<'_>,
    ) -> PyResult<PyQueryResult> {
        let _params = if let Some(p) = params {
            let mut map = HashMap::new();
            for (key, value) in p.iter() {
                let key_str: String = key.extract()?;
                let val = PyValue::from_py(&value).map_err(PyGrafeoError::from)?;
                map.insert(key_str, val);
            }
            map
        } else {
            HashMap::new()
        };

        let db = self.inner.read();
        let result = db.execute_sparql(query).map_err(PyGrafeoError::from)?;

        // SPARQL results don't have LPG nodes/edges, so pass empty vectors
        Ok(PyQueryResult::new(
            result.columns,
            result.rows,
            Vec::new(),
            Vec::new(),
        ))
    }

    /// Create a node.
    #[pyo3(signature = (labels, properties=None))]
    fn create_node(
        &self,
        labels: Vec<String>,
        properties: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<PyNode> {
        let db = self.inner.read();

        // Convert labels from Vec<String> to Vec<&str>
        let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

        // Create node with or without properties
        let id = if let Some(p) = properties {
            // Convert properties
            let mut props: Vec<(
                grafeo_common::types::PropertyKey,
                grafeo_common::types::Value,
            )> = Vec::new();
            for (key, value) in p.iter() {
                let key_str: String = key.extract()?;
                let val = PyValue::from_py(&value).map_err(PyGrafeoError::from)?;
                props.push((grafeo_common::types::PropertyKey::new(key_str), val));
            }
            db.create_node_with_props(&label_refs, props)
        } else {
            db.create_node(&label_refs)
        };

        // Fetch the node back to get the full representation
        if let Some(node) = db.get_node(id) {
            let labels: Vec<String> = node.labels.iter().map(|s| s.to_string()).collect();
            let properties: HashMap<String, grafeo_common::types::Value> = node
                .properties
                .into_iter()
                .map(|(k, v)| (k.as_str().to_string(), v))
                .collect();
            Ok(PyNode::new(id, labels, properties))
        } else {
            Err(PyGrafeoError::Database("Failed to create node".into()).into())
        }
    }

    /// Create an edge between two nodes.
    #[pyo3(signature = (source_id, target_id, edge_type, properties=None))]
    fn create_edge(
        &self,
        source_id: u64,
        target_id: u64,
        edge_type: String,
        properties: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<PyEdge> {
        let db = self.inner.read();
        let src = NodeId(source_id);
        let dst = NodeId(target_id);

        // Create edge with or without properties
        let id = if let Some(p) = properties {
            // Convert properties
            let mut props: Vec<(
                grafeo_common::types::PropertyKey,
                grafeo_common::types::Value,
            )> = Vec::new();
            for (key, value) in p.iter() {
                let key_str: String = key.extract()?;
                let val = PyValue::from_py(&value).map_err(PyGrafeoError::from)?;
                props.push((grafeo_common::types::PropertyKey::new(key_str), val));
            }
            db.create_edge_with_props(src, dst, &edge_type, props)
        } else {
            db.create_edge(src, dst, &edge_type)
        };

        // Fetch the edge back to get the full representation
        if let Some(edge) = db.get_edge(id) {
            let properties: HashMap<String, grafeo_common::types::Value> = edge
                .properties
                .into_iter()
                .map(|(k, v)| (k.as_str().to_string(), v))
                .collect();
            Ok(PyEdge::new(
                id,
                edge.edge_type.to_string(),
                edge.src,
                edge.dst,
                properties,
            ))
        } else {
            Err(PyGrafeoError::Database("Failed to create edge".into()).into())
        }
    }

    /// Get a node by ID.
    fn get_node(&self, id: u64) -> PyResult<Option<PyNode>> {
        let db = self.inner.read();
        let node_id = NodeId(id);

        if let Some(node) = db.get_node(node_id) {
            let labels: Vec<String> = node.labels.iter().map(|s| s.to_string()).collect();
            let properties: HashMap<String, grafeo_common::types::Value> = node
                .properties
                .into_iter()
                .map(|(k, v)| (k.as_str().to_string(), v))
                .collect();
            Ok(Some(PyNode::new(node_id, labels, properties)))
        } else {
            Ok(None)
        }
    }

    /// Get an edge by ID.
    fn get_edge(&self, id: u64) -> PyResult<Option<PyEdge>> {
        let db = self.inner.read();
        let edge_id = EdgeId(id);

        if let Some(edge) = db.get_edge(edge_id) {
            let properties: HashMap<String, grafeo_common::types::Value> = edge
                .properties
                .into_iter()
                .map(|(k, v)| (k.as_str().to_string(), v))
                .collect();
            Ok(Some(PyEdge::new(
                edge_id,
                edge.edge_type.to_string(),
                edge.src,
                edge.dst,
                properties,
            )))
        } else {
            Ok(None)
        }
    }

    /// Delete a node by ID.
    fn delete_node(&self, id: u64) -> PyResult<bool> {
        let db = self.inner.read();
        Ok(db.delete_node(NodeId(id)))
    }

    /// Delete an edge by ID.
    fn delete_edge(&self, id: u64) -> PyResult<bool> {
        let db = self.inner.read();
        Ok(db.delete_edge(EdgeId(id)))
    }

    /// Set a property on a node.
    ///
    /// Example:
    /// ```python
    /// db.set_node_property(node_id, "name", "Alice")
    /// db.set_node_property(node_id, "age", 30)
    /// ```
    fn set_node_property(
        &self,
        node_id: u64,
        key: &str,
        value: &Bound<'_, pyo3::prelude::PyAny>,
    ) -> PyResult<()> {
        let db = self.inner.read();
        let val = PyValue::from_py(value).map_err(PyGrafeoError::from)?;
        db.set_node_property(NodeId(node_id), key, val);
        Ok(())
    }

    /// Add a label to an existing node.
    ///
    /// Returns True if the label was added, False if the node doesn't exist
    /// or already has the label.
    ///
    /// Example:
    /// ```python
    /// alice = db.create_node(["Person"], {"name": "Alice"})
    /// db.add_node_label(alice.id, "Employee")  # Now has Person and Employee
    /// ```
    fn add_node_label(&self, node_id: u64, label: &str) -> PyResult<bool> {
        let db = self.inner.read();
        Ok(db.add_node_label(NodeId(node_id), label))
    }

    /// Remove a label from a node.
    ///
    /// Returns True if the label was removed, False if the node doesn't exist
    /// or doesn't have the label.
    ///
    /// Example:
    /// ```python
    /// db.remove_node_label(alice.id, "Contractor")  # Remove Contractor label
    /// ```
    fn remove_node_label(&self, node_id: u64, label: &str) -> PyResult<bool> {
        let db = self.inner.read();
        Ok(db.remove_node_label(NodeId(node_id), label))
    }

    /// Get all labels for a node.
    ///
    /// Returns a list of label names, or None if the node doesn't exist.
    ///
    /// Example:
    /// ```python
    /// labels = db.get_node_labels(alice.id)
    /// if labels:
    ///     print(f"Alice has labels: {labels}")
    /// ```
    fn get_node_labels(&self, node_id: u64) -> PyResult<Option<Vec<String>>> {
        let db = self.inner.read();
        Ok(db.get_node_labels(NodeId(node_id)))
    }

    /// Set a property on an edge.
    ///
    /// Example:
    /// ```python
    /// db.set_edge_property(edge_id, "weight", 1.5)
    /// db.set_edge_property(edge_id, "since", "2024-01-01")
    /// ```
    fn set_edge_property(
        &self,
        edge_id: u64,
        key: &str,
        value: &Bound<'_, pyo3::prelude::PyAny>,
    ) -> PyResult<()> {
        let db = self.inner.read();
        let val = PyValue::from_py(value).map_err(PyGrafeoError::from)?;
        db.set_edge_property(EdgeId(edge_id), key, val);
        Ok(())
    }

    /// Remove a property from a node.
    ///
    /// Returns True if the property existed and was removed, False otherwise.
    ///
    /// Example:
    /// ```python
    /// if db.remove_node_property(node_id, "deprecated_field"):
    ///     print("Property removed")
    /// ```
    fn remove_node_property(&self, node_id: u64, key: &str) -> PyResult<bool> {
        let db = self.inner.read();
        Ok(db.remove_node_property(NodeId(node_id), key))
    }

    /// Remove a property from an edge.
    ///
    /// Returns True if the property existed and was removed, False otherwise.
    ///
    /// Example:
    /// ```python
    /// if db.remove_edge_property(edge_id, "temporary"):
    ///     print("Property removed")
    /// ```
    fn remove_edge_property(&self, edge_id: u64, key: &str) -> PyResult<bool> {
        let db = self.inner.read();
        Ok(db.remove_edge_property(EdgeId(edge_id), key))
    }

    /// Begin a transaction.
    ///
    /// Returns a Transaction object that can be used as a context manager.
    /// The transaction provides snapshot isolation - all queries within the
    /// transaction see a consistent view of the database.
    ///
    /// Example:
    /// ```python
    /// with db.begin_transaction() as tx:
    ///     tx.execute("CREATE (n:Person {name: 'Alice'})")
    ///     tx.execute("CREATE (n:Person {name: 'Bob'})")
    ///     tx.commit()  # Both nodes created atomically
    /// ```
    fn begin_transaction(&self) -> PyResult<PyTransaction> {
        PyTransaction::new(self.inner.clone())
    }

    /// Get database statistics.
    fn stats(&self) -> PyResult<PyDbStats> {
        let db = self.inner.read();
        Ok(PyDbStats {
            node_count: db.node_count() as u64,
            edge_count: db.edge_count() as u64,
            label_count: db.label_count() as u64,
            property_count: db.property_key_count() as u64,
        })
    }

    // =========================================================================
    // ADMIN API
    // =========================================================================

    /// Returns high-level database information.
    ///
    /// Returns:
    ///     dict with keys: mode, node_count, edge_count, is_persistent, path,
    ///     wal_enabled, version
    ///
    /// Example:
    ///     info = db.info()
    ///     print(f"Nodes: {info['node_count']}, Edges: {info['edge_count']}")
    fn info(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let db = self.inner.read();
        let info = db.info();

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("mode", info.mode.to_string())?;
        dict.set_item("node_count", info.node_count)?;
        dict.set_item("edge_count", info.edge_count)?;
        dict.set_item("is_persistent", info.is_persistent)?;
        dict.set_item("path", info.path.map(|p| p.to_string_lossy().to_string()))?;
        dict.set_item("wal_enabled", info.wal_enabled)?;
        dict.set_item("version", info.version)?;

        Ok(dict.into())
    }

    /// Returns detailed database statistics.
    ///
    /// Returns:
    ///     dict with keys: node_count, edge_count, label_count, edge_type_count,
    ///     property_key_count, index_count, memory_bytes, disk_bytes
    ///
    /// Example:
    ///     stats = db.detailed_stats()
    ///     print(f"Memory: {stats['memory_bytes']} bytes")
    fn detailed_stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let db = self.inner.read();
        let stats = db.detailed_stats();

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("node_count", stats.node_count)?;
        dict.set_item("edge_count", stats.edge_count)?;
        dict.set_item("label_count", stats.label_count)?;
        dict.set_item("edge_type_count", stats.edge_type_count)?;
        dict.set_item("property_key_count", stats.property_key_count)?;
        dict.set_item("index_count", stats.index_count)?;
        dict.set_item("memory_bytes", stats.memory_bytes)?;
        dict.set_item("disk_bytes", stats.disk_bytes)?;

        Ok(dict.into())
    }

    /// Returns schema information (labels, edge types, property keys).
    ///
    /// Returns:
    ///     dict with keys: labels (list of dicts), edge_types (list of dicts),
    ///     property_keys (list of strings)
    ///
    /// Example:
    ///     schema = db.schema()
    ///     for label in schema['labels']:
    ///         print(f"{label['name']}: {label['count']} nodes")
    fn schema(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let db = self.inner.read();
        let schema = db.schema();

        let dict = pyo3::types::PyDict::new(py);

        match schema {
            grafeo_engine::SchemaInfo::Lpg(lpg) => {
                let labels = pyo3::types::PyList::empty(py);
                for label in lpg.labels {
                    let label_dict = pyo3::types::PyDict::new(py);
                    label_dict.set_item("name", label.name)?;
                    label_dict.set_item("count", label.count)?;
                    labels.append(label_dict)?;
                }
                dict.set_item("labels", labels)?;

                let edge_types = pyo3::types::PyList::empty(py);
                for et in lpg.edge_types {
                    let et_dict = pyo3::types::PyDict::new(py);
                    et_dict.set_item("name", et.name)?;
                    et_dict.set_item("count", et.count)?;
                    edge_types.append(et_dict)?;
                }
                dict.set_item("edge_types", edge_types)?;

                dict.set_item("property_keys", lpg.property_keys)?;
            }
            grafeo_engine::SchemaInfo::Rdf(rdf) => {
                let predicates = pyo3::types::PyList::empty(py);
                for pred in rdf.predicates {
                    let pred_dict = pyo3::types::PyDict::new(py);
                    pred_dict.set_item("iri", pred.iri)?;
                    pred_dict.set_item("count", pred.count)?;
                    predicates.append(pred_dict)?;
                }
                dict.set_item("predicates", predicates)?;
                dict.set_item("named_graphs", rdf.named_graphs)?;
                dict.set_item("subject_count", rdf.subject_count)?;
                dict.set_item("object_count", rdf.object_count)?;
            }
        }

        Ok(dict.into())
    }

    /// Validates database integrity.
    ///
    /// Returns:
    ///     list of error dicts (empty = valid). Each error has keys:
    ///     code, message, context
    ///
    /// Example:
    ///     errors = db.validate()
    ///     if not errors:
    ///         print("Database is valid")
    fn validate(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let db = self.inner.read();
        let result = db.validate();

        let errors = pyo3::types::PyList::empty(py);
        for error in result.errors {
            let error_dict = pyo3::types::PyDict::new(py);
            error_dict.set_item("code", error.code)?;
            error_dict.set_item("message", error.message)?;
            error_dict.set_item("context", error.context)?;
            errors.append(error_dict)?;
        }

        Ok(errors.into())
    }

    /// Returns WAL (Write-Ahead Log) status.
    ///
    /// Returns:
    ///     dict with keys: enabled, path, size_bytes, record_count,
    ///     last_checkpoint, current_epoch
    ///
    /// Example:
    ///     wal = db.wal_status()
    ///     print(f"WAL size: {wal['size_bytes']} bytes")
    fn wal_status(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let db = self.inner.read();
        let status = db.wal_status();

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("enabled", status.enabled)?;
        dict.set_item("path", status.path.map(|p| p.to_string_lossy().to_string()))?;
        dict.set_item("size_bytes", status.size_bytes)?;
        dict.set_item("record_count", status.record_count)?;
        dict.set_item("last_checkpoint", status.last_checkpoint)?;
        dict.set_item("current_epoch", status.current_epoch)?;

        Ok(dict.into())
    }

    /// Forces a WAL checkpoint.
    ///
    /// Flushes all pending WAL records to the main storage.
    ///
    /// Example:
    ///     db.wal_checkpoint()
    fn wal_checkpoint(&self) -> PyResult<()> {
        let db = self.inner.read();
        db.wal_checkpoint().map_err(PyGrafeoError::from)?;
        Ok(())
    }

    /// Saves the database to a file path.
    ///
    /// - If in-memory: creates a new persistent database at path
    /// - If file-backed: creates a copy at the new path
    ///
    /// The original database remains unchanged.
    ///
    /// Example:
    ///     db = GrafeoDB()  # in-memory
    ///     db.create_node(["Person"], {"name": "Alice"})
    ///     db.save("./mydb")  # save to file
    fn save(&self, path: String) -> PyResult<()> {
        let db = self.inner.read();
        db.save(path).map_err(PyGrafeoError::from)?;
        Ok(())
    }

    /// Creates an in-memory copy of this database.
    ///
    /// Returns a new database that is completely independent.
    /// Changes to the copy do not affect the original.
    ///
    /// Example:
    ///     file_db = GrafeoDB("./production.db")
    ///     test_db = file_db.to_memory()  # safe copy
    ///     test_db.create_node(...)  # doesn't affect production
    fn to_memory(&self) -> PyResult<Self> {
        let db = self.inner.read();
        let new_db = db.to_memory().map_err(PyGrafeoError::from)?;

        Ok(Self {
            inner: Arc::new(RwLock::new(new_db)),
        })
    }

    /// Opens a database file and loads it entirely into memory.
    ///
    /// The returned database has no connection to the original file.
    /// Changes will NOT be written back to the file.
    ///
    /// Example:
    ///     db = GrafeoDB.open_in_memory("./mydb")
    ///     db.create_node(...)  # doesn't affect file
    #[staticmethod]
    fn open_in_memory(path: String) -> PyResult<Self> {
        let db = grafeo_engine::GrafeoDB::open_in_memory(path).map_err(PyGrafeoError::from)?;

        Ok(Self {
            inner: Arc::new(RwLock::new(db)),
        })
    }

    /// Returns true if this database is backed by a file (persistent).
    ///
    /// In-memory databases return False.
    #[getter]
    fn is_persistent(&self) -> bool {
        let db = self.inner.read();
        db.is_persistent()
    }

    /// Returns the database file path, if persistent.
    ///
    /// In-memory databases return None.
    #[getter]
    fn path(&self) -> Option<String> {
        let db = self.inner.read();
        db.path().map(|p| p.to_string_lossy().to_string())
    }

    /// Close the database.
    fn close(&self) -> PyResult<()> {
        let db = self.inner.read();
        db.close().map_err(PyGrafeoError::from)?;
        Ok(())
    }

    /// Get the algorithms interface.
    ///
    /// Returns an Algorithms object providing access to all graph algorithms.
    ///
    /// Example:
    ///     pr = db.algorithms.pagerank()
    ///     path = db.algorithms.dijkstra(1, 5)
    #[getter]
    fn algorithms(&self) -> PyAlgorithms {
        PyAlgorithms::new(self.inner.clone())
    }

    /// Get a NetworkX-compatible view of the graph.
    ///
    /// Args:
    ///     directed: Whether to treat as directed (default: True)
    ///
    /// Returns:
    ///     NetworkXAdapter that can be used with NetworkX algorithms
    ///     or converted to a NetworkX graph with to_networkx().
    ///
    /// Example:
    ///     nx_adapter = db.as_networkx()
    ///     G = nx_adapter.to_networkx()  # Convert to NetworkX graph
    ///     pr = nx_adapter.pagerank()    # Use native Grafeo algorithms
    #[pyo3(signature = (directed=true))]
    fn as_networkx(&self, directed: bool) -> PyNetworkXAdapter {
        PyNetworkXAdapter::new(self.inner.clone(), directed)
    }

    /// Get a solvOR-compatible adapter for OR-style algorithms.
    ///
    /// Returns:
    ///     SolvORAdapter providing Operations Research style algorithms.
    ///
    /// Example:
    ///     solvor = db.as_solvor()
    ///     distance, path = solvor.shortest_path(1, 5)
    ///     result = solvor.max_flow(source=1, sink=10)
    fn as_solvor(&self) -> PySolvORAdapter {
        PySolvORAdapter::new(self.inner.clone())
    }

    /// Get number of nodes.
    #[getter]
    fn node_count(&self) -> usize {
        let db = self.inner.read();
        db.node_count()
    }

    /// Get number of edges.
    #[getter]
    fn edge_count(&self) -> usize {
        let db = self.inner.read();
        db.edge_count()
    }

    fn __repr__(&self) -> String {
        "GrafeoDB()".to_string()
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        self.close()?;
        Ok(false)
    }
}

/// Groups multiple operations into an atomic unit.
///
/// Use as a context manager - changes are isolated until you commit, and
/// automatically rolled back if an exception occurs:
///
/// ```python
/// with db.begin_transaction() as tx:
///     tx.execute("INSERT (:Person {name: 'Alice'})")
///     tx.execute("INSERT (:Person {name: 'Bob'})")
///     tx.commit()  # Both or neither
/// ```
///
/// Other connections see a consistent snapshot while you work.
#[pyclass(name = "Transaction")]
pub struct PyTransaction {
    db: Arc<RwLock<GrafeoDB>>,
    session: parking_lot::Mutex<Option<grafeo_engine::session::Session>>,
    committed: bool,
    rolled_back: bool,
}

impl PyTransaction {
    /// Create a new transaction, starting a Rust transaction internally.
    fn new(db: Arc<RwLock<GrafeoDB>>) -> PyResult<Self> {
        // Create session from db, but drop the read guard before moving db
        let mut session = {
            let db_guard = db.read();
            db_guard.session()
        };

        // Begin the transaction in the Rust session
        session.begin_tx().map_err(PyGrafeoError::from)?;

        Ok(Self {
            db,
            session: parking_lot::Mutex::new(Some(session)),
            committed: false,
            rolled_back: false,
        })
    }
}

#[pymethods]
impl PyTransaction {
    /// Commit the transaction.
    ///
    /// Makes all changes permanent. Raises an error if the transaction is
    /// already completed or if there's a write-write conflict.
    fn commit(&mut self) -> PyResult<()> {
        if self.committed || self.rolled_back {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Transaction already completed",
            ));
        }

        let mut session_guard = self.session.lock();
        if let Some(ref mut session) = *session_guard {
            session.commit().map_err(PyGrafeoError::from)?;
        }
        *session_guard = None; // Drop the session
        self.committed = true;
        Ok(())
    }

    /// Rollback the transaction.
    ///
    /// Discards all changes made within this transaction.
    fn rollback(&mut self) -> PyResult<()> {
        if self.committed || self.rolled_back {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Transaction already completed",
            ));
        }

        let mut session_guard = self.session.lock();
        if let Some(ref mut session) = *session_guard {
            session.rollback().map_err(PyGrafeoError::from)?;
        }
        *session_guard = None; // Drop the session
        self.rolled_back = true;
        Ok(())
    }

    /// Execute a query within this transaction.
    ///
    /// All queries executed through this method see the same snapshot
    /// and their changes are isolated until commit.
    #[pyo3(signature = (query, params=None))]
    fn execute(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        _py: Python<'_>,
    ) -> PyResult<PyQueryResult> {
        if self.committed || self.rolled_back {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Cannot execute on completed transaction",
            ));
        }

        let db = self.db.read();
        let mut session_guard = self.session.lock();
        let session = session_guard.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Transaction session not available")
        })?;

        let result = if let Some(p) = params {
            // Convert Python params to Rust HashMap
            let mut param_map = HashMap::new();
            for (key, value) in p.iter() {
                let key_str: String = key.extract()?;
                let val = PyValue::from_py(&value).map_err(PyGrafeoError::from)?;
                param_map.insert(key_str, val);
            }
            session
                .execute_with_params(query, param_map)
                .map_err(PyGrafeoError::from)?
        } else {
            session.execute(query).map_err(PyGrafeoError::from)?
        };

        // Extract nodes and edges based on column types
        let (nodes, edges) = extract_entities(&result, &db);

        Ok(PyQueryResult::new(
            result.columns,
            result.rows,
            nodes,
            edges,
        ))
    }

    /// Execute a Gremlin query within this transaction.
    ///
    /// All queries executed through this method see the same snapshot
    /// and their changes are isolated until commit.
    #[cfg(feature = "gremlin")]
    #[pyo3(signature = (query, params=None))]
    fn execute_gremlin(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        _py: Python<'_>,
    ) -> PyResult<PyQueryResult> {
        if self.committed || self.rolled_back {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Cannot execute on completed transaction",
            ));
        }

        let db = self.db.read();
        let mut session_guard = self.session.lock();
        let session = session_guard.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Transaction session not available")
        })?;

        let result = if let Some(p) = params {
            // Convert Python params to Rust HashMap
            let mut param_map = HashMap::new();
            for (key, value) in p.iter() {
                let key_str: String = key.extract()?;
                let val = PyValue::from_py(&value).map_err(PyGrafeoError::from)?;
                param_map.insert(key_str, val);
            }
            session
                .execute_gremlin_with_params(query, param_map)
                .map_err(PyGrafeoError::from)?
        } else {
            session
                .execute_gremlin(query)
                .map_err(PyGrafeoError::from)?
        };

        // Extract nodes and edges based on column types
        let (nodes, edges) = extract_entities(&result, &db);

        Ok(PyQueryResult::new(
            result.columns,
            result.rows,
            nodes,
            edges,
        ))
    }

    /// Execute a GraphQL query within this transaction.
    ///
    /// All queries executed through this method see the same snapshot
    /// and their changes are isolated until commit.
    #[cfg(feature = "graphql")]
    #[pyo3(signature = (query, params=None))]
    fn execute_graphql(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        _py: Python<'_>,
    ) -> PyResult<PyQueryResult> {
        if self.committed || self.rolled_back {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Cannot execute on completed transaction",
            ));
        }

        let db = self.db.read();
        let mut session_guard = self.session.lock();
        let session = session_guard.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Transaction session not available")
        })?;

        let result = if let Some(p) = params {
            // Convert Python params to Rust HashMap
            let mut param_map = HashMap::new();
            for (key, value) in p.iter() {
                let key_str: String = key.extract()?;
                let val = PyValue::from_py(&value).map_err(PyGrafeoError::from)?;
                param_map.insert(key_str, val);
            }
            session
                .execute_graphql_with_params(query, param_map)
                .map_err(PyGrafeoError::from)?
        } else {
            session
                .execute_graphql(query)
                .map_err(PyGrafeoError::from)?
        };

        // Extract nodes and edges based on column types
        let (nodes, edges) = extract_entities(&result, &db);

        Ok(PyQueryResult::new(
            result.columns,
            result.rows,
            nodes,
            edges,
        ))
    }

    /// Execute a SPARQL query within this transaction.
    ///
    /// SPARQL is the W3C standard query language for RDF data.
    /// All queries executed through this method see the same snapshot
    /// and their changes are isolated until commit.
    ///
    /// Example:
    ///     with db.begin_transaction() as tx:
    ///         tx.execute_sparql("INSERT DATA { <http://ex.org/s> <http://ex.org/p> 'value' }")
    ///         result = tx.execute_sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")
    ///         tx.commit()
    #[cfg(feature = "sparql")]
    #[pyo3(signature = (query, params=None))]
    fn execute_sparql(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        _py: Python<'_>,
    ) -> PyResult<PyQueryResult> {
        if self.committed || self.rolled_back {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Cannot execute on completed transaction",
            ));
        }

        let _db = self.db.read();
        let mut session_guard = self.session.lock();
        let session = session_guard.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Transaction session not available")
        })?;

        let result = if let Some(p) = params {
            // Convert Python params to Rust HashMap
            let mut param_map = HashMap::new();
            for (key, value) in p.iter() {
                let key_str: String = key.extract()?;
                let val = PyValue::from_py(&value).map_err(PyGrafeoError::from)?;
                param_map.insert(key_str, val);
            }
            session
                .execute_sparql_with_params(query, param_map)
                .map_err(PyGrafeoError::from)?
        } else {
            session.execute_sparql(query).map_err(PyGrafeoError::from)?
        };

        // SPARQL results don't have LPG nodes/edges, so pass empty vectors
        Ok(PyQueryResult::new(
            result.columns,
            result.rows,
            Vec::new(),
            Vec::new(),
        ))
    }

    /// Check if transaction is active.
    #[getter]
    fn is_active(&self) -> bool {
        !self.committed && !self.rolled_back
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &mut self,
        exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        if !self.committed && !self.rolled_back {
            if exc_type.is_some() {
                self.rollback()?;
            } else {
                // Auto-commit on successful exit (no exception)
                self.commit()?;
            }
        }
        Ok(false)
    }

    fn __repr__(&self) -> String {
        let status = if self.committed {
            "committed"
        } else if self.rolled_back {
            "rolled_back"
        } else {
            "active"
        };
        format!("Transaction(status={})", status)
    }
}

/// Quick stats about your database - node count, edge count, and more.
#[pyclass(name = "DbStats")]
pub struct PyDbStats {
    #[pyo3(get)]
    node_count: u64,
    #[pyo3(get)]
    edge_count: u64,
    #[pyo3(get)]
    label_count: u64,
    #[pyo3(get)]
    property_count: u64,
}

#[pymethods]
impl PyDbStats {
    fn __repr__(&self) -> String {
        format!(
            "DbStats(nodes={}, edges={}, labels={}, properties={})",
            self.node_count, self.edge_count, self.label_count, self.property_count
        )
    }
}

/// Pulls nodes and edges out of query results so Python can work with them.
fn extract_entities(result: &QueryResult, db: &GrafeoDB) -> (Vec<PyNode>, Vec<PyEdge>) {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut seen_node_ids = HashSet::new();
    let mut seen_edge_ids = HashSet::new();

    // Find which columns contain Node/Edge types
    let node_cols: Vec<usize> = result
        .column_types
        .iter()
        .enumerate()
        .filter_map(|(i, t)| {
            if *t == LogicalType::Node {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    let edge_cols: Vec<usize> = result
        .column_types
        .iter()
        .enumerate()
        .filter_map(|(i, t)| {
            if *t == LogicalType::Edge {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    // Extract unique nodes and edges from result rows
    for row in &result.rows {
        // Extract nodes
        for &col_idx in &node_cols {
            if let Some(Value::Int64(id)) = row.get(col_idx) {
                let node_id = NodeId(*id as u64);
                if !seen_node_ids.contains(&node_id) {
                    seen_node_ids.insert(node_id);
                    if let Some(node) = db.get_node(node_id) {
                        let labels: Vec<String> =
                            node.labels.iter().map(|s| s.to_string()).collect();
                        let properties: HashMap<String, Value> = node
                            .properties
                            .into_iter()
                            .map(|(k, v)| (k.as_str().to_string(), v))
                            .collect();
                        nodes.push(PyNode::new(node_id, labels, properties));
                    }
                }
            }
        }

        // Extract edges
        for &col_idx in &edge_cols {
            if let Some(Value::Int64(id)) = row.get(col_idx) {
                let edge_id = EdgeId(*id as u64);
                if !seen_edge_ids.contains(&edge_id) {
                    seen_edge_ids.insert(edge_id);
                    if let Some(edge) = db.get_edge(edge_id) {
                        let properties: HashMap<String, Value> = edge
                            .properties
                            .into_iter()
                            .map(|(k, v)| (k.as_str().to_string(), v))
                            .collect();
                        edges.push(PyEdge::new(
                            edge_id,
                            edge.edge_type.to_string(),
                            edge.src,
                            edge.dst,
                            properties,
                        ));
                    }
                }
            }
        }
    }

    (nodes, edges)
}
