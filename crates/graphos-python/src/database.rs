//! Python database interface.

use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

use graphos_common::types::{EdgeId, NodeId};
use graphos_engine::config::Config;
use graphos_engine::database::GraphosDB;

use crate::error::PyGraphosError;
use crate::graph::{PyEdge, PyNode};
use crate::query::{PyQueryBuilder, PyQueryResult};
use crate::types::PyValue;

/// Python wrapper for GraphosDB.
#[pyclass(name = "GraphosDB")]
pub struct PyGraphosDB {
    inner: Arc<RwLock<GraphosDB>>,
}

#[pymethods]
impl PyGraphosDB {
    /// Create a new in-memory database.
    #[new]
    #[pyo3(signature = (path=None))]
    fn new(path: Option<String>) -> PyResult<Self> {
        let config = if let Some(p) = path {
            Config::persistent(p)
        } else {
            Config::in_memory()
        };

        let db = GraphosDB::with_config(config);

        Ok(Self {
            inner: Arc::new(RwLock::new(db)),
        })
    }

    /// Open an existing database.
    #[staticmethod]
    fn open(path: String) -> PyResult<Self> {
        let config = Config::persistent(path);
        let db = GraphosDB::with_config(config);

        Ok(Self {
            inner: Arc::new(RwLock::new(db)),
        })
    }

    /// Execute a GQL query.
    #[pyo3(signature = (query, params=None))]
    fn execute(
        &self,
        query: &str,
        params: Option<&Bound<'_, pyo3::types::PyDict>>,
        _py: Python<'_>,
    ) -> PyResult<PyQueryResult> {
        let _params = if let Some(p) = params {
            let mut map = HashMap::new();
            for (key, value) in p.iter() {
                let key_str: String = key.extract()?;
                let val = PyValue::from_py(&value).map_err(PyGraphosError::from)?;
                map.insert(key_str, val);
            }
            map
        } else {
            HashMap::new()
        };

        // TODO: Actually execute the query when engine is implemented
        let _ = query;

        Ok(PyQueryResult::empty())
    }

    /// Execute a query and return a query builder.
    fn query(&self, query: String) -> PyQueryBuilder {
        PyQueryBuilder::create(query)
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
            let mut props: Vec<(graphos_common::types::PropertyKey, graphos_common::types::Value)> = Vec::new();
            for (key, value) in p.iter() {
                let key_str: String = key.extract()?;
                let val = PyValue::from_py(&value).map_err(PyGraphosError::from)?;
                props.push((graphos_common::types::PropertyKey::new(key_str), val));
            }
            db.create_node_with_props(&label_refs, props)
        } else {
            db.create_node(&label_refs)
        };

        // Fetch the node back to get the full representation
        if let Some(node) = db.get_node(id) {
            let labels: Vec<String> = node.labels.iter().map(|s| s.to_string()).collect();
            let properties: HashMap<String, graphos_common::types::Value> = node
                .properties
                .into_iter()
                .map(|(k, v)| (k.as_str().to_string(), v))
                .collect();
            Ok(PyNode::new(id, labels, properties))
        } else {
            Err(PyGraphosError::Database("Failed to create node".into()).into())
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
            let mut props: Vec<(graphos_common::types::PropertyKey, graphos_common::types::Value)> = Vec::new();
            for (key, value) in p.iter() {
                let key_str: String = key.extract()?;
                let val = PyValue::from_py(&value).map_err(PyGraphosError::from)?;
                props.push((graphos_common::types::PropertyKey::new(key_str), val));
            }
            db.create_edge_with_props(src, dst, &edge_type, props)
        } else {
            db.create_edge(src, dst, &edge_type)
        };

        // Fetch the edge back to get the full representation
        if let Some(edge) = db.get_edge(id) {
            let properties: HashMap<String, graphos_common::types::Value> = edge
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
            Err(PyGraphosError::Database("Failed to create edge".into()).into())
        }
    }

    /// Get a node by ID.
    fn get_node(&self, id: u64) -> PyResult<Option<PyNode>> {
        let db = self.inner.read();
        let node_id = NodeId(id);

        if let Some(node) = db.get_node(node_id) {
            let labels: Vec<String> = node.labels.iter().map(|s| s.to_string()).collect();
            let properties: HashMap<String, graphos_common::types::Value> = node
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
            let properties: HashMap<String, graphos_common::types::Value> = edge
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

    /// Begin a transaction.
    fn begin_transaction(&self) -> PyResult<PyTransaction> {
        // TODO: Implement transactions
        Ok(PyTransaction {
            _db: self.inner.clone(),
            committed: false,
        })
    }

    /// Get database statistics.
    fn stats(&self) -> PyResult<PyDbStats> {
        let db = self.inner.read();
        Ok(PyDbStats {
            node_count: db.node_count() as u64,
            edge_count: db.edge_count() as u64,
            label_count: 0, // TODO: Add label_count to GraphosDB
            property_count: 0, // TODO: Add property_count to GraphosDB
        })
    }

    /// Close the database.
    fn close(&self) -> PyResult<()> {
        // TODO: Properly close
        Ok(())
    }

    fn __repr__(&self) -> String {
        "GraphosDB()".to_string()
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

/// Transaction wrapper.
#[pyclass(name = "Transaction")]
pub struct PyTransaction {
    _db: Arc<RwLock<GraphosDB>>,
    committed: bool,
}

#[pymethods]
impl PyTransaction {
    /// Commit the transaction.
    fn commit(&mut self) -> PyResult<()> {
        // TODO: Implement
        self.committed = true;
        Ok(())
    }

    /// Rollback the transaction.
    fn rollback(&mut self) -> PyResult<()> {
        // TODO: Implement
        self.committed = false;
        Ok(())
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
        if exc_type.is_some() {
            self.rollback()?;
        } else if !self.committed {
            self.commit()?;
        }
        Ok(false)
    }
}

/// Database statistics.
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
