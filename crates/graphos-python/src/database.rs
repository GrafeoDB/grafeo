//! Python database interface.

use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

use graphos_common::types::{EdgeId, NodeId, Value};
use graphos_engine::config::GraphosConfig;
use graphos_engine::database::GraphosDB;

use crate::error::{PyGraphosError, PyGraphosResult};
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
            GraphosConfig::default().with_data_dir(p.into())
        } else {
            GraphosConfig::default()
        };

        let db = GraphosDB::new(config);

        Ok(Self {
            inner: Arc::new(RwLock::new(db)),
        })
    }

    /// Open an existing database.
    #[staticmethod]
    fn open(path: String) -> PyResult<Self> {
        let config = GraphosConfig::default().with_data_dir(path.into());
        let db = GraphosDB::new(config);

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
        py: Python<'_>,
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
        // For now, return a placeholder result
        let _ = py;
        let _ = query;

        Ok(PyQueryResult::empty())
    }

    /// Execute a query and return a query builder.
    fn query(&self, query: String) -> PyQueryBuilder {
        PyQueryBuilder::new(query)
    }

    /// Create a node.
    #[pyo3(signature = (labels, properties=None))]
    fn create_node(
        &self,
        labels: Vec<String>,
        properties: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<PyNode> {
        let props = if let Some(p) = properties {
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

        // TODO: Actually create in database
        // For now, return a placeholder
        let id = NodeId(1);

        Ok(PyNode::new(id, labels, props))
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
        let props = if let Some(p) = properties {
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

        // TODO: Actually create in database
        let id = EdgeId(1);

        Ok(PyEdge::new(
            id,
            edge_type,
            NodeId(source_id),
            NodeId(target_id),
            props,
        ))
    }

    /// Get a node by ID.
    fn get_node(&self, id: u64) -> PyResult<Option<PyNode>> {
        // TODO: Actually fetch from database
        let _ = id;
        Ok(None)
    }

    /// Get an edge by ID.
    fn get_edge(&self, id: u64) -> PyResult<Option<PyEdge>> {
        // TODO: Actually fetch from database
        let _ = id;
        Ok(None)
    }

    /// Delete a node by ID.
    fn delete_node(&self, id: u64) -> PyResult<bool> {
        // TODO: Actually delete from database
        let _ = id;
        Ok(false)
    }

    /// Delete an edge by ID.
    fn delete_edge(&self, id: u64) -> PyResult<bool> {
        // TODO: Actually delete from database
        let _ = id;
        Ok(false)
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
        // TODO: Get actual stats
        Ok(PyDbStats {
            node_count: 0,
            edge_count: 0,
            label_count: 0,
            property_count: 0,
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
