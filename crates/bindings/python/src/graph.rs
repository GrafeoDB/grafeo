//! Graph elements exposed to Python - nodes and edges with their properties.

use std::collections::HashMap;

use pyo3::prelude::*;

use grafeo_common::types::{EdgeId, NodeId, Value};

use crate::types::PyValue;

/// A node in your graph with labels and properties.
///
/// Access properties with `node["name"]` or `node.get("name")`. Check labels
/// with `node.has_label("Person")`. Nodes are returned by queries like
/// `MATCH (n:Person) RETURN n`.
#[pyclass(name = "Node")]
#[derive(Clone, Debug)]
pub struct PyNode {
    pub(crate) id: NodeId,
    pub(crate) labels: Vec<String>,
    pub(crate) properties: HashMap<String, Value>,
}

#[pymethods]
impl PyNode {
    /// Get the node ID.
    #[getter]
    fn id(&self) -> u64 {
        self.id.0
    }

    /// Get the node labels.
    #[getter]
    fn labels(&self) -> Vec<String> {
        self.labels.clone()
    }

    /// Get a property value.
    fn get(&self, key: &str) -> Option<PyValue> {
        self.properties.get(key).map(|v| PyValue::from(v.clone()))
    }

    /// Get all properties as a dictionary.
    ///
    /// # Panics
    ///
    /// Panics on memory exhaustion during Python dict allocation.
    fn properties(&self, py: Python<'_>) -> Py<PyAny> {
        let dict = pyo3::types::PyDict::new(py);
        for (k, v) in &self.properties {
            dict.set_item(k, PyValue::to_py(v, py))
                .expect("dict.set_item only fails on memory exhaustion");
        }
        dict.unbind().into_any()
    }

    /// Check if node has a label.
    fn has_label(&self, label: &str) -> bool {
        self.labels.iter().any(|l| l == label)
    }

    fn __repr__(&self) -> String {
        format!(
            "Node(id={}, labels={:?}, properties={{...}})",
            self.id.0, self.labels
        )
    }

    fn __str__(&self) -> String {
        format!("(:{} {{id: {}}})", self.labels.join(":"), self.id.0)
    }

    fn __getitem__(&self, key: &str) -> PyResult<PyValue> {
        self.properties
            .get(key)
            .map(|v| PyValue::from(v.clone()))
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!("Property '{}' not found", key))
            })
    }

    fn __contains__(&self, key: &str) -> bool {
        self.properties.contains_key(key)
    }
}

impl PyNode {
    /// Creates a new Python node wrapper (used internally).
    pub fn new(id: NodeId, labels: Vec<String>, properties: HashMap<String, Value>) -> Self {
        Self {
            id,
            labels,
            properties,
        }
    }
}

/// A relationship between two nodes with a type and properties.
///
/// Access properties with `edge["weight"]` or `edge.get("weight")`. Check the
/// relationship type with `edge.edge_type`. Edges connect a `source_id` to a
/// `target_id` and are returned by queries like `MATCH ()-[r:WORKS_AT]->() RETURN r`.
#[pyclass(name = "Edge")]
#[derive(Clone, Debug)]
pub struct PyEdge {
    pub(crate) id: EdgeId,
    pub(crate) edge_type: String,
    pub(crate) source_id: NodeId,
    pub(crate) target_id: NodeId,
    pub(crate) properties: HashMap<String, Value>,
}

#[pymethods]
impl PyEdge {
    /// Get the edge ID.
    #[getter]
    fn id(&self) -> u64 {
        self.id.0
    }

    /// Get the edge type.
    #[getter]
    fn edge_type(&self) -> &str {
        &self.edge_type
    }

    /// Get the source node ID.
    #[getter]
    fn source_id(&self) -> u64 {
        self.source_id.0
    }

    /// Get the target node ID.
    #[getter]
    fn target_id(&self) -> u64 {
        self.target_id.0
    }

    /// Get a property value.
    fn get(&self, key: &str) -> Option<PyValue> {
        self.properties.get(key).map(|v| PyValue::from(v.clone()))
    }

    /// Get all properties as a dictionary.
    ///
    /// # Panics
    ///
    /// Panics on memory exhaustion during Python dict allocation.
    fn properties(&self, py: Python<'_>) -> Py<PyAny> {
        let dict = pyo3::types::PyDict::new(py);
        for (k, v) in &self.properties {
            dict.set_item(k, PyValue::to_py(v, py))
                .expect("dict.set_item only fails on memory exhaustion");
        }
        dict.unbind().into_any()
    }

    fn __repr__(&self) -> String {
        format!(
            "Edge(id={}, type='{}', source={}, target={})",
            self.id.0, self.edge_type, self.source_id.0, self.target_id.0
        )
    }

    fn __str__(&self) -> String {
        format!("()-[:{}]->() (id={})", self.edge_type, self.id.0)
    }

    fn __getitem__(&self, key: &str) -> PyResult<PyValue> {
        self.properties
            .get(key)
            .map(|v| PyValue::from(v.clone()))
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!("Property '{}' not found", key))
            })
    }

    fn __contains__(&self, key: &str) -> bool {
        self.properties.contains_key(key)
    }
}

impl PyEdge {
    /// Creates a new Python edge wrapper (used internally).
    pub fn new(
        id: EdgeId,
        edge_type: String,
        source_id: NodeId,
        target_id: NodeId,
        properties: HashMap<String, Value>,
    ) -> Self {
        Self {
            id,
            edge_type,
            source_id,
            target_id,
            properties,
        }
    }
}
