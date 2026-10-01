//! `db.graph(name)`: work in one named graph without switching the database's
//! current graph.

use std::sync::Arc;

use parking_lot::RwLock;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use grafeo_common::types::{EdgeId, NodeId};
use grafeo_engine::session::Session;
use grafeo_engine::{GrafeoDB, GraphHandle};

use crate::database::PyTransaction;
use crate::direct;
use crate::error::PyGrafeoError;
use crate::graph::{PyEdge, PyNode};
use crate::query::PyQueryResult;
use crate::types::PyValue;

/// A handle on one named graph, from `db.graph(name)`.
///
/// It has the query and direct API of `GrafeoDB`, working in its graph
/// whatever graph `db.set_graph()` selects. Each call is its own
/// transaction (use `begin_transaction()` to group several), and one handle
/// can be shared by several threads. Every call raises if the graph no
/// longer exists.
///
/// Example:
///     db.create_graph("model")
///     model = db.graph("model")
///     billing = model.create_node(["Component"], {"name": "Billing"})
///     model.execute("MATCH (n:Component) RETURN n.name")
#[pyclass(name = "GraphHandle", frozen)]
pub struct PyGraphHandle {
    db: Arc<RwLock<GrafeoDB>>,
    schema: Option<String>,
    name: String,
}

impl PyGraphHandle {
    /// A handle on the named graph of the database's current schema.
    pub(crate) fn open(db: Arc<RwLock<GrafeoDB>>, name: &str) -> PyResult<Self> {
        let schema = {
            let guard = db.read();
            let handle = guard.graph(name).map_err(PyGrafeoError::from)?;
            handle.schema().map(ToString::to_string)
        };
        Ok(Self {
            db,
            schema,
            name: name.to_string(),
        })
    }

    /// A session working in this graph.
    fn session(&self, db: &GrafeoDB) -> PyResult<Session> {
        Ok(db
            .graph_in(self.schema.as_deref(), &self.name)
            .and_then(|graph| graph.session())
            .map_err(PyGrafeoError::from)?)
    }

    /// Runs `call` with a session working in this graph.
    fn with_session<T>(&self, call: impl FnOnce(&Session) -> PyResult<T>) -> PyResult<T> {
        let db = self.db.read();
        call(&self.session(&db)?)
    }

    /// Runs `call` with the engine's handle on this graph, whose direct calls
    /// commit on their own without a session.
    fn with_graph<T>(&self, call: impl FnOnce(&GraphHandle<'_>) -> PyResult<T>) -> PyResult<T> {
        let db = self.db.read();
        let graph = db
            .graph_in(self.schema.as_deref(), &self.name)
            .map_err(PyGrafeoError::from)?;
        call(&graph)
    }
}

#[pymethods]
impl PyGraphHandle {
    /// The graph's name.
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    fn __repr__(&self) -> String {
        format!("GraphHandle({:?})", self.name)
    }

    /// Runs a GQL query in this graph.
    #[pyo3(signature = (query, params=None))]
    fn execute(&self, query: &str, params: Option<&Bound<'_, PyDict>>) -> PyResult<PyQueryResult> {
        self.with_session(|session| direct::run(session, "gql", query, params))
    }

    /// Runs a Cypher query in this graph.
    #[cfg(feature = "cypher")]
    #[pyo3(signature = (query, params=None))]
    fn execute_cypher(
        &self,
        query: &str,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyQueryResult> {
        self.with_session(|session| direct::run(session, "cypher", query, params))
    }

    /// Runs a query in a named language (for example `"cypher"`) in this graph.
    #[pyo3(signature = (language, query, params=None))]
    fn execute_language(
        &self,
        language: &str,
        query: &str,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyQueryResult> {
        self.with_session(|session| direct::run(session, language, query, params))
    }

    /// Begins a transaction in this graph.
    #[pyo3(signature = (isolation_level=None))]
    fn begin_transaction(&self, isolation_level: Option<&str>) -> PyResult<PyTransaction> {
        let session = self.session(&self.db.read())?;
        PyTransaction::begin(Arc::clone(&self.db), session, isolation_level)
    }

    /// Creates a node; raises if it breaks a constraint or node type.
    #[pyo3(signature = (labels, properties=None))]
    fn create_node(
        &self,
        labels: Vec<String>,
        properties: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyNode> {
        self.with_graph(|graph| direct::create_node(graph, &labels, properties))
    }

    /// Creates an edge; raises if an endpoint does not exist or the edge
    /// breaks its edge type.
    #[pyo3(signature = (source_id, target_id, edge_type, properties=None))]
    fn create_edge(
        &self,
        source_id: u64,
        target_id: u64,
        edge_type: &str,
        properties: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyEdge> {
        self.with_graph(|graph| {
            direct::create_edge(
                graph,
                NodeId(source_id),
                NodeId(target_id),
                edge_type,
                properties,
            )
        })
    }

    /// Creates one node per property dict, all or none, each with `label`
    /// (a string or a list of labels).
    fn batch_create_nodes_with_props(
        &self,
        label: &Bound<'_, PyAny>,
        properties_list: &Bound<'_, PyList>,
    ) -> PyResult<Vec<u64>> {
        let labels = direct::labels(label)?;
        let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
        let properties = direct::properties_list(properties_list)?;
        self.with_graph(|graph| {
            let ids = graph
                .batch_create_nodes_with_labels(&labels, properties)
                .map_err(PyGrafeoError::from)?;
            Ok(ids.into_iter().map(|id| id.as_u64()).collect())
        })
    }

    /// Creates edges from `(src, dst, type)` or `(src, dst, type, properties)`
    /// tuples, all or none.
    fn batch_create_edges(&self, edges: &Bound<'_, PyList>) -> PyResult<Vec<u64>> {
        let edges = direct::batch_edges(edges)?;
        self.with_graph(|graph| {
            let ids = graph
                .batch_create_edges(edges)
                .map_err(PyGrafeoError::from)?;
            Ok(ids.into_iter().map(|id| id.as_u64()).collect())
        })
    }

    /// Creates or updates one node per row, matched by `key` and all of
    /// `labels`, in one statement.
    ///
    /// Each row is a dict of properties holding `key`; a row without it is
    /// skipped. By default a row's properties are merged into the node's and
    /// none is removed; with `replace=True` the node's properties become
    /// exactly the row's. Labels are never removed. A key repeated within one
    /// call creates one node, which the later rows update. The call is
    /// checked like a query and writes all rows or none.
    ///
    /// Returns:
    ///     dict with `created`, `updated`, `skipped` and `skipped_rows` (the
    ///     indices of the skipped rows, at most 1,000).
    ///
    /// Example:
    ///     graph.upsert_nodes(["Graph", "File"], [{"id": "f1", "size": 3}])
    #[pyo3(signature = (labels, rows, key="id", replace=false))]
    fn upsert_nodes<'py>(
        &self,
        py: Python<'py>,
        labels: Vec<String>,
        rows: &Bound<'_, PyList>,
        key: &str,
        replace: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.with_graph(|graph| crate::direct::upsert_nodes(py, graph, &labels, rows, key, replace))
    }

    /// Creates or updates one edge of `edge_type` per row between existing
    /// nodes, in one statement.
    ///
    /// Each row names its endpoints in `src_field` and `dst_field` (the
    /// nodes whose `endpoint_key` has that value, with all of
    /// `endpoint_labels` when given) and holds the edge's `key`; every other
    /// field is an edge property. A row is skipped when it lacks the key,
    /// `src_field` or `dst_field`, or when no node or more than one node has
    /// its endpoint key; endpoints are never created. `key`, `src_field` and
    /// `dst_field` must be different fields. An edge is identified by its endpoints, type and key. By
    /// default a row's properties are merged into the edge's; with
    /// `replace=True` they become exactly the row's.
    /// A property index on `endpoint_key` makes the endpoint lookups fast.
    ///
    /// Returns:
    ///     dict with `created`, `updated`, `skipped` and `skipped_rows`.
    ///
    /// Example:
    ///     graph.upsert_edges("USES", [{"src": "f1", "dst": "f2", "id": "u1", "w": 1}])
    #[pyo3(signature = (
        edge_type,
        rows,
        key="id",
        endpoint_key="id",
        endpoint_labels=None,
        src_field="src",
        dst_field="dst",
        replace=false
    ))]
    #[allow(
        clippy::too_many_arguments,
        reason = "each Python keyword argument is a parameter"
    )]
    fn upsert_edges<'py>(
        &self,
        py: Python<'py>,
        edge_type: &str,
        rows: &Bound<'_, PyList>,
        key: &str,
        endpoint_key: &str,
        endpoint_labels: Option<Vec<String>>,
        src_field: &str,
        dst_field: &str,
        replace: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let options = grafeo_engine::database::EdgeUpsertOptions {
            key: key.to_string(),
            endpoint_key: endpoint_key.to_string(),
            endpoint_labels: endpoint_labels.unwrap_or_default(),
            src_field: src_field.to_string(),
            dst_field: dst_field.to_string(),
            replace,
        };
        self.with_graph(|graph| crate::direct::upsert_edges(py, graph, edge_type, rows, &options))
    }

    /// Gets a node by ID, or None.
    fn get_node(&self, id: u64) -> PyResult<Option<PyNode>> {
        self.with_graph(|graph| {
            Ok(graph
                .get_node(NodeId(id))
                .map_err(PyGrafeoError::from)?
                .map(direct::node))
        })
    }

    /// Gets an edge by ID, or None.
    fn get_edge(&self, id: u64) -> PyResult<Option<PyEdge>> {
        self.with_graph(|graph| {
            Ok(graph
                .get_edge(EdgeId(id))
                .map_err(PyGrafeoError::from)?
                .map(direct::edge))
        })
    }

    /// Sets a node property; raises if the node does not exist or the value
    /// breaks a constraint.
    fn set_node_property(&self, node_id: u64, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let value = PyValue::from_py(value)?;
        self.with_graph(|graph| {
            Ok(graph
                .set_node_property(NodeId(node_id), key, value)
                .map_err(PyGrafeoError::from)?)
        })
    }

    /// Sets an edge property; raises if the edge does not exist or the value
    /// breaks its edge type.
    fn set_edge_property(&self, edge_id: u64, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let value = PyValue::from_py(value)?;
        self.with_graph(|graph| {
            Ok(graph
                .set_edge_property(EdgeId(edge_id), key, value)
                .map_err(PyGrafeoError::from)?)
        })
    }

    /// Removes a node property; returns whether the node had it.
    fn remove_node_property(&self, node_id: u64, key: &str) -> PyResult<bool> {
        self.with_graph(|graph| {
            Ok(graph
                .remove_node_property(NodeId(node_id), key)
                .map_err(PyGrafeoError::from)?)
        })
    }

    /// Removes an edge property; returns whether the edge had it.
    fn remove_edge_property(&self, edge_id: u64, key: &str) -> PyResult<bool> {
        self.with_graph(|graph| {
            Ok(graph
                .remove_edge_property(EdgeId(edge_id), key)
                .map_err(PyGrafeoError::from)?)
        })
    }

    /// Adds a label; returns False if the node does not exist or has it.
    fn add_node_label(&self, node_id: u64, label: &str) -> PyResult<bool> {
        self.with_graph(|graph| {
            Ok(graph
                .add_node_label(NodeId(node_id), label)
                .map_err(PyGrafeoError::from)?)
        })
    }

    /// Removes a label; returns False if the node does not exist or lacks it.
    fn remove_node_label(&self, node_id: u64, label: &str) -> PyResult<bool> {
        self.with_graph(|graph| {
            Ok(graph
                .remove_node_label(NodeId(node_id), label)
                .map_err(PyGrafeoError::from)?)
        })
    }

    /// Deletes a node; returns False if it does not exist, raises if it
    /// still has edges.
    fn delete_node(&self, id: u64) -> PyResult<bool> {
        self.with_graph(|graph| Ok(graph.delete_node(NodeId(id)).map_err(PyGrafeoError::from)?))
    }

    /// Deletes an edge; returns False if it does not exist.
    fn delete_edge(&self, id: u64) -> PyResult<bool> {
        self.with_graph(|graph| Ok(graph.delete_edge(EdgeId(id)).map_err(PyGrafeoError::from)?))
    }

    /// Finds the IDs of the nodes in this graph with a property value.
    fn find_nodes_by_property(
        &self,
        property: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<u64>> {
        let value = PyValue::from_py(value)?;
        self.with_session(|session| {
            Ok(session
                .find_nodes_by_property(property, &value)
                .into_iter()
                .map(|id| id.as_u64())
                .collect())
        })
    }

    /// Creates an index on a node property of this graph.
    fn create_property_index(&self, property: &str) -> PyResult<()> {
        self.with_session(|session| {
            session.create_property_index(property);
            Ok(())
        })
    }

    /// Drops the index on a node property of this graph; returns whether
    /// there was one.
    fn drop_property_index(&self, property: &str) -> PyResult<bool> {
        self.with_session(|session| Ok(session.drop_property_index(property)))
    }

    /// Returns whether a node property of this graph has an index.
    fn has_property_index(&self, property: &str) -> PyResult<bool> {
        self.with_session(|session| Ok(session.has_property_index(property)))
    }
}
