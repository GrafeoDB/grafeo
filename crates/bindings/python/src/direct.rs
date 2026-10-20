//! Queries and direct writes shared by `GrafeoDB` (its current graph),
//! `GraphHandle` (its graph) and `Transaction` (its session).

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use grafeo_common::types::{EdgeId, NodeId, PropertyKey, Value};
use grafeo_core::graph::lpg::{Edge, Node};
use grafeo_engine::database::{BatchEdge, EdgeUpsertOptions, QueryResult, UpsertSummary};
use grafeo_engine::{GrafeoDB, GraphHandle, Session};

use crate::error::PyGrafeoError;
use crate::graph::{PyEdge, PyNode};
use crate::query::PyQueryResult;
use crate::types::PyValue;

/// Query parameters from a Python dict.
pub(crate) fn params(
    params: Option<&Bound<'_, PyDict>>,
) -> PyResult<Option<HashMap<String, Value>>> {
    let Some(params) = params else {
        return Ok(None);
    };
    let mut map = HashMap::with_capacity(params.len());
    for (key, value) in params.iter() {
        map.insert(key.extract::<String>()?, PyValue::from_py(&value)?);
    }
    Ok(Some(map))
}

/// Node or edge properties from a Python dict.
pub(crate) fn properties(
    properties: Option<&Bound<'_, PyDict>>,
) -> PyResult<Vec<(PropertyKey, Value)>> {
    let Some(properties) = properties else {
        return Ok(Vec::new());
    };
    properties
        .iter()
        .map(|(key, value)| {
            Ok((
                PropertyKey::new(key.extract::<String>()?),
                PyValue::from_py(&value)?,
            ))
        })
        .collect()
}

/// One property map per node from a Python list of dicts.
pub(crate) fn properties_list(
    list: &Bound<'_, PyList>,
) -> PyResult<Vec<HashMap<PropertyKey, Value>>> {
    list.iter()
        .map(|item| {
            let dict: &Bound<'_, PyDict> = item.cast()?;
            Ok(properties(Some(dict))?.into_iter().collect())
        })
        .collect()
}

/// A query result for Python, with the nodes and edges it returns.
pub(crate) fn query_result(mut result: QueryResult) -> PyQueryResult {
    let (nodes, edges) = grafeo_bindings_common::entity::extract_and_map(
        &result,
        |n| PyNode::new(n.id, n.labels, n.properties),
        |e| PyEdge::new(e.id, e.edge_type, e.source_id, e.target_id, e.properties),
    );
    let columns = std::mem::take(&mut result.columns);
    let execution_time = result.execution_time_ms;
    let rows_scanned = result.rows_scanned;
    let counters = result.counters;
    let profile = result.profile.take();
    let mut query_result = PyQueryResult::with_metrics(
        columns,
        result.into_rows(),
        nodes,
        edges,
        execution_time,
        rows_scanned,
    );
    query_result.counters = counters;
    query_result.profile = profile;
    query_result
}

/// Runs a query in `language` in the session, with parameters converted by
/// [`params`]. It touches no Python object, so it runs with the GIL released.
pub(crate) fn run(
    session: &Session,
    language: &str,
    query: &str,
    query_params: Option<HashMap<String, Value>>,
) -> Result<PyQueryResult, PyGrafeoError> {
    let result = session.execute_language(query, language, query_params)?;
    Ok(query_result(result))
}

pub(crate) fn node(node: Node) -> PyNode {
    let labels = node.labels.iter().map(ToString::to_string).collect();
    PyNode::new(node.id, labels, node.properties.into_iter().collect())
}

pub(crate) fn edge(edge: Edge) -> PyEdge {
    PyEdge::new(
        edge.id,
        edge.edge_type.to_string(),
        edge.src,
        edge.dst,
        edge.properties.into_iter().collect(),
    )
}

/// Where a direct write goes: a session (inside its transaction), the
/// database (its current graph) or a graph handle (its graph). The database
/// and handles commit each call on its own, without a session.
pub(crate) trait DirectTarget {
    fn create_node_with_props(
        &self,
        labels: &[&str],
        properties: Vec<(PropertyKey, Value)>,
    ) -> grafeo_common::utils::error::Result<NodeId>;

    fn create_edge_with_props(
        &self,
        source: NodeId,
        target: NodeId,
        edge_type: &str,
        properties: Vec<(PropertyKey, Value)>,
    ) -> grafeo_common::utils::error::Result<EdgeId>;

    /// The node as the target now sees it; an error when the target itself
    /// is gone (a dropped graph).
    fn node(&self, id: NodeId) -> grafeo_common::utils::error::Result<Option<Node>>;

    /// The edge as the target now sees it (see [`node`](Self::node)).
    fn edge(&self, id: EdgeId) -> grafeo_common::utils::error::Result<Option<Edge>>;

    fn upsert_nodes(
        &self,
        labels: &[&str],
        key: &str,
        rows: Vec<HashMap<PropertyKey, Value>>,
        replace: bool,
    ) -> grafeo_common::utils::error::Result<UpsertSummary>;

    fn upsert_edges(
        &self,
        edge_type: &str,
        rows: Vec<HashMap<PropertyKey, Value>>,
        options: &EdgeUpsertOptions,
    ) -> grafeo_common::utils::error::Result<UpsertSummary>;
}

impl DirectTarget for Session {
    fn create_node_with_props(
        &self,
        labels: &[&str],
        properties: Vec<(PropertyKey, Value)>,
    ) -> grafeo_common::utils::error::Result<NodeId> {
        Session::create_node_with_props(self, labels, properties)
    }

    fn create_edge_with_props(
        &self,
        source: NodeId,
        target: NodeId,
        edge_type: &str,
        properties: Vec<(PropertyKey, Value)>,
    ) -> grafeo_common::utils::error::Result<EdgeId> {
        Session::create_edge_with_props(self, source, target, edge_type, properties)
    }

    fn node(&self, id: NodeId) -> grafeo_common::utils::error::Result<Option<Node>> {
        Ok(self.get_node(id))
    }

    fn edge(&self, id: EdgeId) -> grafeo_common::utils::error::Result<Option<Edge>> {
        Ok(self.get_edge(id))
    }

    fn upsert_nodes(
        &self,
        labels: &[&str],
        key: &str,
        rows: Vec<HashMap<PropertyKey, Value>>,
        replace: bool,
    ) -> grafeo_common::utils::error::Result<UpsertSummary> {
        Session::upsert_nodes(self, labels, key, rows, replace)
    }

    fn upsert_edges(
        &self,
        edge_type: &str,
        rows: Vec<HashMap<PropertyKey, Value>>,
        options: &EdgeUpsertOptions,
    ) -> grafeo_common::utils::error::Result<UpsertSummary> {
        Session::upsert_edges(self, edge_type, rows, options)
    }
}

impl DirectTarget for GrafeoDB {
    fn create_node_with_props(
        &self,
        labels: &[&str],
        properties: Vec<(PropertyKey, Value)>,
    ) -> grafeo_common::utils::error::Result<NodeId> {
        GrafeoDB::create_node_with_props(self, labels, properties)
    }

    fn create_edge_with_props(
        &self,
        source: NodeId,
        target: NodeId,
        edge_type: &str,
        properties: Vec<(PropertyKey, Value)>,
    ) -> grafeo_common::utils::error::Result<EdgeId> {
        GrafeoDB::create_edge_with_props(self, source, target, edge_type, properties)
    }

    fn node(&self, id: NodeId) -> grafeo_common::utils::error::Result<Option<Node>> {
        Ok(self.get_node(id))
    }

    fn edge(&self, id: EdgeId) -> grafeo_common::utils::error::Result<Option<Edge>> {
        Ok(self.get_edge(id))
    }

    fn upsert_nodes(
        &self,
        labels: &[&str],
        key: &str,
        rows: Vec<HashMap<PropertyKey, Value>>,
        replace: bool,
    ) -> grafeo_common::utils::error::Result<UpsertSummary> {
        GrafeoDB::upsert_nodes(self, labels, key, rows, replace)
    }

    fn upsert_edges(
        &self,
        edge_type: &str,
        rows: Vec<HashMap<PropertyKey, Value>>,
        options: &EdgeUpsertOptions,
    ) -> grafeo_common::utils::error::Result<UpsertSummary> {
        GrafeoDB::upsert_edges(self, edge_type, rows, options)
    }
}

impl DirectTarget for GraphHandle<'_> {
    fn create_node_with_props(
        &self,
        labels: &[&str],
        properties: Vec<(PropertyKey, Value)>,
    ) -> grafeo_common::utils::error::Result<NodeId> {
        GraphHandle::create_node_with_props(self, labels, properties)
    }

    fn create_edge_with_props(
        &self,
        source: NodeId,
        target: NodeId,
        edge_type: &str,
        properties: Vec<(PropertyKey, Value)>,
    ) -> grafeo_common::utils::error::Result<EdgeId> {
        GraphHandle::create_edge_with_props(self, source, target, edge_type, properties)
    }

    fn node(&self, id: NodeId) -> grafeo_common::utils::error::Result<Option<Node>> {
        self.get_node(id)
    }

    fn edge(&self, id: EdgeId) -> grafeo_common::utils::error::Result<Option<Edge>> {
        self.get_edge(id)
    }

    fn upsert_nodes(
        &self,
        labels: &[&str],
        key: &str,
        rows: Vec<HashMap<PropertyKey, Value>>,
        replace: bool,
    ) -> grafeo_common::utils::error::Result<UpsertSummary> {
        GraphHandle::upsert_nodes(self, labels, key, rows, replace)
    }

    fn upsert_edges(
        &self,
        edge_type: &str,
        rows: Vec<HashMap<PropertyKey, Value>>,
        options: &EdgeUpsertOptions,
    ) -> grafeo_common::utils::error::Result<UpsertSummary> {
        GraphHandle::upsert_edges(self, edge_type, rows, options)
    }
}

/// An upsert summary as a dict: `created`, `updated`, `skipped` and
/// `skipped_rows`.
pub(crate) fn upsert_summary<'py>(
    py: Python<'py>,
    summary: &UpsertSummary,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("created", summary.created)?;
    dict.set_item("updated", summary.updated)?;
    dict.set_item("skipped", summary.skipped)?;
    dict.set_item("skipped_rows", summary.skipped_rows.clone())?;
    Ok(dict)
}

/// Node labels from a string (one label) or a list of strings.
pub(crate) fn labels(value: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if let Ok(label) = value.extract::<String>() {
        return Ok(vec![label]);
    }
    value.extract::<Vec<String>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err("labels must be a string or a list of strings")
    })
}

/// The direction of `get_node_edges`: `"out"`, `"in"` or `"both"`.
pub(crate) fn direction(value: &str) -> PyResult<grafeo_engine::Direction> {
    match value {
        "out" | "outgoing" => Ok(grafeo_engine::Direction::Outgoing),
        "in" | "incoming" => Ok(grafeo_engine::Direction::Incoming),
        "both" => Ok(grafeo_engine::Direction::Both),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "direction must be \"out\", \"in\" or \"both\", not {other:?}"
        ))),
    }
}

/// The edge types of `get_node_edges`: none for every type, a type, or a
/// list or tuple of types.
pub(crate) fn edge_types(value: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<String>> {
    match value {
        None => Ok(Vec::new()),
        Some(value) if value.is_none() => Ok(Vec::new()),
        Some(value) => {
            if let Ok(edge_type) = value.extract::<String>() {
                return Ok(vec![edge_type]);
            }
            value.extract::<Vec<String>>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "edge_types must be a string or a list of strings",
                )
            })
        }
    }
}

/// `(neighbor_id, edge_id)` pairs for Python.
pub(crate) fn edge_pairs(edges: Vec<(NodeId, EdgeId)>) -> Vec<(u64, u64)> {
    edges
        .into_iter()
        .map(|(neighbor, edge)| (neighbor.as_u64(), edge.as_u64()))
        .collect()
}

/// Edges for a batch from `(src, dst, type)` or `(src, dst, type, properties)`
/// tuples.
pub(crate) fn batch_edges(edges: &Bound<'_, PyList>) -> PyResult<Vec<BatchEdge>> {
    edges
        .iter()
        .map(|edge| {
            let (src, dst, edge_type, props): (u64, u64, String, Option<Bound<'_, PyDict>>) =
                if let Ok((src, dst, edge_type)) = edge.extract::<(u64, u64, String)>() {
                    (src, dst, edge_type, None)
                } else {
                    edge.extract().map_err(|_| {
                        pyo3::exceptions::PyTypeError::new_err(
                            "each edge must be a (src, dst, type) or (src, dst, type, properties) tuple",
                        )
                    })?
                };
            let mut batch_edge = BatchEdge::new(NodeId(src), NodeId(dst), edge_type);
            if let Some(props) = props {
                batch_edge = batch_edge.with_properties(properties(Some(&props))?);
            }
            Ok(batch_edge)
        })
        .collect()
}

/// Creates a node and returns it as the target now sees it.
pub(crate) fn create_node(
    target: &impl DirectTarget,
    labels: &[String],
    node_properties: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyNode> {
    let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
    let id = target
        .create_node_with_props(&labels, properties(node_properties)?)
        .map_err(PyGrafeoError::from)?;
    target
        .node(id)
        .map_err(PyGrafeoError::from)?
        .map(node)
        .ok_or_else(|| PyGrafeoError::database("Failed to create node").into())
}

/// Creates an edge and returns it as the target now sees it.
pub(crate) fn create_edge(
    target: &impl DirectTarget,
    source: NodeId,
    target_node: NodeId,
    edge_type: &str,
    edge_properties: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyEdge> {
    let id = target
        .create_edge_with_props(source, target_node, edge_type, properties(edge_properties)?)
        .map_err(PyGrafeoError::from)?;
    target
        .edge(id)
        .map_err(PyGrafeoError::from)?
        .map(edge)
        .ok_or_else(|| PyGrafeoError::database("Failed to create edge").into())
}
