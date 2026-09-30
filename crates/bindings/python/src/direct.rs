//! Queries and direct writes shared by `GrafeoDB` (its current graph),
//! `GraphHandle` (its graph) and `Transaction` (its session).

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use grafeo_common::types::{EdgeId, NodeId, PropertyKey, Value};
use grafeo_core::graph::lpg::{Edge, Node};
use grafeo_engine::database::QueryResult;
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
    let mut query_result = PyQueryResult::with_metrics(
        columns,
        result.into_rows(),
        nodes,
        edges,
        execution_time,
        rows_scanned,
    );
    query_result.counters = counters;
    query_result
}

/// Runs a query in `language` in the session.
pub(crate) fn run(
    session: &Session,
    language: &str,
    query: &str,
    query_params: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyQueryResult> {
    let result = session
        .execute_language(query, language, params(query_params)?)
        .map_err(PyGrafeoError::from)?;
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

    fn node(&self, id: NodeId) -> Option<Node>;

    fn edge(&self, id: EdgeId) -> Option<Edge>;
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

    fn node(&self, id: NodeId) -> Option<Node> {
        self.get_node(id)
    }

    fn edge(&self, id: EdgeId) -> Option<Edge> {
        self.get_edge(id)
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

    fn node(&self, id: NodeId) -> Option<Node> {
        self.get_node(id)
    }

    fn edge(&self, id: EdgeId) -> Option<Edge> {
        self.get_edge(id)
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

    fn node(&self, id: NodeId) -> Option<Node> {
        self.get_node(id).ok().flatten()
    }

    fn edge(&self, id: EdgeId) -> Option<Edge> {
        self.get_edge(id).ok().flatten()
    }
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
        .map(edge)
        .ok_or_else(|| PyGrafeoError::database("Failed to create edge").into())
}
