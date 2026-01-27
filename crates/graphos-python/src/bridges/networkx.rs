//! NetworkX-compatible adapter for Graphos.
//!
//! This module provides an adapter that allows Graphos graphs to be
//! used with NetworkX algorithms. The adapter implements a view that
//! exposes Graphos data in NetworkX-compatible format.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use graphos_common::types::NodeId;
use graphos_core::graph::Direction;
use graphos_engine::database::GraphosDB;

use crate::error::PyGraphosError;

/// NetworkX-compatible adapter for Graphos.
///
/// Provides a view of a Graphos graph that can be used with NetworkX
/// algorithms or converted to a NetworkX graph.
///
/// Usage:
/// ```python
/// db = graphos.GraphosDB()
/// # ... populate graph ...
///
/// # Get NetworkX adapter
/// nx_adapter = db.as_networkx()
///
/// # Use with NetworkX (requires networkx installed)
/// import networkx as nx
/// G = nx_adapter.to_networkx()  # Convert to NetworkX graph
/// pr = nx.pagerank(G)
///
/// # Or use native Graphos algorithms with NetworkX-style API
/// pr = nx_adapter.pagerank()
/// ```
#[pyclass(name = "NetworkXAdapter")]
pub struct PyNetworkXAdapter {
    db: Arc<RwLock<GraphosDB>>,
    directed: bool,
}

impl PyNetworkXAdapter {
    /// Creates a new NetworkX adapter for the given database.
    pub fn new(db: Arc<RwLock<GraphosDB>>, directed: bool) -> Self {
        Self { db, directed }
    }
}

#[pymethods]
impl PyNetworkXAdapter {
    /// Get number of nodes.
    #[getter]
    fn number_of_nodes(&self) -> PyResult<usize> {
        let db = self.db.read();
        Ok(db.node_count())
    }

    /// Get number of edges.
    #[getter]
    fn number_of_edges(&self) -> PyResult<usize> {
        let db = self.db.read();
        Ok(db.edge_count())
    }

    /// Check if graph is directed.
    #[getter]
    fn is_directed(&self) -> bool {
        self.directed
    }

    /// Get list of all node IDs.
    fn nodes(&self) -> PyResult<Vec<u64>> {
        let db = self.db.read();
        let store = db.store();
        Ok(store.node_ids().into_iter().map(|n| n.0).collect())
    }

    /// Get list of all edges as (source, target) tuples.
    fn edges(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = db.store();
        let nodes = store.node_ids();

        let mut edges: Vec<(u64, u64)> = Vec::new();
        for node in nodes {
            for (neighbor, _) in store.edges_from(node, Direction::Outgoing) {
                edges.push((node.0, neighbor.0));
            }
        }

        Ok(edges.into_pyobject(py)?.into_any().unbind())
    }

    /// Get neighbors of a node.
    fn neighbors(&self, node_id: u64) -> PyResult<Vec<u64>> {
        let db = self.db.read();
        let store = db.store();
        let neighbors: Vec<u64> = store
            .edges_from(NodeId::new(node_id), Direction::Outgoing)
            .map(|(n, _)| n.0)
            .collect();
        Ok(neighbors)
    }

    /// Get in-degree of a node.
    fn in_degree(&self, node_id: u64) -> PyResult<usize> {
        let db = self.db.read();
        let store = db.store();
        let nodes = store.node_ids();
        let mut count = 0;
        for other in nodes {
            for (neighbor, _) in store.edges_from(other, Direction::Outgoing) {
                if neighbor.0 == node_id {
                    count += 1;
                }
            }
        }
        Ok(count)
    }

    /// Get out-degree of a node.
    fn out_degree(&self, node_id: u64) -> PyResult<usize> {
        let db = self.db.read();
        let store = db.store();
        Ok(store
            .edges_from(NodeId::new(node_id), Direction::Outgoing)
            .count())
    }

    /// Get degree of a node (in + out for directed, total for undirected).
    fn degree(&self, node_id: u64) -> PyResult<usize> {
        if self.directed {
            Ok(self.in_degree(node_id)? + self.out_degree(node_id)?)
        } else {
            // For undirected, count each edge once
            self.out_degree(node_id)
        }
    }

    /// Check if an edge exists.
    fn has_edge(&self, source: u64, target: u64) -> PyResult<bool> {
        let db = self.db.read();
        let store = db.store();
        for (neighbor, _) in store.edges_from(NodeId::new(source), Direction::Outgoing) {
            if neighbor.0 == target {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Check if a node exists.
    fn has_node(&self, node_id: u64) -> PyResult<bool> {
        let db = self.db.read();
        Ok(db.get_node(NodeId::new(node_id)).is_some())
    }

    /// Convert to a NetworkX graph object.
    ///
    /// Requires networkx to be installed.
    fn to_networkx(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let nx = py.import("networkx")?;

        // Create appropriate graph type
        let graph = if self.directed {
            nx.call_method0("DiGraph")?
        } else {
            nx.call_method0("Graph")?
        };

        let db = self.db.read();
        let store = db.store();
        let nodes = store.node_ids();

        // Add nodes with properties
        for node_id in &nodes {
            if let Some(node) = store.get_node(*node_id) {
                let attrs = PyDict::new(py);

                // Add labels
                let labels: Vec<String> = node.labels.iter().map(|s| s.to_string()).collect();
                attrs.set_item("labels", labels)?;

                // Add properties
                for (key, value) in &node.properties {
                    attrs.set_item(key.as_str(), crate::types::PyValue::to_py(value, py))?;
                }

                graph.call_method("add_node", (node_id.0,), Some(&attrs))?;
            }
        }

        // Add edges with properties
        for node_id in &nodes {
            for (neighbor, edge_id) in store.edges_from(*node_id, Direction::Outgoing) {
                if let Some(edge) = store.get_edge(edge_id) {
                    let attrs = PyDict::new(py);

                    // Add edge type
                    attrs.set_item("type", edge.edge_type.to_string())?;

                    // Add properties
                    for (key, value) in &edge.properties {
                        attrs.set_item(key.as_str(), crate::types::PyValue::to_py(value, py))?;
                    }

                    graph.call_method("add_edge", (node_id.0, neighbor.0), Some(&attrs))?;
                }
            }
        }

        Ok(graph.into_any().unbind())
    }

    /// Create a Graphos database from a NetworkX graph.
    ///
    /// Args:
    ///     G: NetworkX graph object
    ///
    /// Returns:
    ///     New PyNetworkXAdapter wrapping the imported graph
    #[staticmethod]
    fn from_networkx(g: &Bound<'_, PyAny>, _py: Python<'_>) -> PyResult<Self> {
        use graphos_engine::config::Config;

        // Create new in-memory database
        let db = GraphosDB::with_config(Config::in_memory());
        let db = Arc::new(RwLock::new(db));

        // Check if directed
        let is_directed: bool = g.call_method0("is_directed")?.extract()?;

        // Import nodes
        let nodes = g.call_method0("nodes")?;
        let mut node_map: HashMap<i64, NodeId> = HashMap::new();

        for node in nodes.try_iter()? {
            let node = node?;
            let py_id: i64 = node.extract()?;

            // Get node attributes by indexing the graph
            let node_data = g.get_item(py_id)?;

            let db_guard = db.read();

            // Try to get labels from node data
            let labels: Vec<String> = if let Ok(labels_attr) = node_data.get_item("labels") {
                labels_attr.extract().unwrap_or_else(|_| vec!["Node".to_string()])
            } else {
                vec!["Node".to_string()]
            };

            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            let graphos_id = db_guard.create_node(&label_refs);
            node_map.insert(py_id, graphos_id);
        }

        // Import edges
        let edges = g.call_method0("edges")?;
        for edge in edges.try_iter()? {
            let edge = edge?;
            let edge_tuple: &Bound<'_, PyTuple> = edge.downcast_exact()?;
            let src: i64 = edge_tuple.get_item(0)?.extract()?;
            let dst: i64 = edge_tuple.get_item(1)?.extract()?;

            if let (Some(&src_id), Some(&dst_id)) = (node_map.get(&src), node_map.get(&dst)) {
                let db_guard = db.read();

                // Get edge attributes
                let edge_data = g.get_item((src, dst))?;
                let edge_type: String = if let Ok(t) = edge_data.get_item("type") {
                    t.extract().unwrap_or_else(|_| "EDGE".to_string())
                } else {
                    "EDGE".to_string()
                };

                db_guard.create_edge(src_id, dst_id, &edge_type);
            }
        }

        Ok(Self {
            db,
            directed: is_directed,
        })
    }

    // ==========================================================================
    // NetworkX-style algorithm methods
    // ==========================================================================

    /// Compute PageRank (NetworkX-compatible).
    #[pyo3(signature = (alpha=0.85, max_iter=100, tol=1e-6))]
    fn pagerank(&self, alpha: f64, max_iter: usize, tol: f64) -> PyResult<HashMap<u64, f64>> {
        use graphos_adapters::plugins::algorithms;

        let db = self.db.read();
        let store = db.store();
        let result = algorithms::pagerank(store, alpha, max_iter, tol);
        Ok(result.into_iter().map(|(n, s)| (n.0, s)).collect())
    }

    /// Compute betweenness centrality (NetworkX-compatible).
    #[pyo3(signature = (normalized=true))]
    fn betweenness_centrality(&self, normalized: bool) -> PyResult<HashMap<u64, f64>> {
        use graphos_adapters::plugins::algorithms;

        let db = self.db.read();
        let store = db.store();
        let result = algorithms::betweenness_centrality(store, normalized);
        Ok(result.into_iter().map(|(n, s)| (n.0, s)).collect())
    }

    /// Compute closeness centrality (NetworkX-compatible).
    #[pyo3(signature = (wf_improved=false))]
    fn closeness_centrality(&self, wf_improved: bool) -> PyResult<HashMap<u64, f64>> {
        use graphos_adapters::plugins::algorithms;

        let db = self.db.read();
        let store = db.store();
        let result = algorithms::closeness_centrality(store, wf_improved);
        Ok(result.into_iter().map(|(n, s)| (n.0, s)).collect())
    }

    /// Find connected components (NetworkX-compatible).
    fn connected_components(&self) -> PyResult<Vec<Vec<u64>>> {
        use graphos_adapters::plugins::algorithms;

        let db = self.db.read();
        let store = db.store();
        let components = algorithms::connected_components(store);

        // Group by component
        let mut grouped: HashMap<u64, Vec<u64>> = HashMap::new();
        for (node, comp) in components {
            grouped.entry(comp).or_default().push(node.0);
        }

        Ok(grouped.into_values().collect())
    }

    /// Compute shortest path using Dijkstra (NetworkX-compatible).
    #[pyo3(signature = (source, target=None, weight=None))]
    fn shortest_path(
        &self,
        source: u64,
        target: Option<u64>,
        weight: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        use graphos_adapters::plugins::algorithms;

        let db = self.db.read();
        let store = db.store();

        if let Some(target_id) = target {
            match algorithms::dijkstra_path(
                store,
                NodeId::new(source),
                NodeId::new(target_id),
                weight,
            ) {
                Some((_, path)) => {
                    let path_list: Vec<u64> = path.into_iter().map(|n| n.0).collect();
                    Ok(path_list.into_pyobject(py)?.into_any().unbind())
                }
                None => Err(PyGraphosError::InvalidArgument("No path found".into()).into()),
            }
        } else {
            // Return paths to all reachable nodes
            let result = algorithms::dijkstra(store, NodeId::new(source), weight);
            let dict = PyDict::new(py);

            for (target_node, _) in &result.distances {
                if let Some(path) = result.path_to(NodeId::new(source), *target_node) {
                    let path_list: Vec<u64> = path.into_iter().map(|n| n.0).collect();
                    dict.set_item(target_node.0, path_list)?;
                }
            }

            Ok(dict.into_any().unbind())
        }
    }

    /// Compute shortest path length (NetworkX-compatible).
    #[pyo3(signature = (source, target=None, weight=None))]
    fn shortest_path_length(
        &self,
        source: u64,
        target: Option<u64>,
        weight: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        use graphos_adapters::plugins::algorithms;

        let db = self.db.read();
        let store = db.store();

        if let Some(target_id) = target {
            match algorithms::dijkstra_path(
                store,
                NodeId::new(source),
                NodeId::new(target_id),
                weight,
            ) {
                Some((dist, _)) => Ok(dist.into_pyobject(py)?.into_any().unbind()),
                None => Err(PyGraphosError::InvalidArgument("No path found".into()).into()),
            }
        } else {
            let result = algorithms::dijkstra(store, NodeId::new(source), weight);
            let distances: HashMap<u64, f64> =
                result.distances.into_iter().map(|(n, d)| (n.0, d)).collect();
            Ok(distances.into_pyobject(py)?.into_any().unbind())
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "NetworkXAdapter(directed={}, nodes={}, edges={})",
            self.directed,
            self.number_of_nodes().unwrap_or(0),
            self.number_of_edges().unwrap_or(0)
        )
    }
}
