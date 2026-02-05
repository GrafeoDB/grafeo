//! Bridge to the [solvOR](https://pypi.org/project/solvor/) Python library style.
//!
//! solvOR is a Python library for Operations Research algorithms. This adapter
//! provides a compatible API for classic OR problems - shortest paths, network
//! flow, minimum spanning trees. Results come back in OR-friendly formats
//! (distances with paths, flows with edge assignments).

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use grafeo_common::types::NodeId;
use grafeo_engine::database::GrafeoDB;

use crate::error::PyGrafeoError;

/// Solve classic OR problems on your graph.
///
/// Get this via `db.as_solvor()`. Algorithms return results in OR-friendly
/// formats - distances with paths, flows with edge assignments, etc.
///
/// ```python
/// solvor = db.as_solvor()
///
/// # Shortest paths
/// dist, path = solvor.shortest_path(1, 5)
///
/// # Network flow
/// result = solvor.max_flow(source=1, sink=10)
/// print(f"Max flow: {result['max_flow']}")
///
/// # Minimum spanning tree
/// mst = solvor.minimum_spanning_tree()
/// ```
#[pyclass(name = "SolvORAdapter")]
pub struct PySolvORAdapter {
    db: Arc<RwLock<GrafeoDB>>,
}

impl PySolvORAdapter {
    /// Creates a new solvOR adapter for the given database.
    pub fn new(db: Arc<RwLock<GrafeoDB>>) -> Self {
        Self { db }
    }
}

#[pymethods]
impl PySolvORAdapter {
    // ==========================================================================
    // Shortest Path Algorithms (solvOR-style)
    // ==========================================================================

    /// Compute shortest path between two nodes.
    ///
    /// Args:
    ///     source: Source node ID
    ///     target: Target node ID
    ///     weight: Optional edge property name for weights (default: 1.0)
    ///     method: Algorithm to use: "dijkstra", "bellman_ford", or "astar" (default: "dijkstra")
    ///
    /// Returns:
    ///     Tuple of (distance, path) where path is a list of node IDs,
    ///     or None if no path exists.
    #[pyo3(signature = (source, target, weight=None, method="dijkstra"))]
    fn shortest_path(
        &self,
        source: u64,
        target: u64,
        weight: Option<&str>,
        method: &str,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        use grafeo_adapters::plugins::algorithms;

        let db = self.db.read();
        let store = db.store();

        let result = match method {
            "dijkstra" => {
                algorithms::dijkstra_path(store, NodeId::new(source), NodeId::new(target), weight)
            }
            "bellman_ford" => {
                let bf_result = algorithms::bellman_ford(store, NodeId::new(source), weight);
                if bf_result.has_negative_cycle {
                    return Err(PyGrafeoError::InvalidArgument(
                        "Graph contains negative cycle".into(),
                    )
                    .into());
                }
                bf_result.path_to(NodeId::new(target)).map(|path| {
                    let dist = bf_result
                        .distances
                        .get(&NodeId::new(target))
                        .copied()
                        .unwrap_or(f64::INFINITY);
                    (dist, path)
                })
            }
            "astar" => {
                // A* with zero heuristic (same as Dijkstra)
                algorithms::astar(
                    store,
                    NodeId::new(source),
                    NodeId::new(target),
                    weight,
                    |_| 0.0,
                )
            }
            _ => {
                return Err(PyGrafeoError::InvalidArgument(format!(
                    "Unknown method: {}. Use 'dijkstra', 'bellman_ford', or 'astar'",
                    method
                ))
                .into());
            }
        };

        match result {
            Some((dist, path)) => {
                let path_list: Vec<u64> = path.into_iter().map(|n| n.0).collect();
                Ok((dist, path_list).into_pyobject(py)?.into_any().unbind())
            }
            None => Ok(py.None()),
        }
    }

    /// Compute all-pairs shortest paths.
    ///
    /// Args:
    ///     weight: Optional edge property name for weights (default: 1.0)
    ///
    /// Returns:
    ///     Dict with structure:
    ///         {(source, target): distance}
    ///     for all reachable pairs.
    #[pyo3(signature = (weight=None))]
    fn all_pairs_shortest_paths(
        &self,
        weight: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        use grafeo_adapters::plugins::algorithms;

        let db = self.db.read();
        let store = db.store();

        let result = algorithms::floyd_warshall(store, weight);

        let dict = PyDict::new(py);
        let nodes = result.nodes();
        for from_node in nodes {
            for to_node in nodes {
                if let Some(dist) = result.distance(*from_node, *to_node) {
                    let key = (from_node.0, to_node.0);
                    dict.set_item(key, dist)?;
                }
            }
        }

        Ok(dict.into_any().unbind())
    }

    // ==========================================================================
    // Network Flow Algorithms (solvOR-style)
    // ==========================================================================

    /// Compute maximum flow in a network.
    ///
    /// Uses the Edmonds-Karp algorithm (BFS-based Ford-Fulkerson).
    ///
    /// Args:
    ///     source: Source node ID
    ///     sink: Sink node ID
    ///     capacity: Optional edge property name for capacities (default: 1.0)
    ///
    /// Returns:
    ///     Dict with 'max_flow' and 'flow_edges' keys.
    ///     'flow_edges' is a list of (src, dst, flow) tuples.
    #[pyo3(signature = (source, sink, capacity=None))]
    fn max_flow(
        &self,
        source: u64,
        sink: u64,
        capacity: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        use grafeo_adapters::plugins::algorithms;

        let db = self.db.read();
        let store = db.store();

        match algorithms::max_flow(store, NodeId::new(source), NodeId::new(sink), capacity) {
            Some(result) => {
                let flow_edges: Vec<(u64, u64, f64)> = result
                    .flow_edges
                    .iter()
                    .map(|(src, dst, f)| (src.0, dst.0, *f))
                    .collect();

                let dict = PyDict::new(py);
                dict.set_item("max_flow", result.max_flow)?;
                dict.set_item("flow_edges", flow_edges.into_pyobject(py)?)?;

                Ok(dict.into_any().unbind())
            }
            None => {
                Err(PyGrafeoError::InvalidArgument("Invalid source or sink node".into()).into())
            }
        }
    }

    /// Compute minimum cost maximum flow.
    ///
    /// Args:
    ///     source: Source node ID
    ///     sink: Sink node ID
    ///     capacity: Optional edge property name for capacities (default: 1.0)
    ///     cost: Optional edge property name for costs (default: 0.0)
    ///
    /// Returns:
    ///     Dict with 'max_flow', 'total_cost', and 'flow_edges' keys.
    ///     'flow_edges' is a list of (src, dst, flow, cost) tuples.
    #[pyo3(signature = (source, sink, capacity=None, cost=None))]
    fn min_cost_max_flow(
        &self,
        source: u64,
        sink: u64,
        capacity: Option<&str>,
        cost: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        use grafeo_adapters::plugins::algorithms;

        let db = self.db.read();
        let store = db.store();

        match algorithms::min_cost_max_flow(
            store,
            NodeId::new(source),
            NodeId::new(sink),
            capacity,
            cost,
        ) {
            Some(result) => {
                let flow_edges: Vec<(u64, u64, f64, f64)> = result
                    .flow_edges
                    .iter()
                    .map(|(src, dst, f, c)| (src.0, dst.0, *f, *c))
                    .collect();

                let dict = PyDict::new(py);
                dict.set_item("max_flow", result.max_flow)?;
                dict.set_item("total_cost", result.total_cost)?;
                dict.set_item("flow_edges", flow_edges.into_pyobject(py)?)?;

                Ok(dict.into_any().unbind())
            }
            None => {
                Err(PyGrafeoError::InvalidArgument("Invalid source or sink node".into()).into())
            }
        }
    }

    // ==========================================================================
    // Minimum Spanning Tree (solvOR-style)
    // ==========================================================================

    /// Compute minimum spanning tree.
    ///
    /// Args:
    ///     weight: Optional edge property name for weights (default: 1.0)
    ///     method: Algorithm to use: "kruskal" or "prim" (default: "kruskal")
    ///
    /// Returns:
    ///     Dict with 'edges' and 'total_weight' keys.
    ///     'edges' is a list of (src, dst, weight) tuples.
    #[pyo3(signature = (weight=None, method="kruskal"))]
    fn minimum_spanning_tree(
        &self,
        weight: Option<&str>,
        method: &str,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        use grafeo_adapters::plugins::algorithms;

        let db = self.db.read();
        let store = db.store();

        let result = match method {
            "kruskal" => algorithms::kruskal(store, weight),
            "prim" => algorithms::prim(store, weight, None),
            _ => {
                return Err(PyGrafeoError::InvalidArgument(format!(
                    "Unknown method: {}. Use 'kruskal' or 'prim'",
                    method
                ))
                .into());
            }
        };

        let edges: Vec<(u64, u64, f64)> = result
            .edges
            .iter()
            .map(|(src, dst, _, w)| (src.0, dst.0, *w))
            .collect();

        let dict = PyDict::new(py);
        dict.set_item("edges", edges.into_pyobject(py)?)?;
        dict.set_item("total_weight", result.total_weight)?;

        Ok(dict.into_any().unbind())
    }

    // ==========================================================================
    // Component Analysis (solvOR-style)
    // ==========================================================================

    /// Find connected components.
    ///
    /// Returns:
    ///     Dict mapping node ID to component ID.
    fn connected_components(&self) -> PyResult<HashMap<u64, u64>> {
        use grafeo_adapters::plugins::algorithms;

        let db = self.db.read();
        let store = db.store();
        let result = algorithms::connected_components(store);
        Ok(result.into_iter().map(|(n, c)| (n.0, c)).collect())
    }

    /// Find strongly connected components.
    ///
    /// Returns:
    ///     Dict mapping node ID to SCC ID.
    fn strongly_connected_components(&self) -> PyResult<HashMap<u64, u64>> {
        use grafeo_adapters::plugins::algorithms;

        let db = self.db.read();
        let store = db.store();
        let result = algorithms::strongly_connected_components(store);
        Ok(result.into_iter().map(|(n, c)| (n.0, c)).collect())
    }

    /// Topological sort of the graph.
    ///
    /// Returns:
    ///     List of node IDs in topological order, or None if graph has cycle.
    fn topological_sort(&self) -> PyResult<Option<Vec<u64>>> {
        use grafeo_adapters::plugins::algorithms;

        let db = self.db.read();
        let store = db.store();
        Ok(algorithms::topological_sort(store).map(|v| v.into_iter().map(|n| n.0).collect()))
    }

    // ==========================================================================
    // Centrality Algorithms (solvOR-style)
    // ==========================================================================

    /// Compute PageRank scores.
    ///
    /// Args:
    ///     damping: Damping factor (default: 0.85)
    ///     max_iter: Maximum iterations (default: 100)
    ///     tol: Convergence tolerance (default: 1e-6)
    ///
    /// Returns:
    ///     Dict mapping node ID to PageRank score.
    #[pyo3(signature = (damping=0.85, max_iter=100, tol=1e-6))]
    fn pagerank(&self, damping: f64, max_iter: usize, tol: f64) -> PyResult<HashMap<u64, f64>> {
        use grafeo_adapters::plugins::algorithms;

        let db = self.db.read();
        let store = db.store();
        let result = algorithms::pagerank(store, damping, max_iter, tol);
        Ok(result.into_iter().map(|(n, s)| (n.0, s)).collect())
    }

    /// Compute betweenness centrality.
    ///
    /// Args:
    ///     normalized: If True, normalize scores (default: True)
    ///
    /// Returns:
    ///     Dict mapping node ID to betweenness score.
    #[pyo3(signature = (normalized=true))]
    fn betweenness_centrality(&self, normalized: bool) -> PyResult<HashMap<u64, f64>> {
        use grafeo_adapters::plugins::algorithms;

        let db = self.db.read();
        let store = db.store();
        let result = algorithms::betweenness_centrality(store, normalized);
        Ok(result.into_iter().map(|(n, s)| (n.0, s)).collect())
    }

    // ==========================================================================
    // Community Detection (solvOR-style)
    // ==========================================================================

    /// Detect communities using Louvain algorithm.
    ///
    /// Args:
    ///     resolution: Resolution parameter (default: 1.0)
    ///
    /// Returns:
    ///     Dict with 'communities', 'modularity', and 'num_communities' keys.
    #[pyo3(signature = (resolution=1.0))]
    fn louvain(&self, resolution: f64, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use grafeo_adapters::plugins::algorithms;

        let db = self.db.read();
        let store = db.store();
        let result = algorithms::louvain(store, resolution);

        let communities: HashMap<u64, u64> = result
            .communities
            .into_iter()
            .map(|(n, c)| (n.0, c))
            .collect();

        let dict = PyDict::new(py);
        dict.set_item("communities", communities.into_pyobject(py)?)?;
        dict.set_item("modularity", result.modularity)?;
        dict.set_item("num_communities", result.num_communities)?;

        Ok(dict.into_any().unbind())
    }

    // ==========================================================================
    // Graph Structure (solvOR-style)
    // ==========================================================================

    /// Find articulation points (cut vertices).
    ///
    /// Returns:
    ///     List of node IDs that are articulation points.
    fn articulation_points(&self) -> PyResult<Vec<u64>> {
        use grafeo_adapters::plugins::algorithms;

        let db = self.db.read();
        let store = db.store();
        let result = algorithms::articulation_points(store);
        Ok(result.into_iter().map(|n| n.0).collect())
    }

    /// Find bridges (cut edges).
    ///
    /// Returns:
    ///     List of (source, target) tuples representing bridges.
    fn bridges(&self) -> PyResult<Vec<(u64, u64)>> {
        use grafeo_adapters::plugins::algorithms;

        let db = self.db.read();
        let store = db.store();
        let result = algorithms::bridges(store);
        Ok(result.into_iter().map(|(s, t)| (s.0, t.0)).collect())
    }

    /// Get graph statistics.
    ///
    /// Returns:
    ///     Dict with 'nodes', 'edges', 'density', and 'components' keys.
    fn graph_stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use grafeo_adapters::plugins::algorithms;

        let db = self.db.read();
        let store = db.store();

        let n = db.node_count();
        let e = db.edge_count();
        let density = if n > 1 {
            (e as f64) / ((n * (n - 1)) as f64)
        } else {
            0.0
        };
        let components = algorithms::connected_component_count(store);

        let dict = PyDict::new(py);
        dict.set_item("nodes", n)?;
        dict.set_item("edges", e)?;
        dict.set_item("density", density)?;
        dict.set_item("components", components)?;

        Ok(dict.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        let db = self.db.read();
        format!(
            "SolvORAdapter(nodes={}, edges={})",
            db.node_count(),
            db.edge_count()
        )
    }
}
