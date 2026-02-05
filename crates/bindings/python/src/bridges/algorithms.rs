//! Run graph algorithms directly from Python with Rust performance.
//!
//! Access via `db.algorithms` - all the classic algorithms are here:
//! traversals, shortest paths, centrality measures, community detection,
//! spanning trees, and network flow.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use grafeo_adapters::plugins::algorithms;
use grafeo_common::types::NodeId;
use grafeo_engine::database::GrafeoDB;

use crate::error::PyGrafeoError;

/// Run graph algorithms at Rust speed from Python.
///
/// Get this via `db.algorithms`. All algorithms run directly on the Rust
/// graph store - no copying to Python data structures. Results come back
/// as Python dicts and lists.
#[pyclass(name = "Algorithms")]
pub struct PyAlgorithms {
    db: Arc<RwLock<GrafeoDB>>,
}

impl PyAlgorithms {
    /// Creates a new algorithms interface for the given database.
    pub fn new(db: Arc<RwLock<GrafeoDB>>) -> Self {
        Self { db }
    }
}

#[pymethods]
impl PyAlgorithms {
    // ==========================================================================
    // Traversal Algorithms
    // ==========================================================================

    /// Breadth-first search from a starting node.
    ///
    /// Args:
    ///     start: Starting node ID
    ///
    /// Returns:
    ///     List of node IDs in BFS order
    fn bfs(&self, start: u64) -> PyResult<Vec<u64>> {
        let db = self.db.read();
        let store = db.store();
        let result = algorithms::bfs(store, NodeId::new(start));
        Ok(result.into_iter().map(|n| n.0).collect())
    }

    /// BFS layers - returns nodes grouped by distance from start.
    ///
    /// Args:
    ///     start: Starting node ID
    ///
    /// Returns:
    ///     List of lists, where result[i] contains nodes at distance i
    fn bfs_layers(&self, start: u64) -> PyResult<Vec<Vec<u64>>> {
        let db = self.db.read();
        let store = db.store();
        let layers = algorithms::bfs_layers(store, NodeId::new(start));
        Ok(layers
            .into_iter()
            .map(|layer| layer.into_iter().map(|n| n.0).collect())
            .collect())
    }

    /// Depth-first search from a starting node.
    ///
    /// Args:
    ///     start: Starting node ID
    ///
    /// Returns:
    ///     List of node IDs in post-order (finished order)
    fn dfs(&self, start: u64) -> PyResult<Vec<u64>> {
        let db = self.db.read();
        let store = db.store();
        let result = algorithms::dfs(store, NodeId::new(start));
        Ok(result.into_iter().map(|n| n.0).collect())
    }

    /// DFS visiting all nodes in the graph.
    ///
    /// Returns:
    ///     List of all node IDs in DFS post-order
    fn dfs_all(&self) -> PyResult<Vec<u64>> {
        let db = self.db.read();
        let store = db.store();
        let result = algorithms::dfs_all(store);
        Ok(result.into_iter().map(|n| n.0).collect())
    }

    // ==========================================================================
    // Component Algorithms
    // ==========================================================================

    /// Find connected components (treating graph as undirected).
    ///
    /// Returns:
    ///     Dict mapping node ID to component ID
    fn connected_components(&self) -> PyResult<HashMap<u64, u64>> {
        let db = self.db.read();
        let store = db.store();
        let result = algorithms::connected_components(store);
        Ok(result.into_iter().map(|(n, c)| (n.0, c)).collect())
    }

    /// Count the number of connected components.
    fn connected_component_count(&self) -> PyResult<usize> {
        let db = self.db.read();
        let store = db.store();
        Ok(algorithms::connected_component_count(store))
    }

    /// Find strongly connected components.
    ///
    /// Returns:
    ///     List of lists, each inner list is a strongly connected component
    fn strongly_connected_components(&self) -> PyResult<Vec<Vec<u64>>> {
        let db = self.db.read();
        let store = db.store();
        let result = algorithms::strongly_connected_components(store);

        // Group nodes by component ID
        let mut grouped: HashMap<u64, Vec<u64>> = HashMap::new();
        for (node, comp_id) in result {
            grouped.entry(comp_id).or_default().push(node.0);
        }

        Ok(grouped.into_values().collect())
    }

    /// Topological sort of the graph.
    ///
    /// Returns:
    ///     List of node IDs in topological order, or None if graph has cycle
    fn topological_sort(&self) -> PyResult<Option<Vec<u64>>> {
        let db = self.db.read();
        let store = db.store();
        Ok(algorithms::topological_sort(store).map(|v| v.into_iter().map(|n| n.0).collect()))
    }

    /// Check if the graph is a DAG.
    fn is_dag(&self) -> PyResult<bool> {
        let db = self.db.read();
        let store = db.store();
        Ok(algorithms::is_dag(store))
    }

    // ==========================================================================
    // Shortest Path Algorithms
    // ==========================================================================

    /// Dijkstra's shortest path algorithm.
    ///
    /// Args:
    ///     source: Source node ID
    ///     target: Optional target node ID (returns single path if provided)
    ///     weight: Optional edge property name for weights (default: 1.0)
    ///
    /// Returns:
    ///     If target is None: Dict mapping node ID to distance
    ///     If target is provided: Tuple of (distance, path) or None if unreachable
    #[pyo3(signature = (source, target=None, weight=None))]
    fn dijkstra(
        &self,
        source: u64,
        target: Option<u64>,
        weight: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = db.store();

        if let Some(target_id) = target {
            match algorithms::dijkstra_path(
                store,
                NodeId::new(source),
                NodeId::new(target_id),
                weight,
            ) {
                Some((dist, path)) => {
                    let path_list: Vec<u64> = path.into_iter().map(|n| n.0).collect();
                    Ok((dist, path_list).into_pyobject(py)?.into_any().unbind())
                }
                None => Ok(py.None()),
            }
        } else {
            let result = algorithms::dijkstra(store, NodeId::new(source), weight);
            let distances: HashMap<u64, f64> = result
                .distances
                .into_iter()
                .map(|(n, d)| (n.0, d))
                .collect();
            Ok(distances.into_pyobject(py)?.into_any().unbind())
        }
    }

    /// A* shortest path algorithm.
    ///
    /// Args:
    ///     source: Source node ID
    ///     target: Target node ID
    ///     heuristic: Optional dict mapping node ID to heuristic value
    ///     weight: Optional edge property name for weights
    ///
    /// Returns:
    ///     Tuple of (distance, path) or None if unreachable
    #[pyo3(signature = (source, target, heuristic=None, weight=None))]
    fn astar(
        &self,
        source: u64,
        target: u64,
        heuristic: Option<&Bound<'_, PyDict>>,
        weight: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = db.store();

        // Build heuristic function
        let h_map: HashMap<u64, f64> = if let Some(h) = heuristic {
            let mut map = HashMap::new();
            for (k, v) in h.iter() {
                let node_id: u64 = k.extract()?;
                let value: f64 = v.extract()?;
                map.insert(node_id, value);
            }
            map
        } else {
            HashMap::new()
        };

        let heuristic_fn = |n: NodeId| -> f64 { h_map.get(&n.0).copied().unwrap_or(0.0) };

        match algorithms::astar(
            store,
            NodeId::new(source),
            NodeId::new(target),
            weight,
            heuristic_fn,
        ) {
            Some((dist, path)) => {
                let path_list: Vec<u64> = path.into_iter().map(|n| n.0).collect();
                Ok((dist, path_list).into_pyobject(py)?.into_any().unbind())
            }
            None => Ok(py.None()),
        }
    }

    /// Bellman-Ford shortest path algorithm (handles negative weights).
    ///
    /// Args:
    ///     source: Source node ID
    ///     weight: Optional edge property name for weights
    ///
    /// Returns:
    ///     Dict with 'distances', 'predecessors', and 'has_negative_cycle' keys
    #[pyo3(signature = (source, weight=None))]
    fn bellman_ford(
        &self,
        source: u64,
        weight: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = db.store();

        let result = algorithms::bellman_ford(store, NodeId::new(source), weight);

        let distances: HashMap<u64, f64> = result
            .distances
            .into_iter()
            .map(|(n, d)| (n.0, d))
            .collect();
        let predecessors: HashMap<u64, u64> = result
            .predecessors
            .into_iter()
            .map(|(n, p)| (n.0, p.0))
            .collect();

        let dict = PyDict::new(py);
        dict.set_item("distances", distances.into_pyobject(py)?)?;
        dict.set_item("predecessors", predecessors.into_pyobject(py)?)?;
        dict.set_item("has_negative_cycle", result.has_negative_cycle)?;

        Ok(dict.into_any().unbind())
    }

    /// Floyd-Warshall all-pairs shortest paths.
    ///
    /// Args:
    ///     weight: Optional edge property name for weights
    ///
    /// Returns:
    ///     Dict mapping (source, target) tuples to distances
    #[pyo3(signature = (weight=None))]
    fn floyd_warshall(&self, weight: Option<&str>, py: Python<'_>) -> PyResult<Py<PyAny>> {
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
    // Centrality Algorithms
    // ==========================================================================

    /// Compute degree centrality for all nodes.
    ///
    /// Args:
    ///     normalized: If True, normalize by (n-1) (default: False)
    ///
    /// Returns:
    ///     Dict mapping node ID to centrality score
    #[pyo3(signature = (normalized=false))]
    fn degree_centrality(&self, normalized: bool, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = db.store();

        if normalized {
            let result = algorithms::degree_centrality_normalized(store);
            let scores: HashMap<u64, f64> = result.into_iter().map(|(n, s)| (n.0, s)).collect();
            Ok(scores.into_pyobject(py)?.into_any().unbind())
        } else {
            let result = algorithms::degree_centrality(store);
            let dict = PyDict::new(py);
            for (node, total) in result.total_degree {
                let in_d = *result.in_degree.get(&node).unwrap_or(&0);
                let out_d = *result.out_degree.get(&node).unwrap_or(&0);
                let node_dict = PyDict::new(py);
                node_dict.set_item("in_degree", in_d)?;
                node_dict.set_item("out_degree", out_d)?;
                node_dict.set_item("total_degree", total)?;
                dict.set_item(node.0, node_dict)?;
            }
            Ok(dict.into_any().unbind())
        }
    }

    /// Compute PageRank scores.
    ///
    /// Args:
    ///     damping: Damping factor (default: 0.85)
    ///     max_iterations: Maximum iterations (default: 100)
    ///     tolerance: Convergence tolerance (default: 1e-6)
    ///
    /// Returns:
    ///     Dict mapping node ID to PageRank score
    #[pyo3(signature = (damping=0.85, max_iterations=100, tolerance=1e-6))]
    fn pagerank(
        &self,
        damping: f64,
        max_iterations: usize,
        tolerance: f64,
    ) -> PyResult<HashMap<u64, f64>> {
        let db = self.db.read();
        let store = db.store();
        let result = algorithms::pagerank(store, damping, max_iterations, tolerance);
        Ok(result.into_iter().map(|(n, s)| (n.0, s)).collect())
    }

    /// Compute betweenness centrality using Brandes' algorithm.
    ///
    /// Args:
    ///     normalized: If True, normalize scores (default: True)
    ///
    /// Returns:
    ///     Dict mapping node ID to betweenness score
    #[pyo3(signature = (normalized=true))]
    fn betweenness_centrality(&self, normalized: bool) -> PyResult<HashMap<u64, f64>> {
        let db = self.db.read();
        let store = db.store();
        let result = algorithms::betweenness_centrality(store, normalized);
        Ok(result.into_iter().map(|(n, s)| (n.0, s)).collect())
    }

    /// Compute closeness centrality.
    ///
    /// Args:
    ///     wf_improved: Use Wasserman-Faust formula (default: False)
    ///
    /// Returns:
    ///     Dict mapping node ID to closeness score
    #[pyo3(signature = (wf_improved=false))]
    fn closeness_centrality(&self, wf_improved: bool) -> PyResult<HashMap<u64, f64>> {
        let db = self.db.read();
        let store = db.store();
        let result = algorithms::closeness_centrality(store, wf_improved);
        Ok(result.into_iter().map(|(n, s)| (n.0, s)).collect())
    }

    // ==========================================================================
    // Community Detection
    // ==========================================================================

    /// Detect communities using Label Propagation.
    ///
    /// Args:
    ///     max_iterations: Maximum iterations (default: 100, 0 for unlimited)
    ///
    /// Returns:
    ///     Dict mapping node ID to community ID
    #[pyo3(signature = (max_iterations=100))]
    fn label_propagation(&self, max_iterations: usize) -> PyResult<HashMap<u64, u64>> {
        let db = self.db.read();
        let store = db.store();
        let result = algorithms::label_propagation(store, max_iterations);
        Ok(result.into_iter().map(|(n, c)| (n.0, c)).collect())
    }

    /// Detect communities using Louvain algorithm.
    ///
    /// Args:
    ///     resolution: Resolution parameter (default: 1.0)
    ///
    /// Returns:
    ///     Dict with 'communities', 'modularity', and 'num_communities' keys
    #[pyo3(signature = (resolution=1.0))]
    fn louvain(&self, resolution: f64, py: Python<'_>) -> PyResult<Py<PyAny>> {
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
    // Minimum Spanning Tree
    // ==========================================================================

    /// Compute MST using Kruskal's algorithm.
    ///
    /// Args:
    ///     weight: Edge property name for weights (default: 1.0)
    ///
    /// Returns:
    ///     Dict with 'edges' (list of (src, dst, weight)) and 'total_weight'
    #[pyo3(signature = (weight=None))]
    fn kruskal(&self, weight: Option<&str>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = db.store();
        let result = algorithms::kruskal(store, weight);

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

    /// Compute MST using Prim's algorithm.
    ///
    /// Args:
    ///     weight: Edge property name for weights (default: 1.0)
    ///     start: Starting node ID (optional)
    ///
    /// Returns:
    ///     Dict with 'edges' (list of (src, dst, weight)) and 'total_weight'
    #[pyo3(signature = (weight=None, start=None))]
    fn prim(
        &self,
        weight: Option<&str>,
        start: Option<u64>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = db.store();
        let start_node = start.map(NodeId::new);
        let result = algorithms::prim(store, weight, start_node);

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
    // Network Flow
    // ==========================================================================

    /// Compute maximum flow using Edmonds-Karp.
    ///
    /// Args:
    ///     source: Source node ID
    ///     sink: Sink node ID
    ///     capacity: Edge property name for capacities (default: 1.0)
    ///
    /// Returns:
    ///     Dict with 'max_flow' and 'flow_edges' (list of (src, dst, flow))
    #[pyo3(signature = (source, sink, capacity=None))]
    fn max_flow(
        &self,
        source: u64,
        sink: u64,
        capacity: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
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
    ///     capacity: Edge property name for capacities (default: 1.0)
    ///     cost: Edge property name for costs (default: 0.0)
    ///
    /// Returns:
    ///     Dict with 'max_flow', 'total_cost', and 'flow_edges'
    #[pyo3(signature = (source, sink, capacity=None, cost=None))]
    fn min_cost_max_flow(
        &self,
        source: u64,
        sink: u64,
        capacity: Option<&str>,
        cost: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
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
    // Clustering Algorithms
    // ==========================================================================

    /// Compute local and global clustering coefficients.
    ///
    /// The clustering coefficient measures how close a node's neighbors are
    /// to being a complete graph (clique).
    ///
    /// Args:
    ///     parallel: Enable parallel computation (default: True)
    ///
    /// Returns:
    ///     Dict with 'coefficients', 'triangle_counts', 'total_triangles',
    ///     and 'global_coefficient' keys
    #[pyo3(signature = (parallel=true))]
    fn clustering_coefficient(&self, parallel: bool, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = db.store();

        let result = if parallel {
            algorithms::clustering_coefficient_parallel(store, 50)
        } else {
            algorithms::clustering_coefficient(store)
        };

        let coefficients: HashMap<u64, f64> = result
            .coefficients
            .into_iter()
            .map(|(n, c)| (n.0, c))
            .collect();
        let triangle_counts: HashMap<u64, u64> = result
            .triangle_counts
            .into_iter()
            .map(|(n, t)| (n.0, t))
            .collect();

        let dict = PyDict::new(py);
        dict.set_item("coefficients", coefficients.into_pyobject(py)?)?;
        dict.set_item("triangle_counts", triangle_counts.into_pyobject(py)?)?;
        dict.set_item("total_triangles", result.total_triangles)?;
        dict.set_item("global_coefficient", result.global_coefficient)?;

        Ok(dict.into_any().unbind())
    }

    /// Count the number of triangles containing each node.
    ///
    /// Returns:
    ///     Dict mapping node ID to triangle count
    fn triangle_count(&self) -> PyResult<HashMap<u64, u64>> {
        let db = self.db.read();
        let store = db.store();
        let result = algorithms::triangle_count(store);
        Ok(result.into_iter().map(|(n, t)| (n.0, t)).collect())
    }

    /// Get the total number of unique triangles in the graph.
    ///
    /// Each triangle is counted exactly once.
    ///
    /// Returns:
    ///     Total unique triangle count
    fn total_triangles(&self) -> PyResult<u64> {
        let db = self.db.read();
        let store = db.store();
        Ok(algorithms::total_triangles(store))
    }

    /// Compute the global (average) clustering coefficient.
    ///
    /// Returns:
    ///     Average clustering coefficient across all nodes (0.0 to 1.0)
    fn global_clustering_coefficient(&self) -> PyResult<f64> {
        let db = self.db.read();
        let store = db.store();
        Ok(algorithms::global_clustering_coefficient(store))
    }

    /// Compute local clustering coefficients for each node.
    ///
    /// Returns:
    ///     Dict mapping node ID to local clustering coefficient (0.0 to 1.0)
    fn local_clustering_coefficient(&self) -> PyResult<HashMap<u64, f64>> {
        let db = self.db.read();
        let store = db.store();
        let result = algorithms::local_clustering_coefficient(store);
        Ok(result.into_iter().map(|(n, c)| (n.0, c)).collect())
    }

    // ==========================================================================
    // Structure Analysis
    // ==========================================================================

    /// Find articulation points (cut vertices).
    ///
    /// Returns:
    ///     List of node IDs that are articulation points
    fn articulation_points(&self) -> PyResult<Vec<u64>> {
        let db = self.db.read();
        let store = db.store();
        let result = algorithms::articulation_points(store);
        Ok(result.into_iter().map(|n| n.0).collect())
    }

    /// Find bridges (cut edges).
    ///
    /// Returns:
    ///     List of (source, target) tuples representing bridges
    fn bridges(&self) -> PyResult<Vec<(u64, u64)>> {
        let db = self.db.read();
        let store = db.store();
        let result = algorithms::bridges(store);
        Ok(result.into_iter().map(|(s, t)| (s.0, t.0)).collect())
    }

    /// Compute k-core decomposition.
    ///
    /// Args:
    ///     k: If provided, return only nodes in the k-core
    ///
    /// Returns:
    ///     If k is None: Dict mapping node ID to core number
    ///     If k is provided: List of node IDs in the k-core
    #[pyo3(signature = (k=None))]
    fn kcore(&self, k: Option<usize>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = db.store();
        let result = algorithms::kcore_decomposition(store);

        if let Some(k_val) = k {
            let nodes: Vec<u64> = result.k_core(k_val).into_iter().map(|n| n.0).collect();
            Ok(nodes.into_pyobject(py)?.into_any().unbind())
        } else {
            let dict = PyDict::new(py);
            for (node, core) in result.core_numbers {
                dict.set_item(node.0, core)?;
            }
            dict.set_item("max_core", result.max_core)?;
            Ok(dict.into_any().unbind())
        }
    }

    fn __repr__(&self) -> String {
        "Algorithms()".to_string()
    }
}
