//! Run graph algorithms directly from Python with Rust performance.
//!
//! Access via `db.algorithms` - all the classic algorithms are here:
//! traversals, shortest paths, centrality measures, community detection,
//! spanning trees, and network flow.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use parking_lot::RwLock;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use grafeo_adapters::plugins::algorithms;
use grafeo_common::types::NodeId;
use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind};
use grafeo_core::graph::GraphStoreSearch;
use grafeo_engine::database::GrafeoDB;

use crate::error::PyGrafeoError;
use crate::types::PyValue;

/// The nodes of a store in the order of a key (`key=`), with their values.
struct KeyOrder {
    nodes: Vec<NodeId>,
    values: HashMap<NodeId, Value>,
}

/// The order of `key` over `store`, or `None` without a key. A node without
/// the key, or two with equal values, is an error, raised before the
/// algorithm runs.
fn key_order(store: &dyn GraphStoreSearch, key: Option<&str>) -> PyResult<Option<KeyOrder>> {
    let Some(key) = key else {
        return Ok(None);
    };
    let ordered = algorithms::order_by_key(store, key).map_err(|error| {
        PyErr::from(PyGrafeoError::from(Error::Query(QueryError::new(
            QueryErrorKind::Semantic,
            error.to_string(),
        ))))
    })?;
    let nodes = ordered.iter().map(|(node, _)| *node).collect();
    Ok(Some(KeyOrder {
        nodes,
        values: ordered.into_iter().collect(),
    }))
}

/// A per-node result as a dict: by node id in node-id order, or with `key=`
/// by key value in key order.
fn node_dict<'py, T>(
    py: Python<'py>,
    result: impl IntoIterator<Item = (NodeId, T)>,
    order: Option<&KeyOrder>,
) -> PyResult<Py<PyAny>>
where
    T: IntoPyObject<'py>,
{
    let mut by_node: BTreeMap<NodeId, T> = result.into_iter().collect();
    let dict = PyDict::new(py);
    match order {
        Some(order) => {
            for node in &order.nodes {
                if let Some(value) = by_node.remove(node) {
                    dict.set_item(PyValue::to_py(&order.values[node], py), value)?;
                }
            }
        }
        None => {
            for (node, value) in by_node {
                dict.set_item(node.0, value)?;
            }
        }
    }
    Ok(dict.into_any().unbind())
}

/// A set of nodes as a list: node ids in node-id order, or with `key=` key
/// values in key order.
fn node_list(
    py: Python<'_>,
    nodes: impl IntoIterator<Item = NodeId>,
    order: Option<&KeyOrder>,
) -> PyResult<Py<PyAny>> {
    match order {
        Some(order) => {
            let members: HashSet<NodeId> = nodes.into_iter().collect();
            let values: Vec<Py<PyAny>> = order
                .nodes
                .iter()
                .filter(|node| members.contains(node))
                .map(|node| PyValue::to_py(&order.values[node], py))
                .collect();
            Ok(values.into_pyobject(py)?.into_any().unbind())
        }
        None => {
            let mut ids: Vec<u64> = nodes.into_iter().map(|node| node.0).collect();
            ids.sort_unstable();
            Ok(ids.into_pyobject(py)?.into_any().unbind())
        }
    }
}

/// Run graph algorithms at Rust speed from Python.
///
/// Get this via `db.algorithms` or `db.graph(name).algorithms`. All
/// algorithms run directly on the Rust graph store, with no copying to Python
/// data structures; results come back as Python dicts and lists.
///
/// Which graph they read: `db.algorithms` reads the graph `set_graph()`
/// selects (the default graph when none is selected), like `execute()` and
/// `CALL grafeo.<algorithm>()`; `db.graph(name).algorithms` reads that graph.
/// Every method takes a keyword-only `projection=` that names a projection
/// (`create_projection()`) to read instead. The methods that return a value
/// per node or a set of nodes also take `key=`: a node property every node in
/// scope holds once (an id from outside the database), which keys the result
/// instead of the node id. PageRank, Louvain and label propagation then also
/// run in key order, so their results do not depend on the order the nodes
/// and edges were inserted in; the others only change their keys.
#[pyclass(name = "Algorithms")]
pub struct PyAlgorithms {
    db: Arc<RwLock<GrafeoDB>>,
    /// The named graph (schema, name) of a `GraphHandle`; `None` reads the
    /// graph `set_graph()` selects.
    graph: Option<(Option<String>, String)>,
}

impl PyAlgorithms {
    /// The algorithms of the database: they read the graph `set_graph()`
    /// selects, or the default graph when none is selected.
    pub fn new(db: Arc<RwLock<GrafeoDB>>) -> Self {
        Self { db, graph: None }
    }

    /// The algorithms of one named graph (`db.graph(name).algorithms`).
    pub fn for_graph(db: Arc<RwLock<GrafeoDB>>, schema: Option<String>, name: String) -> Self {
        Self {
            db,
            graph: Some((schema, name)),
        }
    }

    /// The store an algorithm reads: the projection when the call names one
    /// (projection names are database-wide, so it wins over the graph), else
    /// the handle's graph, else the selected graph. A graph or projection that
    /// does not exist is an error, never another graph.
    fn store_for(
        &self,
        db: &GrafeoDB,
        projection: Option<&str>,
    ) -> PyResult<Arc<dyn GraphStoreSearch>> {
        if let Some(name) = projection {
            return db.projection(name).ok_or_else(|| {
                PyGrafeoError::from(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    format!("Projection '{name}' does not exist"),
                )))
                .into()
            });
        }
        let store = match &self.graph {
            Some((schema, name)) => db
                .graph_in(schema.as_deref(), name)
                .and_then(|graph| graph.graph_store()),
            None => db.selected_graph_store(),
        };
        Ok(store.map_err(PyGrafeoError::from)?)
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
    #[pyo3(signature = (start, *, projection=None))]
    fn bfs(&self, start: u64, projection: Option<&str>) -> PyResult<Vec<u64>> {
        let db = self.db.read();
        let store = self.store_for(&db, projection)?;
        let result = algorithms::bfs(&*store, NodeId::new(start));
        Ok(result.into_iter().map(|n| n.0).collect())
    }

    /// BFS layers - returns nodes grouped by distance from start.
    ///
    /// Args:
    ///     start: Starting node ID
    ///
    /// Returns:
    ///     List of lists, where result[i] contains nodes at distance i
    #[pyo3(signature = (start, *, projection=None))]
    fn bfs_layers(&self, start: u64, projection: Option<&str>) -> PyResult<Vec<Vec<u64>>> {
        let db = self.db.read();
        let store = self.store_for(&db, projection)?;
        let layers = algorithms::bfs_layers(&*store, NodeId::new(start));
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
    #[pyo3(signature = (start, *, projection=None))]
    fn dfs(&self, start: u64, projection: Option<&str>) -> PyResult<Vec<u64>> {
        let db = self.db.read();
        let store = self.store_for(&db, projection)?;
        let result = algorithms::dfs(&*store, NodeId::new(start));
        Ok(result.into_iter().map(|n| n.0).collect())
    }

    /// DFS visiting all nodes in the graph.
    ///
    /// Returns:
    ///     List of all node IDs in DFS post-order
    #[pyo3(signature = (*, projection=None))]
    fn dfs_all(&self, projection: Option<&str>) -> PyResult<Vec<u64>> {
        let db = self.db.read();
        let store = self.store_for(&db, projection)?;
        let result = algorithms::dfs_all(&*store);
        Ok(result.into_iter().map(|n| n.0).collect())
    }

    // ==========================================================================
    // Component Algorithms
    // ==========================================================================

    /// Find connected components (treating graph as undirected).
    ///
    /// Args:
    ///     key: A node property every node in scope holds, each with its own
    ///         value (an id from outside the database). The result is keyed by
    ///         it instead of the node id, in key order. A node without it, or
    ///         two with equal values, raise GrafeoError.
    ///
    /// Returns:
    ///     Dict mapping node ID (or key) to component ID
    #[pyo3(signature = (*, projection=None, key=None))]
    fn connected_components(
        &self,
        projection: Option<&str>,
        key: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = self.store_for(&db, projection)?;
        let order = key_order(&*store, key)?;
        let result = algorithms::connected_components(&*store);
        node_dict(py, result, order.as_ref())
    }

    /// Count the number of connected components.
    #[pyo3(signature = (*, projection=None))]
    fn connected_component_count(&self, projection: Option<&str>) -> PyResult<usize> {
        let db = self.db.read();
        let store = self.store_for(&db, projection)?;
        Ok(algorithms::connected_component_count(&*store))
    }

    /// Find strongly connected components.
    ///
    /// Returns:
    ///     List of lists, each inner list is a strongly connected component
    #[pyo3(signature = (*, projection=None))]
    fn strongly_connected_components(&self, projection: Option<&str>) -> PyResult<Vec<Vec<u64>>> {
        let db = self.db.read();
        let store = self.store_for(&db, projection)?;
        let result = algorithms::strongly_connected_components(&*store);

        // Group nodes by component ID
        let mut grouped: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
        for (node, comp_id) in result {
            grouped.entry(comp_id).or_default().push(node.0);
        }

        Ok(grouped.into_values().collect())
    }

    /// Topological sort of the graph.
    ///
    /// Returns:
    ///     List of node IDs in topological order, or None if graph has cycle
    #[pyo3(signature = (*, projection=None))]
    fn topological_sort(&self, projection: Option<&str>) -> PyResult<Option<Vec<u64>>> {
        let db = self.db.read();
        let store = self.store_for(&db, projection)?;
        Ok(algorithms::topological_sort(&*store).map(|v| v.into_iter().map(|n| n.0).collect()))
    }

    /// Check if the graph is a DAG.
    #[pyo3(signature = (*, projection=None))]
    fn is_dag(&self, projection: Option<&str>) -> PyResult<bool> {
        let db = self.db.read();
        let store = self.store_for(&db, projection)?;
        Ok(algorithms::is_dag(&*store))
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
    #[pyo3(signature = (source, target=None, weight=None, *, projection=None))]
    fn dijkstra(
        &self,
        source: u64,
        target: Option<u64>,
        weight: Option<&str>,
        projection: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = self.store_for(&db, projection)?;

        if let Some(target_id) = target {
            match algorithms::dijkstra_path(
                &*store,
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
            let result = algorithms::dijkstra(&*store, NodeId::new(source), weight);
            let distances: BTreeMap<u64, f64> = result
                .distances
                .into_iter()
                .map(|(n, d)| (n.0, d))
                .collect();
            Ok(distances.into_pyobject(py)?.into_any().unbind())
        }
    }

    /// Single-source shortest paths with string node name support.
    ///
    /// LDBC Graphanalytics-compatible API: accepts node names (or numeric IDs
    /// as strings) and returns distances keyed by node name.
    ///
    /// Args:
    ///     source: Source node name (or numeric ID as string)
    ///     weight_attr: Optional edge property name for weights (default: 1.0)
    ///
    /// Returns:
    ///     Dict mapping node name (str) to distance (float)
    #[pyo3(signature = (source, weight_attr=None, *, projection=None))]
    fn sssp(
        &self,
        source: &str,
        weight_attr: Option<&str>,
        projection: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = self.store_for(&db, projection)?;

        // Resolve source: try integer parse first, then name property lookup
        let source_id = if let Ok(id) = source.parse::<u64>() {
            NodeId::new(id)
        } else {
            let candidates = store.find_nodes_by_property("name", &Value::from(source));
            match candidates.len() {
                0 => {
                    return Err(PyGrafeoError::InvalidArgument(format!(
                        "No node found with name '{source}'"
                    ))
                    .into());
                }
                1 => candidates[0],
                _ => {
                    return Err(PyGrafeoError::InvalidArgument(format!(
                        "Multiple nodes found with name '{source}', use node ID instead"
                    ))
                    .into());
                }
            }
        };

        let result = algorithms::dijkstra(&*store, source_id, weight_attr);

        // Map node IDs to names (falling back to string ID)
        let distances: BTreeMap<String, f64> = result
            .distances
            .into_iter()
            .map(|(node, dist)| {
                let name = store
                    .get_node(node)
                    .and_then(|n| n.get_property("name").cloned())
                    .and_then(|v| {
                        if let Value::String(s) = v {
                            Some(s.to_string())
                        } else {
                            None
                        }
                    })
                    .unwrap_or_else(|| node.0.to_string());
                (name, dist)
            })
            .collect();

        Ok(distances.into_pyobject(py)?.into_any().unbind())
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
    #[pyo3(signature = (source, target, heuristic=None, weight=None, *, projection=None))]
    fn astar(
        &self,
        source: u64,
        target: u64,
        heuristic: Option<&Bound<'_, PyDict>>,
        weight: Option<&str>,
        projection: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = self.store_for(&db, projection)?;

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
            &*store,
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
    #[pyo3(signature = (source, weight=None, *, projection=None))]
    fn bellman_ford(
        &self,
        source: u64,
        weight: Option<&str>,
        projection: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = self.store_for(&db, projection)?;

        let result = algorithms::bellman_ford(&*store, NodeId::new(source), weight);

        let distances: BTreeMap<u64, f64> = result
            .distances
            .into_iter()
            .map(|(n, d)| (n.0, d))
            .collect();
        let predecessors: BTreeMap<u64, u64> = result
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
    #[pyo3(signature = (weight=None, *, projection=None))]
    fn floyd_warshall(
        &self,
        weight: Option<&str>,
        projection: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = self.store_for(&db, projection)?;

        let result = algorithms::floyd_warshall(&*store, weight);

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
    ///     key: A node property every node in scope holds, each with its own
    ///         value (an id from outside the database). The result is keyed by
    ///         it instead of the node id, in key order. A node without it, or
    ///         two with equal values, raise GrafeoError.
    ///
    /// Returns:
    ///     Dict mapping node ID (or key) to centrality score
    #[pyo3(signature = (normalized=false, *, projection=None, key=None))]
    fn degree_centrality(
        &self,
        normalized: bool,
        projection: Option<&str>,
        key: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = self.store_for(&db, projection)?;
        let order = key_order(&*store, key)?;

        if normalized {
            let result = algorithms::degree_centrality_normalized(&*store);
            node_dict(py, result, order.as_ref())
        } else {
            let result = algorithms::degree_centrality(&*store);
            let mut degrees = Vec::with_capacity(result.total_degree.len());
            for (&node, &total) in &result.total_degree {
                let in_d = *result.in_degree.get(&node).unwrap_or(&0);
                let out_d = *result.out_degree.get(&node).unwrap_or(&0);
                let node_dict = PyDict::new(py);
                node_dict.set_item("in_degree", in_d)?;
                node_dict.set_item("out_degree", out_d)?;
                node_dict.set_item("total_degree", total)?;
                degrees.push((node, node_dict));
            }
            // In node-id (or key) order, the same in every process.
            node_dict(py, degrees, order.as_ref())
        }
    }

    /// Compute PageRank scores.
    ///
    /// Args:
    ///     damping: Damping factor (default: 0.85)
    ///     max_iterations: Maximum iterations (default: 100)
    ///     tolerance: Convergence tolerance (default: 1e-6)
    ///     directed: Follow edge direction (default: True). False walks the
    ///         simple undirected graph: each pair of connected nodes once, in
    ///         both directions, whatever the edge types or count; self-loops
    ///         are ignored.
    ///     key: A node property every node in scope holds, each with its own
    ///         value (an id from outside the database). The result is keyed by
    ///         it instead of the node id, in key order, and the algorithm runs
    ///         in key order (visits, sums, ties and numbering), so the result
    ///         does not depend on the order nodes and edges were inserted in.
    ///         A node without it, or two with equal values, raise GrafeoError.
    ///
    /// Returns:
    ///     Dict mapping node ID (or key) to PageRank score
    #[pyo3(signature = (damping=0.85, max_iterations=100, tolerance=1e-6, directed=true, *, projection=None, key=None))]
    fn pagerank(
        &self,
        damping: f64,
        max_iterations: usize,
        tolerance: f64,
        directed: bool,
        projection: Option<&str>,
        key: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = self.store_for(&db, projection)?;
        let order = key_order(&*store, key)?;
        let result = match &order {
            Some(order) => algorithms::pagerank_in_order(
                &*store,
                &order.nodes,
                damping,
                max_iterations,
                tolerance,
                directed,
            ),
            None => algorithms::pagerank(&*store, damping, max_iterations, tolerance, directed),
        };
        Python::attach(|py| node_dict(py, result, order.as_ref()))
    }

    /// Compute betweenness centrality using Brandes' algorithm.
    ///
    /// Args:
    ///     normalized: If True, normalize scores (default: True)
    ///     key: A node property every node in scope holds, each with its own
    ///         value (an id from outside the database). The result is keyed by
    ///         it instead of the node id, in key order. A node without it, or
    ///         two with equal values, raise GrafeoError.
    ///         The scores follow the internal node order either way.
    ///
    /// Returns:
    ///     Dict mapping node ID (or key) to betweenness score
    #[pyo3(signature = (normalized=true, *, projection=None, key=None))]
    fn betweenness_centrality(
        &self,
        normalized: bool,
        projection: Option<&str>,
        key: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = self.store_for(&db, projection)?;
        let order = key_order(&*store, key)?;
        let result = algorithms::betweenness_centrality(&*store, normalized);
        node_dict(py, result, order.as_ref())
    }

    /// Compute closeness centrality.
    ///
    /// Args:
    ///     wf_improved: Use Wasserman-Faust formula (default: False)
    ///     key: A node property every node in scope holds, each with its own
    ///         value (an id from outside the database). The result is keyed by
    ///         it instead of the node id, in key order. A node without it, or
    ///         two with equal values, raise GrafeoError.
    ///         The scores follow the internal node order either way.
    ///
    /// Returns:
    ///     Dict mapping node ID (or key) to closeness score
    #[pyo3(signature = (wf_improved=false, *, projection=None, key=None))]
    fn closeness_centrality(
        &self,
        wf_improved: bool,
        projection: Option<&str>,
        key: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = self.store_for(&db, projection)?;
        let order = key_order(&*store, key)?;
        let result = algorithms::closeness_centrality(&*store, wf_improved);
        node_dict(py, result, order.as_ref())
    }

    // ==========================================================================
    // Community Detection
    // ==========================================================================

    /// Detect communities using Label Propagation.
    ///
    /// Args:
    ///     max_iterations: Maximum iterations (default: 100, 0 for unlimited)
    ///     key: A node property every node in scope holds, each with its own
    ///         value (an id from outside the database). The result is keyed by
    ///         it instead of the node id, in key order, and the algorithm runs
    ///         in key order (visits, sums, ties and numbering), so the result
    ///         does not depend on the order nodes and edges were inserted in.
    ///         A node without it, or two with equal values, raise GrafeoError.
    ///
    /// Returns:
    ///     Dict mapping node ID (or key) to community ID; communities are
    ///     numbered 0, 1, 2, ... in the order of their smallest node ID (or key)
    #[pyo3(signature = (max_iterations=100, *, projection=None, key=None))]
    fn label_propagation(
        &self,
        max_iterations: usize,
        projection: Option<&str>,
        key: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = self.store_for(&db, projection)?;
        let order = key_order(&*store, key)?;
        let result = match &order {
            Some(order) => {
                algorithms::label_propagation_in_order(&*store, &order.nodes, max_iterations)
            }
            None => algorithms::label_propagation(&*store, max_iterations),
        };
        node_dict(py, result, order.as_ref())
    }

    /// Detect communities using Louvain algorithm.
    ///
    /// Args:
    ///     resolution: Resolution parameter (default: 1.0)
    ///     key: A node property every node in scope holds, each with its own
    ///         value (an id from outside the database). The result is keyed by
    ///         it instead of the node id, in key order, and the algorithm runs
    ///         in key order (visits, sums, ties and numbering), so the result
    ///         does not depend on the order nodes and edges were inserted in.
    ///         A node without it, or two with equal values, raise GrafeoError.
    ///
    /// Returns:
    ///     Dict with 'communities' (node ID or key to community, numbered 0, 1,
    ///     2, ... in the order of their smallest node ID or key), 'modularity',
    ///     and 'num_communities' keys
    #[pyo3(signature = (resolution=1.0, *, projection=None, key=None))]
    fn louvain(
        &self,
        resolution: f64,
        projection: Option<&str>,
        key: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = self.store_for(&db, projection)?;
        let order = key_order(&*store, key)?;
        let result = match &order {
            Some(order) => algorithms::louvain_in_order(&*store, &order.nodes, resolution),
            None => algorithms::louvain(&*store, resolution),
        };

        let dict = PyDict::new(py);
        dict.set_item(
            "communities",
            node_dict(py, result.communities, order.as_ref())?,
        )?;
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
    #[pyo3(signature = (weight=None, *, projection=None))]
    fn kruskal(
        &self,
        weight: Option<&str>,
        projection: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = self.store_for(&db, projection)?;
        let result = algorithms::kruskal(&*store, weight);

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
    #[pyo3(signature = (weight=None, start=None, *, projection=None))]
    fn prim(
        &self,
        weight: Option<&str>,
        start: Option<u64>,
        projection: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = self.store_for(&db, projection)?;
        let start_node = start.map(NodeId::new);
        let result = algorithms::prim(&*store, weight, start_node);

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
    #[pyo3(signature = (source, sink, capacity=None, *, projection=None))]
    fn max_flow(
        &self,
        source: u64,
        sink: u64,
        capacity: Option<&str>,
        projection: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = self.store_for(&db, projection)?;

        match algorithms::max_flow(&*store, NodeId::new(source), NodeId::new(sink), capacity) {
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
    #[pyo3(signature = (source, sink, capacity=None, cost=None, *, projection=None))]
    fn min_cost_max_flow(
        &self,
        source: u64,
        sink: u64,
        capacity: Option<&str>,
        cost: Option<&str>,
        projection: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = self.store_for(&db, projection)?;

        match algorithms::min_cost_max_flow(
            &*store,
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
    #[pyo3(signature = (parallel=true, *, projection=None))]
    fn clustering_coefficient(
        &self,
        parallel: bool,
        projection: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = self.store_for(&db, projection)?;

        let result = if parallel {
            algorithms::clustering_coefficient_parallel(&*store, 50)
        } else {
            algorithms::clustering_coefficient(&*store)
        };

        let coefficients: BTreeMap<u64, f64> = result
            .coefficients
            .into_iter()
            .map(|(n, c)| (n.0, c))
            .collect();
        let triangle_counts: BTreeMap<u64, u64> = result
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
    /// Args:
    ///     key: A node property every node in scope holds, each with its own
    ///         value (an id from outside the database). The result is keyed by
    ///         it instead of the node id, in key order. A node without it, or
    ///         two with equal values, raise GrafeoError.
    ///
    /// Returns:
    ///     Dict mapping node ID (or key) to triangle count
    #[pyo3(signature = (*, projection=None, key=None))]
    fn triangle_count(
        &self,
        projection: Option<&str>,
        key: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = self.store_for(&db, projection)?;
        let order = key_order(&*store, key)?;
        let result = algorithms::triangle_count(&*store);
        node_dict(py, result, order.as_ref())
    }

    /// Get the total number of unique triangles in the graph.
    ///
    /// Each triangle is counted exactly once.
    ///
    /// Returns:
    ///     Total unique triangle count
    #[pyo3(signature = (*, projection=None))]
    fn total_triangles(&self, projection: Option<&str>) -> PyResult<u64> {
        let db = self.db.read();
        let store = self.store_for(&db, projection)?;
        Ok(algorithms::total_triangles(&*store))
    }

    /// Compute the global (average) clustering coefficient.
    ///
    /// Returns:
    ///     Average clustering coefficient across all nodes (0.0 to 1.0)
    #[pyo3(signature = (*, projection=None))]
    fn global_clustering_coefficient(&self, projection: Option<&str>) -> PyResult<f64> {
        let db = self.db.read();
        let store = self.store_for(&db, projection)?;
        Ok(algorithms::global_clustering_coefficient(&*store))
    }

    /// Compute local clustering coefficients for each node.
    ///
    /// Args:
    ///     key: A node property every node in scope holds, each with its own
    ///         value (an id from outside the database). The result is keyed by
    ///         it instead of the node id, in key order. A node without it, or
    ///         two with equal values, raise GrafeoError.
    ///
    /// Returns:
    ///     Dict mapping node ID (or key) to local clustering coefficient (0.0
    ///     to 1.0)
    #[pyo3(signature = (*, projection=None, key=None))]
    fn local_clustering_coefficient(
        &self,
        projection: Option<&str>,
        key: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = self.store_for(&db, projection)?;
        let order = key_order(&*store, key)?;
        let result = algorithms::local_clustering_coefficient(&*store);
        node_dict(py, result, order.as_ref())
    }

    // ==========================================================================
    // Structure Analysis
    // ==========================================================================

    /// Find articulation points (cut vertices).
    ///
    /// Args:
    ///     key: A node property every node in scope holds, each with its own
    ///         value (an id from outside the database). The result is keyed by
    ///         it instead of the node id, in key order. A node without it, or
    ///         two with equal values, raise GrafeoError.
    ///
    /// Returns:
    ///     List of node IDs (or keys) that are articulation points, in node-id
    ///     (or key) order
    #[pyo3(signature = (*, projection=None, key=None))]
    fn articulation_points(
        &self,
        projection: Option<&str>,
        key: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = self.store_for(&db, projection)?;
        let order = key_order(&*store, key)?;
        let result = algorithms::articulation_points(&*store);
        node_list(py, result, order.as_ref())
    }

    /// Find bridges (cut edges).
    ///
    /// Returns:
    ///     List of (source, target) tuples representing bridges
    #[pyo3(signature = (*, projection=None))]
    fn bridges(&self, projection: Option<&str>) -> PyResult<Vec<(u64, u64)>> {
        let db = self.db.read();
        let store = self.store_for(&db, projection)?;
        let result = algorithms::bridges(&*store);
        Ok(result.into_iter().map(|(s, t)| (s.0, t.0)).collect())
    }

    /// Compute k-core decomposition.
    ///
    /// The graph is treated as simple and undirected: edge direction is
    /// ignored, parallel edges count once and self-loops are ignored.
    ///
    /// Args:
    ///     k: If provided, return only nodes in the k-core
    ///     key: A node property every node in scope holds, each with its own
    ///         value (an id from outside the database). The result is keyed by
    ///         it instead of the node id, in key order. A node without it, or
    ///         two with equal values, raise GrafeoError.
    ///
    /// Returns:
    ///     If k is None: Dict with 'core_numbers' (node ID or key to core
    ///     number) and 'max_core' (the largest core number) keys
    ///     If k is provided: List of node IDs (or keys) in the k-core, in
    ///     node-id (or key) order
    #[pyo3(signature = (k=None, *, projection=None, key=None))]
    fn kcore(
        &self,
        k: Option<usize>,
        projection: Option<&str>,
        key: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let db = self.db.read();
        let store = self.store_for(&db, projection)?;
        let order = key_order(&*store, key)?;
        let result = algorithms::kcore_decomposition(&*store);

        if let Some(k_val) = k {
            node_list(py, result.k_core(k_val), order.as_ref())
        } else {
            // In node-id (or key) order, the same in every process.
            let core_numbers = node_dict(py, result.core_numbers, order.as_ref())?;
            let dict = PyDict::new(py);
            dict.set_item("core_numbers", core_numbers)?;
            dict.set_item("max_core", result.max_core)?;
            Ok(dict.into_any().unbind())
        }
    }

    fn __repr__(&self) -> String {
        "Algorithms()".to_string()
    }
}
