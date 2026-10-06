//! Shortest path algorithms: Dijkstra, A*, Bellman-Ford, Floyd-Warshall.
//!
//! These algorithms find optimal paths in weighted graphs, supporting
//! both single-source and all-pairs variants.

use std::collections::BinaryHeap;
use std::sync::OnceLock;

use grafeo_common::types::{NodeId, Value};
use grafeo_common::utils::error::{Error, Result};
use grafeo_common::utils::hash::FxHashMap;
use grafeo_core::graph::Direction;
use grafeo_core::graph::GraphStore;
#[cfg(all(test, feature = "lpg"))]
use grafeo_core::graph::lpg::LpgStore;

use super::super::{AlgorithmResult, ParameterDef, ParameterType, Parameters};
use super::traits::{GraphAlgorithm, MinScored, impl_algorithm, node_id_from_param};

// ============================================================================
// Edge Weight Extraction
// ============================================================================

/// Extracts edge weight from a property value.
///
/// Supports Int64 and Float64 values, defaulting to 1.0 if no weight property.
fn extract_weight(
    store: &dyn GraphStore,
    edge_id: grafeo_common::types::EdgeId,
    weight_prop: Option<&str>,
) -> f64 {
    if let Some(prop_name) = weight_prop
        && let Some(edge) = store.get_edge(edge_id)
        && let Some(value) = edge.get_property(prop_name)
    {
        return match value {
            Value::Int64(i) => *i as f64,
            Value::Float64(f) => *f,
            _ => 1.0,
        };
    }
    1.0
}

// ============================================================================
// Dijkstra's Algorithm
// ============================================================================

/// Result of Dijkstra's algorithm.
#[derive(Debug, Clone)]
pub struct DijkstraResult {
    /// Distances from source to each reachable node.
    pub distances: FxHashMap<NodeId, f64>,
    /// Predecessor map for path reconstruction.
    pub predecessors: FxHashMap<NodeId, NodeId>,
}

impl DijkstraResult {
    /// Reconstructs the path from source to target.
    ///
    /// Returns `None` if target is unreachable.
    pub fn path_to(&self, source: NodeId, target: NodeId) -> Option<Vec<NodeId>> {
        if !self.distances.contains_key(&target) {
            return None;
        }

        let mut path = Vec::new();
        let mut current = target;

        while current != source {
            path.push(current);
            current = *self.predecessors.get(&current)?;
        }
        path.push(source);
        path.reverse();

        Some(path)
    }

    /// Returns the distance to a target node.
    pub fn distance_to(&self, target: NodeId) -> Option<f64> {
        self.distances.get(&target).copied()
    }
}

/// Runs Dijkstra's algorithm from a source node.
///
/// # Arguments
///
/// * `store` - The graph store
/// * `source` - Starting node ID
/// * `weight_property` - Optional property name for edge weights (defaults to 1.0)
///
/// # Returns
///
/// Distances and predecessors for all reachable nodes.
///
/// # Complexity
///
/// O((V + E) log V) using a binary heap.
pub fn dijkstra(
    store: &dyn GraphStore,
    source: NodeId,
    weight_property: Option<&str>,
) -> DijkstraResult {
    let mut distances: FxHashMap<NodeId, f64> = FxHashMap::default();
    let mut predecessors: FxHashMap<NodeId, NodeId> = FxHashMap::default();
    let mut heap: BinaryHeap<MinScored<f64, NodeId>> = BinaryHeap::new();

    // Check if source exists
    if store.get_node(source).is_none() {
        return DijkstraResult {
            distances,
            predecessors,
        };
    }

    distances.insert(source, 0.0);
    heap.push(MinScored::new(0.0, source));

    while let Some(MinScored(dist, node)) = heap.pop() {
        // Skip if we've found a better path
        if let Some(&best) = distances.get(&node)
            && dist > best
        {
            continue;
        }

        // Explore neighbors
        for (neighbor, edge_id) in store.edges_from(node, Direction::Outgoing) {
            let weight = extract_weight(store, edge_id, weight_property);
            let new_dist = dist + weight;

            let is_better = distances
                .get(&neighbor)
                .map_or(true, |&current| new_dist < current);

            if is_better {
                distances.insert(neighbor, new_dist);
                predecessors.insert(neighbor, node);
                heap.push(MinScored::new(new_dist, neighbor));
            }
        }
    }

    DijkstraResult {
        distances,
        predecessors,
    }
}

/// Runs Dijkstra's algorithm to find shortest path to a specific target.
///
/// Early terminates when target is reached.
pub fn dijkstra_path(
    store: &dyn GraphStore,
    source: NodeId,
    target: NodeId,
    weight_property: Option<&str>,
) -> Option<(f64, Vec<NodeId>)> {
    let mut distances: FxHashMap<NodeId, f64> = FxHashMap::default();
    let mut predecessors: FxHashMap<NodeId, NodeId> = FxHashMap::default();
    let mut heap: BinaryHeap<MinScored<f64, NodeId>> = BinaryHeap::new();

    // Check if source and target exist
    if store.get_node(source).is_none() || store.get_node(target).is_none() {
        return None;
    }

    distances.insert(source, 0.0);
    heap.push(MinScored::new(0.0, source));

    while let Some(MinScored(dist, node)) = heap.pop() {
        // Early termination if we've reached target
        if node == target {
            // Reconstruct path
            let mut path = Vec::new();
            let mut current = target;
            while current != source {
                path.push(current);
                current = *predecessors.get(&current)?;
            }
            path.push(source);
            path.reverse();
            return Some((dist, path));
        }

        // Skip if we've found a better path
        if let Some(&best) = distances.get(&node)
            && dist > best
        {
            continue;
        }

        // Explore neighbors
        for (neighbor, edge_id) in store.edges_from(node, Direction::Outgoing) {
            let weight = extract_weight(store, edge_id, weight_property);
            let new_dist = dist + weight;

            let is_better = distances
                .get(&neighbor)
                .map_or(true, |&current| new_dist < current);

            if is_better {
                distances.insert(neighbor, new_dist);
                predecessors.insert(neighbor, node);
                heap.push(MinScored::new(new_dist, neighbor));
            }
        }
    }

    None // Target not reachable
}

// ============================================================================
// A* Algorithm
// ============================================================================

/// Runs A* algorithm with a heuristic function.
///
/// # Arguments
///
/// * `store` - The graph store
/// * `source` - Starting node ID
/// * `target` - Target node ID
/// * `weight_property` - Optional property name for edge weights
/// * `heuristic` - Function estimating cost from node to target (must be admissible)
///
/// # Returns
///
/// The shortest path distance and path, or `None` if unreachable.
///
/// # Complexity
///
/// O(E) in the best case with a good heuristic, O((V + E) log V) in the worst case.
pub fn astar<H>(
    store: &dyn GraphStore,
    source: NodeId,
    target: NodeId,
    weight_property: Option<&str>,
    heuristic: H,
) -> Option<(f64, Vec<NodeId>)>
where
    H: Fn(NodeId) -> f64,
{
    let mut g_score: FxHashMap<NodeId, f64> = FxHashMap::default();
    let mut predecessors: FxHashMap<NodeId, NodeId> = FxHashMap::default();
    let mut heap: BinaryHeap<MinScored<f64, NodeId>> = BinaryHeap::new();

    // Check if source and target exist
    if store.get_node(source).is_none() || store.get_node(target).is_none() {
        return None;
    }

    g_score.insert(source, 0.0);
    let f_score = heuristic(source);
    heap.push(MinScored::new(f_score, source));

    while let Some(MinScored(_, node)) = heap.pop() {
        if node == target {
            // Reconstruct path
            let mut path = Vec::new();
            let mut current = target;
            while current != source {
                path.push(current);
                current = *predecessors.get(&current)?;
            }
            path.push(source);
            path.reverse();
            return Some((*g_score.get(&target)?, path));
        }

        let current_g = *g_score.get(&node).unwrap_or(&f64::INFINITY);

        // Explore neighbors
        for (neighbor, edge_id) in store.edges_from(node, Direction::Outgoing) {
            let weight = extract_weight(store, edge_id, weight_property);
            let tentative_g = current_g + weight;

            let is_better = g_score
                .get(&neighbor)
                .map_or(true, |&current| tentative_g < current);

            if is_better {
                predecessors.insert(neighbor, node);
                g_score.insert(neighbor, tentative_g);
                let f = tentative_g + heuristic(neighbor);
                heap.push(MinScored::new(f, neighbor));
            }
        }
    }

    None // Target not reachable
}

// ============================================================================
// Bellman-Ford Algorithm
// ============================================================================

/// Result of Bellman-Ford algorithm.
#[derive(Debug, Clone)]
pub struct BellmanFordResult {
    /// Distances from source to each reachable node.
    pub distances: FxHashMap<NodeId, f64>,
    /// Predecessor map for path reconstruction.
    pub predecessors: FxHashMap<NodeId, NodeId>,
    /// Whether a negative cycle was detected.
    pub has_negative_cycle: bool,
    /// The source node used for path reconstruction.
    source: NodeId,
}

impl BellmanFordResult {
    /// Reconstructs the path from source to target.
    pub fn path_to(&self, target: NodeId) -> Option<Vec<NodeId>> {
        if !self.distances.contains_key(&target) {
            return None;
        }

        let mut path = vec![target];
        let mut current = target;

        while current != self.source {
            let pred = self.predecessors.get(&current)?;
            path.push(*pred);
            current = *pred;
        }

        path.reverse();
        Some(path)
    }
}

/// Runs Bellman-Ford algorithm from a source node.
///
/// Unlike Dijkstra, this algorithm handles negative edge weights
/// and detects negative cycles.
///
/// # Arguments
///
/// * `store` - The graph store
/// * `source` - Starting node ID
/// * `weight_property` - Optional property name for edge weights
///
/// # Returns
///
/// Distances, predecessors, and negative cycle detection flag.
///
/// # Complexity
///
/// O(V × E)
pub fn bellman_ford(
    store: &dyn GraphStore,
    source: NodeId,
    weight_property: Option<&str>,
) -> BellmanFordResult {
    let mut distances: FxHashMap<NodeId, f64> = FxHashMap::default();
    let mut predecessors: FxHashMap<NodeId, NodeId> = FxHashMap::default();

    // Check if source exists
    if store.get_node(source).is_none() {
        return BellmanFordResult {
            distances,
            predecessors,
            has_negative_cycle: false,
            source,
        };
    }

    // Collect all nodes and edges
    let nodes: Vec<NodeId> = store.node_ids();
    let edges: Vec<(NodeId, NodeId, grafeo_common::types::EdgeId)> = nodes
        .iter()
        .flat_map(|&node| {
            store
                .edges_from(node, Direction::Outgoing)
                .into_iter()
                .map(move |(neighbor, edge_id)| (node, neighbor, edge_id))
        })
        .collect();

    let n = nodes.len();

    // Initialize distances
    distances.insert(source, 0.0);

    // Relax edges V-1 times
    for _ in 0..n.saturating_sub(1) {
        let mut changed = false;
        for &(u, v, edge_id) in &edges {
            if let Some(&dist_u) = distances.get(&u) {
                let weight = extract_weight(store, edge_id, weight_property);
                let new_dist = dist_u + weight;

                let is_better = distances
                    .get(&v)
                    .map_or(true, |&current| new_dist < current);

                if is_better {
                    distances.insert(v, new_dist);
                    predecessors.insert(v, u);
                    changed = true;
                }
            }
        }
        if !changed {
            break; // Early termination
        }
    }

    // Check for negative cycles
    let mut has_negative_cycle = false;
    for &(u, v, edge_id) in &edges {
        if let Some(&dist_u) = distances.get(&u) {
            let weight = extract_weight(store, edge_id, weight_property);
            if let Some(&dist_v) = distances.get(&v)
                && dist_u + weight < dist_v
            {
                has_negative_cycle = true;
                break;
            }
        }
    }

    BellmanFordResult {
        distances,
        predecessors,
        has_negative_cycle,
        source,
    }
}

// ============================================================================
// Floyd-Warshall Algorithm
// ============================================================================

/// Result of Floyd-Warshall algorithm.
#[derive(Debug, Clone)]
pub struct FloydWarshallResult {
    /// Distance matrix: distances[i][j] is the shortest distance from node i to node j.
    distances: Vec<Vec<f64>>,
    /// Next-hop matrix for path reconstruction.
    next: Vec<Vec<Option<usize>>>,
    /// Mapping from NodeId to matrix index.
    node_to_index: FxHashMap<NodeId, usize>,
    /// Mapping from matrix index to NodeId.
    index_to_node: Vec<NodeId>,
}

impl FloydWarshallResult {
    /// Returns the shortest distance between two nodes.
    pub fn distance(&self, from: NodeId, to: NodeId) -> Option<f64> {
        let i = *self.node_to_index.get(&from)?;
        let j = *self.node_to_index.get(&to)?;
        let dist = self.distances[i][j];
        if dist == f64::INFINITY {
            None
        } else {
            Some(dist)
        }
    }

    /// Reconstructs the shortest path between two nodes.
    pub fn path(&self, from: NodeId, to: NodeId) -> Option<Vec<NodeId>> {
        let i = *self.node_to_index.get(&from)?;
        let j = *self.node_to_index.get(&to)?;

        if self.distances[i][j] == f64::INFINITY {
            return None;
        }

        let mut path = vec![from];
        let mut current = i;

        while current != j {
            current = self.next[current][j]?;
            path.push(self.index_to_node[current]);
        }

        Some(path)
    }

    /// Checks if the graph has a negative cycle.
    pub fn has_negative_cycle(&self) -> bool {
        for i in 0..self.distances.len() {
            if self.distances[i][i] < 0.0 {
                return true;
            }
        }
        false
    }

    /// Returns all nodes in the graph.
    pub fn nodes(&self) -> &[NodeId] {
        &self.index_to_node
    }
}

/// Runs Floyd-Warshall algorithm for all-pairs shortest paths.
///
/// # Arguments
///
/// * `store` - The graph store
/// * `weight_property` - Optional property name for edge weights
///
/// # Returns
///
/// All-pairs shortest path distances and path reconstruction data.
///
/// # Complexity
///
/// O(V³)
pub fn floyd_warshall(
    store: &dyn GraphStore,
    weight_property: Option<&str>,
) -> FloydWarshallResult {
    let nodes: Vec<NodeId> = store.node_ids();
    let n = nodes.len();

    // Build node index mappings
    let mut node_to_index: FxHashMap<NodeId, usize> = FxHashMap::default();
    for (idx, &node) in nodes.iter().enumerate() {
        node_to_index.insert(node, idx);
    }

    // Initialize distance matrix
    let mut distances = vec![vec![f64::INFINITY; n]; n];
    let mut next: Vec<Vec<Option<usize>>> = vec![vec![None; n]; n];

    // Set diagonal to 0
    for i in 0..n {
        distances[i][i] = 0.0;
    }

    // Initialize with direct edges
    for (idx, &node) in nodes.iter().enumerate() {
        for (neighbor, edge_id) in store.edges_from(node, Direction::Outgoing) {
            if let Some(&neighbor_idx) = node_to_index.get(&neighbor) {
                let weight = extract_weight(store, edge_id, weight_property);
                if weight < distances[idx][neighbor_idx] {
                    distances[idx][neighbor_idx] = weight;
                    next[idx][neighbor_idx] = Some(neighbor_idx);
                }
            }
        }
    }

    // Floyd-Warshall main loop
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                let through_k = distances[i][k] + distances[k][j];
                if through_k < distances[i][j] {
                    distances[i][j] = through_k;
                    next[i][j] = next[i][k];
                }
            }
        }
    }

    FloydWarshallResult {
        distances,
        next,
        node_to_index,
        index_to_node: nodes,
    }
}

// ============================================================================
// Algorithm Wrappers for Plugin Registry
// ============================================================================

/// Static parameter definitions for Dijkstra algorithm.
static DIJKSTRA_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn dijkstra_params() -> &'static [ParameterDef] {
    DIJKSTRA_PARAMS.get_or_init(|| {
        vec![
            ParameterDef {
                name: "source".to_string(),
                description: "Source node ID".to_string(),
                param_type: ParameterType::NodeId,
                required: true,
                default: None,
            },
            ParameterDef {
                name: "target".to_string(),
                description: "Target node ID (optional, for single-pair shortest path)".to_string(),
                param_type: ParameterType::NodeId,
                required: false,
                default: None,
            },
            ParameterDef {
                name: "weight".to_string(),
                description: "Edge property name for weights (default: 1.0)".to_string(),
                param_type: ParameterType::String,
                required: false,
                default: None,
            },
        ]
    })
}

/// Dijkstra algorithm wrapper for the plugin registry.
pub struct DijkstraAlgorithm;

impl GraphAlgorithm for DijkstraAlgorithm {
    fn name(&self) -> &str {
        "dijkstra"
    }

    fn description(&self) -> &str {
        "Dijkstra's shortest path algorithm"
    }

    fn parameters(&self) -> &[ParameterDef] {
        dijkstra_params()
    }

    // reason: node IDs are sequential counters, well within i64::MAX
    #[allow(clippy::cast_possible_wrap)]
    fn execute(&self, store: &dyn GraphStore, params: &Parameters) -> Result<AlgorithmResult> {
        let source_id = params
            .get_int("source")
            .ok_or_else(|| Error::InvalidValue("source parameter required".to_string()))?;

        let source = node_id_from_param(source_id, "source")?;
        let weight_prop = params.get_string("weight");

        if let Some(target_id) = params.get_int("target") {
            // Single-pair shortest path
            let target = node_id_from_param(target_id, "target")?;
            match dijkstra_path(store, source, target, weight_prop) {
                Some((distance, path)) => {
                    let mut result = AlgorithmResult::new(vec![
                        "source".to_string(),
                        "target".to_string(),
                        "distance".to_string(),
                        "path".to_string(),
                    ]);

                    let path_str: String = path
                        .iter()
                        .map(|n| n.0.to_string())
                        .collect::<Vec<_>>()
                        .join(" -> ");

                    result.add_row(vec![
                        Value::Int64(source.0 as i64),
                        Value::Int64(target.0 as i64),
                        Value::Float64(distance),
                        Value::String(path_str.into()),
                    ]);

                    Ok(result)
                }
                None => {
                    let mut result = AlgorithmResult::new(vec![
                        "source".to_string(),
                        "target".to_string(),
                        "distance".to_string(),
                        "path".to_string(),
                    ]);
                    result.add_row(vec![
                        Value::Int64(source.0 as i64),
                        Value::Int64(target_id),
                        Value::Null,
                        Value::String("unreachable".into()),
                    ]);
                    Ok(result)
                }
            }
        } else {
            // Single-source shortest paths
            let dijkstra_result = dijkstra(store, source, weight_prop);

            let mut result =
                AlgorithmResult::new(vec!["node_id".to_string(), "distance".to_string()]);

            for (node, distance) in dijkstra_result.distances {
                result.add_row(vec![Value::Int64(node.0 as i64), Value::Float64(distance)]);
            }

            result.sort_by_id_columns(1);
            Ok(result)
        }
    }
}

/// Static parameter definitions for SSSP algorithm.
static SSSP_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn sssp_params() -> &'static [ParameterDef] {
    SSSP_PARAMS.get_or_init(|| {
        vec![
            ParameterDef {
                name: "source".to_string(),
                description: "Source node name (string) or ID (integer as string)".to_string(),
                param_type: ParameterType::String,
                required: true,
                default: None,
            },
            ParameterDef {
                name: "weight".to_string(),
                description: "Edge property name for weights (default: 1.0)".to_string(),
                param_type: ParameterType::String,
                required: false,
                default: None,
            },
        ]
    })
}

/// SSSP (Single-Source Shortest Paths) algorithm for LDBC Graphanalytics compatibility.
///
/// Wraps Dijkstra's algorithm with string-based node name resolution.
/// The source parameter is a string node name (looked up via "name" property)
/// or falls back to integer ID parsing.
pub struct SsspAlgorithm;

impl GraphAlgorithm for SsspAlgorithm {
    fn name(&self) -> &str {
        "sssp"
    }

    fn description(&self) -> &str {
        "Single-source shortest paths (Dijkstra) with string node name support"
    }

    fn parameters(&self) -> &[ParameterDef] {
        sssp_params()
    }

    fn execute(&self, store: &dyn GraphStore, params: &Parameters) -> Result<AlgorithmResult> {
        let source_str = params
            .get_string("source")
            .ok_or_else(|| Error::InvalidValue("source parameter required".to_string()))?;

        // Resolve source: try integer parse first, then name property lookup
        let source = if let Ok(id) = source_str.parse::<u64>() {
            NodeId::new(id)
        } else {
            let candidates = store.find_nodes_by_property("name", &Value::from(source_str));
            match candidates.len() {
                0 => {
                    return Err(Error::InvalidValue(format!(
                        "No node found with name '{source_str}'"
                    )));
                }
                1 => candidates[0],
                _ => {
                    return Err(Error::InvalidValue(format!(
                        "Multiple nodes found with name '{source_str}', use node ID instead"
                    )));
                }
            }
        };

        let weight_prop = params.get_string("weight");
        let dijkstra_result = dijkstra(store, source, weight_prop);

        let mut result = AlgorithmResult::new(vec!["node_id".to_string(), "distance".to_string()]);

        for (node, distance) in dijkstra_result.distances {
            // reason: Node IDs are sequential counters, well within i64::MAX
            #[allow(clippy::cast_possible_wrap)]
            result.add_row(vec![Value::Int64(node.0 as i64), Value::Float64(distance)]);
        }

        result.sort_by_id_columns(1);
        Ok(result)
    }
}

/// Static parameter definitions for Bellman-Ford algorithm.
static BELLMAN_FORD_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn bellman_ford_params() -> &'static [ParameterDef] {
    BELLMAN_FORD_PARAMS.get_or_init(|| {
        vec![
            ParameterDef {
                name: "source".to_string(),
                description: "Source node ID".to_string(),
                param_type: ParameterType::NodeId,
                required: true,
                default: None,
            },
            ParameterDef {
                name: "weight".to_string(),
                description: "Edge property name for weights (default: 1.0)".to_string(),
                param_type: ParameterType::String,
                required: false,
                default: None,
            },
        ]
    })
}

/// Bellman-Ford algorithm wrapper for the plugin registry.
pub struct BellmanFordAlgorithm;

impl_algorithm! {
    BellmanFordAlgorithm,
    name: "bellman_ford",
    description: "Bellman-Ford shortest path algorithm (handles negative weights)",
    params: bellman_ford_params,
    execute(store, params) {
        let source_id = params
            .get_int("source")
            .ok_or_else(|| Error::InvalidValue("source parameter required".to_string()))?;

        let source = node_id_from_param(source_id, "source")?;
        let weight_prop = params.get_string("weight");

        let bf_result = bellman_ford(store, source, weight_prop);

        let mut result = AlgorithmResult::new(vec![
            "node_id".to_string(),
            "distance".to_string(),
            "has_negative_cycle".to_string(),
        ]);

        for (node, distance) in bf_result.distances {
            // reason: Node IDs are sequential counters, well within i64::MAX
            #[allow(clippy::cast_possible_wrap)]
            result.add_row(vec![
                Value::Int64(node.0 as i64),
                Value::Float64(distance),
                Value::Bool(bf_result.has_negative_cycle),
            ]);
        }

        result.sort_by_id_columns(1);
        Ok(result)
    }
}

/// Static parameter definitions for Floyd-Warshall algorithm.
static FLOYD_WARSHALL_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn floyd_warshall_params() -> &'static [ParameterDef] {
    FLOYD_WARSHALL_PARAMS.get_or_init(|| {
        vec![ParameterDef {
            name: "weight".to_string(),
            description: "Edge property name for weights (default: 1.0)".to_string(),
            param_type: ParameterType::String,
            required: false,
            default: None,
        }]
    })
}

/// Floyd-Warshall algorithm wrapper for the plugin registry.
pub struct FloydWarshallAlgorithm;

impl_algorithm! {
    FloydWarshallAlgorithm,
    name: "floyd_warshall",
    description: "Floyd-Warshall all-pairs shortest paths algorithm",
    params: floyd_warshall_params,
    execute(store, params) {
        let weight_prop = params.get_string("weight");

        let fw_result = floyd_warshall(store, weight_prop);

        let mut result = AlgorithmResult::new(vec![
            "source".to_string(),
            "target".to_string(),
            "distance".to_string(),
        ]);

        // Output all pairs with finite distances
        for (i, &from_node) in fw_result.index_to_node.iter().enumerate() {
            for (j, &to_node) in fw_result.index_to_node.iter().enumerate() {
                let dist = fw_result.distances[i][j];
                if dist < f64::INFINITY {
                    // reason: Node IDs are sequential counters, well within i64::MAX
                    #[allow(clippy::cast_possible_wrap)]
                    result.add_row(vec![
                        Value::Int64(from_node.0 as i64),
                        Value::Int64(to_node.0 as i64),
                        Value::Float64(dist),
                    ]);
                }
            }
        }

        Ok(result)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use super::*;

    fn create_weighted_graph() -> LpgStore {
        let store = LpgStore::new().unwrap();

        // Create a weighted graph:
        //   0 --2--> 1 --3--> 2
        //   |        ^        |
        //   4        1        1
        //   v        |        v
        //   3 -------+        4
        //
        // Shortest path from 0 to 2: 0 -> 3 -> 1 -> 2 (cost: 4+1+3 = 8)
        // vs direct: 0 -> 1 -> 2 (cost: 2+3 = 5)
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);
        let n3 = store.create_node(&["Node"]);
        let n4 = store.create_node(&["Node"]);

        // Create edges with weights
        store.create_edge_with_props(n0, n1, "EDGE", [("weight", Value::Float64(2.0))]);
        store.create_edge_with_props(n1, n2, "EDGE", [("weight", Value::Float64(3.0))]);
        store.create_edge_with_props(n0, n3, "EDGE", [("weight", Value::Float64(4.0))]);
        store.create_edge_with_props(n3, n1, "EDGE", [("weight", Value::Float64(1.0))]);
        store.create_edge_with_props(n2, n4, "EDGE", [("weight", Value::Float64(1.0))]);

        store
    }

    #[test]
    fn test_dijkstra_basic() {
        let store = create_weighted_graph();
        let result = dijkstra(&store, NodeId::new(0), Some("weight"));

        // From the graph: 0->1 cost 2, 0->1->2 cost 5, 0->3 cost 4, 0->3->1 cost 5
        assert_eq!(result.distance_to(NodeId::new(0)), Some(0.0));
        assert_eq!(result.distance_to(NodeId::new(1)), Some(2.0));
        assert_eq!(result.distance_to(NodeId::new(2)), Some(5.0));
        assert_eq!(result.distance_to(NodeId::new(3)), Some(4.0));
        assert_eq!(result.distance_to(NodeId::new(4)), Some(6.0)); // 0->1->2->4 = 2+3+1
    }

    #[test]
    fn test_dijkstra_path() {
        let store = create_weighted_graph();
        let result = dijkstra(&store, NodeId::new(0), Some("weight"));

        let path = result.path_to(NodeId::new(0), NodeId::new(2));
        assert!(path.is_some());

        let path = path.unwrap();
        assert_eq!(path[0], NodeId::new(0)); // Start
        assert_eq!(*path.last().unwrap(), NodeId::new(2)); // End
    }

    #[test]
    fn test_dijkstra_single_pair() {
        let store = create_weighted_graph();
        let result = dijkstra_path(&store, NodeId::new(0), NodeId::new(4), Some("weight"));

        assert!(result.is_some());
        let (distance, path) = result.unwrap();
        assert!(distance > 0.0);
        assert_eq!(path[0], NodeId::new(0));
        assert_eq!(*path.last().unwrap(), NodeId::new(4));
    }

    #[test]
    fn test_dijkstra_unreachable() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Node"]);
        let _n1 = store.create_node(&["Node"]); // Disconnected

        let result = dijkstra_path(&store, n0, NodeId::new(1), None);
        assert!(result.is_none());
    }

    #[test]
    fn test_bellman_ford_basic() {
        let store = create_weighted_graph();
        let result = bellman_ford(&store, NodeId::new(0), Some("weight"));

        assert!(!result.has_negative_cycle);
        assert!(result.distances.contains_key(&NodeId::new(0)));
        assert_eq!(*result.distances.get(&NodeId::new(0)).unwrap(), 0.0);
    }

    #[test]
    fn test_bellman_ford_negative_weights() {
        // Graph with a negative edge weight (no cycle):
        //   0 --10--> 1 --(-5)--> 2
        //
        // Shortest: 0 -> 1 -> 2 = 10 + (-5) = 5
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);

        store.create_edge_with_props(n0, n1, "EDGE", [("weight", Value::Float64(10.0))]);
        store.create_edge_with_props(n1, n2, "EDGE", [("weight", Value::Float64(-5.0))]);

        let result = bellman_ford(&store, n0, Some("weight"));

        assert!(!result.has_negative_cycle);
        assert_eq!(*result.distances.get(&n0).unwrap(), 0.0);
        assert_eq!(*result.distances.get(&n1).unwrap(), 10.0);
        assert_eq!(*result.distances.get(&n2).unwrap(), 5.0);

        // Path reconstruction
        let path = result.path_to(n2).unwrap();
        assert_eq!(path, vec![n0, n1, n2]);
    }

    #[test]
    fn test_bellman_ford_negative_weight_shortcut() {
        // Graph where negative edge creates a shorter path:
        //   0 --6--> 1 --2--> 3
        //   |                 ^
        //   +--3--> 2 --(-4)--+
        //
        // Direct: 0->1->3 = 8
        // Via negative: 0->2->3 = 3+(-4) = -1  (shorter!)
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);
        let n3 = store.create_node(&["Node"]);

        store.create_edge_with_props(n0, n1, "EDGE", [("weight", Value::Float64(6.0))]);
        store.create_edge_with_props(n1, n3, "EDGE", [("weight", Value::Float64(2.0))]);
        store.create_edge_with_props(n0, n2, "EDGE", [("weight", Value::Float64(3.0))]);
        store.create_edge_with_props(n2, n3, "EDGE", [("weight", Value::Float64(-4.0))]);

        let result = bellman_ford(&store, n0, Some("weight"));

        assert!(!result.has_negative_cycle);
        assert_eq!(*result.distances.get(&n3).unwrap(), -1.0);

        let path = result.path_to(n3).unwrap();
        assert_eq!(path, vec![n0, n2, n3]);
    }

    #[test]
    fn test_bellman_ford_negative_cycle_detection() {
        // Graph with a negative cycle:
        //   0 --1--> 1 --1--> 2
        //            ^        |
        //            +--(-3)--+
        //
        // Cycle: 1 -> 2 -> 1 with total weight 1 + (-3) = -2 (negative!)
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);

        store.create_edge_with_props(n0, n1, "EDGE", [("weight", Value::Float64(1.0))]);
        store.create_edge_with_props(n1, n2, "EDGE", [("weight", Value::Float64(1.0))]);
        store.create_edge_with_props(n2, n1, "EDGE", [("weight", Value::Float64(-3.0))]);

        let result = bellman_ford(&store, n0, Some("weight"));

        assert!(
            result.has_negative_cycle,
            "Should detect negative cycle: 1->2->1 with total weight -2"
        );
    }

    #[test]
    fn test_floyd_warshall_basic() {
        let store = create_weighted_graph();
        let result = floyd_warshall(&store, Some("weight"));

        // Self-distances should be 0
        assert_eq!(result.distance(NodeId::new(0), NodeId::new(0)), Some(0.0));
        assert_eq!(result.distance(NodeId::new(1), NodeId::new(1)), Some(0.0));

        // Check some paths exist
        assert!(result.distance(NodeId::new(0), NodeId::new(2)).is_some());
    }

    #[test]
    fn test_floyd_warshall_path_reconstruction() {
        let store = create_weighted_graph();
        let result = floyd_warshall(&store, Some("weight"));

        let path = result.path(NodeId::new(0), NodeId::new(2));
        assert!(path.is_some());

        let path = path.unwrap();
        assert_eq!(path[0], NodeId::new(0));
        assert_eq!(*path.last().unwrap(), NodeId::new(2));
    }

    #[test]
    fn test_astar_basic() {
        let store = create_weighted_graph();

        // Simple heuristic: always return 0 (degenerates to Dijkstra)
        let heuristic = |_: NodeId| 0.0;

        let result = astar(
            &store,
            NodeId::new(0),
            NodeId::new(4),
            Some("weight"),
            heuristic,
        );
        assert!(result.is_some());

        let (distance, path) = result.unwrap();
        assert!(distance > 0.0);
        assert_eq!(path[0], NodeId::new(0));
        assert_eq!(*path.last().unwrap(), NodeId::new(4));
    }

    #[test]
    fn test_dijkstra_nonexistent_source() {
        let store = LpgStore::new().unwrap();
        let result = dijkstra(&store, NodeId::new(999), None);
        assert!(result.distances.is_empty());
    }

    #[test]
    fn test_unweighted_defaults() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);
        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n1, n2, "EDGE");

        // Without weight property, should default to 1.0 per edge
        let result = dijkstra(&store, n0, None);
        assert_eq!(result.distance_to(n1), Some(1.0));
        assert_eq!(result.distance_to(n2), Some(2.0));
    }

    #[test]
    fn test_sssp_with_named_nodes() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);
        store.set_node_property(n0, "name", Value::from("alix"));
        store.set_node_property(n1, "name", Value::from("gus"));
        store.set_node_property(n2, "name", Value::from("harm"));
        store.create_edge_with_props(n0, n1, "KNOWS", [("weight", Value::Float64(1.0))]);
        store.create_edge_with_props(n1, n2, "KNOWS", [("weight", Value::Float64(2.0))]);

        let mut params = Parameters::new();
        params.set_string("source", "alix");
        params.set_string("weight", "weight");

        let result = SsspAlgorithm.execute(&store, &params).unwrap();
        assert_eq!(result.columns, vec!["node_id", "distance"]);
        assert_eq!(result.row_count(), 3); // alix, gus, harm
    }

    #[test]
    fn test_sssp_with_numeric_source() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        store.create_edge(n0, n1, "EDGE");

        let mut params = Parameters::new();
        params.set_string("source", n0.0.to_string());

        let result = SsspAlgorithm.execute(&store, &params).unwrap();
        assert_eq!(result.row_count(), 2);
    }

    #[test]
    fn test_sssp_nonexistent_name() {
        let store = LpgStore::new().unwrap();
        let _n0 = store.create_node(&["Node"]);

        let mut params = Parameters::new();
        params.set_string("source", "nonexistent");

        let result = SsspAlgorithm.execute(&store, &params);
        assert!(result.is_err());
    }
}
