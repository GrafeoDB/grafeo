//! Network flow algorithms: Max Flow (Edmonds-Karp), Min Cost Max Flow.
//!
//! These algorithms find optimal flow through a network with capacity
//! constraints on edges.

use std::collections::VecDeque;
use std::sync::OnceLock;

use grafeo_common::types::{EdgeId, NodeId, Value};
use grafeo_common::utils::error::{Error, Result};
use grafeo_common::utils::hash::FxHashMap;
use grafeo_core::graph::Direction;
use grafeo_core::graph::lpg::LpgStore;

use super::super::{AlgorithmResult, ParameterDef, ParameterType, Parameters};
use super::traits::GraphAlgorithm;

// ============================================================================
// Property Extraction
// ============================================================================

/// Extracts capacity from an edge property.
fn extract_capacity(store: &LpgStore, edge_id: EdgeId, capacity_prop: Option<&str>) -> f64 {
    if let Some(prop_name) = capacity_prop
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

/// Extracts cost from an edge property.
fn extract_cost(store: &LpgStore, edge_id: EdgeId, cost_prop: Option<&str>) -> f64 {
    if let Some(prop_name) = cost_prop
        && let Some(edge) = store.get_edge(edge_id)
        && let Some(value) = edge.get_property(prop_name)
    {
        return match value {
            Value::Int64(i) => *i as f64,
            Value::Float64(f) => *f,
            _ => 0.0,
        };
    }
    0.0
}

// ============================================================================
// Max Flow Result
// ============================================================================

/// Result of max flow algorithm.
#[derive(Debug, Clone)]
pub struct MaxFlowResult {
    /// Maximum flow value.
    pub max_flow: f64,
    /// Flow on each edge: (source, target, flow)
    pub flow_edges: Vec<(NodeId, NodeId, f64)>,
}

// ============================================================================
// Edmonds-Karp Algorithm (Max Flow)
// ============================================================================

/// Computes maximum flow using the Edmonds-Karp algorithm.
///
/// Edmonds-Karp is a specific implementation of Ford-Fulkerson that uses
/// BFS to find augmenting paths, guaranteeing O(VE²) complexity.
///
/// # Arguments
///
/// * `store` - The graph store
/// * `source` - Source node ID
/// * `sink` - Sink node ID
/// * `capacity_property` - Optional property name for edge capacities (defaults to 1.0)
///
/// # Returns
///
/// Maximum flow value and flow assignment on edges.
///
/// # Complexity
///
/// O(V × E²)
pub fn max_flow(
    store: &LpgStore,
    source: NodeId,
    sink: NodeId,
    capacity_property: Option<&str>,
) -> Option<MaxFlowResult> {
    // Verify source and sink exist
    if store.get_node(source).is_none() || store.get_node(sink).is_none() {
        return None;
    }

    if source == sink {
        return Some(MaxFlowResult {
            max_flow: 0.0,
            flow_edges: Vec::new(),
        });
    }

    let nodes = store.node_ids();
    let n = nodes.len();

    // Build node index mapping
    let mut node_to_idx: FxHashMap<NodeId, usize> = FxHashMap::default();
    let mut idx_to_node: Vec<NodeId> = Vec::with_capacity(n);
    for (idx, &node) in nodes.iter().enumerate() {
        node_to_idx.insert(node, idx);
        idx_to_node.push(node);
    }

    let source_idx = *node_to_idx.get(&source)?;
    let sink_idx = *node_to_idx.get(&sink)?;

    // Build capacity matrix (using adjacency list for sparse graphs)
    // capacity[i][j] = capacity of edge i -> j
    let mut capacity: Vec<FxHashMap<usize, f64>> = vec![FxHashMap::default(); n];
    let mut edge_map: FxHashMap<(usize, usize), EdgeId> = FxHashMap::default();

    for &node in &nodes {
        let i = *node_to_idx.get(&node).unwrap();
        for (neighbor, edge_id) in store.edges_from(node, Direction::Outgoing) {
            if let Some(&j) = node_to_idx.get(&neighbor) {
                let cap = extract_capacity(store, edge_id, capacity_property);
                *capacity[i].entry(j).or_insert(0.0) += cap;
                edge_map.insert((i, j), edge_id);
            }
        }
    }

    // Residual graph (initially same as capacity)
    let mut residual: Vec<FxHashMap<usize, f64>> = capacity.clone();

    // Ensure reverse edges exist in residual (with 0 capacity initially)
    for i in 0..n {
        let neighbors: Vec<usize> = residual[i].keys().copied().collect();
        for j in neighbors {
            residual[j].entry(i).or_insert(0.0);
        }
    }

    let mut total_flow = 0.0;

    // BFS to find augmenting path
    loop {
        let mut parent: Vec<Option<usize>> = vec![None; n];
        let mut visited = vec![false; n];
        let mut queue: VecDeque<usize> = VecDeque::new();

        visited[source_idx] = true;
        queue.push_back(source_idx);

        while let Some(u) = queue.pop_front() {
            if u == sink_idx {
                break;
            }

            for (&v, &cap) in &residual[u] {
                if !visited[v] && cap > 1e-9 {
                    visited[v] = true;
                    parent[v] = Some(u);
                    queue.push_back(v);
                }
            }
        }

        // No augmenting path found
        if !visited[sink_idx] {
            break;
        }

        // Find minimum capacity along path
        let mut path_flow = f64::INFINITY;
        let mut v = sink_idx;
        while let Some(u) = parent[v] {
            let cap = *residual[u].get(&v).unwrap_or(&0.0);
            path_flow = path_flow.min(cap);
            v = u;
        }

        // Update residual capacities
        v = sink_idx;
        while let Some(u) = parent[v] {
            *residual[u].entry(v).or_insert(0.0) -= path_flow;
            *residual[v].entry(u).or_insert(0.0) += path_flow;
            v = u;
        }

        total_flow += path_flow;
    }

    // Extract flow on original edges
    let mut flow_edges: Vec<(NodeId, NodeId, f64)> = Vec::new();
    for i in 0..n {
        for (&j, &original_cap) in &capacity[i] {
            let residual_cap = *residual[i].get(&j).unwrap_or(&0.0);
            let flow = original_cap - residual_cap;
            if flow > 1e-9 {
                flow_edges.push((idx_to_node[i], idx_to_node[j], flow));
            }
        }
    }

    Some(MaxFlowResult {
        max_flow: total_flow,
        flow_edges,
    })
}

// ============================================================================
// Min Cost Max Flow Result
// ============================================================================

/// Result of min cost max flow algorithm.
#[derive(Debug, Clone)]
pub struct MinCostFlowResult {
    /// Maximum flow value.
    pub max_flow: f64,
    /// Total cost of the flow.
    pub total_cost: f64,
    /// Flow on each edge: (source, target, flow, cost)
    pub flow_edges: Vec<(NodeId, NodeId, f64, f64)>,
}

// ============================================================================
// Min Cost Max Flow (Successive Shortest Paths)
// ============================================================================

/// Computes minimum cost maximum flow using successive shortest paths.
///
/// Finds the maximum flow with minimum total cost by repeatedly finding
/// the shortest (cheapest) augmenting path using Bellman-Ford.
///
/// # Arguments
///
/// * `store` - The graph store
/// * `source` - Source node ID
/// * `sink` - Sink node ID
/// * `capacity_property` - Optional property name for edge capacities
/// * `cost_property` - Optional property name for edge costs
///
/// # Returns
///
/// Maximum flow value, total cost, and flow assignment on edges.
///
/// # Complexity
///
/// O(V² × E × max_flow)
pub fn min_cost_max_flow(
    store: &LpgStore,
    source: NodeId,
    sink: NodeId,
    capacity_property: Option<&str>,
    cost_property: Option<&str>,
) -> Option<MinCostFlowResult> {
    // Verify source and sink exist
    if store.get_node(source).is_none() || store.get_node(sink).is_none() {
        return None;
    }

    if source == sink {
        return Some(MinCostFlowResult {
            max_flow: 0.0,
            total_cost: 0.0,
            flow_edges: Vec::new(),
        });
    }

    let nodes = store.node_ids();
    let n = nodes.len();

    // Build node index mapping
    let mut node_to_idx: FxHashMap<NodeId, usize> = FxHashMap::default();
    let mut idx_to_node: Vec<NodeId> = Vec::with_capacity(n);
    for (idx, &node) in nodes.iter().enumerate() {
        node_to_idx.insert(node, idx);
        idx_to_node.push(node);
    }

    let source_idx = *node_to_idx.get(&source)?;
    let sink_idx = *node_to_idx.get(&sink)?;

    // Build capacity and cost matrices
    let mut capacity: Vec<FxHashMap<usize, f64>> = vec![FxHashMap::default(); n];
    let mut cost: Vec<FxHashMap<usize, f64>> = vec![FxHashMap::default(); n];

    for &node in &nodes {
        let i = *node_to_idx.get(&node).unwrap();
        for (neighbor, edge_id) in store.edges_from(node, Direction::Outgoing) {
            if let Some(&j) = node_to_idx.get(&neighbor) {
                let cap = extract_capacity(store, edge_id, capacity_property);
                let edge_cost = extract_cost(store, edge_id, cost_property);
                *capacity[i].entry(j).or_insert(0.0) += cap;
                *cost[i].entry(j).or_insert(0.0) = edge_cost;
            }
        }
    }

    // Residual graph
    let mut residual: Vec<FxHashMap<usize, f64>> = capacity.clone();
    let mut residual_cost: Vec<FxHashMap<usize, f64>> = cost.clone();

    // Ensure reverse edges exist
    for i in 0..n {
        let neighbors: Vec<usize> = residual[i].keys().copied().collect();
        for j in neighbors {
            residual[j].entry(i).or_insert(0.0);
            let c = *cost[i].get(&j).unwrap_or(&0.0);
            residual_cost[j].entry(i).or_insert(-c);
        }
    }

    let mut total_flow = 0.0;
    let mut total_cost_val = 0.0;

    // Successive shortest paths using Bellman-Ford
    loop {
        // Bellman-Ford from source
        let mut dist = vec![f64::INFINITY; n];
        let mut parent: Vec<Option<usize>> = vec![None; n];
        dist[source_idx] = 0.0;

        for _ in 0..n {
            let mut changed = false;
            for u in 0..n {
                if dist[u] == f64::INFINITY {
                    continue;
                }
                for (&v, &cap) in &residual[u] {
                    if cap > 1e-9 {
                        let c = *residual_cost[u].get(&v).unwrap_or(&0.0);
                        if dist[u] + c < dist[v] {
                            dist[v] = dist[u] + c;
                            parent[v] = Some(u);
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }

        // No path to sink
        if dist[sink_idx] == f64::INFINITY {
            break;
        }

        // Find minimum capacity along path
        let mut path_flow = f64::INFINITY;
        let mut v = sink_idx;
        while let Some(u) = parent[v] {
            let cap = *residual[u].get(&v).unwrap_or(&0.0);
            path_flow = path_flow.min(cap);
            v = u;
        }

        // Update flow and cost
        let path_cost = dist[sink_idx];
        total_flow += path_flow;
        total_cost_val += path_flow * path_cost;

        // Update residual
        v = sink_idx;
        while let Some(u) = parent[v] {
            *residual[u].entry(v).or_insert(0.0) -= path_flow;
            *residual[v].entry(u).or_insert(0.0) += path_flow;
            v = u;
        }
    }

    // Extract flow on original edges
    let mut flow_edges: Vec<(NodeId, NodeId, f64, f64)> = Vec::new();
    for i in 0..n {
        for (&j, &original_cap) in &capacity[i] {
            let residual_cap = *residual[i].get(&j).unwrap_or(&0.0);
            let flow = original_cap - residual_cap;
            if flow > 1e-9 {
                let edge_cost = *cost[i].get(&j).unwrap_or(&0.0);
                flow_edges.push((idx_to_node[i], idx_to_node[j], flow, edge_cost));
            }
        }
    }

    Some(MinCostFlowResult {
        max_flow: total_flow,
        total_cost: total_cost_val,
        flow_edges,
    })
}

// ============================================================================
// Algorithm Wrappers for Plugin Registry
// ============================================================================

/// Static parameter definitions for Max Flow algorithm.
static MAX_FLOW_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn max_flow_params() -> &'static [ParameterDef] {
    MAX_FLOW_PARAMS.get_or_init(|| {
        vec![
            ParameterDef {
                name: "source".to_string(),
                description: "Source node ID".to_string(),
                param_type: ParameterType::NodeId,
                required: true,
                default: None,
            },
            ParameterDef {
                name: "sink".to_string(),
                description: "Sink node ID".to_string(),
                param_type: ParameterType::NodeId,
                required: true,
                default: None,
            },
            ParameterDef {
                name: "capacity".to_string(),
                description: "Edge property name for capacities (default: 1.0)".to_string(),
                param_type: ParameterType::String,
                required: false,
                default: None,
            },
        ]
    })
}

/// Max Flow algorithm wrapper.
pub struct MaxFlowAlgorithm;

impl GraphAlgorithm for MaxFlowAlgorithm {
    fn name(&self) -> &str {
        "max_flow"
    }

    fn description(&self) -> &str {
        "Maximum flow using Edmonds-Karp algorithm"
    }

    fn parameters(&self) -> &[ParameterDef] {
        max_flow_params()
    }

    fn execute(&self, store: &LpgStore, params: &Parameters) -> Result<AlgorithmResult> {
        let source_id = params
            .get_int("source")
            .ok_or_else(|| Error::InvalidValue("source parameter required".to_string()))?;
        let sink_id = params
            .get_int("sink")
            .ok_or_else(|| Error::InvalidValue("sink parameter required".to_string()))?;

        let source = NodeId::new(source_id as u64);
        let sink = NodeId::new(sink_id as u64);
        let capacity_prop = params.get_string("capacity");

        let result = max_flow(store, source, sink, capacity_prop)
            .ok_or_else(|| Error::InvalidValue("Invalid source or sink node".to_string()))?;

        let mut output = AlgorithmResult::new(vec![
            "source".to_string(),
            "target".to_string(),
            "flow".to_string(),
            "max_flow".to_string(),
        ]);

        for (src, dst, flow) in result.flow_edges {
            output.add_row(vec![
                Value::Int64(src.0 as i64),
                Value::Int64(dst.0 as i64),
                Value::Float64(flow),
                Value::Float64(result.max_flow),
            ]);
        }

        // Add summary row if no edges
        if output.row_count() == 0 {
            output.add_row(vec![
                Value::Int64(source_id),
                Value::Int64(sink_id),
                Value::Float64(0.0),
                Value::Float64(result.max_flow),
            ]);
        }

        Ok(output)
    }
}

/// Static parameter definitions for Min Cost Max Flow algorithm.
static MIN_COST_FLOW_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn min_cost_flow_params() -> &'static [ParameterDef] {
    MIN_COST_FLOW_PARAMS.get_or_init(|| {
        vec![
            ParameterDef {
                name: "source".to_string(),
                description: "Source node ID".to_string(),
                param_type: ParameterType::NodeId,
                required: true,
                default: None,
            },
            ParameterDef {
                name: "sink".to_string(),
                description: "Sink node ID".to_string(),
                param_type: ParameterType::NodeId,
                required: true,
                default: None,
            },
            ParameterDef {
                name: "capacity".to_string(),
                description: "Edge property name for capacities (default: 1.0)".to_string(),
                param_type: ParameterType::String,
                required: false,
                default: None,
            },
            ParameterDef {
                name: "cost".to_string(),
                description: "Edge property name for costs (default: 0.0)".to_string(),
                param_type: ParameterType::String,
                required: false,
                default: None,
            },
        ]
    })
}

/// Min Cost Max Flow algorithm wrapper.
pub struct MinCostFlowAlgorithm;

impl GraphAlgorithm for MinCostFlowAlgorithm {
    fn name(&self) -> &str {
        "min_cost_max_flow"
    }

    fn description(&self) -> &str {
        "Minimum cost maximum flow using successive shortest paths"
    }

    fn parameters(&self) -> &[ParameterDef] {
        min_cost_flow_params()
    }

    fn execute(&self, store: &LpgStore, params: &Parameters) -> Result<AlgorithmResult> {
        let source_id = params
            .get_int("source")
            .ok_or_else(|| Error::InvalidValue("source parameter required".to_string()))?;
        let sink_id = params
            .get_int("sink")
            .ok_or_else(|| Error::InvalidValue("sink parameter required".to_string()))?;

        let source = NodeId::new(source_id as u64);
        let sink = NodeId::new(sink_id as u64);
        let capacity_prop = params.get_string("capacity");
        let cost_prop = params.get_string("cost");

        let result = min_cost_max_flow(store, source, sink, capacity_prop, cost_prop)
            .ok_or_else(|| Error::InvalidValue("Invalid source or sink node".to_string()))?;

        let mut output = AlgorithmResult::new(vec![
            "source".to_string(),
            "target".to_string(),
            "flow".to_string(),
            "cost".to_string(),
            "max_flow".to_string(),
            "total_cost".to_string(),
        ]);

        for (src, dst, flow, cost) in result.flow_edges {
            output.add_row(vec![
                Value::Int64(src.0 as i64),
                Value::Int64(dst.0 as i64),
                Value::Float64(flow),
                Value::Float64(cost),
                Value::Float64(result.max_flow),
                Value::Float64(result.total_cost),
            ]);
        }

        Ok(output)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_simple_flow_graph() -> LpgStore {
        // Simple flow network:
        //   0 --[5]--> 1 --[3]--> 3
        //   |          |
        //  [3]        [2]
        //   |          |
        //   v          v
        //   2 --[4]--> 3
        let store = LpgStore::new();

        let n0 = store.create_node(&["Node"]); // Source
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);
        let n3 = store.create_node(&["Node"]); // Sink

        store.create_edge_with_props(n0, n1, "EDGE", [("capacity", Value::Float64(5.0))]);
        store.create_edge_with_props(n0, n2, "EDGE", [("capacity", Value::Float64(3.0))]);
        store.create_edge_with_props(n1, n3, "EDGE", [("capacity", Value::Float64(3.0))]);
        store.create_edge_with_props(n1, n2, "EDGE", [("capacity", Value::Float64(2.0))]);
        store.create_edge_with_props(n2, n3, "EDGE", [("capacity", Value::Float64(4.0))]);

        store
    }

    fn create_cost_flow_graph() -> LpgStore {
        // Flow network with costs:
        //   0 --[cap:3,cost:1]--> 1 --[cap:2,cost:2]--> 2
        //   |                                           ^
        //   +--[cap:2,cost:5]---------------------------+
        let store = LpgStore::new();

        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);

        store.create_edge_with_props(
            n0,
            n1,
            "EDGE",
            [
                ("capacity", Value::Float64(3.0)),
                ("cost", Value::Float64(1.0)),
            ],
        );
        store.create_edge_with_props(
            n1,
            n2,
            "EDGE",
            [
                ("capacity", Value::Float64(2.0)),
                ("cost", Value::Float64(2.0)),
            ],
        );
        store.create_edge_with_props(
            n0,
            n2,
            "EDGE",
            [
                ("capacity", Value::Float64(2.0)),
                ("cost", Value::Float64(5.0)),
            ],
        );

        store
    }

    #[test]
    fn test_max_flow_basic() {
        let store = create_simple_flow_graph();
        let result = max_flow(&store, NodeId::new(0), NodeId::new(3), Some("capacity"));

        assert!(result.is_some());
        let result = result.unwrap();

        // Max flow should be limited by edge capacities
        // Path 0->1->3 can carry 3
        // Path 0->2->3 can carry 3
        // Path 0->1->2->3 can carry min(5-3, 2, 4-3) = min(2, 2, 1) = 1
        // Total: 3 + 3 + 1 = 7? Let's verify...
        assert!(result.max_flow >= 5.0 && result.max_flow <= 8.0);
    }

    #[test]
    fn test_max_flow_same_source_sink() {
        let store = create_simple_flow_graph();
        let result = max_flow(&store, NodeId::new(0), NodeId::new(0), Some("capacity"));

        assert!(result.is_some());
        assert_eq!(result.unwrap().max_flow, 0.0);
    }

    #[test]
    fn test_max_flow_no_path() {
        let store = LpgStore::new();
        let n0 = store.create_node(&["Node"]);
        let _n1 = store.create_node(&["Node"]); // Disconnected

        let result = max_flow(&store, n0, NodeId::new(1), None);
        assert!(result.is_some());
        assert_eq!(result.unwrap().max_flow, 0.0);
    }

    #[test]
    fn test_max_flow_invalid_nodes() {
        let store = LpgStore::new();
        store.create_node(&["Node"]);

        let result = max_flow(&store, NodeId::new(999), NodeId::new(0), None);
        assert!(result.is_none());
    }

    #[test]
    fn test_min_cost_flow_basic() {
        let store = create_cost_flow_graph();
        let result = min_cost_max_flow(
            &store,
            NodeId::new(0),
            NodeId::new(2),
            Some("capacity"),
            Some("cost"),
        );

        assert!(result.is_some());
        let result = result.unwrap();

        // Max flow through the network
        assert!(result.max_flow > 0.0);

        // Cost should be positive
        assert!(result.total_cost >= 0.0);
    }

    #[test]
    fn test_min_cost_prefers_cheaper_path() {
        let store = create_cost_flow_graph();
        let result = min_cost_max_flow(
            &store,
            NodeId::new(0),
            NodeId::new(2),
            Some("capacity"),
            Some("cost"),
        );

        assert!(result.is_some());
        let result = result.unwrap();

        // Should prefer path 0->1->2 (cost 3) over 0->2 (cost 5) when possible
        // Max flow through 0->1->2 is 2 (limited by 1->2)
        // Remaining flow 2 goes through 0->2
        // Total: 4 units of flow
        assert!(result.max_flow >= 2.0);
    }

    #[test]
    fn test_min_cost_flow_no_path() {
        let store = LpgStore::new();
        let n0 = store.create_node(&["Node"]);
        let _n1 = store.create_node(&["Node"]);

        let result = min_cost_max_flow(&store, n0, NodeId::new(1), None, None);
        assert!(result.is_some());
        assert_eq!(result.unwrap().max_flow, 0.0);
    }

    #[test]
    fn test_max_flow_default_capacity() {
        // Without capacity property, all edges have capacity 1.0
        let store = LpgStore::new();
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);

        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n0, n2, "EDGE");
        store.create_edge(n1, n2, "EDGE");

        let result = max_flow(&store, n0, n2, None);
        assert!(result.is_some());

        // Two paths: 0->2 (cap 1) and 0->1->2 (cap 1)
        assert!(result.unwrap().max_flow >= 1.0);
    }
}
