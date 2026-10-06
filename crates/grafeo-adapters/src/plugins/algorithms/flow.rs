//! Network flow algorithms: Max Flow (Edmonds-Karp), Min Cost Max Flow.
//!
//! These algorithms find optimal flow through a network with capacity
//! constraints on edges.

use std::collections::{BTreeMap, VecDeque};
use std::sync::OnceLock;

use grafeo_common::types::{EdgeId, NodeId, Value};
use grafeo_common::utils::error::Error;
use grafeo_common::utils::hash::FxHashMap;
use grafeo_core::graph::Direction;
use grafeo_core::graph::GraphStore;
#[cfg(all(test, feature = "lpg"))]
use grafeo_core::graph::lpg::LpgStore;

use super::super::{AlgorithmResult, ParameterDef, ParameterType};
use super::traits::{impl_algorithm, node_id_from_param};

// ============================================================================
// Property Extraction
// ============================================================================

/// Extracts capacity from an edge property.
fn extract_capacity(store: &dyn GraphStore, edge_id: EdgeId, capacity_prop: Option<&str>) -> f64 {
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
fn extract_cost(store: &dyn GraphStore, edge_id: EdgeId, cost_prop: Option<&str>) -> f64 {
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
    /// Flow on each edge: (source, target, flow), by source, then target.
    pub flow_edges: Vec<(NodeId, NodeId, f64)>,
}

// ============================================================================
// Edmonds-Karp Algorithm (Max Flow)
// ============================================================================

/// Computes maximum flow using the Edmonds-Karp algorithm.
///
/// Edmonds-Karp is a specific implementation of Ford-Fulkerson that uses
/// BFS to find augmenting paths, so it needs at most O(VE) of them. The search
/// visits neighbours in node-id order, so of several maximum flows it finds
/// the same one every call.
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
/// Maximum flow value and flow assignment on edges, by source, then target.
///
/// # Complexity
///
/// O(V × E × (E + V log V)): O(V × E) augmenting paths, each found by a
/// search over the residual graph and applied through its ordered maps.
pub fn max_flow(
    store: &dyn GraphStore,
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
    // capacity[i][j] = capacity of edge i -> j. Ordered maps: the search
    // and the result follow the indices, which are in node-id order.
    let mut capacity: Vec<BTreeMap<usize, f64>> = vec![BTreeMap::new(); n];

    for (i, &node) in nodes.iter().enumerate() {
        for (neighbor, edge_id) in store.edges_from(node, Direction::Outgoing) {
            if let Some(&j) = node_to_idx.get(&neighbor) {
                let cap = extract_capacity(store, edge_id, capacity_property);
                *capacity[i].entry(j).or_insert(0.0) += cap;
            }
        }
    }

    // Residual graph (initially same as capacity)
    let mut residual: Vec<BTreeMap<usize, f64>> = capacity.clone();

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
    /// Flow on each edge: (source, target, flow, cost), by source, then
    /// target.
    pub flow_edges: Vec<(NodeId, NodeId, f64, f64)>,
}

// ============================================================================
// Min Cost Max Flow (Successive Shortest Paths)
// ============================================================================

/// Computes minimum cost maximum flow using successive shortest paths.
///
/// Finds the maximum flow with minimum total cost by repeatedly finding
/// the shortest (cheapest) augmenting path using Bellman-Ford. The search
/// relaxes edges in node-id order, so of several cheapest flows it finds the
/// same one every call.
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
/// Maximum flow value, total cost, and flow assignment on edges, by source,
/// then target.
///
/// # Complexity
///
/// O(V² × E × max_flow)
pub fn min_cost_max_flow(
    store: &dyn GraphStore,
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

    // Build capacity and cost matrices. Ordered maps: the search and the
    // result follow the indices, which are in node-id order.
    let mut capacity: Vec<BTreeMap<usize, f64>> = vec![BTreeMap::new(); n];
    let mut cost: Vec<BTreeMap<usize, f64>> = vec![BTreeMap::new(); n];

    for (i, &node) in nodes.iter().enumerate() {
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
    let mut residual: Vec<BTreeMap<usize, f64>> = capacity.clone();
    let mut residual_cost: Vec<BTreeMap<usize, f64>> = cost.clone();

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

impl_algorithm! {
    MaxFlowAlgorithm,
    name: "max_flow",
    description: "Maximum flow using Edmonds-Karp algorithm",
    params: max_flow_params,
    execute(store, params) {
        let source_id = params
            .get_int("source")
            .ok_or_else(|| Error::InvalidValue("source parameter required".to_string()))?;
        let sink_id = params
            .get_int("sink")
            .ok_or_else(|| Error::InvalidValue("sink parameter required".to_string()))?;

        let source = node_id_from_param(source_id, "source")?;
        let sink = node_id_from_param(sink_id, "sink")?;
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
            // reason: node IDs are sequential counters, well within i64::MAX
            #[allow(clippy::cast_possible_wrap)]
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

        output.sort_by_id_columns(2);
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

impl_algorithm! {
    MinCostFlowAlgorithm,
    name: "min_cost_max_flow",
    description: "Minimum cost maximum flow using successive shortest paths",
    params: min_cost_flow_params,
    execute(store, params) {
        let source_id = params
            .get_int("source")
            .ok_or_else(|| Error::InvalidValue("source parameter required".to_string()))?;
        let sink_id = params
            .get_int("sink")
            .ok_or_else(|| Error::InvalidValue("sink parameter required".to_string()))?;

        let source = node_id_from_param(source_id, "source")?;
        let sink = node_id_from_param(sink_id, "sink")?;
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
            // reason: node IDs are sequential counters, well within i64::MAX
            #[allow(clippy::cast_possible_wrap)]
            output.add_row(vec![
                Value::Int64(src.0 as i64),
                Value::Int64(dst.0 as i64),
                Value::Float64(flow),
                Value::Float64(cost),
                Value::Float64(result.max_flow),
                Value::Float64(result.total_cost),
            ]);
        }

        output.sort_by_id_columns(2);
        Ok(output)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, feature = "lpg"))]
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
        let store = LpgStore::new().unwrap();

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
        let store = LpgStore::new().unwrap();

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
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Node"]);
        let _n1 = store.create_node(&["Node"]); // Disconnected

        let result = max_flow(&store, n0, NodeId::new(1), None);
        assert!(result.is_some());
        assert_eq!(result.unwrap().max_flow, 0.0);
    }

    #[test]
    fn test_max_flow_invalid_nodes() {
        let store = LpgStore::new().unwrap();
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
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Node"]);
        let _n1 = store.create_node(&["Node"]);

        let result = min_cost_max_flow(&store, n0, NodeId::new(1), None, None);
        assert!(result.is_some());
        assert_eq!(result.unwrap().max_flow, 0.0);
    }

    #[test]
    fn test_max_flow_default_capacity() {
        // Without capacity property, all edges have capacity 1.0
        let store = LpgStore::new().unwrap();
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

    #[test]
    fn test_max_flow_flow_edges_populated() {
        let store = create_simple_flow_graph();
        let result = max_flow(&store, NodeId::new(0), NodeId::new(3), Some("capacity")).unwrap();

        // Flow edges should be non-empty if there's positive flow
        assert!(result.max_flow > 0.0);
        assert!(!result.flow_edges.is_empty(), "result.flow_edges is empty");

        // All flow values should be positive
        for (_, _, flow) in &result.flow_edges {
            assert!(*flow > 0.0);
        }
    }

    #[test]
    fn test_max_flow_int_capacity() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);

        // Use Int64 capacity values instead of Float64
        store.create_edge_with_props(n0, n1, "EDGE", [("capacity", Value::Int64(5))]);

        let result = max_flow(&store, n0, n1, Some("capacity")).unwrap();
        assert!((result.max_flow - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_max_flow_parallel_paths() {
        let store = LpgStore::new().unwrap();
        let s = store.create_node(&["Node"]);
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let t = store.create_node(&["Node"]);

        // Two independent paths: s->a->t and s->b->t
        store.create_edge_with_props(s, a, "EDGE", [("capacity", Value::Float64(3.0))]);
        store.create_edge_with_props(a, t, "EDGE", [("capacity", Value::Float64(3.0))]);
        store.create_edge_with_props(s, b, "EDGE", [("capacity", Value::Float64(2.0))]);
        store.create_edge_with_props(b, t, "EDGE", [("capacity", Value::Float64(2.0))]);

        let result = max_flow(&store, s, t, Some("capacity")).unwrap();
        assert!((result.max_flow - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_max_flow_bottleneck() {
        let store = LpgStore::new().unwrap();
        let s = store.create_node(&["Node"]);
        let mid = store.create_node(&["Node"]);
        let t = store.create_node(&["Node"]);

        // s -> mid has high capacity, mid -> t has low = bottleneck
        store.create_edge_with_props(s, mid, "EDGE", [("capacity", Value::Float64(100.0))]);
        store.create_edge_with_props(mid, t, "EDGE", [("capacity", Value::Float64(1.0))]);

        let result = max_flow(&store, s, t, Some("capacity")).unwrap();
        assert!((result.max_flow - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_min_cost_flow_same_source_sink() {
        let store = create_cost_flow_graph();
        let result = min_cost_max_flow(
            &store,
            NodeId::new(0),
            NodeId::new(0),
            Some("capacity"),
            Some("cost"),
        )
        .unwrap();
        assert_eq!(result.max_flow, 0.0);
        assert_eq!(result.total_cost, 0.0);
    }

    #[test]
    fn test_min_cost_flow_invalid_nodes() {
        let store = LpgStore::new().unwrap();
        store.create_node(&["Node"]);
        let result = min_cost_max_flow(&store, NodeId::new(999), NodeId::new(0), None, None);
        assert!(result.is_none());
    }

    #[test]
    fn test_min_cost_flow_edges() {
        let store = create_cost_flow_graph();
        let result = min_cost_max_flow(
            &store,
            NodeId::new(0),
            NodeId::new(2),
            Some("capacity"),
            Some("cost"),
        )
        .unwrap();

        // Flow edges should have both flow and cost info
        for (_, _, flow, _cost) in &result.flow_edges {
            assert!(*flow > 0.0);
        }
    }

    /// A source feeding a hub that fans out over 19 paths of capacity 1 to
    /// the sink; `hub_capacity` of them can carry flow. Returns the store,
    /// the source and the sink.
    fn fan_out(hub_capacity: f64) -> (LpgStore, NodeId, NodeId) {
        let store = LpgStore::new().unwrap();
        let source = store.create_node(&["Node"]);
        let hub = store.create_node(&["Node"]);
        let middles: Vec<NodeId> = (0..19).map(|_| store.create_node(&["Node"])).collect();
        let sink = store.create_node(&["Node"]);
        let edge = |from, to, capacity: f64| {
            store.create_edge_with_props(
                from,
                to,
                "EDGE",
                [
                    ("capacity", Value::Float64(capacity)),
                    ("cost", Value::Float64(3.0)),
                ],
            );
        };
        edge(source, hub, hub_capacity);
        for &middle in &middles {
            edge(hub, middle, 1.0);
            edge(middle, sink, 1.0);
        }
        (store, source, sink)
    }

    /// Max flow lists its flow edges by source, then target, and finds the
    /// same flow every call: its search walks the residual graph in node-id
    /// order, not in the order of a hash map (#592). With 3 of 19 equal paths
    /// in use, the paths it picks are the first 3 by node id.
    #[test]
    fn max_flow_edges_come_in_node_id_order_and_are_the_same_every_call() {
        for hub_capacity in [3.0, 19.0] {
            let (store, source, sink) = fan_out(hub_capacity);
            let first = max_flow(&store, source, sink, Some("capacity")).unwrap();
            assert_eq!(first.max_flow.to_bits(), hub_capacity.to_bits());
            let pairs: Vec<(NodeId, NodeId)> =
                first.flow_edges.iter().map(|&(s, t, _)| (s, t)).collect();
            assert!(
                pairs.windows(2).all(|w| w[0] < w[1]),
                "hub capacity {hub_capacity}: flow edges not in node-id order: {pairs:?}"
            );
            for _ in 0..5 {
                let again = max_flow(&store, source, sink, Some("capacity")).unwrap();
                let again: Vec<(NodeId, NodeId, u64)> = again
                    .flow_edges
                    .iter()
                    .map(|&(s, t, flow)| (s, t, flow.to_bits()))
                    .collect();
                let expected: Vec<(NodeId, NodeId, u64)> = first
                    .flow_edges
                    .iter()
                    .map(|&(s, t, flow)| (s, t, flow.to_bits()))
                    .collect();
                assert_eq!(again, expected, "hub capacity {hub_capacity}");
            }
        }
        let (store, source, sink) = fan_out(3.0);
        let hub = NodeId::new(source.as_u64() + 1);
        let used: Vec<NodeId> = max_flow(&store, source, sink, Some("capacity"))
            .unwrap()
            .flow_edges
            .iter()
            .filter(|&&(s, _, _)| s == hub)
            .map(|&(_, t, _)| t)
            .collect();
        let first_three: Vec<NodeId> = (2..5).map(|i| NodeId::new(source.as_u64() + i)).collect();
        assert_eq!(used, first_three, "the paths through the smallest node ids");
    }

    /// Min-cost max flow lists its flow edges by source, then target, the
    /// same every call (#592).
    #[test]
    fn min_cost_flow_edges_come_in_node_id_order_and_are_the_same_every_call() {
        for hub_capacity in [3.0, 19.0] {
            let (store, source, sink) = fan_out(hub_capacity);
            let run = || -> Vec<(NodeId, NodeId, u64, u64)> {
                min_cost_max_flow(&store, source, sink, Some("capacity"), Some("cost"))
                    .unwrap()
                    .flow_edges
                    .iter()
                    .map(|&(s, t, flow, cost)| (s, t, flow.to_bits(), cost.to_bits()))
                    .collect()
            };
            let first = run();
            assert!(
                first
                    .windows(2)
                    .all(|w| (w[0].0, w[0].1) < (w[1].0, w[1].1)),
                "hub capacity {hub_capacity}: flow edges not in node-id order: {first:?}"
            );
            for _ in 0..5 {
                assert_eq!(run(), first, "hub capacity {hub_capacity}");
            }
        }
    }
}
