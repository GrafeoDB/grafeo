//! Minimum Spanning Tree algorithms: Kruskal, Prim.
//!
//! These algorithms find a tree that connects all nodes in an undirected
//! graph with minimum total edge weight.

use std::collections::BinaryHeap;
use std::sync::OnceLock;

use graphos_common::types::{EdgeId, NodeId, Value};
use graphos_common::utils::error::Result;
use graphos_common::utils::hash::FxHashMap;
use graphos_core::graph::lpg::LpgStore;
use graphos_core::graph::Direction;

use super::components::UnionFind;
use super::traits::{GraphAlgorithm, MinScored};
use super::super::{AlgorithmResult, ParameterDef, ParameterType, Parameters};

// ============================================================================
// Edge Weight Extraction
// ============================================================================

/// Extracts edge weight from a property value.
fn extract_weight(store: &LpgStore, edge_id: EdgeId, weight_prop: Option<&str>) -> f64 {
    if let Some(prop_name) = weight_prop {
        if let Some(edge) = store.get_edge(edge_id) {
            if let Some(value) = edge.get_property(prop_name) {
                return match value {
                    Value::Int64(i) => *i as f64,
                    Value::Float64(f) => *f,
                    _ => 1.0,
                };
            }
        }
    }
    1.0
}

// ============================================================================
// MST Result
// ============================================================================

/// Result of MST algorithms.
#[derive(Debug, Clone)]
pub struct MstResult {
    /// Edges in the MST: (source, target, edge_id, weight)
    pub edges: Vec<(NodeId, NodeId, EdgeId, f64)>,
    /// Total weight of the MST.
    pub total_weight: f64,
}

impl MstResult {
    /// Returns the number of edges in the MST.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Returns true if this is a valid spanning tree for n nodes.
    pub fn is_spanning_tree(&self, node_count: usize) -> bool {
        if node_count == 0 {
            return self.edges.is_empty();
        }
        self.edges.len() == node_count - 1
    }
}

// ============================================================================
// Kruskal's Algorithm
// ============================================================================

/// Computes the Minimum Spanning Tree using Kruskal's algorithm.
///
/// Kruskal's algorithm sorts all edges by weight and greedily adds
/// edges that don't create a cycle (using Union-Find).
///
/// # Arguments
///
/// * `store` - The graph store
/// * `weight_property` - Optional property name for edge weights (defaults to 1.0)
///
/// # Returns
///
/// The MST edges and total weight.
///
/// # Complexity
///
/// O(E log E) for sorting edges
pub fn kruskal(store: &LpgStore, weight_property: Option<&str>) -> MstResult {
    let nodes = store.node_ids();
    let n = nodes.len();

    if n == 0 {
        return MstResult {
            edges: Vec::new(),
            total_weight: 0.0,
        };
    }

    // Build node index mapping
    let mut node_to_idx: FxHashMap<NodeId, usize> = FxHashMap::default();
    for (idx, &node) in nodes.iter().enumerate() {
        node_to_idx.insert(node, idx);
    }

    // Collect all edges with weights (treating as undirected)
    let mut edges: Vec<(f64, NodeId, NodeId, EdgeId)> = Vec::new();
    let mut seen_edges: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();

    for &node in &nodes {
        let i = *node_to_idx.get(&node).unwrap();
        for (neighbor, edge_id) in store.edges_from(node, Direction::Outgoing) {
            if let Some(&j) = node_to_idx.get(&neighbor) {
                // For undirected: only add each edge once
                let key = if i < j { (i, j) } else { (j, i) };
                if !seen_edges.contains(&key) {
                    seen_edges.insert(key);
                    let weight = extract_weight(store, edge_id, weight_property);
                    edges.push((weight, node, neighbor, edge_id));
                }
            }
        }
    }

    // Sort edges by weight
    edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Initialize Union-Find
    let mut uf = UnionFind::new(n);

    let mut mst_edges: Vec<(NodeId, NodeId, EdgeId, f64)> = Vec::new();
    let mut total_weight = 0.0;

    for (weight, src, dst, edge_id) in edges {
        let i = *node_to_idx.get(&src).unwrap();
        let j = *node_to_idx.get(&dst).unwrap();

        if uf.find(i) != uf.find(j) {
            uf.union(i, j);
            mst_edges.push((src, dst, edge_id, weight));
            total_weight += weight;

            // MST has n-1 edges
            if mst_edges.len() == n - 1 {
                break;
            }
        }
    }

    MstResult {
        edges: mst_edges,
        total_weight,
    }
}

// ============================================================================
// Prim's Algorithm
// ============================================================================

/// Computes the Minimum Spanning Tree using Prim's algorithm.
///
/// Prim's algorithm grows the MST from a starting node, always adding
/// the minimum weight edge that connects a tree node to a non-tree node.
///
/// # Arguments
///
/// * `store` - The graph store
/// * `weight_property` - Optional property name for edge weights (defaults to 1.0)
/// * `start` - Optional starting node (defaults to first node)
///
/// # Returns
///
/// The MST edges and total weight.
///
/// # Complexity
///
/// O(E log V) using a binary heap
pub fn prim(store: &LpgStore, weight_property: Option<&str>, start: Option<NodeId>) -> MstResult {
    let nodes = store.node_ids();
    let n = nodes.len();

    if n == 0 {
        return MstResult {
            edges: Vec::new(),
            total_weight: 0.0,
        };
    }

    // Start from the first node or specified start
    let start_node = start.unwrap_or(nodes[0]);

    // Verify start node exists
    if store.get_node(start_node).is_none() {
        return MstResult {
            edges: Vec::new(),
            total_weight: 0.0,
        };
    }

    let mut in_tree: FxHashMap<NodeId, bool> = FxHashMap::default();
    let mut mst_edges: Vec<(NodeId, NodeId, EdgeId, f64)> = Vec::new();
    let mut total_weight = 0.0;

    // Priority queue: (weight, source, target, edge_id)
    let mut heap: BinaryHeap<MinScored<f64, (NodeId, NodeId, EdgeId)>> = BinaryHeap::new();

    // Start with the first node
    in_tree.insert(start_node, true);

    // Add edges from start node
    for (neighbor, edge_id) in store.edges_from(start_node, Direction::Outgoing) {
        let weight = extract_weight(store, edge_id, weight_property);
        heap.push(MinScored::new(weight, (start_node, neighbor, edge_id)));
    }

    // Also consider incoming edges (for undirected behavior)
    for &other in &nodes {
        for (neighbor, edge_id) in store.edges_from(other, Direction::Outgoing) {
            if neighbor == start_node {
                let weight = extract_weight(store, edge_id, weight_property);
                heap.push(MinScored::new(weight, (other, start_node, edge_id)));
            }
        }
    }

    while let Some(MinScored(weight, (src, dst, edge_id))) = heap.pop() {
        // Skip if target already in tree
        if *in_tree.get(&dst).unwrap_or(&false) {
            continue;
        }

        // Add edge to MST
        in_tree.insert(dst, true);
        mst_edges.push((src, dst, edge_id, weight));
        total_weight += weight;

        // Add edges from new node
        for (neighbor, new_edge_id) in store.edges_from(dst, Direction::Outgoing) {
            if !*in_tree.get(&neighbor).unwrap_or(&false) {
                let new_weight = extract_weight(store, new_edge_id, weight_property);
                heap.push(MinScored::new(new_weight, (dst, neighbor, new_edge_id)));
            }
        }

        // Also consider incoming edges
        for &other in &nodes {
            if !*in_tree.get(&other).unwrap_or(&false) {
                for (neighbor, new_edge_id) in store.edges_from(other, Direction::Outgoing) {
                    if neighbor == dst {
                        let new_weight = extract_weight(store, new_edge_id, weight_property);
                        heap.push(MinScored::new(new_weight, (other, dst, new_edge_id)));
                    }
                }
            }
        }

        // MST has n-1 edges
        if mst_edges.len() == n - 1 {
            break;
        }
    }

    MstResult {
        edges: mst_edges,
        total_weight,
    }
}

// ============================================================================
// Algorithm Wrappers for Plugin Registry
// ============================================================================

/// Static parameter definitions for Kruskal algorithm.
static KRUSKAL_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn kruskal_params() -> &'static [ParameterDef] {
    KRUSKAL_PARAMS.get_or_init(|| {
        vec![ParameterDef {
            name: "weight".to_string(),
            description: "Edge property name for weights (default: 1.0)".to_string(),
            param_type: ParameterType::String,
            required: false,
            default: None,
        }]
    })
}

/// Kruskal's MST algorithm wrapper.
pub struct KruskalAlgorithm;

impl GraphAlgorithm for KruskalAlgorithm {
    fn name(&self) -> &str {
        "kruskal"
    }

    fn description(&self) -> &str {
        "Kruskal's Minimum Spanning Tree algorithm"
    }

    fn parameters(&self) -> &[ParameterDef] {
        kruskal_params()
    }

    fn execute(&self, store: &LpgStore, params: &Parameters) -> Result<AlgorithmResult> {
        let weight_prop = params.get_string("weight");

        let result = kruskal(store, weight_prop);

        let mut output = AlgorithmResult::new(vec![
            "source".to_string(),
            "target".to_string(),
            "weight".to_string(),
            "total_weight".to_string(),
        ]);

        for (src, dst, _edge_id, weight) in result.edges {
            output.add_row(vec![
                Value::Int64(src.0 as i64),
                Value::Int64(dst.0 as i64),
                Value::Float64(weight),
                Value::Float64(result.total_weight),
            ]);
        }

        Ok(output)
    }
}

/// Static parameter definitions for Prim algorithm.
static PRIM_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn prim_params() -> &'static [ParameterDef] {
    PRIM_PARAMS.get_or_init(|| {
        vec![
            ParameterDef {
                name: "weight".to_string(),
                description: "Edge property name for weights (default: 1.0)".to_string(),
                param_type: ParameterType::String,
                required: false,
                default: None,
            },
            ParameterDef {
                name: "start".to_string(),
                description: "Starting node ID (optional)".to_string(),
                param_type: ParameterType::NodeId,
                required: false,
                default: None,
            },
        ]
    })
}

/// Prim's MST algorithm wrapper.
pub struct PrimAlgorithm;

impl GraphAlgorithm for PrimAlgorithm {
    fn name(&self) -> &str {
        "prim"
    }

    fn description(&self) -> &str {
        "Prim's Minimum Spanning Tree algorithm"
    }

    fn parameters(&self) -> &[ParameterDef] {
        prim_params()
    }

    fn execute(&self, store: &LpgStore, params: &Parameters) -> Result<AlgorithmResult> {
        let weight_prop = params.get_string("weight");
        let start = params.get_int("start").map(|id| NodeId::new(id as u64));

        let result = prim(store, weight_prop, start);

        let mut output = AlgorithmResult::new(vec![
            "source".to_string(),
            "target".to_string(),
            "weight".to_string(),
            "total_weight".to_string(),
        ]);

        for (src, dst, _edge_id, weight) in result.edges {
            output.add_row(vec![
                Value::Int64(src.0 as i64),
                Value::Int64(dst.0 as i64),
                Value::Float64(weight),
                Value::Float64(result.total_weight),
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

    fn create_weighted_triangle() -> LpgStore {
        // Triangle: 0-1-2 with edges
        // 0-1: weight 1
        // 1-2: weight 2
        // 0-2: weight 3
        let store = LpgStore::new();

        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);

        store.create_edge_with_props(n0, n1, "EDGE", [("weight", Value::Float64(1.0))]);
        store.create_edge_with_props(n1, n0, "EDGE", [("weight", Value::Float64(1.0))]);
        store.create_edge_with_props(n1, n2, "EDGE", [("weight", Value::Float64(2.0))]);
        store.create_edge_with_props(n2, n1, "EDGE", [("weight", Value::Float64(2.0))]);
        store.create_edge_with_props(n0, n2, "EDGE", [("weight", Value::Float64(3.0))]);
        store.create_edge_with_props(n2, n0, "EDGE", [("weight", Value::Float64(3.0))]);

        store
    }

    fn create_simple_chain() -> LpgStore {
        // Chain: 0 - 1 - 2 - 3
        let store = LpgStore::new();

        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);
        let n3 = store.create_node(&["Node"]);

        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n1, n0, "EDGE");
        store.create_edge(n1, n2, "EDGE");
        store.create_edge(n2, n1, "EDGE");
        store.create_edge(n2, n3, "EDGE");
        store.create_edge(n3, n2, "EDGE");

        store
    }

    #[test]
    fn test_kruskal_triangle() {
        let store = create_weighted_triangle();
        let result = kruskal(&store, Some("weight"));

        // MST should have 2 edges for 3 nodes
        assert_eq!(result.edges.len(), 2);

        // Total weight should be 1 + 2 = 3 (not including 0-2 with weight 3)
        assert!((result.total_weight - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_kruskal_chain() {
        let store = create_simple_chain();
        let result = kruskal(&store, None);

        // MST should have 3 edges for 4 nodes
        assert_eq!(result.edges.len(), 3);

        // All edges have default weight 1.0
        assert!((result.total_weight - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_kruskal_empty() {
        let store = LpgStore::new();
        let result = kruskal(&store, None);

        assert!(result.edges.is_empty());
        assert_eq!(result.total_weight, 0.0);
    }

    #[test]
    fn test_kruskal_single_node() {
        let store = LpgStore::new();
        store.create_node(&["Node"]);

        let result = kruskal(&store, None);

        assert!(result.edges.is_empty());
        assert!(result.is_spanning_tree(1));
    }

    #[test]
    fn test_prim_triangle() {
        let store = create_weighted_triangle();
        let result = prim(&store, Some("weight"), None);

        // MST should have 2 edges for 3 nodes
        assert_eq!(result.edges.len(), 2);

        // Total weight should be 1 + 2 = 3
        assert!((result.total_weight - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_prim_chain() {
        let store = create_simple_chain();
        let result = prim(&store, None, None);

        // MST should have 3 edges for 4 nodes
        assert_eq!(result.edges.len(), 3);
    }

    #[test]
    fn test_prim_with_start() {
        let store = create_simple_chain();
        let result = prim(&store, None, Some(NodeId::new(2)));

        // Should still find valid MST starting from node 2
        assert_eq!(result.edges.len(), 3);
    }

    #[test]
    fn test_prim_empty() {
        let store = LpgStore::new();
        let result = prim(&store, None, None);

        assert!(result.edges.is_empty());
    }

    #[test]
    fn test_kruskal_prim_same_weight() {
        let store = create_weighted_triangle();

        let kruskal_result = kruskal(&store, Some("weight"));
        let prim_result = prim(&store, Some("weight"), None);

        // Both should have the same total weight
        assert!((kruskal_result.total_weight - prim_result.total_weight).abs() < 0.001);
    }

    #[test]
    fn test_mst_is_spanning_tree() {
        let store = create_simple_chain();
        let result = kruskal(&store, None);

        assert!(result.is_spanning_tree(4));
    }
}
