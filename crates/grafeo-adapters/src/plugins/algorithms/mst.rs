//! Minimum Spanning Tree algorithms: Kruskal, Prim.
//!
//! These algorithms find a tree that connects all nodes in an undirected
//! graph with minimum total edge weight.

use std::collections::BinaryHeap;
use std::sync::OnceLock;

#[cfg(test)]
use grafeo_common::types::PropertyKey;
use grafeo_common::types::{EdgeId, NodeId, Value};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use grafeo_core::graph::Direction;
use grafeo_core::graph::GraphStore;
#[cfg(test)]
#[cfg(test)]
use grafeo_core::graph::rowgroup::RowGroupStore;

use super::super::Domain;
use super::super::{AlgorithmResult, ParameterDef, ParameterType};
use super::components::UnionFind;
use super::traits::{EdgeWeight, Ticker, impl_algorithm, node_id_from_param, score_order};

/// The weight Kruskal and Prim read: a spanning tree of least weight is defined for any finite
/// weights, negative ones too.
const WEIGHT: EdgeWeight = EdgeWeight {
    what: "weight",
    default: 1.0,
    domain: Domain::OfType,
};

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
///
/// # Errors
///
/// The error of a read of the store that fails.
pub fn kruskal(
    store: &dyn GraphStore,
    view: grafeo_common::types::ReadView,
    weight_property: Option<&str>,
) -> grafeo_common::utils::error::Result<MstResult> {
    let graph = super::traits::GraphView::new(store, view);
    let found = kruskal_on(&graph, weight_property);
    graph.finish(found)?
}

/// [`kruskal`] on the graph a [`GraphView`](super::traits::GraphView) reads.
pub(crate) fn kruskal_on(
    store: &super::traits::GraphView<'_>,
    weight_property: Option<&str>,
) -> grafeo_common::utils::error::Result<MstResult> {
    let mut ticker = Ticker::new();
    let nodes = store.node_ids();
    let n = nodes.len();

    if n == 0 {
        return Ok(MstResult {
            edges: Vec::new(),
            total_weight: 0.0,
        });
    }

    // Build node index mapping
    let mut node_to_idx: FxHashMap<NodeId, usize> = FxHashMap::default();
    for (idx, &node) in nodes.iter().enumerate() {
        node_to_idx.insert(node, idx);
    }

    // Every edge once, from its source, in either direction as the tree reads them: of the edges
    // between two nodes, the union-find keeps the lightest and skips the rest.
    let mut edges: Vec<(f64, NodeId, NodeId, EdgeId)> = Vec::new();
    for &node in &nodes {
        for (neighbor, edge_id) in store.edges(node, Direction::Outgoing) {
            ticker.tick()?;
            if node_to_idx.contains_key(&neighbor) {
                let weight = store.edge_weight(edge_id, weight_property, WEIGHT)?;
                edges.push((weight, node, neighbor, edge_id));
            }
        }
    }

    // By weight, then edge id, so of two edges of one weight the lower id is taken.
    edges.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.3.cmp(&b.3)));

    // Initialize Union-Find
    let mut uf = UnionFind::new(n);

    let mut mst_edges: Vec<(NodeId, NodeId, EdgeId, f64)> = Vec::new();
    let mut total_weight = 0.0;

    for (weight, src, dst, edge_id) in edges {
        ticker.tick()?;
        let Some(&i) = node_to_idx.get(&src) else {
            continue;
        };
        let Some(&j) = node_to_idx.get(&dst) else {
            continue;
        };

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

    Ok(MstResult {
        edges: mst_edges,
        total_weight,
    })
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
///
/// # Errors
///
/// The error of a read of the store that fails.
pub fn prim(
    store: &dyn GraphStore,
    view: grafeo_common::types::ReadView,
    weight_property: Option<&str>,
    start: Option<NodeId>,
) -> grafeo_common::utils::error::Result<MstResult> {
    let graph = super::traits::GraphView::new(store, view);
    let found = prim_on(&graph, weight_property, start);
    graph.finish(found)?
}

/// [`prim`] on the graph a [`GraphView`](super::traits::GraphView) reads.
pub(crate) fn prim_on(
    store: &super::traits::GraphView<'_>,
    weight_property: Option<&str>,
    start: Option<NodeId>,
) -> grafeo_common::utils::error::Result<MstResult> {
    let mut ticker = Ticker::new();
    let nodes = store.node_ids();
    let n = nodes.len();

    if n == 0 {
        return Ok(MstResult {
            edges: Vec::new(),
            total_weight: 0.0,
        });
    }

    // Start from the first node or specified start
    let start_node = start.unwrap_or(nodes[0]);

    // Verify start node exists
    if store.node(start_node).is_none() {
        return Ok(MstResult {
            edges: Vec::new(),
            total_weight: 0.0,
        });
    }

    let mut in_tree: FxHashSet<NodeId> = FxHashSet::default();
    let mut mst_edges: Vec<(NodeId, NodeId, EdgeId, f64)> = Vec::new();
    let mut total_weight = 0.0;
    let mut heap: BinaryHeap<Candidate> = BinaryHeap::new();

    // The edges of a node that joins the tree, in either direction, each toward the node it would
    // add, as stored.
    let push_edges = |node: NodeId,
                      in_tree: &FxHashSet<NodeId>,
                      heap: &mut BinaryHeap<Candidate>,
                      ticker: &mut Ticker|
     -> grafeo_common::utils::error::Result<()> {
        for (direction, outgoing) in [(Direction::Outgoing, true), (Direction::Incoming, false)] {
            for (other, edge) in store.edges(node, direction) {
                ticker.tick()?;
                if in_tree.contains(&other) {
                    continue;
                }
                let weight = store.edge_weight(edge, weight_property, WEIGHT)?;
                let (src, dst) = if outgoing {
                    (node, other)
                } else {
                    (other, node)
                };
                heap.push(Candidate {
                    weight,
                    edge,
                    src,
                    dst,
                    adds: other,
                });
            }
        }
        Ok(())
    };

    in_tree.insert(start_node);
    push_edges(start_node, &in_tree, &mut heap, &mut ticker)?;

    while let Some(candidate) = heap.pop() {
        ticker.tick()?;
        if !in_tree.insert(candidate.adds) {
            continue;
        }
        mst_edges.push((
            candidate.src,
            candidate.dst,
            candidate.edge,
            candidate.weight,
        ));
        total_weight += candidate.weight;
        if mst_edges.len() == n - 1 {
            break;
        }
        push_edges(candidate.adds, &in_tree, &mut heap, &mut ticker)?;
    }

    Ok(MstResult {
        edges: mst_edges,
        total_weight,
    })
}

/// An edge Prim's search may take next: the lightest first, then the lowest edge id, so of
/// several edges between two nodes, in either direction, the lightest is taken.
struct Candidate {
    weight: f64,
    edge: EdgeId,
    /// The edge as stored.
    src: NodeId,
    dst: NodeId,
    /// The node outside the tree that it adds.
    adds: NodeId,
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reversed for a min-heap.
        score_order(other.weight, self.weight).then(other.edge.cmp(&self.edge))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for Candidate {}

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
            domain: Domain::OfType,
        }]
    })
}

/// Kruskal's MST algorithm wrapper.
pub struct KruskalAlgorithm;

impl_algorithm! {
    KruskalAlgorithm,
    name: "kruskal",
    description: "Kruskal's Minimum Spanning Tree algorithm",
    params: kruskal_params,
    compute(store, params) {
        let weight_prop = params.get_string("weight");

        let result = kruskal_on(store, weight_prop)?;

        let mut output = AlgorithmResult::new(vec![
            "source".to_string(),
            "target".to_string(),
            "weight".to_string(),
            "total_weight".to_string(),
        ]);

        for (src, dst, _edge_id, weight) in result.edges {
            // reason: Node IDs are sequential counters, well within i64::MAX
            #[allow(clippy::cast_possible_wrap)]
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
                domain: Domain::OfType,
            },
            ParameterDef {
                name: "start".to_string(),
                description: "Starting node ID (optional)".to_string(),
                param_type: ParameterType::NodeId,
                required: false,
                default: None,
                domain: Domain::OfType,
            },
        ]
    })
}

/// Prim's MST algorithm wrapper.
pub struct PrimAlgorithm;

impl_algorithm! {
    PrimAlgorithm,
    name: "prim",
    description: "Prim's Minimum Spanning Tree algorithm",
    params: prim_params,
    compute(store, params) {
        let weight_prop = params.get_string("weight");
        let start = params.get_int("start").map(|id| node_id_from_param(id, "start")).transpose()?;

        let result = prim_on(store, weight_prop, start)?;

        let mut output = AlgorithmResult::new(vec![
            "source".to_string(),
            "target".to_string(),
            "weight".to_string(),
            "total_weight".to_string(),
        ]);

        for (src, dst, _edge_id, weight) in result.edges {
            // reason: Node IDs are sequential counters, well within i64::MAX
            #[allow(clippy::cast_possible_wrap)]
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
    use grafeo_common::types::ReadView;

    fn create_weighted_triangle() -> RowGroupStore {
        // Triangle: 0-1-2 with edges
        // 0-1: weight 1
        // 1-2: weight 2
        // 0-2: weight 3
        let store = RowGroupStore::new();

        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);

        store.create_edge_with_props(
            n0,
            n1,
            "EDGE",
            &[(PropertyKey::new("weight"), Value::Float64(1.0))],
        );
        store.create_edge_with_props(
            n1,
            n0,
            "EDGE",
            &[(PropertyKey::new("weight"), Value::Float64(1.0))],
        );
        store.create_edge_with_props(
            n1,
            n2,
            "EDGE",
            &[(PropertyKey::new("weight"), Value::Float64(2.0))],
        );
        store.create_edge_with_props(
            n2,
            n1,
            "EDGE",
            &[(PropertyKey::new("weight"), Value::Float64(2.0))],
        );
        store.create_edge_with_props(
            n0,
            n2,
            "EDGE",
            &[(PropertyKey::new("weight"), Value::Float64(3.0))],
        );
        store.create_edge_with_props(
            n2,
            n0,
            "EDGE",
            &[(PropertyKey::new("weight"), Value::Float64(3.0))],
        );

        store
    }

    fn create_simple_chain() -> RowGroupStore {
        // Chain: 0 - 1 - 2 - 3
        let store = RowGroupStore::new();

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
        let result = kruskal(&store, ReadView::latest(), Some("weight")).unwrap();

        // MST should have 2 edges for 3 nodes
        assert_eq!(result.edges.len(), 2);

        // Total weight should be 1 + 2 = 3 (not including 0-2 with weight 3)
        assert!((result.total_weight - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_kruskal_chain() {
        let store = create_simple_chain();
        let result = kruskal(&store, ReadView::latest(), None).unwrap();

        // MST should have 3 edges for 4 nodes
        assert_eq!(result.edges.len(), 3);

        // All edges have default weight 1.0
        assert!((result.total_weight - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_kruskal_empty() {
        let store = RowGroupStore::new();
        let result = kruskal(&store, ReadView::latest(), None).unwrap();

        assert!(result.edges.is_empty(), "{:?}", result.edges);
        assert_eq!(result.total_weight, 0.0);
    }

    #[test]
    fn test_kruskal_single_node() {
        let store = RowGroupStore::new();
        store.create_node(&["Node"]);

        let result = kruskal(&store, ReadView::latest(), None).unwrap();

        assert!(result.edges.is_empty(), "{:?}", result.edges);
        assert!(result.is_spanning_tree(1));
    }

    #[test]
    fn test_prim_triangle() {
        let store = create_weighted_triangle();
        let result = prim(&store, ReadView::latest(), Some("weight"), None).unwrap();

        // MST should have 2 edges for 3 nodes
        assert_eq!(result.edges.len(), 2);

        // Total weight should be 1 + 2 = 3
        assert!((result.total_weight - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_prim_chain() {
        let store = create_simple_chain();
        let result = prim(&store, ReadView::latest(), None, None).unwrap();

        // MST should have 3 edges for 4 nodes
        assert_eq!(result.edges.len(), 3);
    }

    #[test]
    fn test_prim_with_start() {
        let store = create_simple_chain();
        let result = prim(&store, ReadView::latest(), None, Some(NodeId::new(2))).unwrap();

        // Should still find valid MST starting from node 2
        assert_eq!(result.edges.len(), 3);
    }

    #[test]
    fn test_prim_empty() {
        let store = RowGroupStore::new();
        let result = prim(&store, ReadView::latest(), None, None).unwrap();

        assert!(result.edges.is_empty(), "{:?}", result.edges);
    }

    #[test]
    fn test_kruskal_prim_same_weight() {
        let store = create_weighted_triangle();

        let kruskal_result = kruskal(&store, ReadView::latest(), Some("weight")).unwrap();
        let prim_result = prim(&store, ReadView::latest(), Some("weight"), None).unwrap();

        // Both should have the same total weight
        assert!((kruskal_result.total_weight - prim_result.total_weight).abs() < 0.001);
    }

    #[test]
    fn test_mst_is_spanning_tree() {
        let store = create_simple_chain();
        let result = kruskal(&store, ReadView::latest(), None).unwrap();

        assert!(result.is_spanning_tree(4));
    }
}
