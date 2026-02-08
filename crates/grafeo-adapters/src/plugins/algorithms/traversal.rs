//! Graph traversal algorithms: BFS and DFS.
//!
//! These algorithms use the visitor pattern to allow flexible customization
//! of traversal behavior, including early termination and edge filtering.

use std::collections::VecDeque;
use std::sync::OnceLock;

use grafeo_common::types::{NodeId, Value};
use grafeo_common::utils::error::Result;
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use grafeo_core::graph::Direction;
use grafeo_core::graph::lpg::LpgStore;

use super::super::{AlgorithmResult, ParameterDef, ParameterType, Parameters};
use super::traits::{Control, GraphAlgorithm, NodeValueResultBuilder, TraversalEvent};

// ============================================================================
// BFS Implementation
// ============================================================================

/// Performs breadth-first search from a starting node.
///
/// Returns the set of visited nodes in BFS order.
///
/// # Arguments
///
/// * `store` - The graph store to traverse
/// * `start` - The starting node ID
///
/// # Returns
///
/// A vector of node IDs in the order they were discovered.
pub fn bfs(store: &LpgStore, start: NodeId) -> Vec<NodeId> {
    let mut visited = Vec::new();
    bfs_with_visitor(store, start, |event| -> Control<()> {
        if let TraversalEvent::Discover(node) = event {
            visited.push(node);
        }
        Control::Continue
    });
    visited
}

/// Performs breadth-first search with a visitor callback.
///
/// The visitor is called for each traversal event, allowing custom
/// behavior such as early termination or path recording.
///
/// # Arguments
///
/// * `store` - The graph store to traverse
/// * `start` - The starting node ID
/// * `visitor` - Callback function receiving traversal events
///
/// # Returns
///
/// `Some(B)` if the visitor returned `Control::Break(B)`, otherwise `None`.
pub fn bfs_with_visitor<B, F>(store: &LpgStore, start: NodeId, mut visitor: F) -> Option<B>
where
    F: FnMut(TraversalEvent) -> Control<B>,
{
    let mut discovered: FxHashSet<NodeId> = FxHashSet::default();
    let mut queue: VecDeque<NodeId> = VecDeque::new();

    // Check if start node exists
    store.get_node(start)?;

    // Discover the start node
    discovered.insert(start);
    queue.push_back(start);

    match visitor(TraversalEvent::Discover(start)) {
        Control::Break(b) => return Some(b),
        Control::Prune => {
            // Prune means don't explore neighbors, but we still finish the node
            match visitor(TraversalEvent::Finish(start)) {
                Control::Break(b) => return Some(b),
                _ => return None,
            }
        }
        Control::Continue => {}
    }

    while let Some(node) = queue.pop_front() {
        // Iterate over outgoing edges
        for (neighbor, edge_id) in store.edges_from(node, Direction::Outgoing) {
            if discovered.insert(neighbor) {
                // Tree edge - neighbor not yet discovered
                match visitor(TraversalEvent::TreeEdge {
                    source: node,
                    target: neighbor,
                    edge: edge_id,
                }) {
                    Control::Break(b) => return Some(b),
                    Control::Prune => continue, // Don't add to queue
                    Control::Continue => {}
                }

                match visitor(TraversalEvent::Discover(neighbor)) {
                    Control::Break(b) => return Some(b),
                    Control::Prune => continue, // Don't explore neighbors
                    Control::Continue => {}
                }

                queue.push_back(neighbor);
            } else {
                // Non-tree edge - neighbor already discovered
                match visitor(TraversalEvent::NonTreeEdge {
                    source: node,
                    target: neighbor,
                    edge: edge_id,
                }) {
                    Control::Break(b) => return Some(b),
                    _ => {}
                }
            }
        }

        // Node processing complete
        match visitor(TraversalEvent::Finish(node)) {
            Control::Break(b) => return Some(b),
            _ => {}
        }
    }

    None
}

/// BFS layers - returns nodes grouped by their distance from the start.
///
/// # Arguments
///
/// * `store` - The graph store to traverse
/// * `start` - The starting node ID
///
/// # Returns
///
/// A vector of vectors, where `result[i]` contains all nodes at distance `i` from start.
pub fn bfs_layers(store: &LpgStore, start: NodeId) -> Vec<Vec<NodeId>> {
    let mut layers: Vec<Vec<NodeId>> = Vec::new();
    let mut discovered: FxHashSet<NodeId> = FxHashSet::default();
    let mut current_layer: Vec<NodeId> = Vec::new();
    let mut next_layer: Vec<NodeId> = Vec::new();

    if store.get_node(start).is_none() {
        return layers;
    }

    discovered.insert(start);
    current_layer.push(start);

    while !current_layer.is_empty() {
        layers.push(current_layer.clone());

        for &node in &current_layer {
            for (neighbor, _) in store.edges_from(node, Direction::Outgoing) {
                if discovered.insert(neighbor) {
                    next_layer.push(neighbor);
                }
            }
        }

        current_layer.clear();
        std::mem::swap(&mut current_layer, &mut next_layer);
    }

    layers
}

// ============================================================================
// DFS Implementation
// ============================================================================

/// Node state during DFS traversal.
#[derive(Clone, Copy, PartialEq, Eq)]
enum NodeColor {
    /// Not yet discovered
    White,
    /// Discovered but not finished (on stack)
    Gray,
    /// Finished processing
    Black,
}

/// Performs depth-first search from a starting node.
///
/// Returns nodes in the order they were finished (post-order).
///
/// # Arguments
///
/// * `store` - The graph store to traverse
/// * `start` - The starting node ID
///
/// # Returns
///
/// A vector of node IDs in post-order (finished order).
pub fn dfs(store: &LpgStore, start: NodeId) -> Vec<NodeId> {
    let mut finished = Vec::new();
    dfs_with_visitor(store, start, |event| -> Control<()> {
        if let TraversalEvent::Finish(node) = event {
            finished.push(node);
        }
        Control::Continue
    });
    finished
}

/// Performs depth-first search with a visitor callback.
///
/// Uses an explicit stack to avoid stack overflow on deep graphs.
///
/// # Arguments
///
/// * `store` - The graph store to traverse
/// * `start` - The starting node ID
/// * `visitor` - Callback function receiving traversal events
///
/// # Returns
///
/// `Some(B)` if the visitor returned `Control::Break(B)`, otherwise `None`.
pub fn dfs_with_visitor<B, F>(store: &LpgStore, start: NodeId, mut visitor: F) -> Option<B>
where
    F: FnMut(TraversalEvent) -> Control<B>,
{
    let mut color: FxHashMap<NodeId, NodeColor> = FxHashMap::default();

    // Stack entries: (node, edge_iterator_index, is_first_visit)
    // We use indices to track progress through neighbors
    let mut stack: Vec<(NodeId, Vec<(NodeId, grafeo_common::types::EdgeId)>, usize)> = Vec::new();

    // Check if start node exists
    store.get_node(start)?;

    // Discover start node
    color.insert(start, NodeColor::Gray);
    match visitor(TraversalEvent::Discover(start)) {
        Control::Break(b) => return Some(b),
        Control::Prune => {
            color.insert(start, NodeColor::Black);
            match visitor(TraversalEvent::Finish(start)) {
                Control::Break(b) => return Some(b),
                _ => return None,
            }
        }
        Control::Continue => {}
    }

    let neighbors: Vec<_> = store.edges_from(start, Direction::Outgoing).collect();
    stack.push((start, neighbors, 0));

    while let Some((node, neighbors, idx)) = stack.last_mut() {
        if *idx >= neighbors.len() {
            // All neighbors processed, finish this node
            let node = *node;
            stack.pop();
            color.insert(node, NodeColor::Black);
            match visitor(TraversalEvent::Finish(node)) {
                Control::Break(b) => return Some(b),
                _ => {}
            }
            continue;
        }

        let (neighbor, edge_id) = neighbors[*idx];
        *idx += 1;

        match color.get(&neighbor).copied().unwrap_or(NodeColor::White) {
            NodeColor::White => {
                // Tree edge - undiscovered node
                match visitor(TraversalEvent::TreeEdge {
                    source: *node,
                    target: neighbor,
                    edge: edge_id,
                }) {
                    Control::Break(b) => return Some(b),
                    Control::Prune => continue,
                    Control::Continue => {}
                }

                color.insert(neighbor, NodeColor::Gray);
                match visitor(TraversalEvent::Discover(neighbor)) {
                    Control::Break(b) => return Some(b),
                    Control::Prune => {
                        color.insert(neighbor, NodeColor::Black);
                        match visitor(TraversalEvent::Finish(neighbor)) {
                            Control::Break(b) => return Some(b),
                            _ => {}
                        }
                        continue;
                    }
                    Control::Continue => {}
                }

                let neighbor_neighbors: Vec<_> =
                    store.edges_from(neighbor, Direction::Outgoing).collect();
                stack.push((neighbor, neighbor_neighbors, 0));
            }
            NodeColor::Gray => {
                // Back edge - node is on the stack (ancestor)
                match visitor(TraversalEvent::BackEdge {
                    source: *node,
                    target: neighbor,
                    edge: edge_id,
                }) {
                    Control::Break(b) => return Some(b),
                    _ => {}
                }
            }
            NodeColor::Black => {
                // Non-tree edge (cross/forward) - already finished
                match visitor(TraversalEvent::NonTreeEdge {
                    source: *node,
                    target: neighbor,
                    edge: edge_id,
                }) {
                    Control::Break(b) => return Some(b),
                    _ => {}
                }
            }
        }
    }

    None
}

/// Performs DFS on all nodes, visiting each connected component.
///
/// Returns nodes in reverse post-order (useful for topological sort).
pub fn dfs_all(store: &LpgStore) -> Vec<NodeId> {
    let mut finished = Vec::new();
    let mut visited: FxHashSet<NodeId> = FxHashSet::default();

    for node_id in store.node_ids() {
        if visited.contains(&node_id) {
            continue;
        }

        dfs_with_visitor(store, node_id, |event| -> Control<()> {
            match event {
                TraversalEvent::Discover(n) => {
                    visited.insert(n);
                }
                TraversalEvent::Finish(n) => {
                    finished.push(n);
                }
                _ => {}
            }
            Control::Continue
        });
    }

    finished
}

// ============================================================================
// Algorithm Wrappers for Plugin Registry
// ============================================================================

/// Static parameter definitions for BFS algorithm.
static BFS_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn bfs_params() -> &'static [ParameterDef] {
    BFS_PARAMS.get_or_init(|| {
        vec![ParameterDef {
            name: "start".to_string(),
            description: "Starting node ID".to_string(),
            param_type: ParameterType::NodeId,
            required: true,
            default: None,
        }]
    })
}

/// BFS algorithm wrapper for the plugin registry.
pub struct BfsAlgorithm;

impl GraphAlgorithm for BfsAlgorithm {
    fn name(&self) -> &str {
        "bfs"
    }

    fn description(&self) -> &str {
        "Breadth-first search traversal from a starting node"
    }

    fn parameters(&self) -> &[ParameterDef] {
        bfs_params()
    }

    fn execute(&self, store: &LpgStore, params: &Parameters) -> Result<AlgorithmResult> {
        let start_id = params.get_int("start").ok_or_else(|| {
            grafeo_common::utils::error::Error::InvalidValue("start parameter required".to_string())
        })?;

        let start = NodeId::new(start_id as u64);
        let layers = bfs_layers(store, start);

        let mut result = AlgorithmResult::new(vec!["node_id".to_string(), "distance".to_string()]);

        for (distance, layer) in layers.iter().enumerate() {
            for &node in layer {
                result.add_row(vec![
                    Value::Int64(node.0 as i64),
                    Value::Int64(distance as i64),
                ]);
            }
        }

        Ok(result)
    }
}

/// Static parameter definitions for DFS algorithm.
static DFS_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn dfs_params() -> &'static [ParameterDef] {
    DFS_PARAMS.get_or_init(|| {
        vec![ParameterDef {
            name: "start".to_string(),
            description: "Starting node ID".to_string(),
            param_type: ParameterType::NodeId,
            required: true,
            default: None,
        }]
    })
}

/// DFS algorithm wrapper for the plugin registry.
pub struct DfsAlgorithm;

impl GraphAlgorithm for DfsAlgorithm {
    fn name(&self) -> &str {
        "dfs"
    }

    fn description(&self) -> &str {
        "Depth-first search traversal from a starting node"
    }

    fn parameters(&self) -> &[ParameterDef] {
        dfs_params()
    }

    fn execute(&self, store: &LpgStore, params: &Parameters) -> Result<AlgorithmResult> {
        let start_id = params.get_int("start").ok_or_else(|| {
            grafeo_common::utils::error::Error::InvalidValue("start parameter required".to_string())
        })?;

        let start = NodeId::new(start_id as u64);
        let finished = dfs(store, start);

        let mut builder = NodeValueResultBuilder::with_capacity("finish_order", finished.len());
        for (order, node) in finished.iter().enumerate() {
            builder.push(*node, Value::Int64(order as i64));
        }

        Ok(builder.build())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> LpgStore {
        let store = LpgStore::new();

        // Create a simple graph:
        //   0 -> 1 -> 2
        //   |    |
        //   v    v
        //   3 -> 4
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);
        let n3 = store.create_node(&["Node"]);
        let n4 = store.create_node(&["Node"]);

        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n0, n3, "EDGE");
        store.create_edge(n1, n2, "EDGE");
        store.create_edge(n1, n4, "EDGE");
        store.create_edge(n3, n4, "EDGE");

        store
    }

    #[test]
    fn test_bfs_simple() {
        let store = create_test_graph();
        let visited = bfs(&store, NodeId::new(0));

        assert!(!visited.is_empty());
        assert_eq!(visited[0], NodeId::new(0));
        // Node 0 should be first
    }

    #[test]
    fn test_bfs_layers() {
        let store = create_test_graph();
        let layers = bfs_layers(&store, NodeId::new(0));

        assert!(!layers.is_empty());
        assert_eq!(layers[0], vec![NodeId::new(0)]);
        // Distance 0: just the start node
    }

    #[test]
    fn test_dfs_simple() {
        let store = create_test_graph();
        let finished = dfs(&store, NodeId::new(0));

        assert!(!finished.is_empty());
        // Post-order means leaves are finished first
    }

    #[test]
    fn test_bfs_nonexistent_start() {
        let store = LpgStore::new();
        let visited = bfs(&store, NodeId::new(999));
        assert!(visited.is_empty());
    }

    #[test]
    fn test_dfs_nonexistent_start() {
        let store = LpgStore::new();
        let finished = dfs(&store, NodeId::new(999));
        assert!(finished.is_empty());
    }

    #[test]
    fn test_bfs_early_termination() {
        let store = create_test_graph();
        let target = NodeId::new(2);

        let found = bfs_with_visitor(&store, NodeId::new(0), |event| {
            if let TraversalEvent::Discover(node) = event
                && node == target
            {
                return Control::Break(true);
            }
            Control::Continue
        });

        assert_eq!(found, Some(true));
    }

    #[test]
    fn test_bfs_visits_all_reachable() {
        let store = create_test_graph();
        let visited = bfs(&store, NodeId::new(0));
        // All 5 nodes are reachable from node 0
        assert_eq!(visited.len(), 5);
    }

    #[test]
    fn test_bfs_layers_distances() {
        let store = create_test_graph();
        let layers = bfs_layers(&store, NodeId::new(0));

        // Layer 0: node 0
        // Layer 1: nodes 1, 3 (direct neighbors)
        // Layer 2: nodes 2, 4 (distance 2)
        assert_eq!(layers.len(), 3);
        assert_eq!(layers[0].len(), 1);
        assert_eq!(layers[1].len(), 2);
        assert_eq!(layers[2].len(), 2);
    }

    #[test]
    fn test_bfs_layers_empty_graph() {
        let store = LpgStore::new();
        let layers = bfs_layers(&store, NodeId::new(0));
        assert!(layers.is_empty());
    }

    #[test]
    fn test_bfs_single_node() {
        let store = LpgStore::new();
        let n0 = store.create_node(&["Node"]);
        let visited = bfs(&store, n0);
        assert_eq!(visited, vec![n0]);
    }

    #[test]
    fn test_bfs_layers_single_node() {
        let store = LpgStore::new();
        let n0 = store.create_node(&["Node"]);
        let layers = bfs_layers(&store, n0);
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0], vec![n0]);
    }

    #[test]
    fn test_bfs_with_visitor_prune_on_start() {
        let store = create_test_graph();
        let result: Option<()> = bfs_with_visitor(&store, NodeId::new(0), |event| {
            if let TraversalEvent::Discover(node) = event
                && node == NodeId::new(0)
            {
                return Control::Prune;
            }
            Control::Continue
        });
        // Pruning start node means no further traversal
        assert!(result.is_none());
    }

    #[test]
    fn test_bfs_with_visitor_collects_tree_edges() {
        let store = create_test_graph();
        let mut tree_edges = Vec::new();

        bfs_with_visitor(&store, NodeId::new(0), |event| -> Control<()> {
            if let TraversalEvent::TreeEdge { source, target, .. } = event {
                tree_edges.push((source, target));
            }
            Control::Continue
        });

        // BFS tree from node 0 has 4 tree edges (one per non-start node)
        assert_eq!(tree_edges.len(), 4);
    }

    #[test]
    fn test_bfs_with_visitor_detects_non_tree_edges() {
        let store = create_test_graph();
        let mut non_tree_edges = Vec::new();

        bfs_with_visitor(&store, NodeId::new(0), |event| -> Control<()> {
            if let TraversalEvent::NonTreeEdge { source, target, .. } = event {
                non_tree_edges.push((source, target));
            }
            Control::Continue
        });

        // There's at least one non-tree edge (3->4 or 1->4)
        assert!(!non_tree_edges.is_empty());
    }

    #[test]
    fn test_dfs_visits_all_reachable() {
        let store = create_test_graph();
        let finished = dfs(&store, NodeId::new(0));
        assert_eq!(finished.len(), 5);
    }

    #[test]
    fn test_dfs_post_order() {
        let store = create_test_graph();
        let finished = dfs(&store, NodeId::new(0));
        // In post-order, the start node is finished last
        assert_eq!(*finished.last().unwrap(), NodeId::new(0));
    }

    #[test]
    fn test_dfs_with_visitor_early_termination() {
        let store = create_test_graph();
        let found = dfs_with_visitor(&store, NodeId::new(0), |event| {
            if let TraversalEvent::Discover(node) = event
                && node == NodeId::new(4)
            {
                return Control::Break(node);
            }
            Control::Continue
        });
        assert_eq!(found, Some(NodeId::new(4)));
    }

    #[test]
    fn test_dfs_with_visitor_prune() {
        let store = create_test_graph();
        let mut discovered = Vec::new();

        dfs_with_visitor(&store, NodeId::new(0), |event| -> Control<()> {
            if let TraversalEvent::Discover(node) = event {
                discovered.push(node);
                if node == NodeId::new(1) {
                    return Control::Prune; // Don't explore node 1's children
                }
            }
            Control::Continue
        });

        // Node 1 is discovered but its children (2, 4) may not all be
        assert!(discovered.contains(&NodeId::new(0)));
        assert!(discovered.contains(&NodeId::new(1)));
        // Node 2 should not be discovered since we pruned node 1
        assert!(!discovered.contains(&NodeId::new(2)));
    }

    #[test]
    fn test_dfs_with_visitor_back_edge() {
        // Create a cycle: 0 -> 1 -> 2 -> 0
        let store = LpgStore::new();
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);
        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n1, n2, "EDGE");
        store.create_edge(n2, n0, "EDGE");

        let mut back_edges = Vec::new();
        dfs_with_visitor(&store, n0, |event| -> Control<()> {
            if let TraversalEvent::BackEdge { source, target, .. } = event {
                back_edges.push((source, target));
            }
            Control::Continue
        });

        // Edge 2->0 is a back edge (0 is ancestor of 2)
        assert_eq!(back_edges.len(), 1);
        assert_eq!(back_edges[0], (n2, n0));
    }

    #[test]
    fn test_dfs_single_node() {
        let store = LpgStore::new();
        let n0 = store.create_node(&["Node"]);
        let finished = dfs(&store, n0);
        assert_eq!(finished, vec![n0]);
    }

    #[test]
    fn test_dfs_all_visits_all_components() {
        let store = LpgStore::new();
        // Component 1: 0 -> 1
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        store.create_edge(n0, n1, "EDGE");

        // Component 2: 2 -> 3
        let n2 = store.create_node(&["Node"]);
        let n3 = store.create_node(&["Node"]);
        store.create_edge(n2, n3, "EDGE");

        let finished = dfs_all(&store);
        assert_eq!(finished.len(), 4);
    }

    #[test]
    fn test_dfs_all_empty_graph() {
        let store = LpgStore::new();
        let finished = dfs_all(&store);
        assert!(finished.is_empty());
    }

    #[test]
    fn test_bfs_prune_tree_edge() {
        let store = create_test_graph();
        let mut discovered = Vec::new();

        bfs_with_visitor(&store, NodeId::new(0), |event| -> Control<()> {
            match event {
                TraversalEvent::TreeEdge { target, .. } => {
                    if target == NodeId::new(1) {
                        return Control::Prune; // Skip node 1
                    }
                    Control::Continue
                }
                TraversalEvent::Discover(node) => {
                    discovered.push(node);
                    Control::Continue
                }
                _ => Control::Continue,
            }
        });

        // Node 1 should not be discovered due to pruned tree edge
        assert!(discovered.contains(&NodeId::new(0)));
        assert!(!discovered.contains(&NodeId::new(1)));
    }

    #[test]
    fn test_dfs_with_visitor_prune_start() {
        let store = create_test_graph();
        let result: Option<()> = dfs_with_visitor(&store, NodeId::new(0), |event| {
            if let TraversalEvent::Discover(node) = event
                && node == NodeId::new(0)
            {
                return Control::Prune;
            }
            Control::Continue
        });
        assert!(result.is_none());
    }
}
