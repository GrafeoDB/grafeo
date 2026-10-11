//! Subgraph isomorphism algorithms.
//!
//! Finds all subgraph isomorphisms of a pattern graph P in a target graph G.
//! Uses a VF2-based backtracking algorithm with feasibility pruning.
//!
//! This is the core algorithm for the GraphChallenge Static Graph benchmark,
//! where triangle counting is a special case (P = K_3).

use std::sync::OnceLock;

use grafeo_common::types::{NodeId, Value};
use grafeo_common::utils::error::Result;
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use grafeo_core::graph::Direction;
use grafeo_core::graph::GraphStore;
#[cfg(all(test, feature = "lpg"))]
use grafeo_core::graph::lpg::LpgStore;

use super::super::{AlgorithmResult, ParameterDef, ParameterType, Parameters};
use super::traits::{GraphAlgorithm, visible_edges_from};

/// Counts all subgraph isomorphisms of `pattern` in `target`.
///
/// Both graphs are treated as undirected. The count includes all distinct
/// mappings from pattern nodes to target nodes that preserve adjacency.
///
/// For an undirected pattern with k nodes, each distinct subgraph is counted
/// `|Aut(P)|` times (once per automorphism of the pattern). To get the number
/// of distinct subgraphs, divide by the automorphism count.
///
/// # Complexity
///
/// O(n^k) worst case where n = |V_target|, k = |V_pattern|. Much better with
/// feasibility pruning on real graphs.
pub fn subgraph_isomorphism_count(target: &dyn GraphStore, pattern: &dyn GraphStore) -> u64 {
    let mut state = VF2State::new(target, pattern);
    state.count_all()
}

/// Enumerates subgraph isomorphisms (up to a limit).
///
/// Each mapping is a Vec of (pattern_node, target_node) pairs.
///
/// # Arguments
///
/// * `target` - The graph to search in.
/// * `pattern` - The subgraph pattern to find.
/// * `limit` - Maximum number of mappings to return (None for all).
pub fn subgraph_isomorphism(
    target: &dyn GraphStore,
    pattern: &dyn GraphStore,
    limit: Option<usize>,
) -> Vec<Vec<(NodeId, NodeId)>> {
    let mut state = VF2State::new(target, pattern);
    state.enumerate(limit)
}

/// Counts subgraph isomorphisms using an edge list to define the pattern.
///
/// This is convenient for the GraphChallenge benchmark where patterns are
/// specified as small edge lists.
///
/// # Arguments
///
/// * `target` - The graph to search in.
/// * `pattern_edges` - Edges of the pattern graph as (source_idx, target_idx) pairs
///   where indices are 0-based.
/// * `pattern_node_count` - Number of nodes in the pattern.
pub fn subgraph_isomorphism_count_from_edges(
    target: &dyn GraphStore,
    pattern_edges: &[(usize, usize)],
    pattern_node_count: usize,
) -> u64 {
    let pattern_adj = build_pattern_adjacency(pattern_edges, pattern_node_count);
    let mut state = VF2StateFromAdj::new(target, &pattern_adj, pattern_node_count);
    state.count_all()
}

// ============================================================================
// VF2 State (GraphStore-based pattern)
// ============================================================================

/// VF2 backtracking state for subgraph isomorphism.
struct VF2State<'a> {
    target_nodes: Vec<NodeId>,
    target_adj: FxHashMap<NodeId, FxHashSet<NodeId>>,
    pattern_nodes: Vec<NodeId>,
    pattern_adj: FxHashMap<NodeId, FxHashSet<NodeId>>,
    /// Current mapping: pattern_node -> target_node.
    mapping: FxHashMap<NodeId, NodeId>,
    /// Reverse mapping: target_node -> pattern_node (for quick membership checks).
    reverse_mapping: FxHashSet<NodeId>,
    _target: &'a dyn GraphStore,
}

impl<'a> VF2State<'a> {
    fn new(target: &'a dyn GraphStore, pattern: &'a dyn GraphStore) -> Self {
        let target_nodes = target.node_ids();
        let pattern_nodes = pattern.node_ids();

        let target_adj = build_undirected_adj(target);
        let pattern_adj = build_undirected_adj(pattern);

        Self {
            target_nodes,
            target_adj,
            pattern_nodes,
            pattern_adj,
            mapping: FxHashMap::default(),
            reverse_mapping: FxHashSet::default(),
            _target: target,
        }
    }

    fn count_all(&mut self) -> u64 {
        if self.pattern_nodes.is_empty() {
            return 1;
        }
        self.backtrack_count(0)
    }

    fn enumerate(&mut self, limit: Option<usize>) -> Vec<Vec<(NodeId, NodeId)>> {
        let mut results = Vec::new();
        if self.pattern_nodes.is_empty() {
            return results;
        }
        self.backtrack_enumerate(0, &mut results, limit);
        results
    }

    fn backtrack_count(&mut self, depth: usize) -> u64 {
        if depth == self.pattern_nodes.len() {
            return 1; // Complete mapping found.
        }

        let pattern_node = self.pattern_nodes[depth];
        let mut count = 0u64;

        for i in 0..self.target_nodes.len() {
            let target_node = self.target_nodes[i];

            if self.reverse_mapping.contains(&target_node) {
                continue;
            }

            if self.is_feasible(pattern_node, target_node) {
                self.mapping.insert(pattern_node, target_node);
                self.reverse_mapping.insert(target_node);

                count += self.backtrack_count(depth + 1);

                self.mapping.remove(&pattern_node);
                self.reverse_mapping.remove(&target_node);
            }
        }

        count
    }

    fn backtrack_enumerate(
        &mut self,
        depth: usize,
        results: &mut Vec<Vec<(NodeId, NodeId)>>,
        limit: Option<usize>,
    ) {
        if let Some(max) = limit
            && results.len() >= max
        {
            return;
        }

        if depth == self.pattern_nodes.len() {
            let mapping: Vec<(NodeId, NodeId)> =
                self.mapping.iter().map(|(&p, &t)| (p, t)).collect();
            results.push(mapping);
            return;
        }

        let pattern_node = self.pattern_nodes[depth];

        for i in 0..self.target_nodes.len() {
            let target_node = self.target_nodes[i];

            if self.reverse_mapping.contains(&target_node) {
                continue;
            }

            if self.is_feasible(pattern_node, target_node) {
                self.mapping.insert(pattern_node, target_node);
                self.reverse_mapping.insert(target_node);

                self.backtrack_enumerate(depth + 1, results, limit);

                self.mapping.remove(&pattern_node);
                self.reverse_mapping.remove(&target_node);

                if let Some(max) = limit
                    && results.len() >= max
                {
                    return;
                }
            }
        }
    }

    /// Checks if mapping pattern_node -> target_node is feasible.
    ///
    /// For each already-mapped neighbor of pattern_node, the corresponding
    /// target node must be a neighbor of target_node.
    fn is_feasible(&self, pattern_node: NodeId, target_node: NodeId) -> bool {
        let Some(pattern_neighbors) = self.pattern_adj.get(&pattern_node) else {
            return true;
        };

        let Some(target_neighbors) = self.target_adj.get(&target_node) else {
            // Target node has no neighbors: only feasible if pattern node also has none
            // among already-mapped neighbors.
            return !pattern_neighbors
                .iter()
                .any(|on| self.mapping.contains_key(on));
        };

        // Connectivity check: every mapped neighbor of pattern_node must map to
        // a neighbor of target_node.
        for &on in pattern_neighbors {
            if let Some(&tn) = self.mapping.get(&on)
                && !target_neighbors.contains(&tn)
            {
                return false;
            }
        }

        // Degree pruning: target_node must have enough neighbors to potentially
        // satisfy all pattern_node's constraints.
        if target_neighbors.len() < pattern_neighbors.len() {
            return false;
        }

        true
    }
}

// ============================================================================
// VF2 State (edge-list-based pattern)
// ============================================================================

/// VF2 state using a pre-built adjacency matrix for the pattern.
struct VF2StateFromAdj<'a> {
    target_nodes: Vec<NodeId>,
    target_adj: FxHashMap<NodeId, FxHashSet<NodeId>>,
    pattern_adj: Vec<FxHashSet<usize>>,
    pattern_node_count: usize,
    /// Current mapping: pattern_idx -> target_node.
    mapping: Vec<Option<NodeId>>,
    /// Set of mapped target nodes.
    mapped_targets: FxHashSet<NodeId>,
    _target: &'a dyn GraphStore,
}

impl<'a> VF2StateFromAdj<'a> {
    fn new(
        target: &'a dyn GraphStore,
        pattern_adj: &[FxHashSet<usize>],
        pattern_node_count: usize,
    ) -> Self {
        let target_nodes = target.node_ids();
        let target_adj = build_undirected_adj(target);

        Self {
            target_nodes,
            target_adj,
            pattern_adj: pattern_adj.to_vec(),
            pattern_node_count,
            mapping: vec![None; pattern_node_count],
            mapped_targets: FxHashSet::default(),
            _target: target,
        }
    }

    fn count_all(&mut self) -> u64 {
        if self.pattern_node_count == 0 {
            return 1;
        }
        self.backtrack_count(0)
    }

    fn backtrack_count(&mut self, depth: usize) -> u64 {
        if depth == self.pattern_node_count {
            return 1;
        }

        let mut count = 0u64;

        for i in 0..self.target_nodes.len() {
            let target_node = self.target_nodes[i];

            if self.mapped_targets.contains(&target_node) {
                continue;
            }

            if self.is_feasible_adj(depth, target_node) {
                self.mapping[depth] = Some(target_node);
                self.mapped_targets.insert(target_node);

                count += self.backtrack_count(depth + 1);

                self.mapping[depth] = None;
                self.mapped_targets.remove(&target_node);
            }
        }

        count
    }

    fn is_feasible_adj(&self, pattern_idx: usize, target_node: NodeId) -> bool {
        let pattern_neighbors = &self.pattern_adj[pattern_idx];

        let Some(target_neighbors) = self.target_adj.get(&target_node) else {
            return !pattern_neighbors
                .iter()
                .any(|&on| self.mapping[on].is_some());
        };

        if target_neighbors.len() < pattern_neighbors.len() {
            return false;
        }

        for &on_idx in pattern_neighbors {
            if let Some(tn) = self.mapping[on_idx]
                && !target_neighbors.contains(&tn)
            {
                return false;
            }
        }

        true
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn build_undirected_adj(store: &dyn GraphStore) -> FxHashMap<NodeId, FxHashSet<NodeId>> {
    let nodes = store.node_ids();
    let mut adj: FxHashMap<NodeId, FxHashSet<NodeId>> = FxHashMap::default();
    for &node in &nodes {
        adj.insert(node, FxHashSet::default());
    }
    for &node in &nodes {
        for (neighbor, _) in visible_edges_from(store, node, Direction::Outgoing) {
            if let Some(set) = adj.get_mut(&node) {
                set.insert(neighbor);
            }
            if let Some(set) = adj.get_mut(&neighbor) {
                set.insert(node);
            }
        }
    }
    adj
}

fn build_pattern_adjacency(edges: &[(usize, usize)], node_count: usize) -> Vec<FxHashSet<usize>> {
    let mut adj = vec![FxHashSet::default(); node_count];
    for &(u, v) in edges {
        if u < node_count && v < node_count {
            adj[u].insert(v);
            adj[v].insert(u);
        }
    }
    adj
}

// ============================================================================
// Algorithm Wrapper
// ============================================================================

static SUBGRAPH_ISO_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn subgraph_iso_params() -> &'static [ParameterDef] {
    SUBGRAPH_ISO_PARAMS.get_or_init(|| {
        vec![
            ParameterDef {
                name: "pattern_edges".to_string(),
                description: "Pattern as comma-separated edge pairs: '0-1,1-2,2-0'".to_string(),
                param_type: ParameterType::String,
                required: true,
                default: None,
            },
            ParameterDef {
                name: "pattern_nodes".to_string(),
                description: "Number of nodes in the pattern".to_string(),
                param_type: ParameterType::Integer,
                required: true,
                default: None,
            },
        ]
    })
}

/// Subgraph Isomorphism algorithm wrapper.
pub struct SubgraphIsomorphismAlgorithm;

impl GraphAlgorithm for SubgraphIsomorphismAlgorithm {
    fn name(&self) -> &str {
        "subgraph_isomorphism"
    }

    fn description(&self) -> &str {
        "Count subgraph isomorphisms of a pattern in the graph"
    }

    fn parameters(&self) -> &[ParameterDef] {
        subgraph_iso_params()
    }

    fn execute(&self, store: &dyn GraphStore, params: &Parameters) -> Result<AlgorithmResult> {
        let edges_str = params.get_string("pattern_edges").unwrap_or("");
        let node_count = usize::try_from(params.get_int("pattern_nodes").unwrap_or(0)).unwrap_or(0);

        let edges: Vec<(usize, usize)> = edges_str
            .split(',')
            .filter(|s| !s.is_empty())
            .filter_map(|s| {
                let parts: Vec<&str> = s.split('-').collect();
                if parts.len() == 2 {
                    Some((parts[0].parse().ok()?, parts[1].parse().ok()?))
                } else {
                    None
                }
            })
            .collect();

        let count = subgraph_isomorphism_count_from_edges(store, &edges, node_count);

        let mut output = AlgorithmResult::new(vec!["count".to_string()]);
        output.add_row(vec![Value::Int64(i64::try_from(count).unwrap_or(i64::MAX))]);
        Ok(output)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use super::*;

    fn create_triangle() -> LpgStore {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);

        store.create_edge(n0, n1, "E");
        store.create_edge(n1, n0, "E");
        store.create_edge(n1, n2, "E");
        store.create_edge(n2, n1, "E");
        store.create_edge(n2, n0, "E");
        store.create_edge(n0, n2, "E");

        store
    }

    fn create_k4() -> LpgStore {
        let store = LpgStore::new().unwrap();
        let nodes: Vec<NodeId> = (0..4).map(|_| store.create_node(&["Node"])).collect();
        for i in 0..4 {
            for j in (i + 1)..4 {
                store.create_edge(nodes[i], nodes[j], "E");
                store.create_edge(nodes[j], nodes[i], "E");
            }
        }
        store
    }

    #[test]
    fn test_triangle_in_triangle() {
        let target = create_triangle();
        let pattern = create_triangle();

        // A triangle has 6 automorphisms (3! = 6), so counting all
        // isomorphisms of a triangle in a triangle gives 6.
        let count = subgraph_isomorphism_count(&target, &pattern);
        assert_eq!(count, 6, "Triangle in triangle should have 6 mappings (3!)");
    }

    #[test]
    fn test_triangle_in_k4() {
        let target = create_k4();
        let pattern = create_triangle();

        // K_4 has C(4,3) = 4 triangles. Each has 3! = 6 automorphisms.
        // Total mappings = 4 * 6 = 24.
        let count = subgraph_isomorphism_count(&target, &pattern);
        assert_eq!(count, 24, "K_4 should have 24 triangle mappings (4 * 3!)");
    }

    #[test]
    fn test_edge_in_k4() {
        // Pattern: single edge (2 nodes)
        let pattern = LpgStore::new().unwrap();
        let p0 = pattern.create_node(&["Node"]);
        let p1 = pattern.create_node(&["Node"]);
        pattern.create_edge(p0, p1, "E");
        pattern.create_edge(p1, p0, "E");

        let target = create_k4();

        // K_4 has 6 edges. Each edge has 2! = 2 automorphisms.
        // Total = 6 * 2 = 12.
        let count = subgraph_isomorphism_count(&target, &pattern);
        assert_eq!(count, 12, "K_4 should have 12 edge mappings (6 * 2!)");
    }

    #[test]
    fn test_from_edges_triangle_in_k4() {
        let target = create_k4();

        // Triangle pattern as edge list.
        let pattern_edges = [(0, 1), (1, 2), (2, 0)];
        let count = subgraph_isomorphism_count_from_edges(&target, &pattern_edges, 3);

        assert_eq!(count, 24, "Edge-list triangle in K_4 should give 24");
    }

    #[test]
    fn test_k4_in_k4() {
        let target = create_k4();
        let pattern = create_k4();

        // K_4 in K_4: 4! = 24 automorphisms.
        let count = subgraph_isomorphism_count(&target, &pattern);
        assert_eq!(count, 24, "K_4 in K_4 should have 24 mappings (4!)");
    }

    #[test]
    fn test_enumerate_triangle_in_triangle() {
        let target = create_triangle();
        let pattern = create_triangle();

        let mappings = subgraph_isomorphism(&target, &pattern, None);
        assert_eq!(mappings.len(), 6);

        // Each mapping should have 3 entries.
        for m in &mappings {
            assert_eq!(m.len(), 3);
        }
    }

    #[test]
    fn test_enumerate_with_limit() {
        let target = create_k4();
        let pattern = create_triangle();

        let mappings = subgraph_isomorphism(&target, &pattern, Some(3));
        assert_eq!(mappings.len(), 3);
    }

    #[test]
    fn test_no_match() {
        // Target is a path (no triangles), pattern is a triangle.
        let target = LpgStore::new().unwrap();
        let n0 = target.create_node(&["Node"]);
        let n1 = target.create_node(&["Node"]);
        let n2 = target.create_node(&["Node"]);
        target.create_edge(n0, n1, "E");
        target.create_edge(n1, n0, "E");
        target.create_edge(n1, n2, "E");
        target.create_edge(n2, n1, "E");
        // No edge between n0 and n2: not a triangle.

        let pattern = create_triangle();
        let count = subgraph_isomorphism_count(&target, &pattern);
        assert_eq!(count, 0, "Path graph should have 0 triangle isomorphisms");
    }

    #[test]
    fn test_empty_pattern() {
        let target = create_k4();
        let pattern = LpgStore::new().unwrap();

        let count = subgraph_isomorphism_count(&target, &pattern);
        assert_eq!(count, 1, "Empty pattern should have 1 mapping (empty)");
    }

    #[test]
    fn test_empty_target() {
        let target = LpgStore::new().unwrap();
        let pattern = create_triangle();

        let count = subgraph_isomorphism_count(&target, &pattern);
        assert_eq!(count, 0, "Empty target should have 0 mappings");
    }

    #[test]
    fn test_algorithm_wrapper() {
        use super::super::traits::GraphAlgorithm;

        let store = create_k4();
        let algo = SubgraphIsomorphismAlgorithm;

        let mut params = Parameters::new();
        params.set_string("pattern_edges", "0-1,1-2,2-0");
        params.set_int("pattern_nodes", 3);

        let result = algo.execute(&store, &params).unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows[0][0], Value::Int64(24));
    }
}
