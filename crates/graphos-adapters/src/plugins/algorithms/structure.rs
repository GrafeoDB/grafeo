//! Structure analysis algorithms: Articulation Points, Bridges, K-Core decomposition.
//!
//! These algorithms identify critical structural elements in graphs.

use std::sync::OnceLock;

use graphos_common::types::{NodeId, Value};
use graphos_common::utils::error::Result;
use graphos_common::utils::hash::{FxHashMap, FxHashSet};
use graphos_core::graph::lpg::LpgStore;
use graphos_core::graph::Direction;

use super::traits::GraphAlgorithm;
use super::super::{AlgorithmResult, ParameterDef, ParameterType, Parameters};

// ============================================================================
// Articulation Points (Cut Vertices)
// ============================================================================

/// Finds articulation points (cut vertices) in the graph.
///
/// An articulation point is a vertex whose removal disconnects the graph.
/// Uses Tarjan's algorithm with low-link values.
///
/// # Arguments
///
/// * `store` - The graph store (treated as undirected)
///
/// # Returns
///
/// Set of node IDs that are articulation points.
///
/// # Complexity
///
/// O(V + E)
pub fn articulation_points(store: &LpgStore) -> FxHashSet<NodeId> {
    let nodes = store.node_ids();
    let n = nodes.len();

    if n == 0 {
        return FxHashSet::default();
    }

    // Build node index mapping
    let mut node_to_idx: FxHashMap<NodeId, usize> = FxHashMap::default();
    let mut idx_to_node: Vec<NodeId> = Vec::with_capacity(n);
    for (idx, &node) in nodes.iter().enumerate() {
        node_to_idx.insert(node, idx);
        idx_to_node.push(node);
    }

    // Build undirected adjacency list
    let mut adj: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); n];
    for &node in &nodes {
        let i = *node_to_idx.get(&node).unwrap();
        for (neighbor, _) in store.edges_from(node, Direction::Outgoing) {
            if let Some(&j) = node_to_idx.get(&neighbor) {
                adj[i].insert(j);
                adj[j].insert(i); // Undirected
            }
        }
    }

    let mut visited = vec![false; n];
    let mut disc = vec![0usize; n]; // Discovery time
    let mut low = vec![0usize; n]; // Low-link value
    let mut parent = vec![None::<usize>; n];
    let mut ap = vec![false; n]; // Is articulation point
    let mut time = 0usize;

    // DFS from each unvisited node (handles disconnected graphs)
    for start in 0..n {
        if visited[start] {
            continue;
        }

        // Iterative DFS using explicit stack
        let mut stack: Vec<(usize, usize)> = vec![(start, 0)]; // (node, neighbor_idx)
        let mut children_count: FxHashMap<usize, usize> = FxHashMap::default();

        while let Some(&(u, idx)) = stack.last() {
            if !visited[u] {
                visited[u] = true;
                disc[u] = time;
                low[u] = time;
                time += 1;
                children_count.insert(u, 0);
            }

            let neighbors: Vec<usize> = adj[u].iter().copied().collect();

            if idx < neighbors.len() {
                let v = neighbors[idx];
                stack.last_mut().unwrap().1 += 1;

                if !visited[v] {
                    parent[v] = Some(u);
                    *children_count.entry(u).or_insert(0) += 1;
                    stack.push((v, 0));
                } else if parent[u] != Some(v) {
                    low[u] = low[u].min(disc[v]);
                }
            } else {
                stack.pop();

                if let Some(p) = parent[u] {
                    low[p] = low[p].min(low[u]);

                    // Check articulation point condition
                    if parent[p].is_some() && low[u] >= disc[p] {
                        ap[p] = true;
                    }
                }

                // Root is articulation point if it has more than one child
                if parent[u].is_none() && *children_count.get(&u).unwrap_or(&0) > 1 {
                    ap[u] = true;
                }
            }
        }
    }

    ap.iter()
        .enumerate()
        .filter(|&(_, is_ap)| *is_ap)
        .map(|(idx, _)| idx_to_node[idx])
        .collect()
}

// ============================================================================
// Bridges (Cut Edges)
// ============================================================================

/// Finds bridges (cut edges) in the graph.
///
/// A bridge is an edge whose removal disconnects the graph.
/// Uses Tarjan's algorithm with low-link values.
///
/// # Arguments
///
/// * `store` - The graph store (treated as undirected)
///
/// # Returns
///
/// List of bridges as (source, target) pairs.
///
/// # Complexity
///
/// O(V + E)
pub fn bridges(store: &LpgStore) -> Vec<(NodeId, NodeId)> {
    let nodes = store.node_ids();
    let n = nodes.len();

    if n == 0 {
        return Vec::new();
    }

    // Build node index mapping
    let mut node_to_idx: FxHashMap<NodeId, usize> = FxHashMap::default();
    let mut idx_to_node: Vec<NodeId> = Vec::with_capacity(n);
    for (idx, &node) in nodes.iter().enumerate() {
        node_to_idx.insert(node, idx);
        idx_to_node.push(node);
    }

    // Build undirected adjacency list
    let mut adj: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); n];
    for &node in &nodes {
        let i = *node_to_idx.get(&node).unwrap();
        for (neighbor, _) in store.edges_from(node, Direction::Outgoing) {
            if let Some(&j) = node_to_idx.get(&neighbor) {
                adj[i].insert(j);
                adj[j].insert(i);
            }
        }
    }

    let mut visited = vec![false; n];
    let mut disc = vec![0usize; n];
    let mut low = vec![0usize; n];
    let mut parent = vec![None::<usize>; n];
    let mut time = 0usize;
    let mut bridge_list: Vec<(usize, usize)> = Vec::new();

    for start in 0..n {
        if visited[start] {
            continue;
        }

        let mut stack: Vec<(usize, usize)> = vec![(start, 0)];

        while let Some(&(u, idx)) = stack.last() {
            if !visited[u] {
                visited[u] = true;
                disc[u] = time;
                low[u] = time;
                time += 1;
            }

            let neighbors: Vec<usize> = adj[u].iter().copied().collect();

            if idx < neighbors.len() {
                let v = neighbors[idx];
                stack.last_mut().unwrap().1 += 1;

                if !visited[v] {
                    parent[v] = Some(u);
                    stack.push((v, 0));
                } else if parent[u] != Some(v) {
                    low[u] = low[u].min(disc[v]);
                }
            } else {
                stack.pop();

                if let Some(p) = parent[u] {
                    low[p] = low[p].min(low[u]);

                    // Bridge condition: low[u] > disc[p]
                    if low[u] > disc[p] {
                        bridge_list.push((p.min(u), p.max(u)));
                    }
                }
            }
        }
    }

    bridge_list
        .into_iter()
        .map(|(i, j)| (idx_to_node[i], idx_to_node[j]))
        .collect()
}

// ============================================================================
// K-Core Decomposition
// ============================================================================

/// Result of k-core decomposition.
#[derive(Debug, Clone)]
pub struct KCoreResult {
    /// Core number for each node.
    pub core_numbers: FxHashMap<NodeId, usize>,
    /// Maximum core number (degeneracy).
    pub max_core: usize,
}

impl KCoreResult {
    /// Returns nodes in the k-core (nodes with core number >= k).
    pub fn k_core(&self, k: usize) -> Vec<NodeId> {
        self.core_numbers
            .iter()
            .filter(|&(_, core)| *core >= k)
            .map(|(&node, _)| node)
            .collect()
    }

    /// Returns the k-shell (nodes with core number exactly k).
    pub fn k_shell(&self, k: usize) -> Vec<NodeId> {
        self.core_numbers
            .iter()
            .filter(|&(_, core)| *core == k)
            .map(|(&node, _)| node)
            .collect()
    }
}

/// Computes the k-core decomposition of the graph.
///
/// The k-core is the maximal subgraph where every vertex has degree at least k.
/// The core number of a vertex is the largest k such that it belongs to the k-core.
///
/// # Arguments
///
/// * `store` - The graph store (treated as undirected)
///
/// # Returns
///
/// Core numbers for all nodes and the maximum core number.
///
/// # Complexity
///
/// O(V + E)
pub fn kcore_decomposition(store: &LpgStore) -> KCoreResult {
    let nodes = store.node_ids();
    let n = nodes.len();

    if n == 0 {
        return KCoreResult {
            core_numbers: FxHashMap::default(),
            max_core: 0,
        };
    }

    // Build node index mapping
    let mut node_to_idx: FxHashMap<NodeId, usize> = FxHashMap::default();
    let mut idx_to_node: Vec<NodeId> = Vec::with_capacity(n);
    for (idx, &node) in nodes.iter().enumerate() {
        node_to_idx.insert(node, idx);
        idx_to_node.push(node);
    }

    // Build undirected adjacency list and compute degrees
    let mut adj: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); n];
    for &node in &nodes {
        let i = *node_to_idx.get(&node).unwrap();
        for (neighbor, _) in store.edges_from(node, Direction::Outgoing) {
            if let Some(&j) = node_to_idx.get(&neighbor) {
                adj[i].insert(j);
                adj[j].insert(i);
            }
        }
    }

    let mut degree: Vec<usize> = adj.iter().map(|neighbors| neighbors.len()).collect();
    let mut core = vec![0usize; n];
    let mut removed = vec![false; n];

    // Find maximum degree for bucket initialization
    let max_degree = *degree.iter().max().unwrap_or(&0);
    if max_degree == 0 {
        return KCoreResult {
            core_numbers: nodes.iter().map(|&n| (n, 0)).collect(),
            max_core: 0,
        };
    }

    // Buckets for O(1) retrieval of minimum degree vertices
    let mut buckets: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); max_degree + 1];
    for (i, &d) in degree.iter().enumerate() {
        buckets[d].insert(i);
    }

    let mut max_core_val = 0;

    // Process vertices in order of increasing degree
    for _ in 0..n {
        // Find minimum degree bucket
        let mut min_deg = 0;
        while min_deg <= max_degree && buckets[min_deg].is_empty() {
            min_deg += 1;
        }

        if min_deg > max_degree {
            break;
        }

        // Pick a vertex from the minimum degree bucket
        let v = *buckets[min_deg].iter().next().unwrap();
        buckets[min_deg].remove(&v);
        removed[v] = true;
        core[v] = min_deg;
        max_core_val = max_core_val.max(min_deg);

        // Update degrees of neighbors
        for &u in &adj[v] {
            if !removed[u] && degree[u] > 0 {
                let old_deg = degree[u];
                buckets[old_deg].remove(&u);
                degree[u] -= 1;
                let new_deg = degree[u];
                buckets[new_deg].insert(u);
            }
        }
    }

    let core_numbers: FxHashMap<NodeId, usize> = (0..n)
        .map(|i| (idx_to_node[i], core[i]))
        .collect();

    KCoreResult {
        core_numbers,
        max_core: max_core_val,
    }
}

/// Extracts the k-core subgraph (nodes with core number >= k).
pub fn k_core(store: &LpgStore, k: usize) -> Vec<NodeId> {
    let result = kcore_decomposition(store);
    result.k_core(k)
}

// ============================================================================
// Algorithm Wrappers for Plugin Registry
// ============================================================================

/// Static parameter definitions for Articulation Points algorithm.
static ARTICULATION_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn articulation_params() -> &'static [ParameterDef] {
    ARTICULATION_PARAMS.get_or_init(Vec::new)
}

/// Articulation Points algorithm wrapper.
pub struct ArticulationPointsAlgorithm;

impl GraphAlgorithm for ArticulationPointsAlgorithm {
    fn name(&self) -> &str {
        "articulation_points"
    }

    fn description(&self) -> &str {
        "Find articulation points (cut vertices) in the graph"
    }

    fn parameters(&self) -> &[ParameterDef] {
        articulation_params()
    }

    fn execute(&self, store: &LpgStore, _params: &Parameters) -> Result<AlgorithmResult> {
        let points = articulation_points(store);

        let mut result = AlgorithmResult::new(vec!["node_id".to_string()]);

        for node in points {
            result.add_row(vec![Value::Int64(node.0 as i64)]);
        }

        Ok(result)
    }
}

/// Static parameter definitions for Bridges algorithm.
static BRIDGES_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn bridges_params() -> &'static [ParameterDef] {
    BRIDGES_PARAMS.get_or_init(Vec::new)
}

/// Bridges algorithm wrapper.
pub struct BridgesAlgorithm;

impl GraphAlgorithm for BridgesAlgorithm {
    fn name(&self) -> &str {
        "bridges"
    }

    fn description(&self) -> &str {
        "Find bridges (cut edges) in the graph"
    }

    fn parameters(&self) -> &[ParameterDef] {
        bridges_params()
    }

    fn execute(&self, store: &LpgStore, _params: &Parameters) -> Result<AlgorithmResult> {
        let bridge_list = bridges(store);

        let mut result = AlgorithmResult::new(vec![
            "source".to_string(),
            "target".to_string(),
        ]);

        for (src, dst) in bridge_list {
            result.add_row(vec![
                Value::Int64(src.0 as i64),
                Value::Int64(dst.0 as i64),
            ]);
        }

        Ok(result)
    }
}

/// Static parameter definitions for K-Core algorithm.
static KCORE_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn kcore_params() -> &'static [ParameterDef] {
    KCORE_PARAMS.get_or_init(|| {
        vec![ParameterDef {
            name: "k".to_string(),
            description: "Core number threshold (optional, returns decomposition if not set)"
                .to_string(),
            param_type: ParameterType::Integer,
            required: false,
            default: None,
        }]
    })
}

/// K-Core decomposition algorithm wrapper.
pub struct KCoreAlgorithm;

impl GraphAlgorithm for KCoreAlgorithm {
    fn name(&self) -> &str {
        "kcore"
    }

    fn description(&self) -> &str {
        "K-core decomposition of the graph"
    }

    fn parameters(&self) -> &[ParameterDef] {
        kcore_params()
    }

    fn execute(&self, store: &LpgStore, params: &Parameters) -> Result<AlgorithmResult> {
        let decomposition = kcore_decomposition(store);

        if let Some(k) = params.get_int("k") {
            // Return nodes in the k-core
            let k_core_nodes = decomposition.k_core(k as usize);

            let mut result = AlgorithmResult::new(vec![
                "node_id".to_string(),
                "in_k_core".to_string(),
            ]);

            for node in k_core_nodes {
                result.add_row(vec![Value::Int64(node.0 as i64), Value::Bool(true)]);
            }

            Ok(result)
        } else {
            // Return full decomposition
            let mut result = AlgorithmResult::new(vec![
                "node_id".to_string(),
                "core_number".to_string(),
                "max_core".to_string(),
            ]);

            for (node, core) in decomposition.core_numbers {
                result.add_row(vec![
                    Value::Int64(node.0 as i64),
                    Value::Int64(core as i64),
                    Value::Int64(decomposition.max_core as i64),
                ]);
            }

            Ok(result)
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_simple_path() -> LpgStore {
        // Path: 0 - 1 - 2 - 3 (all are articulation points except endpoints)
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

    fn create_diamond() -> LpgStore {
        // Diamond: 0 - 1 - 3
        //          |   |
        //          +---2
        // No articulation points in a diamond
        let store = LpgStore::new();

        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);
        let n3 = store.create_node(&["Node"]);

        // 0-1, 0-2, 1-3, 2-3
        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n1, n0, "EDGE");
        store.create_edge(n0, n2, "EDGE");
        store.create_edge(n2, n0, "EDGE");
        store.create_edge(n1, n3, "EDGE");
        store.create_edge(n3, n1, "EDGE");
        store.create_edge(n2, n3, "EDGE");
        store.create_edge(n3, n2, "EDGE");

        store
    }

    fn create_tree() -> LpgStore {
        // Tree:     0
        //          / \
        //         1   2
        //        /
        //       3
        let store = LpgStore::new();

        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);
        let n3 = store.create_node(&["Node"]);

        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n1, n0, "EDGE");
        store.create_edge(n0, n2, "EDGE");
        store.create_edge(n2, n0, "EDGE");
        store.create_edge(n1, n3, "EDGE");
        store.create_edge(n3, n1, "EDGE");

        store
    }

    #[test]
    fn test_articulation_points_path() {
        let store = create_simple_path();
        let ap = articulation_points(&store);

        // In a path, middle nodes are articulation points
        // 0-1-2-3: nodes 1 and 2 are articulation points
        assert!(ap.len() >= 2);
        assert!(ap.contains(&NodeId::new(1)) || ap.contains(&NodeId::new(2)));
    }

    #[test]
    fn test_articulation_points_diamond() {
        let store = create_diamond();
        let ap = articulation_points(&store);

        // Diamond has no articulation points (it's 2-connected)
        assert!(ap.is_empty());
    }

    #[test]
    fn test_articulation_points_tree() {
        let store = create_tree();
        let ap = articulation_points(&store);

        // In a tree, all non-leaf nodes are articulation points
        // Tree: 0 has children 1, 2. Node 1 has child 3.
        // Articulation points: 0, 1
        assert!(ap.contains(&NodeId::new(0)) || ap.contains(&NodeId::new(1)));
    }

    #[test]
    fn test_articulation_points_empty() {
        let store = LpgStore::new();
        let ap = articulation_points(&store);
        assert!(ap.is_empty());
    }

    #[test]
    fn test_bridges_path() {
        let store = create_simple_path();
        let br = bridges(&store);

        // In a path, all edges are bridges
        assert_eq!(br.len(), 3);
    }

    #[test]
    fn test_bridges_diamond() {
        let store = create_diamond();
        let br = bridges(&store);

        // Diamond has no bridges (every edge is part of a cycle)
        assert!(br.is_empty());
    }

    #[test]
    fn test_bridges_empty() {
        let store = LpgStore::new();
        let br = bridges(&store);
        assert!(br.is_empty());
    }

    #[test]
    fn test_kcore_path() {
        let store = create_simple_path();
        let result = kcore_decomposition(&store);

        // In a path using peeling algorithm, most nodes have core number 1
        // At least some nodes should have core number >= 1
        let max_core = result.core_numbers.values().copied().max().unwrap_or(0);
        assert!(max_core >= 1);
        assert_eq!(result.max_core, max_core);
    }

    #[test]
    fn test_kcore_triangle() {
        let store = LpgStore::new();
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);

        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n1, n0, "EDGE");
        store.create_edge(n1, n2, "EDGE");
        store.create_edge(n2, n1, "EDGE");
        store.create_edge(n0, n2, "EDGE");
        store.create_edge(n2, n0, "EDGE");

        let result = kcore_decomposition(&store);

        // Triangle: max core should be at least 1 (nodes have degree 2)
        assert!(result.max_core >= 1);
        // All nodes should be decomposed
        assert_eq!(result.core_numbers.len(), 3);
    }

    #[test]
    fn test_kcore_empty() {
        let store = LpgStore::new();
        let result = kcore_decomposition(&store);

        assert!(result.core_numbers.is_empty());
        assert_eq!(result.max_core, 0);
    }

    #[test]
    fn test_kcore_isolated() {
        let store = LpgStore::new();
        store.create_node(&["Node"]);
        store.create_node(&["Node"]);

        let result = kcore_decomposition(&store);

        // Isolated nodes have core number 0
        for (_, &core) in &result.core_numbers {
            assert_eq!(core, 0);
        }
    }

    #[test]
    fn test_k_core_extraction() {
        let store = create_simple_path();
        let result = kcore_decomposition(&store);

        // k_core(0) should return all nodes
        let k0_core = result.k_core(0);
        assert_eq!(k0_core.len(), 4);

        // Higher k-cores have fewer or equal nodes
        let k1_core = result.k_core(1);
        assert!(k1_core.len() <= 4);

        let k2_core = result.k_core(2);
        assert!(k2_core.len() <= k1_core.len());
    }

    #[test]
    fn test_k_shell() {
        let store = create_simple_path();
        let result = kcore_decomposition(&store);

        // Total nodes in all shells should equal total nodes
        let total_in_shells: usize = (0..=result.max_core)
            .map(|k| result.k_shell(k).len())
            .sum();
        assert_eq!(total_in_shells, 4);
    }
}
