//! Clustering coefficient algorithms: Local, Global, and Triangle counting.
//!
//! These algorithms measure how tightly connected the neighbors of each node are.
//! A high clustering coefficient indicates that neighbors tend to be connected to each other.

#[cfg(feature = "parallel")]
use std::sync::Arc;
use std::sync::OnceLock;

use grafeo_common::types::{NodeId, Value};
use grafeo_common::utils::error::Result;
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use grafeo_core::graph::Direction;
use grafeo_core::graph::lpg::LpgStore;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::super::{AlgorithmResult, ParameterDef, ParameterType, Parameters};
use super::traits::GraphAlgorithm;
#[cfg(feature = "parallel")]
use super::traits::ParallelGraphAlgorithm;

// ============================================================================
// Result Types
// ============================================================================

/// Result of clustering coefficient computation.
#[derive(Debug, Clone)]
pub struct ClusteringCoefficientResult {
    /// Local clustering coefficient for each node (0.0 to 1.0).
    pub coefficients: FxHashMap<NodeId, f64>,
    /// Number of triangles containing each node.
    pub triangle_counts: FxHashMap<NodeId, u64>,
    /// Total number of unique triangles in the graph.
    pub total_triangles: u64,
    /// Global (average) clustering coefficient.
    pub global_coefficient: f64,
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Builds undirected neighbor sets for all nodes.
///
/// Treats the graph as undirected by combining both outgoing and incoming edges.
fn build_undirected_neighbors(store: &LpgStore) -> FxHashMap<NodeId, FxHashSet<NodeId>> {
    let nodes = store.node_ids();
    let mut neighbors: FxHashMap<NodeId, FxHashSet<NodeId>> = FxHashMap::default();

    // Initialize all nodes with empty sets
    for &node in &nodes {
        neighbors.insert(node, FxHashSet::default());
    }

    // Add edges in both directions (undirected treatment)
    for &node in &nodes {
        // Outgoing edges: node -> neighbor
        for (neighbor, _) in store.edges_from(node, Direction::Outgoing) {
            if let Some(set) = neighbors.get_mut(&node) {
                set.insert(neighbor);
            }
            // Add reverse direction for undirected
            if let Some(set) = neighbors.get_mut(&neighbor) {
                set.insert(node);
            }
        }

        // Incoming edges: neighbor -> node (ensures we capture all connections)
        for (neighbor, _) in store.edges_from(node, Direction::Incoming) {
            if let Some(set) = neighbors.get_mut(&node) {
                set.insert(neighbor);
            }
            if let Some(set) = neighbors.get_mut(&neighbor) {
                set.insert(node);
            }
        }
    }

    neighbors
}

/// Counts triangles for a single node given its neighbors and the full neighbor map.
fn count_node_triangles(
    node_neighbors: &FxHashSet<NodeId>,
    all_neighbors: &FxHashMap<NodeId, FxHashSet<NodeId>>,
) -> u64 {
    let neighbor_list: Vec<NodeId> = node_neighbors.iter().copied().collect();
    let k = neighbor_list.len();
    let mut triangles = 0u64;

    // For each pair of neighbors, check if they're connected
    for i in 0..k {
        for j in (i + 1)..k {
            let u = neighbor_list[i];
            let w = neighbor_list[j];

            // Check if u and w are neighbors (completing a triangle)
            if let Some(u_neighbors) = all_neighbors.get(&u)
                && u_neighbors.contains(&w)
            {
                triangles += 1;
            }
        }
    }

    triangles
}

// ============================================================================
// Core Algorithm Functions
// ============================================================================

/// Counts the number of triangles containing each node.
///
/// A triangle is a set of three nodes where each pair is connected.
/// Each triangle is counted once for each of its three vertices.
///
/// # Arguments
///
/// * `store` - The graph store (treated as undirected)
///
/// # Returns
///
/// Map from NodeId to the number of triangles containing that node.
///
/// # Complexity
///
/// O(V * d^2) where d is the average degree
pub fn triangle_count(store: &LpgStore) -> FxHashMap<NodeId, u64> {
    let neighbors = build_undirected_neighbors(store);
    let mut counts: FxHashMap<NodeId, u64> = FxHashMap::default();

    for (&node, node_neighbors) in &neighbors {
        let triangles = count_node_triangles(node_neighbors, &neighbors);
        counts.insert(node, triangles);
    }

    counts
}

/// Computes the local clustering coefficient for each node.
///
/// The local clustering coefficient measures how close a node's neighbors are
/// to being a complete graph (clique). For a node v with degree k and T triangles:
///
/// C(v) = 2T / (k * (k-1)) for undirected graphs
///
/// Nodes with degree < 2 have coefficient 0.0 (cannot form triangles).
///
/// # Arguments
///
/// * `store` - The graph store (treated as undirected)
///
/// # Returns
///
/// Map from NodeId to local clustering coefficient (0.0 to 1.0).
///
/// # Complexity
///
/// O(V * d^2) where d is the average degree
pub fn local_clustering_coefficient(store: &LpgStore) -> FxHashMap<NodeId, f64> {
    let neighbors = build_undirected_neighbors(store);
    let mut coefficients: FxHashMap<NodeId, f64> = FxHashMap::default();

    for (&node, node_neighbors) in &neighbors {
        let k = node_neighbors.len();

        if k < 2 {
            // Cannot form triangles with fewer than 2 neighbors
            coefficients.insert(node, 0.0);
        } else {
            let triangles = count_node_triangles(node_neighbors, &neighbors);
            let max_triangles = (k * (k - 1)) / 2;
            let coefficient = triangles as f64 / max_triangles as f64;
            coefficients.insert(node, coefficient);
        }
    }

    coefficients
}

/// Computes the global (average) clustering coefficient.
///
/// The global clustering coefficient is the average of all local coefficients
/// across all nodes in the graph.
///
/// # Arguments
///
/// * `store` - The graph store (treated as undirected)
///
/// # Returns
///
/// Average clustering coefficient (0.0 to 1.0).
///
/// # Complexity
///
/// O(V * d^2) where d is the average degree
pub fn global_clustering_coefficient(store: &LpgStore) -> f64 {
    let local = local_clustering_coefficient(store);

    if local.is_empty() {
        return 0.0;
    }

    let sum: f64 = local.values().sum();
    sum / local.len() as f64
}

/// Counts the total number of unique triangles in the graph.
///
/// Each triangle is counted exactly once (not three times).
///
/// # Arguments
///
/// * `store` - The graph store (treated as undirected)
///
/// # Returns
///
/// Total number of unique triangles.
///
/// # Complexity
///
/// O(V * d^2) where d is the average degree
pub fn total_triangles(store: &LpgStore) -> u64 {
    let per_node = triangle_count(store);
    // Each triangle is counted 3 times (once per vertex), so divide by 3
    per_node.values().sum::<u64>() / 3
}

/// Computes all clustering metrics in a single pass.
///
/// More efficient than calling each function separately since it builds
/// the neighbor structure only once.
///
/// # Arguments
///
/// * `store` - The graph store (treated as undirected)
///
/// # Returns
///
/// Complete clustering coefficient result including local coefficients,
/// triangle counts, total triangles, and global coefficient.
///
/// # Complexity
///
/// O(V * d^2) where d is the average degree
pub fn clustering_coefficient(store: &LpgStore) -> ClusteringCoefficientResult {
    let neighbors = build_undirected_neighbors(store);
    let n = neighbors.len();

    let mut coefficients: FxHashMap<NodeId, f64> = FxHashMap::default();
    let mut triangle_counts: FxHashMap<NodeId, u64> = FxHashMap::default();

    for (&node, node_neighbors) in &neighbors {
        let k = node_neighbors.len();
        let triangles = count_node_triangles(node_neighbors, &neighbors);

        triangle_counts.insert(node, triangles);

        let coefficient = if k < 2 {
            0.0
        } else {
            let max_triangles = (k * (k - 1)) / 2;
            triangles as f64 / max_triangles as f64
        };
        coefficients.insert(node, coefficient);
    }

    let total_triangles = triangle_counts.values().sum::<u64>() / 3;
    let global_coefficient = if n == 0 {
        0.0
    } else {
        coefficients.values().sum::<f64>() / n as f64
    };

    ClusteringCoefficientResult {
        coefficients,
        triangle_counts,
        total_triangles,
        global_coefficient,
    }
}

// ============================================================================
// Parallel Implementation
// ============================================================================

/// Computes clustering coefficients in parallel using rayon.
///
/// Automatically falls back to sequential execution for small graphs
/// to avoid parallelization overhead.
///
/// # Arguments
///
/// * `store` - The graph store (treated as undirected)
/// * `parallel_threshold` - Minimum node count to enable parallelism
///
/// # Returns
///
/// Complete clustering coefficient result.
///
/// # Complexity
///
/// O(V * d^2 / threads) where d is the average degree
#[cfg(feature = "parallel")]
pub fn clustering_coefficient_parallel(
    store: &LpgStore,
    parallel_threshold: usize,
) -> ClusteringCoefficientResult {
    let neighbors = build_undirected_neighbors(store);
    let n = neighbors.len();

    if n < parallel_threshold {
        // Fall back to sequential for small graphs
        return clustering_coefficient(store);
    }

    // Use Arc for shared neighbor data across threads
    let neighbors = Arc::new(neighbors);
    let nodes: Vec<NodeId> = neighbors.keys().copied().collect();

    // Parallel computation using fold-reduce pattern
    let (coefficients, triangle_counts): (FxHashMap<NodeId, f64>, FxHashMap<NodeId, u64>) = nodes
        .par_iter()
        .fold(
            || (FxHashMap::default(), FxHashMap::default()),
            |(mut coeffs, mut triangles), &node| {
                let node_neighbors = neighbors.get(&node).unwrap();
                let k = node_neighbors.len();

                let t = count_node_triangles(node_neighbors, &neighbors);

                triangles.insert(node, t);

                let coefficient = if k < 2 {
                    0.0
                } else {
                    let max_triangles = (k * (k - 1)) / 2;
                    t as f64 / max_triangles as f64
                };
                coeffs.insert(node, coefficient);

                (coeffs, triangles)
            },
        )
        .reduce(
            || (FxHashMap::default(), FxHashMap::default()),
            |(mut c1, mut t1), (c2, t2)| {
                c1.extend(c2);
                t1.extend(t2);
                (c1, t1)
            },
        );

    let total_triangles = triangle_counts.values().sum::<u64>() / 3;
    let global_coefficient = if n == 0 {
        0.0
    } else {
        coefficients.values().sum::<f64>() / n as f64
    };

    ClusteringCoefficientResult {
        coefficients,
        triangle_counts,
        total_triangles,
        global_coefficient,
    }
}

// ============================================================================
// Algorithm Wrapper for Plugin Registry
// ============================================================================

/// Static parameter definitions for Clustering Coefficient algorithm.
static CLUSTERING_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn clustering_params() -> &'static [ParameterDef] {
    CLUSTERING_PARAMS.get_or_init(|| {
        vec![
            ParameterDef {
                name: "parallel".to_string(),
                description: "Enable parallel computation (default: true)".to_string(),
                param_type: ParameterType::Boolean,
                required: false,
                default: Some("true".to_string()),
            },
            ParameterDef {
                name: "parallel_threshold".to_string(),
                description: "Minimum nodes for parallel execution (default: 50)".to_string(),
                param_type: ParameterType::Integer,
                required: false,
                default: Some("50".to_string()),
            },
        ]
    })
}

/// Clustering Coefficient algorithm wrapper for the plugin registry.
pub struct ClusteringCoefficientAlgorithm;

impl GraphAlgorithm for ClusteringCoefficientAlgorithm {
    fn name(&self) -> &str {
        "clustering_coefficient"
    }

    fn description(&self) -> &str {
        "Local and global clustering coefficients with triangle counts"
    }

    fn parameters(&self) -> &[ParameterDef] {
        clustering_params()
    }

    fn execute(&self, store: &LpgStore, params: &Parameters) -> Result<AlgorithmResult> {
        #[cfg(feature = "parallel")]
        let result = {
            let parallel = params.get_bool("parallel").unwrap_or(true);
            let threshold = params.get_int("parallel_threshold").unwrap_or(50) as usize;

            if parallel {
                clustering_coefficient_parallel(store, threshold)
            } else {
                clustering_coefficient(store)
            }
        };

        #[cfg(not(feature = "parallel"))]
        let result = {
            let _ = params; // suppress unused warning
            clustering_coefficient(store)
        };

        let mut output = AlgorithmResult::new(vec![
            "node_id".to_string(),
            "clustering_coefficient".to_string(),
            "triangle_count".to_string(),
        ]);

        for (node, coefficient) in result.coefficients {
            let triangles = *result.triangle_counts.get(&node).unwrap_or(&0);
            output.add_row(vec![
                Value::Int64(node.0 as i64),
                Value::Float64(coefficient),
                Value::Int64(triangles as i64),
            ]);
        }

        Ok(output)
    }
}

#[cfg(feature = "parallel")]
impl ParallelGraphAlgorithm for ClusteringCoefficientAlgorithm {
    fn parallel_threshold(&self) -> usize {
        50
    }

    fn execute_parallel(
        &self,
        store: &LpgStore,
        _params: &Parameters,
        _num_threads: usize,
    ) -> Result<AlgorithmResult> {
        let result = clustering_coefficient_parallel(store, self.parallel_threshold());

        let mut output = AlgorithmResult::new(vec![
            "node_id".to_string(),
            "clustering_coefficient".to_string(),
            "triangle_count".to_string(),
        ]);

        for (node, coefficient) in result.coefficients {
            let triangles = *result.triangle_counts.get(&node).unwrap_or(&0);
            output.add_row(vec![
                Value::Int64(node.0 as i64),
                Value::Float64(coefficient),
                Value::Int64(triangles as i64),
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

    fn create_triangle_graph() -> LpgStore {
        // Simple triangle: 0 - 1 - 2 - 0
        let store = LpgStore::new();
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);

        // Bidirectional edges for undirected treatment
        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n1, n0, "EDGE");
        store.create_edge(n1, n2, "EDGE");
        store.create_edge(n2, n1, "EDGE");
        store.create_edge(n2, n0, "EDGE");
        store.create_edge(n0, n2, "EDGE");

        store
    }

    fn create_star_graph() -> LpgStore {
        // Star: center (0) connected to leaves (1, 2, 3, 4)
        // No triangles because leaves don't connect to each other
        let store = LpgStore::new();
        let center = store.create_node(&["Center"]);

        for _ in 0..4 {
            let leaf = store.create_node(&["Leaf"]);
            store.create_edge(center, leaf, "EDGE");
            store.create_edge(leaf, center, "EDGE");
        }

        store
    }

    fn create_complete_graph(n: usize) -> LpgStore {
        // K_n: complete graph with n nodes (all pairs connected)
        let store = LpgStore::new();
        let nodes: Vec<NodeId> = (0..n).map(|_| store.create_node(&["Node"])).collect();

        for i in 0..n {
            for j in (i + 1)..n {
                store.create_edge(nodes[i], nodes[j], "EDGE");
                store.create_edge(nodes[j], nodes[i], "EDGE");
            }
        }

        store
    }

    fn create_path_graph() -> LpgStore {
        // Path: 0 - 1 - 2 - 3
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
    fn test_triangle_graph_clustering() {
        let store = create_triangle_graph();
        let result = clustering_coefficient(&store);

        // All nodes in a triangle have coefficient 1.0
        for (_, coeff) in &result.coefficients {
            assert!(
                (*coeff - 1.0).abs() < 1e-10,
                "Expected coefficient 1.0, got {}",
                coeff
            );
        }

        // One unique triangle
        assert_eq!(result.total_triangles, 1);

        // Global coefficient should be 1.0
        assert!(
            (result.global_coefficient - 1.0).abs() < 1e-10,
            "Expected global 1.0, got {}",
            result.global_coefficient
        );
    }

    #[test]
    fn test_star_graph_clustering() {
        let store = create_star_graph();
        let result = clustering_coefficient(&store);

        // All coefficients should be 0 (no triangles in a star)
        for (_, coeff) in &result.coefficients {
            assert_eq!(*coeff, 0.0);
        }

        assert_eq!(result.total_triangles, 0);
        assert_eq!(result.global_coefficient, 0.0);
    }

    #[test]
    fn test_complete_graph_clustering() {
        let store = create_complete_graph(5);
        let result = clustering_coefficient(&store);

        // In a complete graph, all coefficients are 1.0
        for (_, coeff) in &result.coefficients {
            assert!((*coeff - 1.0).abs() < 1e-10, "Expected 1.0, got {}", coeff);
        }

        // K_5 has C(5,3) = 10 triangles
        assert_eq!(result.total_triangles, 10);
    }

    #[test]
    fn test_path_graph_clustering() {
        let store = create_path_graph();
        let result = clustering_coefficient(&store);

        // Path has no triangles
        assert_eq!(result.total_triangles, 0);

        // All coefficients should be 0 (endpoints have degree 1, middle have no triangles)
        for (_, coeff) in &result.coefficients {
            assert_eq!(*coeff, 0.0);
        }
    }

    #[test]
    fn test_empty_graph() {
        let store = LpgStore::new();
        let result = clustering_coefficient(&store);

        assert!(result.coefficients.is_empty());
        assert!(result.triangle_counts.is_empty());
        assert_eq!(result.total_triangles, 0);
        assert_eq!(result.global_coefficient, 0.0);
    }

    #[test]
    fn test_single_node() {
        let store = LpgStore::new();
        let n0 = store.create_node(&["Node"]);

        let result = clustering_coefficient(&store);

        assert_eq!(result.coefficients.len(), 1);
        assert_eq!(*result.coefficients.get(&n0).unwrap(), 0.0);
        assert_eq!(result.total_triangles, 0);
    }

    #[test]
    fn test_two_connected_nodes() {
        let store = LpgStore::new();
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n1, n0, "EDGE");

        let result = clustering_coefficient(&store);

        // Both nodes have degree 1, so coefficient is 0
        assert_eq!(*result.coefficients.get(&n0).unwrap(), 0.0);
        assert_eq!(*result.coefficients.get(&n1).unwrap(), 0.0);
        assert_eq!(result.total_triangles, 0);
    }

    #[test]
    fn test_triangle_count_function() {
        let store = create_triangle_graph();
        let counts = triangle_count(&store);

        // Each node in a triangle has 1 triangle
        for (_, count) in counts {
            assert_eq!(count, 1);
        }
    }

    #[test]
    fn test_local_clustering_coefficient_function() {
        let store = create_complete_graph(4);
        let coefficients = local_clustering_coefficient(&store);

        // K_4: all nodes have coefficient 1.0
        for (_, coeff) in coefficients {
            assert!((coeff - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_global_clustering_coefficient_function() {
        let store = create_triangle_graph();
        let global = global_clustering_coefficient(&store);
        assert!((global - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_total_triangles_function() {
        let store = create_complete_graph(4);
        let total = total_triangles(&store);
        // K_4 has C(4,3) = 4 triangles
        assert_eq!(total, 4);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_matches_sequential() {
        let store = create_complete_graph(20);

        let sequential = clustering_coefficient(&store);
        let parallel = clustering_coefficient_parallel(&store, 1); // Force parallel

        // Results should match
        for (node, seq_coeff) in &sequential.coefficients {
            let par_coeff = parallel.coefficients.get(node).unwrap();
            assert!(
                (seq_coeff - par_coeff).abs() < 1e-10,
                "Mismatch for node {:?}: seq={}, par={}",
                node,
                seq_coeff,
                par_coeff
            );
        }

        assert_eq!(sequential.total_triangles, parallel.total_triangles);
        assert!((sequential.global_coefficient - parallel.global_coefficient).abs() < 1e-10);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_threshold_fallback() {
        let store = create_triangle_graph();

        // With threshold higher than node count, should use sequential
        let result = clustering_coefficient_parallel(&store, 100);

        assert_eq!(result.coefficients.len(), 3);
        assert_eq!(result.total_triangles, 1);
    }

    #[test]
    fn test_algorithm_wrapper() {
        let store = create_triangle_graph();
        let algo = ClusteringCoefficientAlgorithm;

        assert_eq!(algo.name(), "clustering_coefficient");
        assert!(!algo.description().is_empty());
        assert_eq!(algo.parameters().len(), 2);

        let params = Parameters::new();
        let result = algo.execute(&store, &params).unwrap();

        assert_eq!(result.columns.len(), 3);
        assert_eq!(result.row_count(), 3);
    }

    #[test]
    fn test_algorithm_wrapper_sequential() {
        let store = create_triangle_graph();
        let algo = ClusteringCoefficientAlgorithm;

        let mut params = Parameters::new();
        params.set_bool("parallel", false);

        let result = algo.execute(&store, &params).unwrap();
        assert_eq!(result.row_count(), 3);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_algorithm_trait() {
        let store = create_complete_graph(10);
        let algo = ClusteringCoefficientAlgorithm;

        assert_eq!(algo.parallel_threshold(), 50);

        let params = Parameters::new();
        let result = algo.execute_parallel(&store, &params, 4).unwrap();

        assert_eq!(result.row_count(), 10);
    }

    #[test]
    fn test_two_triangles_sharing_edge() {
        // Two triangles sharing edge 0-1:
        //     2
        //    / \
        //   0---1
        //    \ /
        //     3
        let store = LpgStore::new();
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);
        let n3 = store.create_node(&["Node"]);

        // Triangle 0-1-2
        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n1, n0, "EDGE");
        store.create_edge(n1, n2, "EDGE");
        store.create_edge(n2, n1, "EDGE");
        store.create_edge(n2, n0, "EDGE");
        store.create_edge(n0, n2, "EDGE");

        // Triangle 0-1-3
        store.create_edge(n1, n3, "EDGE");
        store.create_edge(n3, n1, "EDGE");
        store.create_edge(n3, n0, "EDGE");
        store.create_edge(n0, n3, "EDGE");

        let result = clustering_coefficient(&store);

        // 2 unique triangles
        assert_eq!(result.total_triangles, 2);

        // Nodes 0 and 1 are in both triangles
        assert_eq!(*result.triangle_counts.get(&n0).unwrap(), 2);
        assert_eq!(*result.triangle_counts.get(&n1).unwrap(), 2);

        // Nodes 2 and 3 are in one triangle each
        assert_eq!(*result.triangle_counts.get(&n2).unwrap(), 1);
        assert_eq!(*result.triangle_counts.get(&n3).unwrap(), 1);

        // Node 0 has 3 neighbors (1, 2, 3), 2 triangles
        // max_triangles = 3*2/2 = 3, coefficient = 2/3
        let coeff_0 = *result.coefficients.get(&n0).unwrap();
        assert!(
            (coeff_0 - 2.0 / 3.0).abs() < 1e-10,
            "Expected 2/3, got {}",
            coeff_0
        );
    }
}
