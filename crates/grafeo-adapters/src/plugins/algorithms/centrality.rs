//! Centrality algorithms: PageRank, Betweenness, Closeness, Degree.
//!
//! These algorithms measure node importance in a graph based on various
//! criteria including connectivity, path involvement, and link structure.

use std::collections::VecDeque;
use std::sync::OnceLock;

use grafeo_common::types::{NodeId, Value};
use grafeo_common::utils::error::Result;
use grafeo_common::utils::hash::FxHashMap;
use grafeo_core::graph::Direction;
use grafeo_core::graph::lpg::LpgStore;

use super::super::{AlgorithmResult, ParameterDef, ParameterType, Parameters};
use super::traits::{GraphAlgorithm, NodeValueResultBuilder};

// ============================================================================
// Degree Centrality
// ============================================================================

/// Result of degree centrality computation.
#[derive(Debug, Clone)]
pub struct DegreeCentralityResult {
    /// In-degree for each node.
    pub in_degree: FxHashMap<NodeId, usize>,
    /// Out-degree for each node.
    pub out_degree: FxHashMap<NodeId, usize>,
    /// Total degree (in + out) for each node.
    pub total_degree: FxHashMap<NodeId, usize>,
}

/// Computes degree centrality for all nodes.
///
/// Degree centrality is the simplest centrality measure, counting the
/// number of edges connected to each node.
///
/// # Arguments
///
/// * `store` - The graph store
///
/// # Returns
///
/// In-degree, out-degree, and total degree for each node.
///
/// # Complexity
///
/// O(V + E)
pub fn degree_centrality(store: &LpgStore) -> DegreeCentralityResult {
    let mut in_degree: FxHashMap<NodeId, usize> = FxHashMap::default();
    let mut out_degree: FxHashMap<NodeId, usize> = FxHashMap::default();

    let nodes = store.node_ids();

    // Initialize all nodes
    for &node in &nodes {
        in_degree.insert(node, 0);
        out_degree.insert(node, 0);
    }

    // Count degrees
    for &node in &nodes {
        let out_count = store.edges_from(node, Direction::Outgoing).count();
        out_degree.insert(node, out_count);

        // For incoming edges, we count edges targeting this node
        for (neighbor, _) in store.edges_from(node, Direction::Outgoing) {
            *in_degree.entry(neighbor).or_insert(0) += 1;
        }
    }

    // Compute total degree
    let total_degree: FxHashMap<NodeId, usize> = nodes
        .iter()
        .map(|&n| {
            let in_d = *in_degree.get(&n).unwrap_or(&0);
            let out_d = *out_degree.get(&n).unwrap_or(&0);
            (n, in_d + out_d)
        })
        .collect();

    DegreeCentralityResult {
        in_degree,
        out_degree,
        total_degree,
    }
}

/// Computes normalized degree centrality.
///
/// Normalizes by dividing by (n-1) where n is the node count.
pub fn degree_centrality_normalized(store: &LpgStore) -> FxHashMap<NodeId, f64> {
    let result = degree_centrality(store);
    let n = result.total_degree.len();

    if n <= 1 {
        return result
            .total_degree
            .into_iter()
            .map(|(k, _)| (k, 0.0))
            .collect();
    }

    let norm = (n - 1) as f64;
    result
        .total_degree
        .into_iter()
        .map(|(k, v)| (k, v as f64 / norm))
        .collect()
}

// ============================================================================
// PageRank
// ============================================================================

/// Computes PageRank for all nodes using power iteration.
///
/// PageRank measures node importance based on the link structure,
/// where a node is important if it's linked to by other important nodes.
///
/// # Arguments
///
/// * `store` - The graph store
/// * `damping` - Damping factor (typically 0.85)
/// * `max_iterations` - Maximum number of iterations
/// * `tolerance` - Convergence tolerance (stop when change < tolerance)
///
/// # Returns
///
/// PageRank score for each node.
///
/// # Complexity
///
/// O(iterations × (V + E))
pub fn pagerank(
    store: &LpgStore,
    damping: f64,
    max_iterations: usize,
    tolerance: f64,
) -> FxHashMap<NodeId, f64> {
    let nodes = store.node_ids();
    let n = nodes.len();

    if n == 0 {
        return FxHashMap::default();
    }

    // Build node index mapping for efficient access
    let mut node_to_idx: FxHashMap<NodeId, usize> = FxHashMap::default();
    for (idx, &node) in nodes.iter().enumerate() {
        node_to_idx.insert(node, idx);
    }

    // Build adjacency structure
    let mut out_edges: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut out_degree: Vec<usize> = vec![0; n];

    for (idx, &node) in nodes.iter().enumerate() {
        let edges: Vec<usize> = store
            .edges_from(node, Direction::Outgoing)
            .filter_map(|(neighbor, _)| node_to_idx.get(&neighbor).copied())
            .collect();
        out_degree[idx] = edges.len();
        out_edges[idx] = edges;
    }

    // Initialize PageRank scores
    let initial_score = 1.0 / n as f64;
    let mut scores = vec![initial_score; n];
    let mut new_scores = vec![0.0; n];

    // Identify dangling nodes (no outgoing edges)
    let dangling: Vec<usize> = (0..n).filter(|&i| out_degree[i] == 0).collect();

    // Power iteration
    for _ in 0..max_iterations {
        // Compute dangling contribution
        let dangling_sum: f64 = dangling.iter().map(|&i| scores[i]).sum();
        let dangling_contrib = damping * dangling_sum / n as f64;

        // Reset new scores with teleportation and dangling contribution
        let teleport = (1.0 - damping) / n as f64;
        for score in &mut new_scores {
            *score = teleport + dangling_contrib;
        }

        // Add contributions from incoming edges
        for (i, edges) in out_edges.iter().enumerate() {
            if !edges.is_empty() {
                let contrib = damping * scores[i] / edges.len() as f64;
                for &j in edges {
                    new_scores[j] += contrib;
                }
            }
        }

        // Check convergence
        let max_diff: f64 = scores
            .iter()
            .zip(new_scores.iter())
            .map(|(old, new)| (old - new).abs())
            .fold(0.0, f64::max);

        std::mem::swap(&mut scores, &mut new_scores);

        if max_diff < tolerance {
            break;
        }
    }

    // Convert back to NodeId map
    nodes
        .into_iter()
        .enumerate()
        .map(|(idx, node)| (node, scores[idx]))
        .collect()
}

// ============================================================================
// Betweenness Centrality (Brandes' Algorithm)
// ============================================================================

/// Computes betweenness centrality using Brandes' algorithm.
///
/// Betweenness centrality measures how often a node lies on shortest
/// paths between other nodes.
///
/// # Arguments
///
/// * `store` - The graph store
/// * `normalized` - Whether to normalize by 2/((n-1)(n-2)) for directed graphs
///
/// # Returns
///
/// Betweenness centrality score for each node.
///
/// # Complexity
///
/// O(V × E) for unweighted graphs
pub fn betweenness_centrality(store: &LpgStore, normalized: bool) -> FxHashMap<NodeId, f64> {
    let nodes = store.node_ids();
    let n = nodes.len();

    let mut centrality: FxHashMap<NodeId, f64> = FxHashMap::default();
    for &node in &nodes {
        centrality.insert(node, 0.0);
    }

    if n <= 2 {
        return centrality;
    }

    // Brandes' algorithm: run BFS from each source
    for &source in &nodes {
        // BFS data structures
        let mut stack: Vec<NodeId> = Vec::new();
        let mut predecessors: FxHashMap<NodeId, Vec<NodeId>> = FxHashMap::default();
        let mut sigma: FxHashMap<NodeId, f64> = FxHashMap::default(); // Number of shortest paths
        let mut dist: FxHashMap<NodeId, i64> = FxHashMap::default();

        // Initialize
        for &node in &nodes {
            predecessors.insert(node, Vec::new());
            sigma.insert(node, 0.0);
            dist.insert(node, -1);
        }
        sigma.insert(source, 1.0);
        dist.insert(source, 0);

        // BFS
        let mut queue: VecDeque<NodeId> = VecDeque::new();
        queue.push_back(source);

        while let Some(v) = queue.pop_front() {
            stack.push(v);
            let dist_v = *dist.get(&v).unwrap();

            for (w, _) in store.edges_from(v, Direction::Outgoing) {
                // First visit?
                if *dist.get(&w).unwrap() < 0 {
                    dist.insert(w, dist_v + 1);
                    queue.push_back(w);
                }

                // Shortest path to w via v?
                if *dist.get(&w).unwrap() == dist_v + 1 {
                    let sigma_v = *sigma.get(&v).unwrap();
                    *sigma.entry(w).or_insert(0.0) += sigma_v;
                    predecessors.entry(w).or_default().push(v);
                }
            }
        }

        // Accumulation
        let mut delta: FxHashMap<NodeId, f64> = FxHashMap::default();
        for &node in &nodes {
            delta.insert(node, 0.0);
        }

        while let Some(w) = stack.pop() {
            if w == source {
                continue;
            }

            let sigma_w = *sigma.get(&w).unwrap();
            let delta_w = *delta.get(&w).unwrap();

            for v in predecessors.get(&w).unwrap_or(&Vec::new()) {
                let sigma_v = *sigma.get(v).unwrap();
                let coeff = (sigma_v / sigma_w) * (1.0 + delta_w);
                *delta.entry(*v).or_insert(0.0) += coeff;
            }

            *centrality.entry(w).or_insert(0.0) += delta_w;
        }
    }

    // Normalize if requested
    if normalized && n > 2 {
        let norm = 2.0 / ((n - 1) * (n - 2)) as f64;
        for (_, v) in &mut centrality {
            *v *= norm;
        }
    }

    centrality
}

// ============================================================================
// Closeness Centrality
// ============================================================================

/// Computes closeness centrality for all nodes.
///
/// Closeness centrality is the reciprocal of the average shortest path
/// distance from a node to all other reachable nodes.
///
/// # Arguments
///
/// * `store` - The graph store
/// * `wf_improved` - Use Wasserman-Faust improved formula for disconnected graphs
///
/// # Returns
///
/// Closeness centrality score for each node.
///
/// # Complexity
///
/// O(V × (V + E))
pub fn closeness_centrality(store: &LpgStore, wf_improved: bool) -> FxHashMap<NodeId, f64> {
    let nodes = store.node_ids();
    let n = nodes.len();

    let mut centrality: FxHashMap<NodeId, f64> = FxHashMap::default();

    if n <= 1 {
        for &node in &nodes {
            centrality.insert(node, 0.0);
        }
        return centrality;
    }

    for &source in &nodes {
        // BFS to find shortest paths
        let mut dist: FxHashMap<NodeId, usize> = FxHashMap::default();
        let mut queue: VecDeque<NodeId> = VecDeque::new();

        dist.insert(source, 0);
        queue.push_back(source);

        while let Some(v) = queue.pop_front() {
            let dist_v = *dist.get(&v).unwrap();

            for (w, _) in store.edges_from(v, Direction::Outgoing) {
                if !dist.contains_key(&w) {
                    dist.insert(w, dist_v + 1);
                    queue.push_back(w);
                }
            }
        }

        // Compute closeness
        let reachable = dist.len() - 1; // Exclude source
        let total_dist: usize = dist.values().sum();

        let closeness = if total_dist > 0 && reachable > 0 {
            if wf_improved {
                // Wasserman-Faust: (reachable/(n-1)) * (reachable/total_dist)
                let reachable_f = reachable as f64;
                let n_minus_1 = (n - 1) as f64;
                (reachable_f / n_minus_1) * (reachable_f / total_dist as f64)
            } else {
                // Standard: reachable / total_dist
                reachable as f64 / total_dist as f64
            }
        } else {
            0.0
        };

        centrality.insert(source, closeness);
    }

    centrality
}

// ============================================================================
// Algorithm Wrappers for Plugin Registry
// ============================================================================

/// Static parameter definitions for PageRank algorithm.
static PAGERANK_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn pagerank_params() -> &'static [ParameterDef] {
    PAGERANK_PARAMS.get_or_init(|| {
        vec![
            ParameterDef {
                name: "damping".to_string(),
                description: "Damping factor (default: 0.85)".to_string(),
                param_type: ParameterType::Float,
                required: false,
                default: Some("0.85".to_string()),
            },
            ParameterDef {
                name: "max_iterations".to_string(),
                description: "Maximum iterations (default: 100)".to_string(),
                param_type: ParameterType::Integer,
                required: false,
                default: Some("100".to_string()),
            },
            ParameterDef {
                name: "tolerance".to_string(),
                description: "Convergence tolerance (default: 1e-6)".to_string(),
                param_type: ParameterType::Float,
                required: false,
                default: Some("1e-6".to_string()),
            },
        ]
    })
}

/// PageRank algorithm wrapper for the plugin registry.
pub struct PageRankAlgorithm;

impl GraphAlgorithm for PageRankAlgorithm {
    fn name(&self) -> &str {
        "pagerank"
    }

    fn description(&self) -> &str {
        "PageRank algorithm for measuring node importance"
    }

    fn parameters(&self) -> &[ParameterDef] {
        pagerank_params()
    }

    fn execute(&self, store: &LpgStore, params: &Parameters) -> Result<AlgorithmResult> {
        let damping = params.get_float("damping").unwrap_or(0.85);
        let max_iter = params.get_int("max_iterations").unwrap_or(100) as usize;
        let tolerance = params.get_float("tolerance").unwrap_or(1e-6);

        let scores = pagerank(store, damping, max_iter, tolerance);

        let mut builder = NodeValueResultBuilder::with_capacity("pagerank", scores.len());
        for (node, score) in scores {
            builder.push(node, Value::Float64(score));
        }

        Ok(builder.build())
    }
}

/// Static parameter definitions for Betweenness Centrality algorithm.
static BETWEENNESS_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn betweenness_params() -> &'static [ParameterDef] {
    BETWEENNESS_PARAMS.get_or_init(|| {
        vec![ParameterDef {
            name: "normalized".to_string(),
            description: "Normalize scores (default: true)".to_string(),
            param_type: ParameterType::Boolean,
            required: false,
            default: Some("true".to_string()),
        }]
    })
}

/// Betweenness centrality algorithm wrapper.
pub struct BetweennessCentralityAlgorithm;

impl GraphAlgorithm for BetweennessCentralityAlgorithm {
    fn name(&self) -> &str {
        "betweenness_centrality"
    }

    fn description(&self) -> &str {
        "Betweenness centrality using Brandes' algorithm"
    }

    fn parameters(&self) -> &[ParameterDef] {
        betweenness_params()
    }

    fn execute(&self, store: &LpgStore, params: &Parameters) -> Result<AlgorithmResult> {
        let normalized = params.get_bool("normalized").unwrap_or(true);

        let scores = betweenness_centrality(store, normalized);

        let mut builder = NodeValueResultBuilder::with_capacity("betweenness", scores.len());
        for (node, score) in scores {
            builder.push(node, Value::Float64(score));
        }

        Ok(builder.build())
    }
}

/// Static parameter definitions for Closeness Centrality algorithm.
static CLOSENESS_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn closeness_params() -> &'static [ParameterDef] {
    CLOSENESS_PARAMS.get_or_init(|| {
        vec![ParameterDef {
            name: "wf_improved".to_string(),
            description: "Use Wasserman-Faust formula for disconnected graphs (default: false)"
                .to_string(),
            param_type: ParameterType::Boolean,
            required: false,
            default: Some("false".to_string()),
        }]
    })
}

/// Closeness centrality algorithm wrapper.
pub struct ClosenessCentralityAlgorithm;

impl GraphAlgorithm for ClosenessCentralityAlgorithm {
    fn name(&self) -> &str {
        "closeness_centrality"
    }

    fn description(&self) -> &str {
        "Closeness centrality based on shortest path distances"
    }

    fn parameters(&self) -> &[ParameterDef] {
        closeness_params()
    }

    fn execute(&self, store: &LpgStore, params: &Parameters) -> Result<AlgorithmResult> {
        let wf_improved = params.get_bool("wf_improved").unwrap_or(false);

        let scores = closeness_centrality(store, wf_improved);

        let mut builder = NodeValueResultBuilder::with_capacity("closeness", scores.len());
        for (node, score) in scores {
            builder.push(node, Value::Float64(score));
        }

        Ok(builder.build())
    }
}

/// Static parameter definitions for Degree Centrality algorithm.
static DEGREE_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn degree_params() -> &'static [ParameterDef] {
    DEGREE_PARAMS.get_or_init(|| {
        vec![ParameterDef {
            name: "normalized".to_string(),
            description: "Normalize by (n-1) (default: false)".to_string(),
            param_type: ParameterType::Boolean,
            required: false,
            default: Some("false".to_string()),
        }]
    })
}

/// Degree centrality algorithm wrapper.
pub struct DegreeCentralityAlgorithm;

impl GraphAlgorithm for DegreeCentralityAlgorithm {
    fn name(&self) -> &str {
        "degree_centrality"
    }

    fn description(&self) -> &str {
        "Degree centrality (node connectivity measure)"
    }

    fn parameters(&self) -> &[ParameterDef] {
        degree_params()
    }

    fn execute(&self, store: &LpgStore, params: &Parameters) -> Result<AlgorithmResult> {
        let normalized = params.get_bool("normalized").unwrap_or(false);

        if normalized {
            let scores = degree_centrality_normalized(store);

            let mut builder =
                NodeValueResultBuilder::with_capacity("degree_centrality", scores.len());
            for (node, score) in scores {
                builder.push(node, Value::Float64(score));
            }
            Ok(builder.build())
        } else {
            let result = degree_centrality(store);

            let mut output = AlgorithmResult::new(vec![
                "node_id".to_string(),
                "in_degree".to_string(),
                "out_degree".to_string(),
                "total_degree".to_string(),
            ]);

            for (&node, &total) in &result.total_degree {
                let in_d = *result.in_degree.get(&node).unwrap_or(&0);
                let out_d = *result.out_degree.get(&node).unwrap_or(&0);

                output.add_row(vec![
                    Value::Int64(node.0 as i64),
                    Value::Int64(in_d as i64),
                    Value::Int64(out_d as i64),
                    Value::Int64(total as i64),
                ]);
            }

            Ok(output)
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

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

    fn create_pagerank_graph() -> LpgStore {
        // Simple graph for PageRank testing
        // A -> B -> C
        // A -> C
        let store = LpgStore::new();

        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);

        store.create_edge(a, b, "LINK");
        store.create_edge(b, c, "LINK");
        store.create_edge(a, c, "LINK");

        store
    }

    #[test]
    fn test_degree_centrality() {
        let store = create_test_graph();
        let result = degree_centrality(&store);

        // Node 0 has 2 outgoing edges
        assert_eq!(*result.out_degree.get(&NodeId::new(0)).unwrap(), 2);

        // Node 4 has 0 outgoing edges but receives from 1 and 3
        assert_eq!(*result.out_degree.get(&NodeId::new(4)).unwrap(), 0);
        assert_eq!(*result.in_degree.get(&NodeId::new(4)).unwrap(), 2);
    }

    #[test]
    fn test_degree_centrality_normalized() {
        let store = create_test_graph();
        let result = degree_centrality_normalized(&store);

        // All normalized values should be between 0 and 1
        for (_, &score) in &result {
            assert!((0.0..=1.0).contains(&score));
        }
    }

    #[test]
    fn test_pagerank_basic() {
        let store = create_pagerank_graph();
        let scores = pagerank(&store, 0.85, 100, 1e-6);

        assert_eq!(scores.len(), 3);

        // All scores should be positive
        for (_, &score) in &scores {
            assert!(score > 0.0);
        }

        // Scores should sum to approximately 1
        let total: f64 = scores.values().sum();
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_pagerank_dangling() {
        // Graph with dangling node (no outgoing edges)
        let store = LpgStore::new();
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        store.create_edge(a, b, "EDGE");
        // b is dangling

        let scores = pagerank(&store, 0.85, 100, 1e-6);
        assert_eq!(scores.len(), 2);

        // Dangling node should still have positive PageRank
        assert!(*scores.get(&b).unwrap() > 0.0);
    }

    #[test]
    fn test_pagerank_empty() {
        let store = LpgStore::new();
        let scores = pagerank(&store, 0.85, 100, 1e-6);
        assert!(scores.is_empty());
    }

    #[test]
    fn test_betweenness_centrality() {
        let store = create_test_graph();
        let scores = betweenness_centrality(&store, false);

        assert_eq!(scores.len(), 5);

        // All scores should be non-negative
        for (_, &score) in &scores {
            assert!(score >= 0.0);
        }
    }

    #[test]
    fn test_betweenness_centrality_normalized() {
        let store = create_test_graph();
        let scores = betweenness_centrality(&store, true);

        // Normalized scores should be between 0 and 1
        for (_, &score) in &scores {
            assert!(score >= 0.0);
        }
    }

    #[test]
    fn test_closeness_centrality() {
        let store = create_test_graph();
        let scores = closeness_centrality(&store, false);

        assert_eq!(scores.len(), 5);

        // All scores should be non-negative
        for (_, &score) in &scores {
            assert!(score >= 0.0);
        }

        // Source node (0) should have positive closeness (can reach others)
        assert!(*scores.get(&NodeId::new(0)).unwrap() > 0.0);
    }

    #[test]
    fn test_closeness_wf_improved() {
        let store = create_test_graph();
        let scores_standard = closeness_centrality(&store, false);
        let scores_wf = closeness_centrality(&store, true);

        // WF improved scores may differ but should still be valid
        for (node, &wf_score) in &scores_wf {
            assert!(wf_score >= 0.0);
            // WF formula typically gives different but related values
            let _std_score = scores_standard.get(node).unwrap();
        }
    }

    #[test]
    fn test_closeness_disconnected() {
        // Graph with isolated node
        let store = LpgStore::new();
        let a = store.create_node(&["Node"]);
        let _b = store.create_node(&["Node"]); // Isolated

        let scores = closeness_centrality(&store, false);

        // Isolated node has 0 closeness
        assert_eq!(*scores.get(&a).unwrap(), 0.0);
    }

    #[test]
    fn test_single_node() {
        let store = LpgStore::new();
        store.create_node(&["Node"]);

        let degree = degree_centrality(&store);
        assert_eq!(degree.total_degree.len(), 1);

        let pr = pagerank(&store, 0.85, 100, 1e-6);
        assert_eq!(pr.len(), 1);

        let bc = betweenness_centrality(&store, false);
        assert_eq!(bc.len(), 1);

        let cc = closeness_centrality(&store, false);
        assert_eq!(cc.len(), 1);
    }

    // ========================================================================
    // Algorithm Wrapper Tests (for coverage of GraphAlgorithm trait impls)
    // ========================================================================

    #[test]
    fn test_pagerank_algorithm_wrapper() {
        let store = create_pagerank_graph();
        let algo = PageRankAlgorithm;

        // Test trait methods
        assert_eq!(algo.name(), "pagerank");
        assert!(!algo.description().is_empty());
        assert_eq!(algo.parameters().len(), 3);

        // Test execute with default params
        let params = Parameters::new();
        let result = algo.execute(&store, &params).unwrap();
        assert_eq!(result.columns.len(), 2); // node_id, pagerank
        assert_eq!(result.row_count(), 3);

        // Test execute with custom params
        let mut params = Parameters::new();
        params.set_float("damping", 0.9);
        params.set_int("max_iterations", 50);
        params.set_float("tolerance", 1e-4);
        let result = algo.execute(&store, &params).unwrap();
        assert_eq!(result.row_count(), 3);
    }

    #[test]
    fn test_betweenness_algorithm_wrapper() {
        let store = create_test_graph();
        let algo = BetweennessCentralityAlgorithm;

        // Test trait methods
        assert_eq!(algo.name(), "betweenness_centrality");
        assert!(!algo.description().is_empty());
        assert_eq!(algo.parameters().len(), 1);

        // Test execute with default params (normalized=true)
        let params = Parameters::new();
        let result = algo.execute(&store, &params).unwrap();
        assert_eq!(result.columns.len(), 2); // node_id, betweenness
        assert_eq!(result.row_count(), 5);

        // Test execute with normalized=false
        let mut params = Parameters::new();
        params.set_bool("normalized", false);
        let result = algo.execute(&store, &params).unwrap();
        assert_eq!(result.row_count(), 5);
    }

    #[test]
    fn test_closeness_algorithm_wrapper() {
        let store = create_test_graph();
        let algo = ClosenessCentralityAlgorithm;

        // Test trait methods
        assert_eq!(algo.name(), "closeness_centrality");
        assert!(!algo.description().is_empty());
        assert_eq!(algo.parameters().len(), 1);

        // Test execute with default params (wf_improved=false)
        let params = Parameters::new();
        let result = algo.execute(&store, &params).unwrap();
        assert_eq!(result.columns.len(), 2); // node_id, closeness
        assert_eq!(result.row_count(), 5);

        // Test execute with wf_improved=true
        let mut params = Parameters::new();
        params.set_bool("wf_improved", true);
        let result = algo.execute(&store, &params).unwrap();
        assert_eq!(result.row_count(), 5);
    }

    #[test]
    fn test_degree_algorithm_wrapper() {
        let store = create_test_graph();
        let algo = DegreeCentralityAlgorithm;

        // Test trait methods
        assert_eq!(algo.name(), "degree_centrality");
        assert!(!algo.description().is_empty());
        assert_eq!(algo.parameters().len(), 1);

        // Test execute with default params (normalized=false) - returns 4 columns
        let params = Parameters::new();
        let result = algo.execute(&store, &params).unwrap();
        assert_eq!(result.columns.len(), 4); // node_id, in_degree, out_degree, total_degree
        assert_eq!(result.row_count(), 5);

        // Test execute with normalized=true - returns 2 columns
        let mut params = Parameters::new();
        params.set_bool("normalized", true);
        let result = algo.execute(&store, &params).unwrap();
        assert_eq!(result.columns.len(), 2); // node_id, degree_centrality
        assert_eq!(result.row_count(), 5);
    }

    #[test]
    fn test_betweenness_small_graph() {
        // Test n <= 2 edge case
        let store = LpgStore::new();
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        store.create_edge(a, b, "EDGE");

        let scores = betweenness_centrality(&store, false);
        assert_eq!(scores.len(), 2);
        // With only 2 nodes, betweenness is 0 for both
        assert_eq!(*scores.get(&a).unwrap(), 0.0);
        assert_eq!(*scores.get(&b).unwrap(), 0.0);

        let scores_norm = betweenness_centrality(&store, true);
        assert_eq!(scores_norm.len(), 2);
    }

    #[test]
    fn test_closeness_empty_graph() {
        let store = LpgStore::new();
        let scores = closeness_centrality(&store, false);
        assert!(scores.is_empty());

        let scores_wf = closeness_centrality(&store, true);
        assert!(scores_wf.is_empty());
    }

    #[test]
    fn test_closeness_single_node() {
        let store = LpgStore::new();
        let a = store.create_node(&["Node"]);

        let scores = closeness_centrality(&store, false);
        assert_eq!(scores.len(), 1);
        assert_eq!(*scores.get(&a).unwrap(), 0.0);

        let scores_wf = closeness_centrality(&store, true);
        assert_eq!(scores_wf.len(), 1);
        assert_eq!(*scores_wf.get(&a).unwrap(), 0.0);
    }

    #[test]
    fn test_degree_empty_graph() {
        let store = LpgStore::new();
        let result = degree_centrality(&store);
        assert!(result.total_degree.is_empty());

        let normalized = degree_centrality_normalized(&store);
        assert!(normalized.is_empty());
    }

    #[test]
    fn test_pagerank_convergence() {
        // Test that PageRank converges with tight tolerance
        let store = create_pagerank_graph();
        let scores_tight = pagerank(&store, 0.85, 1000, 1e-10);
        let scores_loose = pagerank(&store, 0.85, 1000, 1e-2);

        // Both should produce valid results
        assert_eq!(scores_tight.len(), 3);
        assert_eq!(scores_loose.len(), 3);

        // Tight tolerance should sum closer to 1.0
        let sum_tight: f64 = scores_tight.values().sum();
        let sum_loose: f64 = scores_loose.values().sum();
        assert!((sum_tight - 1.0).abs() <= (sum_loose - 1.0).abs() + 0.001);
    }

    #[test]
    fn test_pagerank_low_damping() {
        // Test with low damping (more teleportation)
        let store = create_pagerank_graph();
        let scores_low = pagerank(&store, 0.5, 100, 1e-6);
        let scores_high = pagerank(&store, 0.99, 100, 1e-6);

        // Both should be valid
        assert_eq!(scores_low.len(), 3);
        assert_eq!(scores_high.len(), 3);

        // Low damping should give more uniform distribution
        let variance_low: f64 = {
            let mean = scores_low.values().sum::<f64>() / 3.0;
            scores_low
                .values()
                .map(|&v| (v - mean).powi(2))
                .sum::<f64>()
                / 3.0
        };
        let variance_high: f64 = {
            let mean = scores_high.values().sum::<f64>() / 3.0;
            scores_high
                .values()
                .map(|&v| (v - mean).powi(2))
                .sum::<f64>()
                / 3.0
        };
        assert!(variance_low <= variance_high + 0.01);
    }

    #[test]
    fn test_betweenness_linear_graph() {
        // Linear graph: A -> B -> C -> D
        // B and C should have highest betweenness
        let store = LpgStore::new();
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let d = store.create_node(&["Node"]);

        store.create_edge(a, b, "EDGE");
        store.create_edge(b, c, "EDGE");
        store.create_edge(c, d, "EDGE");

        let scores = betweenness_centrality(&store, false);

        // Middle nodes should have higher betweenness than endpoints
        let bc_b = *scores.get(&b).unwrap();
        let bc_c = *scores.get(&c).unwrap();
        let bc_a = *scores.get(&a).unwrap();
        let bc_d = *scores.get(&d).unwrap();

        assert!(bc_b >= bc_a);
        assert!(bc_c >= bc_d);
    }
}
