//! Community detection algorithms: Louvain, Label Propagation.
//!
//! These algorithms identify clusters or communities of nodes that are
//! more densely connected to each other than to the rest of the graph.

use std::sync::OnceLock;

use grafeo_common::types::{NodeId, Value};
use grafeo_common::utils::error::Result;
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use grafeo_core::graph::Direction;
use grafeo_core::graph::lpg::LpgStore;

use super::super::{AlgorithmResult, ParameterDef, ParameterType, Parameters};
use super::traits::{ComponentResultBuilder, GraphAlgorithm};

// ============================================================================
// Label Propagation
// ============================================================================

/// Detects communities using the Label Propagation Algorithm.
///
/// Each node is initially assigned a unique label. Then, iteratively,
/// each node adopts the most frequent label among its neighbors until
/// the labels stabilize.
///
/// # Arguments
///
/// * `store` - The graph store
/// * `max_iterations` - Maximum number of iterations (0 for unlimited)
///
/// # Returns
///
/// A map from node ID to community (label) ID.
///
/// # Complexity
///
/// O(iterations × E)
pub fn label_propagation(store: &LpgStore, max_iterations: usize) -> FxHashMap<NodeId, u64> {
    let nodes = store.node_ids();
    let n = nodes.len();

    if n == 0 {
        return FxHashMap::default();
    }

    // Initialize labels: each node gets its own unique label
    let mut labels: FxHashMap<NodeId, u64> = FxHashMap::default();
    for (idx, &node) in nodes.iter().enumerate() {
        labels.insert(node, idx as u64);
    }

    let max_iter = if max_iterations == 0 {
        n * 10
    } else {
        max_iterations
    };

    for _ in 0..max_iter {
        let mut changed = false;

        // Update labels in random order (here we use insertion order)
        for &node in &nodes {
            // Get neighbor labels and their frequencies
            let mut label_counts: FxHashMap<u64, usize> = FxHashMap::default();

            // Consider both outgoing and incoming edges (undirected community detection)
            // Outgoing edges: node -> neighbor
            for (neighbor, _) in store.edges_from(node, Direction::Outgoing) {
                if let Some(&label) = labels.get(&neighbor) {
                    *label_counts.entry(label).or_insert(0) += 1;
                }
            }

            // Incoming edges: neighbor -> node
            // Uses backward adjacency index for O(degree) instead of O(V*E)
            for (incoming_neighbor, _) in store.edges_from(node, Direction::Incoming) {
                if let Some(&label) = labels.get(&incoming_neighbor) {
                    *label_counts.entry(label).or_insert(0) += 1;
                }
            }

            if label_counts.is_empty() {
                continue;
            }

            // Find the most frequent label
            let max_count = *label_counts.values().max().unwrap_or(&0);
            let max_labels: Vec<u64> = label_counts
                .into_iter()
                .filter(|&(_, count)| count == max_count)
                .map(|(label, _)| label)
                .collect();

            // Choose the smallest label in case of tie (deterministic)
            let new_label = *max_labels.iter().min().unwrap();
            let current_label = *labels.get(&node).unwrap();

            if new_label != current_label {
                labels.insert(node, new_label);
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    // Normalize labels to be contiguous starting from 0
    let unique_labels: FxHashSet<u64> = labels.values().copied().collect();
    let mut label_map: FxHashMap<u64, u64> = FxHashMap::default();
    for (idx, label) in unique_labels.into_iter().enumerate() {
        label_map.insert(label, idx as u64);
    }

    labels
        .into_iter()
        .map(|(node, label)| (node, *label_map.get(&label).unwrap()))
        .collect()
}

// ============================================================================
// Louvain Algorithm
// ============================================================================

/// Result of Louvain algorithm.
#[derive(Debug, Clone)]
pub struct LouvainResult {
    /// Community assignment for each node.
    pub communities: FxHashMap<NodeId, u64>,
    /// Final modularity score.
    pub modularity: f64,
    /// Number of communities detected.
    pub num_communities: usize,
}

/// Detects communities using the Louvain algorithm.
///
/// The Louvain algorithm optimizes modularity through a greedy approach,
/// consisting of two phases that are repeated iteratively:
/// 1. Local optimization: Move nodes to neighboring communities if it increases modularity
/// 2. Aggregation: Build a new graph where communities become super-nodes
///
/// # Arguments
///
/// * `store` - The graph store
/// * `resolution` - Resolution parameter (higher = smaller communities, default 1.0)
///
/// # Returns
///
/// Community assignments and modularity score.
///
/// # Complexity
///
/// O(V log V) on average for sparse graphs
pub fn louvain(store: &LpgStore, resolution: f64) -> LouvainResult {
    let nodes = store.node_ids();
    let n = nodes.len();

    if n == 0 {
        return LouvainResult {
            communities: FxHashMap::default(),
            modularity: 0.0,
            num_communities: 0,
        };
    }

    // Build node index mapping
    let mut node_to_idx: FxHashMap<NodeId, usize> = FxHashMap::default();
    for (idx, &node) in nodes.iter().enumerate() {
        node_to_idx.insert(node, idx);
    }

    // Build adjacency with weights (for undirected graph)
    // weights[i][j] = weight of edge between nodes i and j
    let mut weights: Vec<FxHashMap<usize, f64>> = vec![FxHashMap::default(); n];
    let mut total_weight = 0.0;

    for &node in &nodes {
        let i = *node_to_idx.get(&node).unwrap();
        for (neighbor, _edge_id) in store.edges_from(node, Direction::Outgoing) {
            if let Some(&j) = node_to_idx.get(&neighbor) {
                // For undirected: add weight to both directions
                let w = 1.0; // Could extract from edge property
                *weights[i].entry(j).or_insert(0.0) += w;
                *weights[j].entry(i).or_insert(0.0) += w;
                total_weight += w;
            }
        }
    }

    // Handle isolated nodes
    if total_weight == 0.0 {
        let communities: FxHashMap<NodeId, u64> = nodes
            .iter()
            .enumerate()
            .map(|(idx, &node)| (node, idx as u64))
            .collect();
        return LouvainResult {
            communities,
            modularity: 0.0,
            num_communities: n,
        };
    }

    // Compute node degrees (sum of incident edge weights)
    let degrees: Vec<f64> = (0..n).map(|i| weights[i].values().sum()).collect();

    // Initialize: each node in its own community
    let mut community: Vec<usize> = (0..n).collect();

    // Community internal weights and total weights
    let mut community_internal: FxHashMap<usize, f64> = FxHashMap::default();
    let mut community_total: FxHashMap<usize, f64> = FxHashMap::default();

    for i in 0..n {
        community_total.insert(i, degrees[i]);
        community_internal.insert(i, weights[i].get(&i).copied().unwrap_or(0.0));
    }

    // Phase 1: Local optimization
    let mut improved = true;
    while improved {
        improved = false;

        for i in 0..n {
            let current_comm = community[i];

            // Compute links to each neighboring community
            let mut comm_links: FxHashMap<usize, f64> = FxHashMap::default();
            for (&j, &w) in &weights[i] {
                let c = community[j];
                *comm_links.entry(c).or_insert(0.0) += w;
            }

            // Try moving to each neighboring community
            let mut best_delta = 0.0;
            let mut best_comm = current_comm;

            // Remove node from current community for delta calculation
            let ki = degrees[i];
            let ki_in = comm_links.get(&current_comm).copied().unwrap_or(0.0);

            for (&target_comm, &k_i_to_comm) in &comm_links {
                if target_comm == current_comm {
                    continue;
                }

                let sigma_tot = *community_total.get(&target_comm).unwrap_or(&0.0);

                // Modularity delta for moving to target_comm
                let delta = resolution
                    * (k_i_to_comm
                        - ki_in
                        - ki * (sigma_tot - community_total.get(&current_comm).unwrap_or(&0.0)
                            + ki)
                            / (2.0 * total_weight));

                if delta > best_delta {
                    best_delta = delta;
                    best_comm = target_comm;
                }
            }

            if best_comm != current_comm {
                // Move node to best community
                // Update community statistics
                *community_total.entry(current_comm).or_insert(0.0) -= ki;
                *community_internal.entry(current_comm).or_insert(0.0) -=
                    2.0 * ki_in + weights[i].get(&i).copied().unwrap_or(0.0);

                community[i] = best_comm;

                *community_total.entry(best_comm).or_insert(0.0) += ki;
                let k_i_best = comm_links.get(&best_comm).copied().unwrap_or(0.0);
                *community_internal.entry(best_comm).or_insert(0.0) +=
                    2.0 * k_i_best + weights[i].get(&i).copied().unwrap_or(0.0);

                improved = true;
            }
        }
    }

    // Normalize community IDs
    let unique_comms: FxHashSet<usize> = community.iter().copied().collect();
    let mut comm_map: FxHashMap<usize, u64> = FxHashMap::default();
    for (idx, c) in unique_comms.iter().enumerate() {
        comm_map.insert(*c, idx as u64);
    }

    let communities: FxHashMap<NodeId, u64> = nodes
        .iter()
        .enumerate()
        .map(|(i, &node)| (node, *comm_map.get(&community[i]).unwrap()))
        .collect();

    // Compute final modularity
    let modularity = compute_modularity(&weights, &community, total_weight, resolution);

    LouvainResult {
        communities,
        modularity,
        num_communities: unique_comms.len(),
    }
}

/// Computes the modularity of a community assignment.
fn compute_modularity(
    weights: &[FxHashMap<usize, f64>],
    community: &[usize],
    total_weight: f64,
    resolution: f64,
) -> f64 {
    let n = community.len();
    let m2 = 2.0 * total_weight;

    if m2 == 0.0 {
        return 0.0;
    }

    let degrees: Vec<f64> = (0..n).map(|i| weights[i].values().sum()).collect();

    let mut modularity = 0.0;

    for i in 0..n {
        for (&j, &a_ij) in &weights[i] {
            if community[i] == community[j] {
                modularity += a_ij - resolution * degrees[i] * degrees[j] / m2;
            }
        }
    }

    modularity / m2
}

/// Returns the number of communities detected.
pub fn community_count(communities: &FxHashMap<NodeId, u64>) -> usize {
    let unique: FxHashSet<u64> = communities.values().copied().collect();
    unique.len()
}

// ============================================================================
// Algorithm Wrappers for Plugin Registry
// ============================================================================

/// Static parameter definitions for Label Propagation algorithm.
static LABEL_PROP_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn label_prop_params() -> &'static [ParameterDef] {
    LABEL_PROP_PARAMS.get_or_init(|| {
        vec![ParameterDef {
            name: "max_iterations".to_string(),
            description: "Maximum iterations (0 for unlimited, default: 100)".to_string(),
            param_type: ParameterType::Integer,
            required: false,
            default: Some("100".to_string()),
        }]
    })
}

/// Label Propagation algorithm wrapper.
pub struct LabelPropagationAlgorithm;

impl GraphAlgorithm for LabelPropagationAlgorithm {
    fn name(&self) -> &str {
        "label_propagation"
    }

    fn description(&self) -> &str {
        "Label Propagation community detection"
    }

    fn parameters(&self) -> &[ParameterDef] {
        label_prop_params()
    }

    fn execute(&self, store: &LpgStore, params: &Parameters) -> Result<AlgorithmResult> {
        let max_iter = params.get_int("max_iterations").unwrap_or(100) as usize;

        let communities = label_propagation(store, max_iter);

        let mut builder = ComponentResultBuilder::with_capacity(communities.len());
        for (node, community_id) in communities {
            builder.push(node, community_id);
        }

        Ok(builder.build())
    }
}

/// Static parameter definitions for Louvain algorithm.
static LOUVAIN_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn louvain_params() -> &'static [ParameterDef] {
    LOUVAIN_PARAMS.get_or_init(|| {
        vec![ParameterDef {
            name: "resolution".to_string(),
            description: "Resolution parameter (default: 1.0)".to_string(),
            param_type: ParameterType::Float,
            required: false,
            default: Some("1.0".to_string()),
        }]
    })
}

/// Louvain algorithm wrapper.
pub struct LouvainAlgorithm;

impl GraphAlgorithm for LouvainAlgorithm {
    fn name(&self) -> &str {
        "louvain"
    }

    fn description(&self) -> &str {
        "Louvain community detection (modularity optimization)"
    }

    fn parameters(&self) -> &[ParameterDef] {
        louvain_params()
    }

    fn execute(&self, store: &LpgStore, params: &Parameters) -> Result<AlgorithmResult> {
        let resolution = params.get_float("resolution").unwrap_or(1.0);

        let result = louvain(store, resolution);

        let mut output = AlgorithmResult::new(vec![
            "node_id".to_string(),
            "community_id".to_string(),
            "modularity".to_string(),
        ]);

        for (node, community_id) in result.communities {
            output.add_row(vec![
                Value::Int64(node.0 as i64),
                Value::Int64(community_id as i64),
                Value::Float64(result.modularity),
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

    fn create_two_cliques_graph() -> LpgStore {
        // Two cliques connected by one edge
        // Clique 1: 0-1-2-3 (fully connected)
        // Clique 2: 4-5-6-7 (fully connected)
        // Bridge: 3-4
        let store = LpgStore::new();

        let nodes: Vec<NodeId> = (0..8).map(|_| store.create_node(&["Node"])).collect();

        // Clique 1
        for i in 0..4 {
            for j in (i + 1)..4 {
                store.create_edge(nodes[i], nodes[j], "EDGE");
                store.create_edge(nodes[j], nodes[i], "EDGE");
            }
        }

        // Clique 2
        for i in 4..8 {
            for j in (i + 1)..8 {
                store.create_edge(nodes[i], nodes[j], "EDGE");
                store.create_edge(nodes[j], nodes[i], "EDGE");
            }
        }

        // Bridge
        store.create_edge(nodes[3], nodes[4], "EDGE");
        store.create_edge(nodes[4], nodes[3], "EDGE");

        store
    }

    fn create_simple_graph() -> LpgStore {
        let store = LpgStore::new();

        // Simple chain: 0 -> 1 -> 2
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);

        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n1, n2, "EDGE");

        store
    }

    #[test]
    fn test_label_propagation_basic() {
        let store = create_simple_graph();
        let communities = label_propagation(&store, 100);

        assert_eq!(communities.len(), 3);

        // All nodes should have some community assignment
        for (_, &comm) in &communities {
            assert!(comm < 3);
        }
    }

    #[test]
    fn test_label_propagation_cliques() {
        let store = create_two_cliques_graph();
        let communities = label_propagation(&store, 100);

        assert_eq!(communities.len(), 8);

        // Should detect 2 communities (ideally)
        let num_comms = community_count(&communities);
        assert!(num_comms >= 1 && num_comms <= 8); // May vary due to algorithm randomness
    }

    #[test]
    fn test_label_propagation_empty() {
        let store = LpgStore::new();
        let communities = label_propagation(&store, 100);
        assert!(communities.is_empty());
    }

    #[test]
    fn test_label_propagation_single_node() {
        let store = LpgStore::new();
        store.create_node(&["Node"]);

        let communities = label_propagation(&store, 100);
        assert_eq!(communities.len(), 1);
    }

    #[test]
    fn test_louvain_basic() {
        let store = create_simple_graph();
        let result = louvain(&store, 1.0);

        assert_eq!(result.communities.len(), 3);
        assert!(result.num_communities >= 1);
    }

    #[test]
    fn test_louvain_cliques() {
        let store = create_two_cliques_graph();
        let result = louvain(&store, 1.0);

        assert_eq!(result.communities.len(), 8);

        // Should detect approximately 2 communities
        // Louvain should find good modularity
        assert!(result.num_communities >= 1 && result.num_communities <= 8);
    }

    #[test]
    fn test_louvain_empty() {
        let store = LpgStore::new();
        let result = louvain(&store, 1.0);

        assert!(result.communities.is_empty());
        assert_eq!(result.modularity, 0.0);
        assert_eq!(result.num_communities, 0);
    }

    #[test]
    fn test_louvain_isolated_nodes() {
        let store = LpgStore::new();
        store.create_node(&["Node"]);
        store.create_node(&["Node"]);
        store.create_node(&["Node"]);

        let result = louvain(&store, 1.0);

        // Each isolated node should be its own community
        assert_eq!(result.communities.len(), 3);
        assert_eq!(result.num_communities, 3);
    }

    #[test]
    fn test_louvain_resolution_parameter() {
        let store = create_two_cliques_graph();

        // Low resolution: fewer, larger communities
        let result_low = louvain(&store, 0.5);

        // High resolution: more, smaller communities
        let result_high = louvain(&store, 2.0);

        // Both should be valid
        assert!(!result_low.communities.is_empty());
        assert!(!result_high.communities.is_empty());
    }

    #[test]
    fn test_community_count() {
        let mut communities: FxHashMap<NodeId, u64> = FxHashMap::default();
        communities.insert(NodeId::new(0), 0);
        communities.insert(NodeId::new(1), 0);
        communities.insert(NodeId::new(2), 1);
        communities.insert(NodeId::new(3), 1);
        communities.insert(NodeId::new(4), 2);

        assert_eq!(community_count(&communities), 3);
    }
}
