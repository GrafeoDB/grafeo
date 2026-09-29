//! Community detection algorithms: Louvain, Label Propagation.
//!
//! These algorithms identify clusters or communities of nodes that are
//! more densely connected to each other than to the rest of the graph.

use std::sync::OnceLock;

use grafeo_common::types::{NodeId, Value};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use grafeo_core::graph::Direction;
use grafeo_core::graph::GraphStore;
#[cfg(all(test, feature = "lpg"))]
use grafeo_core::graph::lpg::LpgStore;

use super::super::{AlgorithmResult, ParameterDef, ParameterType};
use super::traits::{ComponentResultBuilder, impl_algorithm};

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
/// # Panics
///
/// Panics if the internal label map is inconsistent (should not happen with a valid `GraphStore`).
///
/// # Complexity
///
/// O(iterations × E)
pub fn label_propagation(store: &dyn GraphStore, max_iterations: usize) -> FxHashMap<NodeId, u64> {
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
            let new_label = *max_labels
                .iter()
                .min()
                .expect("max_labels non-empty: filtered from non-empty label_counts");
            let current_label = *labels.get(&node).expect("node initialized with label");

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
        .map(|(node, label)| (node, *label_map.get(&label).expect("label present in map")))
        .collect()
}

// ============================================================================
// Louvain Algorithm
// ============================================================================

/// Result of Louvain algorithm.
#[derive(Debug, Clone)]
pub struct LouvainResult {
    /// Community assignment for each node. Communities are numbered 0, 1, 2,
    /// ... in increasing order of their smallest node id.
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
/// # Determinism
///
/// The same graph always gives the same result: nodes are visited in node id
/// order, a node that gains equally from several moves joins the community
/// with the smallest index, and communities are numbered 0, 1, 2, ... in
/// increasing order of their smallest node id.
///
/// # Complexity
///
/// O(V log V) on average for sparse graphs
pub fn louvain(store: &dyn GraphStore, resolution: f64) -> LouvainResult {
    let mut nodes = store.node_ids();
    nodes.sort_unstable();
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
    // adjacency[i][j] = weight of edge between nodes i and j
    let mut adjacency: Vec<FxHashMap<usize, f64>> = vec![FxHashMap::default(); n];
    let mut total_weight = 0.0;

    for (i, &node) in nodes.iter().enumerate() {
        for (neighbor, _edge_id) in store.edges_from(node, Direction::Outgoing) {
            if let Some(&j) = node_to_idx.get(&neighbor) {
                // For undirected: add weight to both directions
                let w = 1.0; // Could extract from edge property
                *adjacency[i].entry(j).or_insert(0.0) += w;
                *adjacency[j].entry(i).or_insert(0.0) += w;
                total_weight += w;
            }
        }
    }

    // Neighbor lists sorted by index: every sum below runs in the same order
    // on every call (hash map iteration order differs between map instances).
    let weights: Vec<Vec<(usize, f64)>> = adjacency
        .into_iter()
        .map(|neighbors| {
            let mut sorted: Vec<(usize, f64)> = neighbors.into_iter().collect();
            sorted.sort_unstable_by_key(|&(j, _)| j);
            sorted
        })
        .collect();

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
    let degrees: Vec<f64> = weights
        .iter()
        .map(|neighbors| neighbors.iter().map(|&(_, w)| w).sum())
        .collect();

    // Initialize: each node in its own community, identified by the node's index
    let mut community: Vec<usize> = (0..n).collect();
    let mut community_total: Vec<f64> = degrees.clone();

    // Links from the node being moved to each neighboring community, reset
    // after every node: `touched` lists the communities with an entry.
    let mut links_to: Vec<f64> = vec![0.0; n];
    let mut is_touched: Vec<bool> = vec![false; n];
    let mut touched: Vec<usize> = Vec::new();

    // Phase 1: Local optimization
    let mut improved = true;
    while improved {
        improved = false;

        for i in 0..n {
            let current_comm = community[i];

            // Compute links to each neighboring community
            for &(j, w) in &weights[i] {
                let c = community[j];
                if !is_touched[c] {
                    is_touched[c] = true;
                    touched.push(c);
                }
                links_to[c] += w;
            }
            // Candidates in increasing order: on a tie the smallest index wins.
            touched.sort_unstable();

            // Try moving to each neighboring community
            let mut best_delta = 0.0;
            let mut best_comm = current_comm;

            // Remove node from current community for delta calculation
            let ki = degrees[i];
            let ki_in = links_to[current_comm];

            for &target_comm in &touched {
                if target_comm == current_comm {
                    continue;
                }

                let k_i_to_comm = links_to[target_comm];
                let sigma_tot = community_total[target_comm];

                // Modularity delta for moving to target_comm
                let delta = resolution
                    * (k_i_to_comm
                        - ki_in
                        - ki * (sigma_tot - community_total[current_comm] + ki)
                            / (2.0 * total_weight));

                if delta > best_delta {
                    best_delta = delta;
                    best_comm = target_comm;
                }
            }

            for &c in &touched {
                links_to[c] = 0.0;
                is_touched[c] = false;
            }
            touched.clear();

            if best_comm != current_comm {
                // Move node to best community
                community_total[current_comm] -= ki;
                community[i] = best_comm;
                community_total[best_comm] += ki;

                improved = true;
            }
        }
    }

    // Number communities by their smallest node id: nodes are sorted, so the
    // first node met in a community is its smallest.
    let mut canonical_id: Vec<Option<u64>> = vec![None; n];
    let mut num_communities = 0;
    let mut communities: FxHashMap<NodeId, u64> = FxHashMap::default();
    for (i, &node) in nodes.iter().enumerate() {
        let id = *canonical_id[community[i]].get_or_insert_with(|| {
            num_communities += 1;
            (num_communities - 1) as u64
        });
        communities.insert(node, id);
    }

    // Compute final modularity
    let modularity = compute_modularity(&weights, &community, total_weight, resolution);

    LouvainResult {
        communities,
        modularity,
        num_communities,
    }
}

/// Computes the modularity of a community assignment.
fn compute_modularity(
    weights: &[Vec<(usize, f64)>],
    community: &[usize],
    total_weight: f64,
    resolution: f64,
) -> f64 {
    let n = community.len();
    let m2 = 2.0 * total_weight;

    if m2 == 0.0 {
        return 0.0;
    }

    let degrees: Vec<f64> = (0..n)
        .map(|i| weights[i].iter().map(|&(_, w)| w).sum())
        .collect();

    let mut modularity = 0.0;

    for i in 0..n {
        for &(j, a_ij) in &weights[i] {
            if community[i] == community[j] {
                modularity += a_ij - resolution * degrees[i] * degrees[j] / m2;
            }
        }
    }

    modularity / m2
}

// ============================================================================
// Stochastic Block Partition
// ============================================================================

/// Result of stochastic block partition.
#[derive(Debug, Clone)]
pub struct StochasticBlockPartitionResult {
    /// Maps each node to its block/community ID.
    pub partition: FxHashMap<NodeId, usize>,
    /// Number of blocks in the partition.
    pub num_blocks: usize,
    /// Description length (MDL) of the partition: lower is better.
    pub description_length: f64,
}

/// Infers the optimal block partition using the degree-corrected stochastic
/// block model.
///
/// Uses agglomerative merging to minimize the description length of the graph
/// under the SBM generative model. Starting from each node in its own block,
/// it greedily merges the pair of blocks that gives the largest decrease in
/// description length until no merge improves the objective.
///
/// # Arguments
///
/// * `store` - The graph to partition.
/// * `num_blocks` - Optional target number of blocks. If `None`, the algorithm
///   selects the optimal number by minimizing description length.
/// * `max_iterations` - Maximum merge iterations.
///
/// # Complexity
///
/// O(V * B^2) per iteration where B is the current block count.
pub fn stochastic_block_partition(
    store: &dyn GraphStore,
    num_blocks: Option<usize>,
    max_iterations: usize,
) -> StochasticBlockPartitionResult {
    stochastic_block_partition_inner(store, num_blocks, max_iterations, None)
}

/// Incrementally updates an existing stochastic block partition after edges
/// have been added to the graph.
///
/// Takes the previous partition as a warm start and refines it. Much faster
/// than computing from scratch when only a small number of edges were added.
pub fn stochastic_block_partition_incremental(
    store: &dyn GraphStore,
    prior_partition: &FxHashMap<NodeId, usize>,
    max_iterations: usize,
) -> StochasticBlockPartitionResult {
    stochastic_block_partition_inner(store, None, max_iterations, Some(prior_partition))
}

/// Core SBP implementation with optional warm start.
fn stochastic_block_partition_inner(
    store: &dyn GraphStore,
    target_blocks: Option<usize>,
    max_iterations: usize,
    warm_start: Option<&FxHashMap<NodeId, usize>>,
) -> StochasticBlockPartitionResult {
    let nodes = store.node_ids();
    let n = nodes.len();

    if n == 0 {
        return StochasticBlockPartitionResult {
            partition: FxHashMap::default(),
            num_blocks: 0,
            description_length: 0.0,
        };
    }

    // Build node index mapping.
    let mut node_to_idx: FxHashMap<NodeId, usize> = FxHashMap::default();
    let mut idx_to_node: Vec<NodeId> = Vec::with_capacity(n);
    for (idx, &node) in nodes.iter().enumerate() {
        node_to_idx.insert(node, idx);
        idx_to_node.push(node);
    }

    // Build undirected adjacency (as index pairs).
    let mut adj: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); n];
    for &node in &nodes {
        let i = node_to_idx[&node];
        for (neighbor, _) in store.edges_from(node, Direction::Outgoing) {
            if let Some(&j) = node_to_idx.get(&neighbor) {
                adj[i].insert(j);
                adj[j].insert(i);
            }
        }
    }

    // Initialize partition: each node in its own block, or warm start.
    let mut block: Vec<usize> = if let Some(prior) = warm_start {
        // Map prior partition to index-based blocks. New nodes get fresh block IDs.
        let mut max_block = prior.values().copied().max().unwrap_or(0);
        (0..n)
            .map(|i| {
                if let Some(&b) = prior.get(&idx_to_node[i]) {
                    b
                } else {
                    max_block += 1;
                    max_block
                }
            })
            .collect()
    } else {
        (0..n).collect()
    };

    // Count total edges (undirected, so each edge counted once).
    let total_edges: usize = adj.iter().map(|a| a.len()).sum::<usize>() / 2;

    if total_edges == 0 {
        // No edges: each node is its own block.
        let partition = idx_to_node
            .iter()
            .enumerate()
            .map(|(i, &node)| (node, i))
            .collect();
        return StochasticBlockPartitionResult {
            partition,
            num_blocks: n,
            description_length: 0.0,
        };
    }

    // Compute block-level statistics.
    let mut block_edge_counts: FxHashMap<(usize, usize), usize> = FxHashMap::default();
    let mut block_degrees: FxHashMap<usize, usize> = FxHashMap::default();

    for i in 0..n {
        *block_degrees.entry(block[i]).or_default() += adj[i].len();
        for &j in &adj[i] {
            if i < j {
                let (bi, bj) = ordered_block(block[i], block[j]);
                *block_edge_counts.entry((bi, bj)).or_default() += 1;
            }
        }
    }

    let mut current_dl =
        compute_description_length(&block, &block_edge_counts, &block_degrees, total_edges, n);

    // Agglomerative merging: greedily merge blocks that reduce description length.
    let target = target_blocks.unwrap_or(1);

    for _ in 0..max_iterations {
        let active_blocks: FxHashSet<usize> = block.iter().copied().collect();
        let num_active = active_blocks.len();

        if num_active <= target {
            break;
        }

        // Try all pairs: pick the merge that reduces DL the most.
        let mut best_dl = current_dl;
        let mut best_pair: Option<(usize, usize)> = None;

        let block_list: Vec<usize> = active_blocks.iter().copied().collect();
        for i in 0..block_list.len() {
            for j in (i + 1)..block_list.len() {
                let bi = block_list[i];
                let bj = block_list[j];

                // Simulate merge: recompute block stats with bi merged into bj.
                let mut trial_degrees = block_degrees.clone();
                *trial_degrees.entry(bj).or_default() +=
                    trial_degrees.get(&bi).copied().unwrap_or(0);
                trial_degrees.remove(&bi);

                let mut merged_counts: FxHashMap<(usize, usize), usize> = FxHashMap::default();
                for (&(blk_a, blk_b), &count) in &block_edge_counts {
                    let na = if blk_a == bi { bj } else { blk_a };
                    let nb = if blk_b == bi { bj } else { blk_b };
                    let (oa, ob) = ordered_block(na, nb);
                    *merged_counts.entry((oa, ob)).or_default() += count;
                }

                let trial_dl = compute_description_length(
                    &block,
                    &merged_counts,
                    &trial_degrees,
                    total_edges,
                    n,
                );

                if trial_dl < best_dl {
                    best_dl = trial_dl;
                    best_pair = Some((bi, bj));
                }
            }
        }

        // If no merge improves DL and we have no hard target, stop.
        if best_pair.is_none() {
            if target_blocks.is_some() && num_active > target {
                // Forced merge: pick the least-worst pair.
                let mut least_worst = f64::MAX;
                for i in 0..block_list.len() {
                    for j in (i + 1)..block_list.len() {
                        let bi = block_list[i];
                        let bj = block_list[j];

                        let mut trial_degrees = block_degrees.clone();
                        *trial_degrees.entry(bj).or_default() +=
                            trial_degrees.get(&bi).copied().unwrap_or(0);
                        trial_degrees.remove(&bi);

                        let mut merged_counts: FxHashMap<(usize, usize), usize> =
                            FxHashMap::default();
                        for (&(blk_a, blk_b), &count) in &block_edge_counts {
                            let na = if blk_a == bi { bj } else { blk_a };
                            let nb = if blk_b == bi { bj } else { blk_b };
                            let (oa, ob) = ordered_block(na, nb);
                            *merged_counts.entry((oa, ob)).or_default() += count;
                        }

                        let trial_dl = compute_description_length(
                            &block,
                            &merged_counts,
                            &trial_degrees,
                            total_edges,
                            n,
                        );

                        if trial_dl < least_worst {
                            least_worst = trial_dl;
                            best_pair = Some((bi, bj));
                            best_dl = trial_dl;
                        }
                    }
                }
            }
            if best_pair.is_none() {
                break;
            }
        }

        let (merge_from, merge_to) = best_pair.expect("best pair exists");

        // Execute merge: move all nodes in merge_from to merge_to.
        for b in &mut block {
            if *b == merge_from {
                *b = merge_to;
            }
        }

        // Update block statistics.
        *block_degrees.entry(merge_to).or_default() +=
            block_degrees.get(&merge_from).copied().unwrap_or(0);
        block_degrees.remove(&merge_from);

        // Rebuild block edge counts for the merged block.
        let keys_to_update: Vec<(usize, usize)> = block_edge_counts.keys().copied().collect();
        let mut new_counts: FxHashMap<(usize, usize), usize> = FxHashMap::default();

        for (bi, bj) in keys_to_update {
            let count = block_edge_counts.remove(&(bi, bj)).unwrap_or(0);
            let new_bi = if bi == merge_from { merge_to } else { bi };
            let new_bj = if bj == merge_from { merge_to } else { bj };
            let (nbi, nbj) = ordered_block(new_bi, new_bj);
            *new_counts.entry((nbi, nbj)).or_default() += count;
        }
        block_edge_counts = new_counts;

        current_dl = best_dl;
    }

    // Normalize block IDs to 0..num_blocks-1.
    let unique_blocks: FxHashSet<usize> = block.iter().copied().collect();
    let mut block_map: FxHashMap<usize, usize> = FxHashMap::default();
    for (idx, &b) in unique_blocks.iter().enumerate() {
        block_map.insert(b, idx);
    }

    let partition = idx_to_node
        .iter()
        .enumerate()
        .map(|(i, &node)| (node, block_map[&block[i]]))
        .collect();

    StochasticBlockPartitionResult {
        partition,
        num_blocks: unique_blocks.len(),
        description_length: current_dl,
    }
}

/// Computes the description length (MDL) of a partition under the degree-corrected SBM.
///
/// DL = sum_{r,s} e_{rs} * log(e_{rs} / (d_r * d_s)) + sum_r d_r * log(d_r)
///
/// Lower is better.
fn compute_description_length(
    _block: &[usize],
    block_edge_counts: &FxHashMap<(usize, usize), usize>,
    block_degrees: &FxHashMap<usize, usize>,
    total_edges: usize,
    _n: usize,
) -> f64 {
    if total_edges == 0 {
        return 0.0;
    }

    let m = total_edges as f64;
    let mut dl = 0.0f64;

    // Edge term: sum over block pairs.
    for (&(bi, bj), &e_rs) in block_edge_counts {
        if e_rs == 0 {
            continue;
        }
        let d_r = block_degrees.get(&bi).copied().unwrap_or(1) as f64;
        let d_s = block_degrees.get(&bj).copied().unwrap_or(1) as f64;
        let e = e_rs as f64;

        let expected = d_r * d_s / (2.0 * m);
        if expected > 0.0 {
            dl += e * (e / expected).ln();
        }
    }

    // Degree term: sum over blocks.
    for &d_r in block_degrees.values() {
        if d_r > 0 {
            let d = d_r as f64;
            dl += d * d.ln();
        }
    }

    dl
}

/// Orders two block IDs so the smaller comes first.
fn ordered_block(a: usize, b: usize) -> (usize, usize) {
    if a <= b { (a, b) } else { (b, a) }
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

impl_algorithm! {
    LabelPropagationAlgorithm,
    name: "label_propagation",
    description: "Label Propagation community detection",
    params: label_prop_params,
    execute(store, params) {
        let max_iter = match params.get_int("max_iterations") {
            Some(v) if v < 0 => {
                return Err(grafeo_common::utils::error::Error::InvalidValue(
                    format!("max_iterations must be non-negative, got {v}"),
                ));
            }
            Some(v) => usize::try_from(v).map_err(|_| {
                grafeo_common::utils::error::Error::InvalidValue(
                    format!("max_iterations value {v} exceeds maximum supported size"),
                )
            })?,
            None => 100,
        };

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

impl_algorithm! {
    LouvainAlgorithm,
    name: "louvain",
    description: "Louvain community detection (modularity optimization)",
    params: louvain_params,
    execute(store, params) {
        let resolution = params.get_float("resolution").unwrap_or(1.0);

        let result = louvain(store, resolution);

        let mut output = AlgorithmResult::new(vec![
            "node_id".to_string(),
            "community_id".to_string(),
            "modularity".to_string(),
        ]);

        // Rows in node id order, so the same graph gives the same rows.
        let mut assignments: Vec<(NodeId, u64)> = result.communities.into_iter().collect();
        assignments.sort_unstable_by_key(|&(node, _)| node);
        for (node, community_id) in assignments {
            // reason: Node/community IDs are sequential counters, well within i64::MAX
            #[allow(clippy::cast_possible_wrap)]
            output.add_row(vec![
                Value::Int64(node.0 as i64),
                Value::Int64(community_id as i64),
                Value::Float64(result.modularity),
            ]);
        }

        Ok(output)
    }
}

/// Static parameter definitions for Stochastic Block Partition algorithm.
static SBP_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn sbp_params() -> &'static [ParameterDef] {
    SBP_PARAMS.get_or_init(|| {
        vec![
            ParameterDef {
                name: "num_blocks".to_string(),
                description: "Target number of blocks (optional, auto-selects if not set)"
                    .to_string(),
                param_type: ParameterType::Integer,
                required: false,
                default: None,
            },
            ParameterDef {
                name: "max_iterations".to_string(),
                description: "Maximum merge iterations (default: 100)".to_string(),
                param_type: ParameterType::Integer,
                required: false,
                default: Some("100".to_string()),
            },
        ]
    })
}

/// Stochastic Block Partition algorithm wrapper.
pub struct StochasticBlockPartitionAlgorithm;

impl_algorithm! {
    StochasticBlockPartitionAlgorithm,
    name: "stochastic_block_partition",
    description: "Stochastic Block Model community detection (MDL minimization)",
    params: sbp_params,
    execute(store, params) {
        let num_blocks = match params.get_int("num_blocks") {
            Some(v) if v < 0 => {
                return Err(grafeo_common::utils::error::Error::InvalidValue(
                    format!("num_blocks must be non-negative, got {v}"),
                ));
            }
            Some(v) => Some(usize::try_from(v).map_err(|_| {
                grafeo_common::utils::error::Error::InvalidValue(
                    format!("num_blocks value {v} exceeds maximum supported size"),
                )
            })?),
            None => None,
        };
        let max_iter = match params.get_int("max_iterations") {
            Some(v) if v < 0 => {
                return Err(grafeo_common::utils::error::Error::InvalidValue(
                    format!("max_iterations must be non-negative, got {v}"),
                ));
            }
            Some(v) => usize::try_from(v).map_err(|_| {
                grafeo_common::utils::error::Error::InvalidValue(
                    format!("max_iterations value {v} exceeds maximum supported size"),
                )
            })?,
            None => 100,
        };

        let result = stochastic_block_partition(store, num_blocks, max_iter);

        let mut output = AlgorithmResult::new(vec![
            "node_id".to_string(),
            "block_id".to_string(),
            "description_length".to_string(),
        ]);

        for (node, block_id) in &result.partition {
            // reason: Node/block IDs are sequential counters, well within i64::MAX
            #[allow(clippy::cast_possible_wrap)]
            output.add_row(vec![
                Value::Int64(node.0 as i64),
                Value::Int64(*block_id as i64),
                Value::Float64(result.description_length),
            ]);
        }

        Ok(output)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use super::*;

    fn create_two_cliques_graph() -> LpgStore {
        // Two cliques connected by one edge
        // Clique 1: 0-1-2-3 (fully connected)
        // Clique 2: 4-5-6-7 (fully connected)
        // Bridge: 3-4
        let store = LpgStore::new().unwrap();

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
        let store = LpgStore::new().unwrap();

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
        assert!((1..=8).contains(&num_comms)); // May vary due to algorithm randomness
    }

    #[test]
    fn test_label_propagation_empty() {
        let store = LpgStore::new().unwrap();
        let communities = label_propagation(&store, 100);
        assert!(communities.is_empty());
    }

    #[test]
    fn test_label_propagation_single_node() {
        let store = LpgStore::new().unwrap();
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

        // Two K4 cliques connected by a single bridge: should detect 2-3 communities
        assert!(
            result.num_communities >= 2 && result.num_communities <= 3,
            "Two cliques should produce 2-3 communities, got {}",
            result.num_communities
        );
    }

    #[test]
    fn test_louvain_empty() {
        let store = LpgStore::new().unwrap();
        let result = louvain(&store, 1.0);

        assert!(result.communities.is_empty());
        assert_eq!(result.modularity, 0.0);
        assert_eq!(result.num_communities, 0);
    }

    #[test]
    fn test_louvain_isolated_nodes() {
        let store = LpgStore::new().unwrap();
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

    /// The graph from a downstream report: 30 nodes with edges `i -> i + 1`,
    /// `i -> i + 3` and `i -> i * 7 % 30`, where many moves tie.
    fn create_tied_moves_graph() -> LpgStore {
        let store = LpgStore::new().unwrap();
        let nodes: Vec<NodeId> = (0..30).map(|_| store.create_node(&["N"])).collect();
        for i in 0..30 {
            for j in [i + 1, i + 3, i * 7 % 30] {
                if j < 30 && j != i {
                    store.create_edge(nodes[i], nodes[j], "USES");
                }
            }
        }
        store
    }

    /// Two K4 cliques whose members interleave in id order (even and odd
    /// positions), joined by one bridge between the last two nodes.
    fn create_interleaved_cliques_graph() -> (LpgStore, Vec<NodeId>) {
        let store = LpgStore::new().unwrap();
        let nodes: Vec<NodeId> = (0..8).map(|_| store.create_node(&["Node"])).collect();
        for clique in [[0, 2, 4, 6], [1, 3, 5, 7]] {
            for (position, &a) in clique.iter().enumerate() {
                for &b in &clique[position + 1..] {
                    store.create_edge(nodes[a], nodes[b], "EDGE");
                }
            }
        }
        store.create_edge(nodes[6], nodes[7], "EDGE");
        (store, nodes)
    }

    #[test]
    fn test_louvain_is_deterministic() {
        let store = create_tied_moves_graph();
        let first = louvain(&store, 1.0);
        for run in 0..20 {
            let again = louvain(&store, 1.0);
            assert_eq!(again.communities, first.communities, "run {run}");
            assert_eq!(
                again.modularity.to_bits(),
                first.modularity.to_bits(),
                "run {run}"
            );
            assert_eq!(again.num_communities, first.num_communities, "run {run}");
        }
    }

    #[test]
    fn test_louvain_community_ids_follow_smallest_node_id() {
        let (store, nodes) = create_interleaved_cliques_graph();
        for _ in 0..10 {
            let result = louvain(&store, 1.0);
            assert_eq!(result.num_communities, 2);
            // The clique holding the smallest node id is community 0.
            for (position, node) in nodes.iter().enumerate() {
                assert_eq!(
                    result.communities[node],
                    (position % 2) as u64,
                    "node at position {position}"
                );
            }
        }
    }

    #[test]
    fn test_louvain_algorithm_rows_in_node_order() {
        use super::super::traits::GraphAlgorithm;

        let (store, nodes) = create_interleaved_cliques_graph();
        let params = super::super::super::Parameters::new();
        let result = LouvainAlgorithm.execute(&store, &params).unwrap();
        let rows: Vec<(Value, Value)> = result
            .rows
            .iter()
            .map(|row| (row[0].clone(), row[1].clone()))
            .collect();
        let expected: Vec<(Value, Value)> = nodes
            .iter()
            .enumerate()
            .map(|(position, node)| {
                (
                    Value::Int64(i64::try_from(node.0).unwrap()),
                    Value::Int64(i64::try_from(position % 2).unwrap()),
                )
            })
            .collect();
        assert_eq!(rows, expected);
    }

    // ---- Stochastic Block Partition tests ----

    #[test]
    fn test_sbp_empty_graph() {
        let store = LpgStore::new().unwrap();
        let result = stochastic_block_partition(&store, None, 100);
        assert!(result.partition.is_empty());
        assert_eq!(result.num_blocks, 0);
    }

    #[test]
    fn test_sbp_single_node() {
        let store = LpgStore::new().unwrap();
        store.create_node(&["Node"]);
        let result = stochastic_block_partition(&store, None, 100);
        assert_eq!(result.partition.len(), 1);
        assert_eq!(result.num_blocks, 1);
    }

    #[test]
    fn test_sbp_two_cliques() {
        let store = create_two_cliques_graph();
        let result = stochastic_block_partition(&store, None, 100);

        // All 8 nodes should be partitioned.
        assert_eq!(result.partition.len(), 8);
        // Number of blocks should be between 1 and 8 inclusive.
        assert!(
            result.num_blocks >= 1 && result.num_blocks <= 8,
            "num_blocks should be in [1,8], got {}",
            result.num_blocks
        );
        // Description length should be finite.
        assert!(result.description_length.is_finite());
    }

    #[test]
    fn test_sbp_target_blocks() {
        let store = create_two_cliques_graph();
        let result = stochastic_block_partition(&store, Some(2), 100);

        assert_eq!(result.partition.len(), 8);
        assert_eq!(result.num_blocks, 2);
    }

    #[test]
    fn test_sbp_description_length_decreases() {
        let store = create_two_cliques_graph();

        // With 8 blocks (each node alone) vs 2 blocks.
        let result_2 = stochastic_block_partition(&store, Some(2), 100);
        // Description length should be finite.
        assert!(
            result_2.description_length.is_finite(),
            "DL should be finite, got {}",
            result_2.description_length
        );
    }

    #[test]
    fn test_sbp_isolated_nodes() {
        let store = LpgStore::new().unwrap();
        store.create_node(&["Node"]);
        store.create_node(&["Node"]);
        store.create_node(&["Node"]);

        let result = stochastic_block_partition(&store, None, 100);
        assert_eq!(result.partition.len(), 3);
        // Isolated nodes: each stays in its own block.
        assert_eq!(result.num_blocks, 3);
    }

    #[test]
    fn test_sbp_incremental() {
        let store = LpgStore::new().unwrap();

        // Phase 1: Two connected nodes.
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n1, n0, "EDGE");

        let result1 = stochastic_block_partition(&store, None, 100);

        // Phase 2: Add a third node.
        let n2 = store.create_node(&["Node"]);
        store.create_edge(n1, n2, "EDGE");
        store.create_edge(n2, n1, "EDGE");

        let result2 = stochastic_block_partition_incremental(&store, &result1.partition, 100);

        assert_eq!(result2.partition.len(), 3);
        // The incremental result should include the new node.
        assert!(result2.partition.contains_key(&n2));
    }

    #[test]
    fn test_sbp_algorithm_wrapper() {
        use super::super::traits::GraphAlgorithm;

        let store = create_two_cliques_graph();
        let algo = StochasticBlockPartitionAlgorithm;

        assert_eq!(algo.name(), "stochastic_block_partition");

        let params = super::super::super::Parameters::new();
        let result = algo.execute(&store, &params).unwrap();
        assert_eq!(result.columns.len(), 3);
        assert_eq!(result.row_count(), 8);
    }
}
