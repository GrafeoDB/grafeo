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
use super::traits::{ComponentResultBuilder, impl_algorithm, visible_edges_from};

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
/// A map from node ID to community (label) ID. Communities are numbered 0, 1,
/// 2, ... in increasing order of their smallest node id.
///
/// # Determinism
///
/// Nodes are visited in node id order, and a tie between labels goes to the
/// smallest, so the same graph always gives the same communities and ids.
///
/// # Panics
///
/// Panics if the internal label map is inconsistent (should not happen with a valid `GraphStore`).
///
/// # Complexity
///
/// O(iterations × E)
pub fn label_propagation(store: &dyn GraphStore, max_iterations: usize) -> FxHashMap<NodeId, u64> {
    let mut nodes = store.node_ids();
    nodes.sort_unstable();
    label_propagation_in_order(store, &nodes, max_iterations)
}

/// [`label_propagation`] over `nodes`, in their order: nodes are visited in
/// that order, a tie between labels goes to the label of the node that comes
/// first, and communities are numbered 0, 1, 2, ... in the order of their
/// first node. Given the nodes in the order of a key
/// ([`order_by_key`](super::order_by_key)), the result does not depend on the
/// order the nodes and edges were inserted in.
///
/// `nodes` must be the nodes of `store`, each once; an edge to a node not in
/// `nodes` is not followed.
///
/// # Panics
///
/// Panics if the internal label map is inconsistent (should not happen with a
/// valid `GraphStore`).
pub fn label_propagation_in_order(
    store: &dyn GraphStore,
    nodes: &[NodeId],
    max_iterations: usize,
) -> FxHashMap<NodeId, u64> {
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

        // Update labels in the order of `nodes`
        for &node in nodes {
            // Get neighbor labels and their frequencies
            let mut label_counts: FxHashMap<u64, usize> = FxHashMap::default();

            // Consider both outgoing and incoming edges (undirected community detection)
            // Outgoing edges: node -> neighbor
            for (neighbor, _) in visible_edges_from(store, node, Direction::Outgoing) {
                if let Some(&label) = labels.get(&neighbor) {
                    *label_counts.entry(label).or_insert(0) += 1;
                }
            }

            // Incoming edges: neighbor -> node
            // Uses backward adjacency index for O(degree) instead of O(V*E)
            for (incoming_neighbor, _) in visible_edges_from(store, node, Direction::Incoming) {
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

    // Number communities 0, 1, 2, ... in the order of their first node in
    // `nodes`, so the same graph gives the same ids every call.
    let mut label_map: FxHashMap<u64, u64> = FxHashMap::default();
    for node in nodes {
        let label = labels[node];
        let next = label_map.len() as u64;
        label_map.entry(label).or_insert(next);
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
/// The Louvain algorithm optimizes modularity greedily, repeating two phases
/// until a level changes nothing:
/// 1. Local moving: each node joins the neighbouring community that gains the
///    most modularity, until no move gains.
/// 2. Aggregation: every community becomes a super-node of a new graph, with
///    the weights between communities summed and the weight inside each
///    community kept as a self-loop.
///
/// The graph is treated as undirected with weight 1 per edge: two edges between
/// the same nodes weigh 2, and a self-loop adds 2 to its node's degree.
///
/// # Arguments
///
/// * `store` - The graph store
/// * `resolution` - Resolution parameter (higher = smaller communities, default 1.0)
///
/// # Returns
///
/// Community assignments and the modularity of the partition at `resolution`.
///
/// # Determinism
///
/// The same graph always gives the same result, at every level: nodes and
/// super-nodes are visited in node id order (a super-node by the smallest node
/// id it holds), a node that gains equally from several moves joins the
/// community with the smallest index, sums run in a fixed order, and
/// communities are numbered 0, 1, 2, ... in increasing order of their smallest
/// node id.
///
/// # Complexity
///
/// O(V log V) on average for sparse graphs
pub fn louvain(store: &dyn GraphStore, resolution: f64) -> LouvainResult {
    let mut nodes = store.node_ids();
    nodes.sort_unstable();
    louvain_in_order(store, &nodes, resolution)
}

/// [`louvain`] over `nodes`, in their order, at every level: nodes and
/// super-nodes are visited in that order (a super-node by its first node), a
/// node that gains equally from several moves joins the community that comes
/// first, sums run in that order, and communities are numbered 0, 1, 2, ...
/// in the order of their first node. Given the nodes in the order of a key
/// ([`order_by_key`](super::order_by_key)), the result does not depend on the
/// order the nodes and edges were inserted in.
///
/// `nodes` must be the nodes of `store`, each once; an edge to a node not in
/// `nodes` is not followed.
pub fn louvain_in_order(
    store: &dyn GraphStore,
    nodes: &[NodeId],
    resolution: f64,
) -> LouvainResult {
    let n = nodes.len();

    if n == 0 {
        return LouvainResult {
            communities: FxHashMap::default(),
            modularity: 0.0,
            num_communities: 0,
        };
    }

    let node_to_idx: FxHashMap<NodeId, usize> = nodes
        .iter()
        .enumerate()
        .map(|(idx, &node)| (node, idx))
        .collect();

    // Undirected weights: each edge adds 1 to both directions, so a self-loop
    // adds 2 to its own entry.
    let mut adjacency: Vec<FxHashMap<usize, f64>> = vec![FxHashMap::default(); n];
    let mut total_weight = 0.0;
    for (i, &node) in nodes.iter().enumerate() {
        for (neighbor, _edge_id) in visible_edges_from(store, node, Direction::Outgoing) {
            if let Some(&j) = node_to_idx.get(&neighbor) {
                *adjacency[i].entry(j).or_insert(0.0) += 1.0;
                *adjacency[j].entry(i).or_insert(0.0) += 1.0;
                total_weight += 1.0;
            }
        }
    }

    // Isolated nodes only: each node is its own community.
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

    let weights = sorted_neighbors(adjacency);

    // `membership[v]` is the super-node holding node v at the current level.
    let mut membership: Vec<usize> = (0..n).collect();
    let mut level = weights.clone();
    loop {
        let Some(community) = move_nodes(&level, total_weight, resolution) else {
            break;
        };
        let (renumbered, count) = number_by_first_member(&community);
        for super_node in &mut membership {
            *super_node = renumbered[*super_node];
        }
        level = aggregate(&level, &renumbered, count);
    }

    // Super-nodes are numbered by their first member, so `membership` already
    // numbers communities in the order of their first node in `nodes`.
    let num_communities = membership.iter().copied().max().map_or(0, |max| max + 1);
    let communities: FxHashMap<NodeId, u64> = nodes
        .iter()
        .zip(&membership)
        .map(|(&node, &community)| (node, community as u64))
        .collect();
    let modularity = compute_modularity(&weights, &membership, total_weight, resolution);

    LouvainResult {
        communities,
        modularity,
        num_communities,
    }
}

/// Turns adjacency maps into neighbour lists sorted by index, so every sum over
/// them runs in the same order (hash map iteration order differs between map
/// instances).
fn sorted_neighbors(adjacency: Vec<FxHashMap<usize, f64>>) -> Vec<Vec<(usize, f64)>> {
    adjacency
        .into_iter()
        .map(|neighbors| {
            let mut sorted: Vec<(usize, f64)> = neighbors.into_iter().collect();
            sorted.sort_unstable_by_key(|&(j, _)| j);
            sorted
        })
        .collect()
}

/// Louvain phase 1 on one level: moves nodes between neighbouring communities
/// until no move gains modularity. Returns the community of each node, or
/// `None` when no node moved.
///
/// A move from community A to B gains, in units of `1 / m`,
/// `k_i,B - k_i,A - resolution * k_i * (Σ_B - Σ_A\i) / 2m`, where `k_i,X` is the
/// weight from node i to community X (its self-loop excluded), `k_i` its degree
/// and `Σ_X` the total degree of X. The gains are compared multiplied by `2m`:
/// on an unweighted graph at resolution 1 they are then whole numbers, so ties
/// are exact; at other resolutions a move must beat staying by more than the
/// rounding error (see [`move_gains`]), so nodes cannot move back and forth.
fn move_nodes(
    graph: &[Vec<(usize, f64)>],
    total_weight: f64,
    resolution: f64,
) -> Option<Vec<usize>> {
    let k = graph.len();
    // Edge weights are whole numbers, so at resolution 1 every gain below is a
    // whole number too and compares exactly (see `move_gains`).
    let exact = resolution.to_bits() == 1.0_f64.to_bits();
    let m2 = 2.0 * total_weight;
    let degrees: Vec<f64> = graph
        .iter()
        .map(|neighbors| neighbors.iter().map(|&(_, w)| w).sum())
        .collect();

    // Each node starts in its own community, identified by the node's index.
    let mut community: Vec<usize> = (0..k).collect();
    let mut community_total: Vec<f64> = degrees.clone();

    // Links from the node being moved to each neighbouring community, reset
    // after every node: `touched` lists the communities with an entry.
    let mut links_to: Vec<f64> = vec![0.0; k];
    let mut is_touched: Vec<bool> = vec![false; k];
    let mut touched: Vec<usize> = Vec::new();

    // Sweeps until one moves nothing. That point comes: `move_gains` lets a node
    // move only when the move raises the exact modularity, so no partition comes
    // back, and a graph has finitely many partitions.
    let mut moved = false;
    loop {
        let mut improved = false;
        for i in 0..k {
            let current = community[i];
            for &(j, w) in &graph[i] {
                if j == i {
                    continue;
                }
                let c = community[j];
                if !is_touched[c] {
                    is_touched[c] = true;
                    touched.push(c);
                }
                links_to[c] += w;
            }
            // Candidates in increasing order: on a tie the smallest index wins.
            touched.sort_unstable();

            let ki = degrees[i];
            // The best neighbouring community (on a tie the smallest index), then
            // the move only if it beats staying for certain.
            let mut best = current;
            let mut best_gain = f64::NEG_INFINITY;
            for &target in &touched {
                if target == current {
                    continue;
                }
                let gain = links_to[target] * m2 - resolution * ki * community_total[target];
                if gain > best_gain {
                    best_gain = gain;
                    best = target;
                }
            }
            if best != current
                && !move_gains(
                    links_to[best] * m2,
                    resolution * ki * community_total[best],
                    links_to[current] * m2,
                    resolution * ki * (community_total[current] - ki),
                    exact,
                )
            {
                best = current;
            }

            for &c in &touched {
                links_to[c] = 0.0;
                is_touched[c] = false;
            }
            touched.clear();

            if best != current {
                community_total[current] -= ki;
                community_total[best] += ki;
                community[i] = best;
                improved = true;
            }
        }
        if !improved {
            break;
        }
        moved = true;
    }

    moved.then_some(community)
}

/// Whether a move gains modularity for certain. The scaled gains of moving
/// (`target_links - target_expected`) and of staying (`stay_links -
/// stay_expected`) are each a link term `k_i,X * 2m` minus an expected term
/// `resolution * k_i * sigma_X`. `exact` says the terms are whole numbers
/// (resolution 1, as edge weights are); below 2^53 they then compare exactly
/// and any positive difference is a real gain. Otherwise the difference must
/// exceed the rounding error of computing it, so a move taken always raises
/// the exact modularity and local moving cannot cycle.
fn move_gains(
    target_links: f64,
    target_expected: f64,
    stay_links: f64,
    stay_expected: f64,
    exact: bool,
) -> bool {
    // 2^53: whole numbers up to here are exact in f64.
    const EXACT_LIMIT: f64 = 9_007_199_254_740_992.0;
    let magnitude =
        target_links.abs() + target_expected.abs() + stay_links.abs() + stay_expected.abs();
    let rounding = if exact && magnitude < EXACT_LIMIT {
        0.0
    } else {
        // Each term carries at most two roundings and the difference three more.
        4.0 * f64::EPSILON * magnitude
    };
    (target_links - target_expected) - (stay_links - stay_expected) > rounding
}

/// Renumbers communities 0, 1, 2, ... in order of their first member. Returns the
/// new number of each old community index and the number of communities.
fn number_by_first_member(community: &[usize]) -> (Vec<usize>, usize) {
    let mut number: Vec<Option<usize>> = vec![None; community.len()];
    let mut count = 0;
    let renumbered = community
        .iter()
        .map(|&c| {
            *number[c].get_or_insert_with(|| {
                count += 1;
                count - 1
            })
        })
        .collect();
    (renumbered, count)
}

/// Louvain phase 2: one super-node per community, with the weights between
/// communities summed and the weight inside a community as its self-loop.
fn aggregate(
    graph: &[Vec<(usize, f64)>],
    community: &[usize],
    count: usize,
) -> Vec<Vec<(usize, f64)>> {
    let mut adjacency: Vec<FxHashMap<usize, f64>> = vec![FxHashMap::default(); count];
    for (i, neighbors) in graph.iter().enumerate() {
        for &(j, w) in neighbors {
            *adjacency[community[i]].entry(community[j]).or_insert(0.0) += w;
        }
    }
    sorted_neighbors(adjacency)
}

/// Computes the modularity of a community assignment:
/// `Σ_c (A_c / 2m - resolution * (Σ_c / 2m)²)`, where `A_c` sums the weights
/// inside community c in both directions (self-loops included) and `Σ_c` is
/// its total degree.
fn compute_modularity(
    weights: &[Vec<(usize, f64)>],
    community: &[usize],
    total_weight: f64,
    resolution: f64,
) -> f64 {
    let m2 = 2.0 * total_weight;
    if m2 == 0.0 {
        return 0.0;
    }

    let count = community.iter().copied().max().map_or(0, |max| max + 1);
    let mut internal = vec![0.0; count];
    let mut total = vec![0.0; count];
    for (i, neighbors) in weights.iter().enumerate() {
        for &(j, w) in neighbors {
            total[community[i]] += w;
            if community[i] == community[j] {
                internal[community[i]] += w;
            }
        }
    }

    internal
        .iter()
        .zip(&total)
        .map(|(&inside, &degree)| inside / m2 - resolution * (degree / m2) * (degree / m2))
        .sum()
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
    // In id order: the order blocks are tried and numbered in.
    let mut nodes = store.node_ids();
    nodes.sort_unstable();
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
        for (neighbor, _) in visible_edges_from(store, node, Direction::Outgoing) {
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

        // In block order, so that of two merges with the same description
        // length the same one is taken on every call.
        let mut block_list: Vec<usize> = active_blocks.iter().copied().collect();
        block_list.sort_unstable();
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

    // Normalize block IDs to 0..num_blocks-1, in the order of each block's
    // first node.
    let mut block_map: FxHashMap<usize, usize> = FxHashMap::default();
    for &b in &block {
        let next = block_map.len();
        block_map.entry(b).or_insert(next);
    }

    let partition = idx_to_node
        .iter()
        .enumerate()
        .map(|(i, &node)| (node, block_map[&block[i]]))
        .collect();

    StochasticBlockPartitionResult {
        partition,
        num_blocks: block_map.len(),
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

    // Both sums run in key order: a float sum depends on the order of its
    // terms, and a map's order changes from one map to the next.
    let mut edge_counts: Vec<((usize, usize), usize)> = block_edge_counts
        .iter()
        .map(|(&pair, &count)| (pair, count))
        .collect();
    edge_counts.sort_unstable();
    let mut degrees: Vec<(usize, usize)> = block_degrees
        .iter()
        .map(|(&block, &degree)| (block, degree))
        .collect();
    degrees.sort_unstable();

    // Edge term: sum over block pairs.
    for ((bi, bj), e_rs) in edge_counts {
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
    for (_, d_r) in degrees {
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

        let mut result = builder.build();
        result.sort_by_id_columns(1);
        Ok(result)
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

        output.sort_by_id_columns(1);
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

        // A three-node path is best left whole: modularity 0, against -0.125 for
        // any split.
        assert_eq!(result.communities.len(), 3);
        assert!(result.communities.values().all(|&c| c == 0));
        assert_eq!(result.num_communities, 1);
        assert!(result.modularity.abs() < 1e-15);
    }

    #[test]
    fn test_louvain_cliques() {
        let store = create_two_cliques_graph();
        let result = louvain(&store, 1.0);

        // Two K4 cliques connected by a single bridge: one community each.
        let mut nodes: Vec<NodeId> = result.communities.keys().copied().collect();
        nodes.sort_unstable();
        assert_eq!(partition(&result, &nodes), vec![0, 0, 0, 0, 1, 1, 1, 1]);
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

        assert_eq!(result_low.communities.len(), 8);
        assert_eq!(result_high.communities.len(), 8);
        assert!(result_low.num_communities <= result_high.num_communities);
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

    /// Builds a store with `n` nodes and one directed edge per pair in `edges`.
    fn store_from_edges(n: usize, edges: &[(usize, usize)]) -> (LpgStore, Vec<NodeId>) {
        let store = LpgStore::new().unwrap();
        let nodes: Vec<NodeId> = (0..n).map(|_| store.create_node(&["Node"])).collect();
        for &(u, v) in edges {
            store.create_edge(nodes[u], nodes[v], "EDGE");
        }
        (store, nodes)
    }

    fn path_edges(n: usize) -> Vec<(usize, usize)> {
        (1..n).map(|i| (i - 1, i)).collect()
    }

    /// Community of each node, in node order.
    fn partition(result: &LouvainResult, nodes: &[NodeId]) -> Vec<u64> {
        nodes.iter().map(|node| result.communities[node]).collect()
    }

    /// Modularity from its definition, every edge undirected with weight 1:
    /// the sum over communities of `L_c / m - resolution * (d_c / 2m)^2`, where a
    /// self-loop counts once in `L_c` and twice in the degree sum `d_c`.
    fn reference_modularity(
        n: usize,
        edges: &[(usize, usize)],
        communities: &[u64],
        resolution: f64,
    ) -> f64 {
        let m = edges.len() as f64;
        let mut degree = vec![0.0; n];
        let mut internal: FxHashMap<u64, f64> = FxHashMap::default();
        for &(u, v) in edges {
            degree[u] += 1.0;
            degree[v] += 1.0;
            if communities[u] == communities[v] {
                *internal.entry(communities[u]).or_insert(0.0) += 1.0;
            }
        }
        let mut degree_sum: FxHashMap<u64, f64> = FxHashMap::default();
        for (v, &d) in degree.iter().enumerate() {
            *degree_sum.entry(communities[v]).or_insert(0.0) += d;
        }
        degree_sum
            .iter()
            .map(|(c, &d)| {
                internal.get(c).copied().unwrap_or(0.0) / m
                    - resolution * (d / (2.0 * m)) * (d / (2.0 * m))
            })
            .sum()
    }

    /// Zachary's karate club (34 members, 78 friendships).
    const KARATE_CLUB: [(usize, usize); 78] = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (0, 7),
        (0, 8),
        (0, 10),
        (0, 11),
        (0, 12),
        (0, 13),
        (0, 17),
        (0, 19),
        (0, 21),
        (0, 31),
        (1, 2),
        (1, 3),
        (1, 7),
        (1, 13),
        (1, 17),
        (1, 19),
        (1, 21),
        (1, 30),
        (2, 3),
        (2, 7),
        (2, 8),
        (2, 9),
        (2, 13),
        (2, 27),
        (2, 28),
        (2, 32),
        (3, 7),
        (3, 12),
        (3, 13),
        (4, 6),
        (4, 10),
        (5, 6),
        (5, 10),
        (5, 16),
        (6, 16),
        (8, 30),
        (8, 32),
        (8, 33),
        (9, 33),
        (13, 33),
        (14, 32),
        (14, 33),
        (15, 32),
        (15, 33),
        (18, 32),
        (18, 33),
        (19, 33),
        (20, 32),
        (20, 33),
        (22, 32),
        (22, 33),
        (23, 25),
        (23, 27),
        (23, 29),
        (23, 32),
        (23, 33),
        (24, 25),
        (24, 27),
        (24, 31),
        (25, 31),
        (26, 29),
        (26, 33),
        (27, 33),
        (28, 31),
        (28, 33),
        (29, 32),
        (29, 33),
        (30, 32),
        (30, 33),
        (31, 32),
        (31, 33),
        (32, 33),
    ];

    #[test]
    fn test_louvain_aggregates_a_long_path() {
        // One level of local moving pairs up neighbours (500 communities, modularity
        // 0.5); merging communities level by level reaches above 0.9.
        let edges = path_edges(1000);
        let (store, nodes) = store_from_edges(1000, &edges);
        let result = louvain(&store, 1.0);
        assert!(result.modularity > 0.9, "modularity {}", result.modularity);
        assert!(
            (10..=100).contains(&result.num_communities),
            "{} communities",
            result.num_communities
        );
        // Every community of a path is a contiguous run of nodes.
        let communities = partition(&result, &nodes);
        assert!(
            communities
                .windows(2)
                .all(|w| w[1] == w[0] || w[1] == w[0] + 1)
        );
    }

    #[test]
    fn test_louvain_keeps_two_joined_five_cliques_apart() {
        let mut edges = Vec::new();
        for clique in [0..5, 5..10] {
            for a in clique.clone() {
                for b in (a + 1)..clique.end {
                    edges.push((a, b));
                }
            }
        }
        edges.push((4, 5));
        let (store, nodes) = store_from_edges(10, &edges);
        let result = louvain(&store, 1.0);
        assert_eq!(
            partition(&result, &nodes),
            vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        );
    }

    #[test]
    fn test_louvain_reaches_known_modularity_on_karate_club() {
        // The best partition of the karate club has modularity 0.4198; Louvain finds
        // 0.4188 or 0.4198 depending on the visiting order.
        let (store, nodes) = store_from_edges(34, &KARATE_CLUB);
        let result = louvain(&store, 1.0);
        assert!(
            result.modularity > 0.418,
            "modularity {}",
            result.modularity
        );
        assert!(
            (3..=5).contains(&result.num_communities),
            "{} communities",
            result.num_communities
        );
        let expected = reference_modularity(34, &KARATE_CLUB, &partition(&result, &nodes), 1.0);
        assert!((result.modularity - expected).abs() < 1e-12);
    }

    #[test]
    fn test_louvain_resolution_changes_the_partition() {
        let (store, nodes) = store_from_edges(34, &KARATE_CLUB);
        let coarse = louvain(&store, 0.05);
        let standard = louvain(&store, 1.0);
        let fine = louvain(&store, 3.0);
        // A low resolution merges the connected club into one community, a high one
        // splits it further than the standard resolution does.
        assert_eq!(coarse.num_communities, 1);
        assert!(
            fine.num_communities > standard.num_communities,
            "{} at 3.0 against {} at 1.0",
            fine.num_communities,
            standard.num_communities
        );
        for (resolution, result) in [(0.05, &coarse), (3.0, &fine)] {
            let expected =
                reference_modularity(34, &KARATE_CLUB, &partition(result, &nodes), resolution);
            assert!(
                (result.modularity - expected).abs() < 1e-12,
                "resolution {resolution}: {} against {expected}",
                result.modularity
            );
        }
    }

    #[test]
    fn test_louvain_handles_self_loops() {
        // Two triangles joined by one edge, with a self-loop on every node.
        let mut edges = vec![(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)];
        edges.extend((0..6).map(|v| (v, v)));
        let (store, nodes) = store_from_edges(6, &edges);
        let result = louvain(&store, 1.0);
        let communities = partition(&result, &nodes);
        assert_eq!(communities, vec![0, 0, 0, 1, 1, 1]);
        let expected = reference_modularity(6, &edges, &communities, 1.0);
        assert!((result.modularity - expected).abs() < 1e-12);
    }

    #[test]
    fn test_louvain_reported_modularity_matches_its_partition() {
        let edges = path_edges(1000);
        let (store, nodes) = store_from_edges(1000, &edges);
        let result = louvain(&store, 1.0);
        let expected = reference_modularity(1000, &edges, &partition(&result, &nodes), 1.0);
        assert!((result.modularity - expected).abs() < 1e-12);
    }

    #[test]
    fn test_louvain_is_deterministic_across_levels() {
        let edges = path_edges(1000);
        let (store, _) = store_from_edges(1000, &edges);
        let first = louvain(&store, 1.0);
        for run in 0..5 {
            let again = louvain(&store, 1.0);
            assert_eq!(again.communities, first.communities, "run {run}");
            assert_eq!(
                again.modularity.to_bits(),
                first.modularity.to_bits(),
                "run {run}"
            );
        }
    }

    #[test]
    fn test_a_move_must_beat_staying_beyond_rounding() {
        // Exact (resolution 1, whole-number terms below 2^53): a gain of 1 wins, a tie does not.
        let big = 1.0e15;
        assert!(move_gains(big + 1.0, big, big, big, true));
        assert!(!move_gains(big, big, big, big, true));
        // Inexact: a difference inside the rounding error is not a gain.
        assert!(!move_gains(10.0, 5.0 + 1.0e-14, 10.0, 5.0, false));
        assert!(!move_gains(10.0, 5.0 - 1.0e-15, 10.0, 5.0, false));
        assert!(move_gains(10.0, 4.0, 10.0, 5.0, false));
        // Past 2^53 even resolution 1 rounds, so the margin applies.
        let huge = 2.0e16;
        assert!(!move_gains(huge + 2.0, huge, huge, huge, true));
        assert!(move_gains(huge + 1.0e3, huge, huge, huge, true));
    }

    /// Louvain's level-0 weights for `edges`, as `louvain` builds them.
    fn level_weights(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<(usize, f64)>> {
        let mut adjacency: Vec<FxHashMap<usize, f64>> = vec![FxHashMap::default(); n];
        for &(u, v) in edges {
            *adjacency[u].entry(v).or_insert(0.0) += 1.0;
            *adjacency[v].entry(u).or_insert(0.0) += 1.0;
        }
        sorted_neighbors(adjacency)
    }

    #[test]
    fn test_local_moving_stops_at_a_local_optimum() {
        // Small whole numbers in the exact check below fit in i32.
        let to_f64 = |x: i128| f64::from(i32::try_from(x).unwrap());
        // xorshift64: a fixed seed keeps the graphs the same on every run.
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = |bound: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            usize::try_from(state % u64::try_from(bound).unwrap()).unwrap()
        };
        // Resolutions p / 20, so gains compare exactly in i128 after scaling by 20.
        for p in [6_i32, 15, 20, 32] {
            let resolution = f64::from(p) / 20.0;
            let p = i128::from(p);
            for case in 0..60 {
                let n = 2 + next(40);
                let edge_count = 1 + next(n * 3);
                let edges: Vec<(usize, usize)> =
                    (0..edge_count).map(|_| (next(n), next(n))).collect();
                let graph = level_weights(n, &edges);
                let m = to_f64(i128::try_from(edges.len()).unwrap());
                let community =
                    move_nodes(&graph, m, resolution).unwrap_or_else(|| (0..n).collect());

                // The same weights as whole numbers: each edge adds 1 both ways.
                let mut weight: Vec<FxHashMap<usize, i128>> = vec![FxHashMap::default(); n];
                for &(u, v) in &edges {
                    *weight[u].entry(v).or_insert(0) += 1;
                    *weight[v].entry(u).or_insert(0) += 1;
                }
                let m2 = 2 * i128::try_from(edges.len()).unwrap();
                let degree: Vec<i128> = weight.iter().map(|nb| nb.values().sum()).collect();
                let mut total = vec![0_i128; n];
                for v in 0..n {
                    total[community[v]] += degree[v];
                }
                for i in 0..n {
                    let mut links: FxHashMap<usize, i128> = FxHashMap::default();
                    for (&j, &w) in &weight[i] {
                        if j != i {
                            *links.entry(community[j]).or_insert(0) += w;
                        }
                    }
                    let own = community[i];
                    let own_links = links.get(&own).copied().unwrap_or(0);
                    // Exact gains: 20 * (links * 2m) - p * k_i * sigma.
                    let stay = 20 * own_links * m2 - p * degree[i] * (total[own] - degree[i]);
                    for (&target, &l) in &links {
                        if target == own {
                            continue;
                        }
                        let gain = 20 * l * m2 - p * degree[i] * total[target];
                        // Nothing better is left, up to the rounding margin of the
                        // comparison (zero at resolution 1, where gains are exact).
                        let margin = if p == 20 {
                            0.0
                        } else {
                            let magnitude = to_f64(20 * (l + own_links) * m2)
                                + to_f64(p * degree[i] * (total[target] + total[own]));
                            4.0 * f64::EPSILON * magnitude
                        };
                        assert!(
                            to_f64(gain - stay) <= margin,
                            "resolution {resolution}, case {case}: node {i} gains {} by moving",
                            gain - stay
                        );
                    }
                }
            }
        }
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

    /// The partition of a graph is the same on every call: its blocks (numbered
    /// in the order of their first node) and its description length to the
    /// last bit, with and without a target. Equal merges were taken in hash
    /// order, so the blocks, and at times the description length, changed
    /// from call to call.
    #[test]
    fn test_sbp_is_the_same_on_every_call() {
        let store = create_two_cliques_graph();
        for target in [None, Some(2), Some(3)] {
            let blocks_of = |result: &StochasticBlockPartitionResult| {
                let mut blocks: Vec<(NodeId, usize)> = result
                    .partition
                    .iter()
                    .map(|(&node, &b)| (node, b))
                    .collect();
                blocks.sort_unstable();
                blocks
            };
            let first = stochastic_block_partition(&store, target, 100);
            let expected = blocks_of(&first);
            // Blocks are numbered in the order of their first node.
            let mut seen = 0;
            for &(_, block) in &expected {
                assert!(block <= seen, "target {target:?}: blocks {expected:?}");
                if block == seen {
                    seen += 1;
                }
            }
            for call in 1..40 {
                let next = stochastic_block_partition(&store, target, 100);
                assert_eq!(
                    blocks_of(&next),
                    expected,
                    "target {target:?}, call {call}: the blocks changed"
                );
                assert_eq!(
                    next.description_length.to_bits(),
                    first.description_length.to_bits(),
                    "target {target:?}, call {call}: {} against {}",
                    next.description_length,
                    first.description_length
                );
            }
        }
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
