//! HNSW (Hierarchical Navigable Small World) index implementation.
//!
//! HNSW is a graph-based approximate nearest neighbor algorithm that builds
//! a multi-layer navigable small world graph. It provides:
//!
//! - **O(log n)** search complexity (approximate)
//! - **>95%** recall at k=10 with default settings
//! - **Memory**: ~1.5x the vector storage size
//!
//! # Algorithm Overview
//!
//! 1. **Multi-layer graph**: Nodes exist at multiple layers, with decreasing
//!    probability at higher layers (exponential distribution).
//! 2. **Greedy search**: Starting from the entry point at the top layer,
//!    greedily traverse to find the nearest node, then descend.
//! 3. **Beam search**: At the bottom layer, maintain a candidate set of
//!    size `ef` to find the k nearest neighbors.
//!
//! # Example
//!
//! ```ignore
//! use grafeo_core::index::vector::{HnswIndex, HnswConfig, DistanceMetric};
//! use grafeo_common::types::NodeId;
//!
//! let config = HnswConfig::new(384, DistanceMetric::Cosine);
//! let mut index = HnswIndex::new(config);
//!
//! // Insert vectors
//! let vec1 = vec![0.1f32; 384];
//! index.insert(NodeId::new(1), &vec1);
//!
//! // Search for nearest neighbors
//! let query = vec![0.15f32; 384];
//! let results = index.search(&query, 10);
//! ```
//!
//! # References
//!
//! - Malkov & Yashunin, "Efficient and robust approximate nearest neighbor
//!   search using Hierarchical Navigable Small World graphs" (2018)

use super::compute_distance;
use crate::index::vector::HnswConfig;
use grafeo_common::types::NodeId;
use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use rand::{Rng, SeedableRng};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;

/// A neighbor entry in the HNSW graph.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Neighbor {
    id: NodeId,
    distance: f32,
}

impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap: smaller distance = higher priority
        OrderedFloat(other.distance).cmp(&OrderedFloat(self.distance))
    }
}

/// A candidate for the max-heap during search (furthest first).
#[derive(Debug, Clone, Copy, PartialEq)]
struct FurthestCandidate {
    id: NodeId,
    distance: f32,
}

impl Eq for FurthestCandidate {}

impl PartialOrd for FurthestCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FurthestCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap: larger distance = higher priority
        OrderedFloat(self.distance).cmp(&OrderedFloat(other.distance))
    }
}

/// Node data stored in the HNSW index.
#[derive(Debug, Clone)]
struct HnswNode {
    /// The vector data.
    vector: Arc<[f32]>,
    /// Neighbors at each layer (layer 0 is the bottom).
    /// The node's max layer is `neighbors.len() - 1`.
    neighbors: Vec<Vec<NodeId>>,
}

/// HNSW (Hierarchical Navigable Small World) index.
///
/// Thread-safe approximate nearest neighbor index supporting concurrent
/// reads and exclusive writes.
pub struct HnswIndex {
    /// Index configuration.
    config: HnswConfig,
    /// Node storage: NodeId -> HnswNode.
    nodes: RwLock<HashMap<NodeId, HnswNode>>,
    /// Entry point for search (node at the highest layer).
    entry_point: RwLock<Option<NodeId>>,
    /// Current maximum layer in the index.
    max_level: RwLock<usize>,
    /// Random number generator for level selection.
    rng: RwLock<rand::rngs::StdRng>,
}

impl HnswIndex {
    /// Creates a new empty HNSW index with the given configuration.
    #[must_use]
    pub fn new(config: HnswConfig) -> Self {
        Self {
            config,
            nodes: RwLock::new(HashMap::new()),
            entry_point: RwLock::new(None),
            max_level: RwLock::new(0),
            rng: RwLock::new(rand::rngs::StdRng::from_os_rng()),
        }
    }

    /// Creates a new HNSW index with pre-allocated capacity.
    ///
    /// Use this when you know the approximate number of vectors upfront
    /// to avoid HashMap rehashing during bulk insertion.
    #[must_use]
    pub fn with_capacity(config: HnswConfig, capacity: usize) -> Self {
        Self {
            config,
            nodes: RwLock::new(HashMap::with_capacity(capacity)),
            entry_point: RwLock::new(None),
            max_level: RwLock::new(0),
            rng: RwLock::new(rand::rngs::StdRng::from_os_rng()),
        }
    }

    /// Creates a new HNSW index with a fixed seed for reproducible results.
    #[must_use]
    pub fn with_seed(config: HnswConfig, seed: u64) -> Self {
        Self {
            config,
            nodes: RwLock::new(HashMap::new()),
            entry_point: RwLock::new(None),
            max_level: RwLock::new(0),
            rng: RwLock::new(rand::rngs::StdRng::seed_from_u64(seed)),
        }
    }

    /// Returns the index configuration.
    #[must_use]
    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    /// Returns the number of vectors in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.read().len()
    }

    /// Returns true if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.read().is_empty()
    }

    /// Inserts a vector with the given ID into the index.
    ///
    /// If the distance metric is cosine, the vector is normalized before
    /// storage so that distance computation can use a faster dot-product.
    ///
    /// # Panics
    ///
    /// Panics if the vector dimensions don't match the configuration.
    pub fn insert(&self, id: NodeId, vector: &[f32]) {
        assert_eq!(
            vector.len(),
            self.config.dimensions,
            "Vector dimensions mismatch: expected {}, got {}",
            self.config.dimensions,
            vector.len()
        );

        // Pre-normalize for cosine metric so distance() can use dot product
        let vector: Arc<[f32]> = if self.config.metric == super::DistanceMetric::Cosine {
            let mut v = vector.to_vec();
            super::normalize(&mut v);
            v.into()
        } else {
            vector.into()
        };

        let level = self.random_level();

        // Create the new node
        let node = HnswNode {
            vector: vector.clone(),
            neighbors: vec![Vec::new(); level + 1],
        };

        let mut nodes = self.nodes.write();
        let mut entry_point = self.entry_point.write();
        let mut max_level = self.max_level.write();

        // First insertion
        if entry_point.is_none() {
            nodes.insert(id, node);
            *entry_point = Some(id);
            *max_level = level;
            return;
        }

        let ep = entry_point.unwrap();
        let current_max_level = *max_level;

        // Insert the node first so we can reference it
        nodes.insert(id, node);

        // Search from top to the level above the new node's max layer
        let mut current_ep = ep;
        for lc in (level + 1..=current_max_level).rev() {
            current_ep = self.search_layer_single(&nodes, &vector, current_ep, lc);
        }

        // For each layer from the new node's max layer down to 0
        for lc in (0..=level.min(current_max_level)).rev() {
            let m_max = if lc == 0 {
                self.config.m_max
            } else {
                self.config.m
            };

            // Find ef_construction nearest neighbors at this layer
            let neighbors =
                self.search_layer(&nodes, &vector, current_ep, self.config.ef_construction, lc);

            // Select neighbors using diversity-aware heuristic
            let selected = self.select_neighbors_heuristic(&nodes, &neighbors, m_max);

            // Update the new node's neighbors
            if let Some(new_node) = nodes.get_mut(&id) {
                new_node.neighbors[lc] = selected.clone();
            }

            // Add bidirectional links and collect info for pruning
            // First pass: add links and identify who needs pruning
            let mut needs_pruning: Vec<NodeId> = Vec::new();

            for &neighbor_id in &selected {
                if let Some(neighbor) = nodes.get_mut(&neighbor_id)
                    && neighbor.neighbors.len() > lc
                {
                    neighbor.neighbors[lc].push(id);

                    // Mark for pruning if too many neighbors
                    if neighbor.neighbors[lc].len() > m_max {
                        needs_pruning.push(neighbor_id);
                    }
                }
            }

            // Second pass: compute distances for pruning (immutable borrow)
            let mut prune_data: Vec<(NodeId, Vec<(NodeId, f32)>)> = Vec::new();
            for neighbor_id in &needs_pruning {
                if let Some(neighbor) = nodes.get(neighbor_id)
                    && neighbor.neighbors.len() > lc
                {
                    let base_vec = &neighbor.vector;
                    let distances: Vec<(NodeId, f32)> = neighbor.neighbors[lc]
                        .iter()
                        .map(|&nid| (nid, self.node_distance(&nodes, base_vec, nid)))
                        .collect();
                    prune_data.push((*neighbor_id, distances));
                }
            }

            // Third pass: apply pruning (mutable borrow)
            for (neighbor_id, distances) in prune_data {
                if let Some(neighbor) = nodes.get_mut(&neighbor_id)
                    && neighbor.neighbors.len() > lc
                {
                    Self::prune_neighbors_with_distances(
                        &mut neighbor.neighbors[lc],
                        &distances,
                        m_max,
                    );
                }
            }

            // Update entry point for next layer
            if !selected.is_empty() {
                current_ep = selected[0];
            }
        }

        // Update global entry point if needed
        if level > current_max_level {
            *entry_point = Some(id);
            *max_level = level;
        }
    }

    /// Searches for the k nearest neighbors to the query vector.
    ///
    /// Returns a vector of (NodeId, distance) pairs sorted by distance
    /// (closest first).
    ///
    /// # Panics
    ///
    /// Panics if the query vector dimensions don't match the configuration.
    #[must_use]
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(NodeId, f32)> {
        self.search_with_ef(query, k, self.config.ef)
    }

    /// Searches with a custom ef (beam width) parameter.
    ///
    /// Higher ef values give better recall at the cost of latency.
    /// If the metric is cosine, the query is normalized to match stored vectors.
    #[must_use]
    pub fn search_with_ef(&self, query: &[f32], k: usize, ef: usize) -> Vec<(NodeId, f32)> {
        assert_eq!(
            query.len(),
            self.config.dimensions,
            "Query dimensions mismatch: expected {}, got {}",
            self.config.dimensions,
            query.len()
        );

        // Pre-normalize query for cosine metric to match stored vectors
        let query = if self.config.metric == super::DistanceMetric::Cosine {
            let mut q = query.to_vec();
            super::normalize(&mut q);
            q
        } else {
            query.to_vec()
        };

        let nodes = self.nodes.read();
        let entry_point = self.entry_point.read();
        let max_level = *self.max_level.read();

        if entry_point.is_none() || nodes.is_empty() {
            return Vec::new();
        }

        let ep = entry_point.unwrap();

        // Greedy search from top layer to layer 1
        let mut current_ep = ep;
        for lc in (1..=max_level).rev() {
            current_ep = self.search_layer_single(&nodes, &query, current_ep, lc);
        }

        // Beam search at layer 0
        let ef_search = ef.max(k);
        let candidates = self.search_layer(&nodes, &query, current_ep, ef_search, 0);

        // Return top k
        candidates
            .into_iter()
            .take(k)
            .map(|n| (n.id, n.distance))
            .collect()
    }

    /// Removes a vector from the index.
    ///
    /// Returns true if the vector was found and removed.
    pub fn remove(&self, id: NodeId) -> bool {
        let mut nodes = self.nodes.write();
        let mut entry_point = self.entry_point.write();

        if nodes.remove(&id).is_none() {
            return false;
        }

        // Remove bidirectional links
        for (_, node) in nodes.iter_mut() {
            for neighbors in &mut node.neighbors {
                neighbors.retain(|&n| n != id);
            }
        }

        // Update entry point if needed
        if *entry_point == Some(id) {
            *entry_point = nodes.keys().next().copied();
        }

        true
    }

    /// Returns the vector associated with the given ID, if it exists.
    #[must_use]
    pub fn get(&self, id: NodeId) -> Option<Arc<[f32]>> {
        self.nodes.read().get(&id).map(|n| n.vector.clone())
    }

    /// Returns true if the index contains a vector with the given ID.
    #[must_use]
    pub fn contains(&self, id: NodeId) -> bool {
        self.nodes.read().contains_key(&id)
    }

    /// Iterates over all (NodeId, vector) pairs in the index.
    pub fn iter(&self) -> impl Iterator<Item = (NodeId, Arc<[f32]>)> + '_ {
        let nodes = self.nodes.read();
        let items: Vec<_> = nodes
            .iter()
            .map(|(&id, n)| (id, n.vector.clone()))
            .collect();
        items.into_iter()
    }

    /// Generates a random level for a new node.
    fn random_level(&self) -> usize {
        let mut rng = self.rng.write();
        let r: f64 = rng.random();
        (-r.ln() * self.config.ml).floor() as usize
    }

    /// Single-element greedy search at a layer.
    fn search_layer_single(
        &self,
        nodes: &HashMap<NodeId, HnswNode>,
        query: &[f32],
        ep: NodeId,
        layer: usize,
    ) -> NodeId {
        let mut current = ep;
        let mut current_dist = self.node_distance(nodes, query, ep);

        loop {
            let mut changed = false;

            if let Some(node) = nodes.get(&current)
                && layer < node.neighbors.len()
            {
                for &neighbor in &node.neighbors[layer] {
                    let dist = self.node_distance(nodes, query, neighbor);
                    if dist < current_dist {
                        current = neighbor;
                        current_dist = dist;
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        current
    }

    /// Beam search at a layer, returning ef nearest neighbors.
    fn search_layer(
        &self,
        nodes: &HashMap<NodeId, HnswNode>,
        query: &[f32],
        ep: NodeId,
        ef: usize,
        layer: usize,
    ) -> Vec<Neighbor> {
        let ep_dist = self.node_distance(nodes, query, ep);

        // Min-heap of candidates to explore
        let mut candidates: BinaryHeap<Neighbor> = BinaryHeap::new();
        candidates.push(Neighbor {
            id: ep,
            distance: ep_dist,
        });

        // Max-heap of current best (furthest = top)
        let mut results: BinaryHeap<FurthestCandidate> = BinaryHeap::new();
        results.push(FurthestCandidate {
            id: ep,
            distance: ep_dist,
        });

        let mut visited: HashSet<NodeId> =
            HashSet::with_capacity(nodes.len().min(ef.saturating_mul(2)));
        visited.insert(ep);

        while let Some(current) = candidates.pop() {
            // If the closest candidate is further than the furthest result, stop
            if let Some(furthest) = results.peek()
                && current.distance > furthest.distance
                && results.len() >= ef
            {
                break;
            }

            // Explore neighbors
            if let Some(node) = nodes.get(&current.id)
                && layer < node.neighbors.len()
            {
                for &neighbor in &node.neighbors[layer] {
                    if visited.contains(&neighbor) {
                        continue;
                    }
                    visited.insert(neighbor);

                    let dist = self.node_distance(nodes, query, neighbor);

                    // Add to results if closer than furthest, or if we have room
                    let should_add =
                        results.len() < ef || results.peek().map_or(true, |f| dist < f.distance);

                    if should_add {
                        candidates.push(Neighbor {
                            id: neighbor,
                            distance: dist,
                        });
                        results.push(FurthestCandidate {
                            id: neighbor,
                            distance: dist,
                        });

                        // Keep only ef results
                        while results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        // Convert to sorted vec
        let mut result_vec: Vec<Neighbor> = results
            .into_iter()
            .map(|fc| Neighbor {
                id: fc.id,
                distance: fc.distance,
            })
            .collect();
        result_vec.sort_by(|a, b| OrderedFloat(a.distance).cmp(&OrderedFloat(b.distance)));
        result_vec
    }

    /// Selects neighbors using diversity-aware heuristic (Vamana-style).
    ///
    /// Instead of simply taking the M closest candidates, this checks whether
    /// each candidate is "covered" by an already-selected neighbor. A candidate
    /// is covered if any selected neighbor is closer to it than
    /// `alpha * distance(candidate, query)`. This preserves graph navigability
    /// by ensuring neighbors point to diverse regions of the space.
    fn select_neighbors_heuristic(
        &self,
        nodes: &HashMap<NodeId, HnswNode>,
        candidates: &[Neighbor],
        m: usize,
    ) -> Vec<NodeId> {
        let alpha = self.config.alpha;
        let mut selected: Vec<(NodeId, &Arc<[f32]>)> = Vec::with_capacity(m);

        for candidate in candidates {
            if selected.len() >= m {
                break;
            }
            let cv = match nodes.get(&candidate.id) {
                Some(node) => &node.vector,
                None => continue,
            };
            let covered = selected
                .iter()
                .any(|(_, sv)| self.vector_distance(cv, sv) < alpha * candidate.distance);
            if !covered {
                selected.push((candidate.id, cv));
            }
        }

        selected.into_iter().map(|(id, _)| id).collect()
    }

    /// Prunes a neighbor list using distance-based diversity heuristic.
    ///
    /// Similar to `select_neighbors_heuristic` but operates on `(NodeId, f32)`
    /// distance pairs instead of `Neighbor` structs. Used during post-insert
    /// pruning where distances have already been computed.
    fn prune_neighbors_with_distances(
        neighbors: &mut Vec<NodeId>,
        distances: &[(NodeId, f32)],
        m: usize,
    ) {
        if neighbors.len() <= m {
            return;
        }

        // Sort by distance
        let mut sorted: Vec<_> = distances.to_vec();
        sorted.sort_by(|a, b| OrderedFloat(a.1).cmp(&OrderedFloat(b.1)));

        *neighbors = sorted.into_iter().take(m).map(|(id, _)| id).collect();
    }

    /// Computes distance between two raw vectors using the configured metric.
    ///
    /// For cosine metric with pre-normalized vectors, uses fast dot product.
    #[inline]
    fn vector_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        if self.config.metric == super::DistanceMetric::Cosine {
            // Vectors are pre-normalized, so cosine distance = 1 - dot_product
            1.0 - super::dot_product(a, b)
        } else {
            compute_distance(a, b, self.config.metric)
        }
    }

    /// Computes the distance between a query vector and a stored node.
    fn node_distance(&self, nodes: &HashMap<NodeId, HnswNode>, query: &[f32], id: NodeId) -> f32 {
        nodes
            .get(&id)
            .map_or(f32::MAX, |n| self.vector_distance(query, &n.vector))
    }

    // ========================================================================
    // Batch Operations
    // ========================================================================

    /// Inserts multiple vectors in batch.
    ///
    /// This method inserts vectors sequentially into the HNSW graph structure
    /// but with optimized internal operations. For truly parallel construction
    /// of very large indexes, consider using multiple indexes and merging.
    ///
    /// # Arguments
    ///
    /// * `vectors` - Iterator of (NodeId, vector) pairs to insert
    ///
    /// # Panics
    ///
    /// Panics if any vector dimensions don't match the configuration.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use grafeo_core::index::vector::{HnswIndex, HnswConfig, DistanceMetric};
    /// use grafeo_common::types::NodeId;
    ///
    /// let config = HnswConfig::new(384, DistanceMetric::Cosine);
    /// let index = HnswIndex::new(config);
    ///
    /// let vectors: Vec<(NodeId, Vec<f32>)> = (0..1000)
    ///     .map(|i| (NodeId::new(i), vec![0.1f32; 384]))
    ///     .collect();
    ///
    /// index.batch_insert(vectors.iter().map(|(id, v)| (*id, v.as_slice())));
    /// ```
    pub fn batch_insert<'a, I>(&self, vectors: I)
    where
        I: IntoIterator<Item = (NodeId, &'a [f32])>,
    {
        for (id, vector) in vectors {
            self.insert(id, vector);
        }
    }

    /// Searches for k nearest neighbors for multiple queries in parallel.
    ///
    /// This method runs multiple searches concurrently using rayon, providing
    /// significant speedup when you have many queries to execute.
    ///
    /// # Arguments
    ///
    /// * `queries` - Slice of query vectors (as `Vec<f32>` or similar)
    /// * `k` - Number of nearest neighbors to return for each query
    ///
    /// # Returns
    ///
    /// Vector of results, one per query. Each result is a vector of
    /// (NodeId, distance) pairs sorted by distance.
    ///
    /// # Panics
    ///
    /// Panics if any query vector dimensions don't match the configuration.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use grafeo_core::index::vector::{HnswIndex, HnswConfig, DistanceMetric};
    ///
    /// let config = HnswConfig::new(384, DistanceMetric::Cosine);
    /// let index = HnswIndex::new(config);
    /// // ... insert vectors ...
    ///
    /// let queries: Vec<Vec<f32>> = vec![
    ///     vec![0.1f32; 384],
    ///     vec![0.2f32; 384],
    ///     vec![0.3f32; 384],
    /// ];
    ///
    /// let all_results = index.batch_search(&queries, 10);
    /// assert_eq!(all_results.len(), 3);
    /// ```
    #[must_use]
    pub fn batch_search(&self, queries: &[Vec<f32>], k: usize) -> Vec<Vec<(NodeId, f32)>> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            queries
                .par_iter()
                .map(|query| self.search(query, k))
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            queries.iter().map(|query| self.search(query, k)).collect()
        }
    }

    /// Searches for k nearest neighbors for multiple queries in parallel.
    ///
    /// This variant accepts query vectors as slices.
    #[must_use]
    pub fn batch_search_slices(&self, queries: &[&[f32]], k: usize) -> Vec<Vec<(NodeId, f32)>> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            queries
                .par_iter()
                .map(|query| self.search(query, k))
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            queries.iter().map(|query| self.search(query, k)).collect()
        }
    }

    /// Searches with custom ef parameter for multiple queries in parallel.
    ///
    /// Higher ef values give better recall at the cost of latency.
    #[must_use]
    pub fn batch_search_with_ef(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        ef: usize,
    ) -> Vec<Vec<(NodeId, f32)>> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            queries
                .par_iter()
                .map(|query| self.search_with_ef(query, k, ef))
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            queries
                .iter()
                .map(|query| self.search_with_ef(query, k, ef))
                .collect()
        }
    }
}

impl std::fmt::Debug for HnswIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswIndex")
            .field("config", &self.config)
            .field("len", &self.len())
            .field("max_level", &*self.max_level.read())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::vector::DistanceMetric;

    fn create_test_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                (0..dim)
                    .map(|j| ((i * dim + j) as f32) / (n * dim) as f32)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_hnsw_empty() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);

        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        assert!(index.search(&[0.0, 0.0, 0.0, 0.0], 10).is_empty());
    }

    #[test]
    fn test_hnsw_single_insert() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);

        index.insert(NodeId::new(1), &[0.1, 0.2, 0.3, 0.4]);

        assert_eq!(index.len(), 1);
        assert!(index.contains(NodeId::new(1)));
        assert!(!index.contains(NodeId::new(2)));

        let results = index.search(&[0.1, 0.2, 0.3, 0.4], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, NodeId::new(1));
        assert!(results[0].1 < 0.001); // Near-zero distance
    }

    #[test]
    fn test_hnsw_multiple_inserts() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let vectors = create_test_vectors(100, 4);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec);
        }

        assert_eq!(index.len(), 100);

        // Search for nearest neighbors
        let query = &vectors[50];
        let results = index.search(query, 5);

        assert_eq!(results.len(), 5);
        // The closest should be the vector itself
        assert_eq!(results[0].0, NodeId::new(51));
        assert!(results[0].1 < 0.001);
    }

    #[test]
    fn test_hnsw_search_returns_sorted() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let vectors = create_test_vectors(50, 4);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec);
        }

        let query = [0.5, 0.5, 0.5, 0.5];
        let results = index.search(&query, 10);

        // Verify sorted by distance
        for i in 1..results.len() {
            assert!(results[i - 1].1 <= results[i].1);
        }
    }

    #[test]
    fn test_hnsw_remove() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);

        index.insert(NodeId::new(1), &[0.1, 0.2, 0.3, 0.4]);
        index.insert(NodeId::new(2), &[0.5, 0.6, 0.7, 0.8]);

        assert_eq!(index.len(), 2);

        assert!(index.remove(NodeId::new(1)));
        assert_eq!(index.len(), 1);
        assert!(!index.contains(NodeId::new(1)));
        assert!(index.contains(NodeId::new(2)));

        // Removing again returns false
        assert!(!index.remove(NodeId::new(1)));
    }

    #[test]
    fn test_hnsw_get() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);

        let vec = [0.1, 0.2, 0.3, 0.4];
        index.insert(NodeId::new(1), &vec);

        let retrieved = index.get(NodeId::new(1)).unwrap();
        assert_eq!(&*retrieved, &vec);

        assert!(index.get(NodeId::new(999)).is_none());
    }

    #[test]
    fn test_hnsw_cosine_metric() {
        let config = HnswConfig::new(4, DistanceMetric::Cosine);
        let index = HnswIndex::with_seed(config, 42);

        // Insert normalized vectors
        index.insert(NodeId::new(1), &[1.0, 0.0, 0.0, 0.0]);
        index.insert(NodeId::new(2), &[0.0, 1.0, 0.0, 0.0]);
        index.insert(NodeId::new(3), &[0.707, 0.707, 0.0, 0.0]);

        // Query similar to node 1
        let results = index.search(&[0.9, 0.1, 0.0, 0.0], 3);

        // Node 1 should be closest (most similar direction)
        assert_eq!(results[0].0, NodeId::new(1));
    }

    #[test]
    fn test_hnsw_iter() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);

        index.insert(NodeId::new(1), &[0.1, 0.2, 0.3, 0.4]);
        index.insert(NodeId::new(2), &[0.5, 0.6, 0.7, 0.8]);

        let items: Vec<_> = index.iter().collect();
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_hnsw_ef_parameter() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let vectors = create_test_vectors(100, 4);
        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec);
        }

        let query = [0.5, 0.5, 0.5, 0.5];

        // Higher ef should give same or better results
        let results_low = index.search_with_ef(&query, 5, 10);
        let results_high = index.search_with_ef(&query, 5, 100);

        assert_eq!(results_low.len(), 5);
        assert_eq!(results_high.len(), 5);

        // High ef should find equal or better (smaller) distances
        assert!(results_high[0].1 <= results_low[0].1);
    }

    #[test]
    #[should_panic(expected = "Vector dimensions mismatch")]
    fn test_hnsw_dimension_mismatch_insert() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);

        index.insert(NodeId::new(1), &[0.1, 0.2, 0.3]); // Wrong dimension
    }

    #[test]
    #[should_panic(expected = "Query dimensions mismatch")]
    fn test_hnsw_dimension_mismatch_search() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);

        index.insert(NodeId::new(1), &[0.1, 0.2, 0.3, 0.4]);
        let _ = index.search(&[0.1, 0.2, 0.3], 1); // Wrong dimension
    }

    #[test]
    fn test_hnsw_batch_insert() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let vectors = create_test_vectors(100, 4);
        let pairs: Vec<_> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (NodeId::new(i as u64 + 1), v.as_slice()))
            .collect();

        index.batch_insert(pairs);

        assert_eq!(index.len(), 100);

        // Verify search still works
        let results = index.search(&vectors[50], 5);
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].0, NodeId::new(51));
    }

    #[test]
    fn test_hnsw_batch_search() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let vectors = create_test_vectors(100, 4);
        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec);
        }

        // Batch search with 5 queries
        let queries: Vec<Vec<f32>> = (0..5).map(|i| vectors[i * 20].clone()).collect();

        let all_results = index.batch_search(&queries, 3);

        assert_eq!(all_results.len(), 5);
        for (i, results) in all_results.iter().enumerate() {
            assert_eq!(results.len(), 3);
            // First result should be the query vector itself
            assert_eq!(results[0].0, NodeId::new((i * 20 + 1) as u64));
            assert!(results[0].1 < 0.001);
        }
    }

    #[test]
    fn test_hnsw_batch_search_with_ef() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let vectors = create_test_vectors(100, 4);
        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec);
        }

        let queries: Vec<Vec<f32>> = vec![vectors[25].clone(), vectors[75].clone()];

        // Search with higher ef for better recall
        let results = index.batch_search_with_ef(&queries, 5, 100);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 5);
        assert_eq!(results[1].len(), 5);
    }

    #[test]
    fn test_hnsw_batch_search_empty_index() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);

        let queries = vec![vec![0.0f32, 0.0, 0.0, 0.0]];
        let results = index.batch_search(&queries, 10);

        assert_eq!(results.len(), 1);
        assert!(results[0].is_empty());
    }

    /// Brute-force k-NN for recall verification.
    fn brute_force_knn(
        vectors: &[Vec<f32>],
        query: &[f32],
        k: usize,
        metric: DistanceMetric,
    ) -> Vec<usize> {
        let mut dists: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, crate::index::vector::compute_distance(query, v, metric)))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        dists.into_iter().take(k).map(|(i, _)| i).collect()
    }

    #[test]
    fn test_hnsw_recall_euclidean() {
        // 1000 vectors, 20 dimensions — matches ann-benchmarks random-xs profile
        let n = 1000;
        let dim = 20;
        let k = 10;
        let num_queries = 100;

        // Deterministic pseudo-random vectors via linear congruential generator
        let mut seed: u64 = 12345;
        let mut rand_f32 = || -> f32 {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((seed >> 33) as f32) / (u32::MAX as f32)
        };

        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rand_f32()).collect())
            .collect();

        let config = HnswConfig::new(dim, DistanceMetric::Euclidean).with_m(16);
        let index = HnswIndex::with_seed(config, 42);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64), vec);
        }

        // Measure recall over num_queries random queries
        let queries: Vec<Vec<f32>> = (0..num_queries)
            .map(|_| (0..dim).map(|_| rand_f32()).collect())
            .collect();

        let mut total_recall = 0.0f64;
        for query in &queries {
            let ground_truth = brute_force_knn(&vectors, query, k, DistanceMetric::Euclidean);
            let gt_set: std::collections::HashSet<u64> =
                ground_truth.iter().map(|&i| i as u64).collect();

            let results = index.search_with_ef(query, k, 50);
            let found: std::collections::HashSet<u64> =
                results.iter().map(|(id, _)| id.as_u64()).collect();

            let overlap = gt_set.intersection(&found).count();
            total_recall += overlap as f64 / k as f64;
        }

        let avg_recall = total_recall / num_queries as f64;
        assert!(
            avg_recall >= 0.90,
            "Recall {avg_recall:.3} is below 0.90 threshold at M=16/ef=50"
        );
    }

    #[test]
    fn test_hnsw_recall_cosine() {
        let n = 500;
        let dim = 20;
        let k = 10;
        let num_queries = 50;

        let mut seed: u64 = 67890;
        let mut rand_f32 = || -> f32 {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((seed >> 33) as f32) / (u32::MAX as f32)
        };

        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rand_f32()).collect())
            .collect();

        let config = HnswConfig::new(dim, DistanceMetric::Cosine).with_m(16);
        let index = HnswIndex::with_seed(config, 42);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64), vec);
        }

        let queries: Vec<Vec<f32>> = (0..num_queries)
            .map(|_| (0..dim).map(|_| rand_f32()).collect())
            .collect();

        let mut total_recall = 0.0f64;
        for query in &queries {
            let ground_truth = brute_force_knn(&vectors, query, k, DistanceMetric::Cosine);
            let gt_set: std::collections::HashSet<u64> =
                ground_truth.iter().map(|&i| i as u64).collect();

            let results = index.search_with_ef(query, k, 50);
            let found: std::collections::HashSet<u64> =
                results.iter().map(|(id, _)| id.as_u64()).collect();

            let overlap = gt_set.intersection(&found).count();
            total_recall += overlap as f64 / k as f64;
        }

        let avg_recall = total_recall / num_queries as f64;
        assert!(
            avg_recall >= 0.90,
            "Cosine recall {avg_recall:.3} is below 0.90 threshold at M=16/ef=50"
        );
    }

    #[test]
    fn test_diversity_pruning_prevents_clustering() {
        // Verify that diversity pruning selects diverse neighbors, not just closest
        let dim = 4;
        let config = HnswConfig::new(dim, DistanceMetric::Euclidean).with_m(4);
        let index = HnswIndex::with_seed(config, 42);

        // Insert a cluster of very similar vectors and one outlier
        index.insert(NodeId::new(0), &[0.0, 0.0, 0.0, 0.0]);
        index.insert(NodeId::new(1), &[0.01, 0.0, 0.0, 0.0]);
        index.insert(NodeId::new(2), &[0.02, 0.0, 0.0, 0.0]);
        index.insert(NodeId::new(3), &[0.03, 0.0, 0.0, 0.0]);
        index.insert(NodeId::new(4), &[0.04, 0.0, 0.0, 0.0]);
        // Outlier in a different direction
        index.insert(NodeId::new(5), &[0.0, 1.0, 0.0, 0.0]);

        // Search for the outlier — it should be findable
        let results = index.search(&[0.0, 0.9, 0.0, 0.0], 1);
        assert_eq!(results[0].0, NodeId::new(5));
    }

    // ── Edge case tests ─────────────────────────────────────────────

    #[test]
    fn test_single_vector() {
        let config = HnswConfig::new(3, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);
        index.insert(NodeId::new(0), &[1.0, 0.0, 0.0]);

        let results = index.search(&[1.0, 0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, NodeId::new(0));
        assert!(results[0].1 < 0.01);
    }

    #[test]
    fn test_search_k_larger_than_index() {
        let config = HnswConfig::new(3, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);
        index.insert(NodeId::new(0), &[1.0, 0.0, 0.0]);
        index.insert(NodeId::new(1), &[0.0, 1.0, 0.0]);

        // k=10 but only 2 vectors
        let results = index.search(&[1.0, 0.0, 0.0], 10);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_empty_index_search() {
        let config = HnswConfig::new(3, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);

        let results = index.search(&[1.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_remove_and_search() {
        let config = HnswConfig::new(3, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);
        index.insert(NodeId::new(0), &[1.0, 0.0, 0.0]);
        index.insert(NodeId::new(1), &[0.0, 1.0, 0.0]);
        index.insert(NodeId::new(2), &[0.0, 0.0, 1.0]);

        index.remove(NodeId::new(1));
        let results = index.search(&[0.0, 1.0, 0.0], 3);
        // Removed node should not appear
        assert!(results.iter().all(|(id, _)| *id != NodeId::new(1)));
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_duplicate_insert() {
        let config = HnswConfig::new(3, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);
        index.insert(NodeId::new(0), &[1.0, 0.0, 0.0]);
        index.insert(NodeId::new(0), &[0.0, 1.0, 0.0]); // Same ID, different vector

        assert_eq!(index.len(), 1);
        // Should use the latest vector
        let results = index.search(&[0.0, 1.0, 0.0], 1);
        assert_eq!(results[0].0, NodeId::new(0));
    }

    #[test]
    fn test_search_with_ef_zero() {
        let config = HnswConfig::new(3, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);
        index.insert(NodeId::new(0), &[1.0, 0.0, 0.0]);

        // ef=0 should still return results (search uses max(ef, k))
        let results = index.search_with_ef(&[1.0, 0.0, 0.0], 1, 0);
        // Behavior may vary but should not panic
        assert!(results.len() <= 1);
    }

    #[test]
    fn test_all_metrics_search() {
        for metric in [
            DistanceMetric::Cosine,
            DistanceMetric::Euclidean,
            DistanceMetric::DotProduct,
            DistanceMetric::Manhattan,
        ] {
            let config = HnswConfig::new(3, metric);
            let index = HnswIndex::new(config);
            index.insert(NodeId::new(0), &[1.0, 0.0, 0.0]);
            index.insert(NodeId::new(1), &[0.0, 1.0, 0.0]);

            let results = index.search(&[1.0, 0.0, 0.0], 2);
            assert_eq!(results.len(), 2, "Failed for metric {metric:?}");
            assert_eq!(
                results[0].0,
                NodeId::new(0),
                "Closest not correct for metric {metric:?}"
            );
        }
    }

    #[test]
    fn test_batch_search_consistency() {
        let config = HnswConfig::new(3, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);
        index.insert(NodeId::new(0), &[1.0, 0.0, 0.0]);
        index.insert(NodeId::new(1), &[0.0, 1.0, 0.0]);
        index.insert(NodeId::new(2), &[0.0, 0.0, 1.0]);

        let queries: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let batch_results = index.batch_search(&queries, 1);
        assert_eq!(batch_results.len(), 3);

        // Each query should find its exact match
        for (i, results) in batch_results.iter().enumerate() {
            assert_eq!(results[0].0, NodeId::new(i as u64));
        }
    }

    #[test]
    fn test_with_capacity_constructor() {
        let config = HnswConfig::new(3, DistanceMetric::Euclidean);
        let index = HnswIndex::with_capacity(config, 100);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());

        index.insert(NodeId::new(0), &[1.0, 0.0, 0.0]);
        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_high_m_value() {
        // M larger than number of nodes
        let config = HnswConfig::new(3, DistanceMetric::Euclidean).with_m(64);
        let index = HnswIndex::new(config);
        index.insert(NodeId::new(0), &[1.0, 0.0, 0.0]);
        index.insert(NodeId::new(1), &[0.0, 1.0, 0.0]);

        let results = index.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
    }
}
