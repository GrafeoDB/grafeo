//! Maximal Marginal Relevance (MMR) for diversity-aware vector search.
//!
//! MMR balances relevance (similarity to query) with diversity (dissimilarity
//! among selected results). Used by LangChain-style vector stores for
//! retrieval-augmented generation (RAG).
//!
//! # Algorithm
//!
//! At each step, the candidate with the highest MMR score is selected:
//!
//! ```text
//! score(doc) = λ * sim(query, doc) - (1 - λ) * max(sim(doc, selected))
//! ```
//!
//! where λ controls the relevance-diversity tradeoff.
//!
//! # References
//!
//! Carbonell & Goldstein, "The Use of MMR, Diversity-Based Reranking for
//! Reordering Documents and Producing Summaries" (1998).

use super::{DistanceMetric, compute_distance};
use grafeo_common::types::NodeId;

/// Converts a distance value to a similarity value for the MMR formula.
///
/// All Grafeo distance functions return lower-is-better values.
/// The MMR formula needs higher-is-better similarities.
#[inline]
fn distance_to_similarity(distance: f32, metric: DistanceMetric) -> f32 {
    match metric {
        // cosine_distance = 1 - cosine_similarity
        DistanceMetric::Cosine => 1.0 - distance,
        // Bounded transform: identical vectors → 1.0, far → ~0.0
        DistanceMetric::Euclidean | DistanceMetric::Manhattan => 1.0 / (1.0 + distance),
        // distance = -dot_product, so similarity = -distance
        DistanceMetric::DotProduct => -distance,
    }
}

/// Selects k items from candidates using Maximal Marginal Relevance.
///
/// Greedy iterative: at each step, picks the candidate with the highest
/// MMR score and adds it to the selected set.
///
/// # Arguments
///
/// * `query` - The query vector
/// * `candidates` - `(NodeId, query_distance, vector)` triples. The distance
///   is precomputed from the query using the same metric.
/// * `k` - Number of items to select
/// * `lambda` - Trade-off in \[0, 1\]: 1.0 = pure relevance, 0.0 = pure diversity
/// * `metric` - Distance metric (for computing inter-document similarity)
///
/// # Returns
///
/// `(NodeId, distance)` pairs in MMR selection order. The f32 is the **original
/// distance** from the query (not the MMR score), matching `vector_search` output.
///
/// # Complexity
///
/// O(k * n) where n = candidates.len(). Negligible for typical k=4, n=20.
#[must_use]
pub fn mmr_select(
    query: &[f32],
    candidates: &[(NodeId, f32, &[f32])],
    k: usize,
    lambda: f32,
    metric: DistanceMetric,
) -> Vec<(NodeId, f32)> {
    if candidates.is_empty() || k == 0 {
        return Vec::new();
    }

    let k = k.min(candidates.len());
    let lambda = lambda.clamp(0.0, 1.0);
    let _ = query; // query distances are precomputed in candidates

    // Precompute query-similarity for each candidate (from precomputed distances)
    let query_similarities: Vec<f32> = candidates
        .iter()
        .map(|(_, dist, _)| distance_to_similarity(*dist, metric))
        .collect();

    let mut selected_indices: Vec<usize> = Vec::with_capacity(k);
    let mut remaining: Vec<usize> = (0..candidates.len()).collect();

    for _ in 0..k {
        let mut best_pos = 0;
        let mut best_mmr = f32::NEG_INFINITY;

        for (pos, &cand_idx) in remaining.iter().enumerate() {
            let relevance = query_similarities[cand_idx];

            let max_sim_to_selected = if selected_indices.is_empty() {
                0.0
            } else {
                selected_indices
                    .iter()
                    .map(|&sel_idx| {
                        let dist =
                            compute_distance(candidates[cand_idx].2, candidates[sel_idx].2, metric);
                        distance_to_similarity(dist, metric)
                    })
                    .fold(f32::NEG_INFINITY, f32::max)
            };

            let mmr_score = lambda * relevance - (1.0 - lambda) * max_sim_to_selected;

            if mmr_score > best_mmr {
                best_mmr = mmr_score;
                best_pos = pos;
            }
        }

        let chosen = remaining.swap_remove(best_pos);
        selected_indices.push(chosen);
    }

    selected_indices
        .iter()
        .map(|&idx| (candidates[idx].0, candidates[idx].1))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_candidates() {
        let result = mmr_select(&[1.0, 0.0], &[], 5, 0.5, DistanceMetric::Euclidean);
        assert!(result.is_empty(), "{result:?}");
    }

    #[test]
    fn test_k_zero() {
        let v = [1.0f32, 0.0];
        let candidates = vec![(NodeId::new(1), 0.0, v.as_slice())];
        let result = mmr_select(&[1.0, 0.0], &candidates, 0, 0.5, DistanceMetric::Euclidean);
        assert!(result.is_empty(), "{result:?}");
    }

    #[test]
    fn test_lambda_one_is_pure_relevance() {
        let query = [1.0f32, 0.0, 0.0];
        let v1 = [0.9f32, 0.1, 0.0]; // closest
        let v2 = [0.5f32, 0.5, 0.0]; // middle
        let v3 = [0.0f32, 1.0, 0.0]; // farthest

        let d1 = compute_distance(&query, &v1, DistanceMetric::Euclidean);
        let d2 = compute_distance(&query, &v2, DistanceMetric::Euclidean);
        let d3 = compute_distance(&query, &v3, DistanceMetric::Euclidean);

        let candidates = vec![
            (NodeId::new(1), d1, v1.as_slice()),
            (NodeId::new(2), d2, v2.as_slice()),
            (NodeId::new(3), d3, v3.as_slice()),
        ];

        let result = mmr_select(&query, &candidates, 3, 1.0, DistanceMetric::Euclidean);
        assert_eq!(result.len(), 3);
        // First selected should be the closest to query
        assert_eq!(result[0].0, NodeId::new(1));
    }

    #[test]
    fn test_diversity_avoids_redundancy() {
        let query = [1.0f32, 0.0, 0.0];
        let v1 = [0.9f32, 0.1, 0.0]; // close to query
        let v2 = [0.89f32, 0.11, 0.0]; // nearly identical to v1
        let v3 = [0.0f32, 0.0, 1.0]; // very different direction

        let d1 = compute_distance(&query, &v1, DistanceMetric::Euclidean);
        let d2 = compute_distance(&query, &v2, DistanceMetric::Euclidean);
        let d3 = compute_distance(&query, &v3, DistanceMetric::Euclidean);

        let candidates = vec![
            (NodeId::new(1), d1, v1.as_slice()),
            (NodeId::new(2), d2, v2.as_slice()),
            (NodeId::new(3), d3, v3.as_slice()),
        ];

        // With lambda=0.5, after selecting v1, v3 should be preferred over v2
        let result = mmr_select(&query, &candidates, 2, 0.5, DistanceMetric::Euclidean);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, NodeId::new(1)); // most relevant first
        assert_eq!(result[1].0, NodeId::new(3)); // diverse second
    }

    #[test]
    fn test_k_larger_than_candidates() {
        let query = [1.0f32, 0.0];
        let v1 = [0.9f32, 0.1];
        let v2 = [0.5f32, 0.5];

        let d1 = compute_distance(&query, &v1, DistanceMetric::Cosine);
        let d2 = compute_distance(&query, &v2, DistanceMetric::Cosine);

        let candidates = vec![
            (NodeId::new(1), d1, v1.as_slice()),
            (NodeId::new(2), d2, v2.as_slice()),
        ];

        let result = mmr_select(&query, &candidates, 10, 0.5, DistanceMetric::Cosine);
        assert_eq!(result.len(), 2); // capped at candidate count
    }

    #[test]
    fn test_returns_original_distances() {
        let query = [1.0f32, 0.0, 0.0];
        let v1 = [0.9f32, 0.1, 0.0];
        let d1 = compute_distance(&query, &v1, DistanceMetric::Euclidean);

        let candidates = vec![(NodeId::new(1), d1, v1.as_slice())];
        let result = mmr_select(&query, &candidates, 1, 0.5, DistanceMetric::Euclidean);

        assert_eq!(result[0].1, d1);
    }

    #[test]
    fn test_all_metrics() {
        for metric in [
            DistanceMetric::Cosine,
            DistanceMetric::Euclidean,
            DistanceMetric::DotProduct,
            DistanceMetric::Manhattan,
        ] {
            let query = [1.0f32, 0.0, 0.0];
            let v1 = [0.9f32, 0.1, 0.0];
            let v2 = [0.0f32, 1.0, 0.0];
            let d1 = compute_distance(&query, &v1, metric);
            let d2 = compute_distance(&query, &v2, metric);

            let candidates = vec![
                (NodeId::new(1), d1, v1.as_slice()),
                (NodeId::new(2), d2, v2.as_slice()),
            ];

            let result = mmr_select(&query, &candidates, 2, 0.5, metric);
            assert_eq!(result.len(), 2, "failed for metric {metric:?}");
        }
    }

    #[test]
    fn test_distance_to_similarity_cosine() {
        assert!((distance_to_similarity(0.0, DistanceMetric::Cosine) - 1.0).abs() < 1e-6);
        assert!((distance_to_similarity(1.0, DistanceMetric::Cosine) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_distance_to_similarity_euclidean() {
        assert!((distance_to_similarity(0.0, DistanceMetric::Euclidean) - 1.0).abs() < 1e-6);
        assert!(distance_to_similarity(1000.0, DistanceMetric::Euclidean) < 0.01);
    }

    #[test]
    fn test_distance_to_similarity_dot_product() {
        // distance = -dot_product, so similarity = -distance
        assert!((distance_to_similarity(-32.0, DistanceMetric::DotProduct) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_single_candidate() {
        let query = [1.0f32, 0.0];
        let v1 = [0.5f32, 0.5];
        let d1 = compute_distance(&query, &v1, DistanceMetric::Cosine);

        let candidates = vec![(NodeId::new(42), d1, v1.as_slice())];
        let result = mmr_select(&query, &candidates, 1, 0.5, DistanceMetric::Cosine);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, NodeId::new(42));
    }
}
