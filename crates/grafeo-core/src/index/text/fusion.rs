//! Score fusion for combining results from multiple search sources.

use grafeo_common::types::NodeId;
use std::collections::HashMap;

/// Method for fusing scores from multiple search sources.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum FusionMethod {
    /// Reciprocal Rank Fusion, parameter-free, robust default.
    ///
    /// Score = sum(1 / (k + rank_i)) across sources.
    /// Higher k values reduce the impact of high-ranked items.
    Rrf {
        /// Smoothing constant (default 60).
        k: usize,
    },
    /// Weighted score combination.
    ///
    /// Scores from each source are normalized to [0, 1] then
    /// combined with explicit weights.
    Weighted {
        /// Weight for each source (must sum to 1.0).
        weights: Vec<f64>,
    },
}

impl Default for FusionMethod {
    fn default() -> Self {
        Self::Rrf { k: 60 }
    }
}

/// Fuses ranked results from multiple search sources into a single ranking.
///
/// Each source provides a `Vec<(NodeId, f64)>` sorted by score descending.
/// Returns up to `k` results sorted by fused score descending.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "hybrid-search")]
/// # {
/// use grafeo_core::index::text::{FusionMethod, fuse_results};
/// use grafeo_common::types::NodeId;
///
/// let text_results = vec![(NodeId::new(1), 2.5), (NodeId::new(2), 1.8)];
/// let vector_results = vec![(NodeId::new(2), 0.95), (NodeId::new(3), 0.80)];
///
/// let fused = fuse_results(
///     &[text_results, vector_results],
///     &FusionMethod::Rrf { k: 60 },
///     10,
/// );
/// // Node 2 appears in both sources, likely ranked first
/// # }
/// ```
pub fn fuse_results(
    sources: &[Vec<(NodeId, f64)>],
    method: &FusionMethod,
    k: usize,
) -> Vec<(NodeId, f64)> {
    if sources.is_empty() {
        return Vec::new();
    }

    // Single source: just truncate
    if sources.len() == 1 {
        let mut results = sources[0].clone();
        results.truncate(k);
        return results;
    }

    match method {
        FusionMethod::Rrf { k: rrf_k } => fuse_rrf(sources, *rrf_k, k),
        FusionMethod::Weighted { weights } => fuse_weighted(sources, weights, k),
    }
}

/// Reciprocal Rank Fusion.
///
/// For each source, assigns score = 1 / (k + rank) where rank is 1-based.
/// Scores are summed across sources.
fn fuse_rrf(sources: &[Vec<(NodeId, f64)>], rrf_k: usize, k: usize) -> Vec<(NodeId, f64)> {
    let mut scores: HashMap<NodeId, f64> = HashMap::new();

    for source in sources {
        for (rank, (node_id, _score)) in source.iter().enumerate() {
            // rank is 0-based, RRF uses 1-based
            let rrf_score = 1.0 / (rrf_k as f64 + (rank + 1) as f64);
            *scores.entry(*node_id).or_insert(0.0) += rrf_score;
        }
    }

    let mut results: Vec<(NodeId, f64)> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

/// Weighted score fusion with min-max normalization.
///
/// Each source's scores are normalized to [0, 1], then weighted
/// and summed. If a node appears in only some sources, missing
/// sources contribute 0 for that node.
fn fuse_weighted(sources: &[Vec<(NodeId, f64)>], weights: &[f64], k: usize) -> Vec<(NodeId, f64)> {
    let mut scores: HashMap<NodeId, f64> = HashMap::new();

    for (i, source) in sources.iter().enumerate() {
        let weight = weights
            .get(i)
            .copied()
            .unwrap_or(1.0 / sources.len() as f64);

        if source.is_empty() {
            continue;
        }

        // Find min/max for normalization
        let min_score = source.iter().map(|(_, s)| *s).fold(f64::INFINITY, f64::min);
        let max_score = source
            .iter()
            .map(|(_, s)| *s)
            .fold(f64::NEG_INFINITY, f64::max);
        let range = max_score - min_score;

        for (node_id, score) in source {
            let normalized = if range > f64::EPSILON {
                (score - min_score) / range
            } else {
                1.0 // All scores equal
            };
            *scores.entry(*node_id).or_insert(0.0) += weight * normalized;
        }
    }

    let mut results: Vec<(NodeId, f64)> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_basic() {
        let source_a = vec![
            (NodeId::new(1), 10.0),
            (NodeId::new(2), 8.0),
            (NodeId::new(3), 6.0),
        ];
        let source_b = vec![
            (NodeId::new(2), 0.9),
            (NodeId::new(3), 0.8),
            (NodeId::new(4), 0.7),
        ];

        let results = fuse_results(&[source_a, source_b], &FusionMethod::Rrf { k: 60 }, 10);

        // Node 2 appears in both at rank 2 and rank 1 → highest fused score
        assert!(!results.is_empty(), "results is empty");
        // Find node 2's score, should be highest since it's in both lists
        let node2_score = results.iter().find(|(id, _)| *id == NodeId::new(2));
        let node1_score = results.iter().find(|(id, _)| *id == NodeId::new(1));
        assert!(node2_score.is_some());
        assert!(node1_score.is_some());
        assert!(node2_score.unwrap().1 > node1_score.unwrap().1);
    }

    #[test]
    fn test_rrf_single_source() {
        let source = vec![(NodeId::new(1), 10.0), (NodeId::new(2), 8.0)];

        let results = fuse_results(
            std::slice::from_ref(&source),
            &FusionMethod::Rrf { k: 60 },
            10,
        );
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, source[0].0);
    }

    #[test]
    fn test_rrf_k_limit() {
        let source_a = vec![
            (NodeId::new(1), 10.0),
            (NodeId::new(2), 8.0),
            (NodeId::new(3), 6.0),
        ];

        let results = fuse_results(&[source_a], &FusionMethod::Rrf { k: 60 }, 1);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_weighted_basic() {
        let text_results = vec![(NodeId::new(1), 2.5), (NodeId::new(2), 1.0)];
        let vector_results = vec![(NodeId::new(2), 0.95), (NodeId::new(3), 0.80)];

        let results = fuse_results(
            &[text_results, vector_results],
            &FusionMethod::Weighted {
                weights: vec![0.5, 0.5],
            },
            10,
        );

        assert!(!results.is_empty(), "results is empty");
        // Node 2 appears in both sources with good scores
        let node2 = results.iter().find(|(id, _)| *id == NodeId::new(2));
        assert!(node2.is_some());
    }

    #[test]
    fn test_weighted_normalization() {
        // Same scores in both sources, equal weights
        let source_a = vec![(NodeId::new(1), 100.0)];
        let source_b = vec![(NodeId::new(1), 0.01)];

        let results = fuse_results(
            &[source_a, source_b],
            &FusionMethod::Weighted {
                weights: vec![0.5, 0.5],
            },
            10,
        );

        // Single element in each → normalized to 1.0 → fused = 0.5 + 0.5 = 1.0
        assert_eq!(results.len(), 1);
        assert!((results[0].1 - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_empty_sources() {
        let results = fuse_results(&[], &FusionMethod::default(), 10);
        assert!(results.is_empty(), "{results:?}");
    }

    #[test]
    fn test_empty_source_lists() {
        let empty: Vec<(NodeId, f64)> = vec![];
        let non_empty = vec![(NodeId::new(1), 1.0)];

        let results = fuse_results(
            &[empty, non_empty],
            &FusionMethod::Weighted {
                weights: vec![0.5, 0.5],
            },
            10,
        );

        assert_eq!(results.len(), 1);
    }
}
