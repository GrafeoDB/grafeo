//! Score fusion for combining results from multiple search sources.

use crate::index::ranking::by_descending_score;
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
/// Returns up to `k` results sorted by fused score descending, equal
/// scores by ascending node id (a NaN score last), so identical calls give
/// the same list and a cut at `k` inside a tie keeps its lowest ids.
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
    results.sort_by(by_descending_score);
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
    results.sort_by(by_descending_score);
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

    /// The ids `offset + 2 * position` for `count` positions, in a
    /// scrambled order (a multiplicative permutation), so a list's rank
    /// says nothing about its ids.
    fn scrambled_ids(count: u64, offset: u64) -> Vec<NodeId> {
        // 7 is coprime with 200, so `i * 7 % count` visits every position.
        assert_eq!(count, 200, "the permutation below assumes 200 positions");
        (0..count)
            .map(|i| NodeId::new(offset + 2 * ((i * 7) % count)))
            .collect()
    }

    /// The `count` fused results of `sources`, for every one of `calls`
    /// identical calls: they must all be the same ordered list.
    fn fused_on_every_call(
        sources: &[Vec<(NodeId, f64)>],
        method: &FusionMethod,
        k: usize,
        calls: usize,
    ) -> Vec<(NodeId, f64)> {
        let first = fuse_results(sources, method, k);
        for call in 1..calls {
            assert_eq!(
                fuse_results(sources, method, k),
                first,
                "call {call} gave another ranking than the first"
            );
        }
        first
    }

    /// Two lists with disjoint nodes: the nodes at the same rank of each
    /// list have the same RRF score, so every rank is a tie of two. The
    /// fused list holds each tie with the lower id first, and a cut inside
    /// a tie keeps the lower id (#619: hash order decided it).
    #[test]
    fn rrf_ties_rank_by_ascending_node_id() {
        let text_ids = scrambled_ids(200, 2);
        let vector_ids = scrambled_ids(200, 1);
        let text: Vec<(NodeId, f64)> = text_ids
            .iter()
            .enumerate()
            .map(|(rank, id)| (*id, 1000.0 - rank as f64))
            .collect();
        let vector: Vec<(NodeId, f64)> = vector_ids
            .iter()
            .enumerate()
            .map(|(rank, id)| (*id, -(rank as f64)))
            .collect();
        let expected: Vec<NodeId> = text_ids
            .iter()
            .zip(&vector_ids)
            .flat_map(|(t, v)| [(*t).min(*v), (*t).max(*v)])
            .collect();

        let sources = [text, vector];
        let method = FusionMethod::Rrf { k: 60 };
        let all = fused_on_every_call(&sources, &method, 400, 10);
        let ids: Vec<NodeId> = all.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, expected);

        // 7 cuts the fourth tie in two: its lower id stays.
        let cut = fused_on_every_call(&sources, &method, 7, 10);
        let ids: Vec<NodeId> = cut.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, expected[..7]);
    }

    /// Every score of a source equal normalizes to 1.0, so two such
    /// sources of disjoint nodes give every node the same weighted score:
    /// the fused list is by ascending id, and a cut keeps the lowest ids.
    #[test]
    fn weighted_ties_rank_by_ascending_node_id() {
        let text: Vec<(NodeId, f64)> = scrambled_ids(200, 2)
            .into_iter()
            .map(|id| (id, 2.5))
            .collect();
        let vector: Vec<(NodeId, f64)> = scrambled_ids(200, 1)
            .into_iter()
            .map(|id| (id, -0.25))
            .collect();
        let sources = [text, vector];
        let method = FusionMethod::Weighted {
            weights: vec![0.5, 0.5],
        };

        let cut = fused_on_every_call(&sources, &method, 9, 10);
        let ids: Vec<u64> = cut.iter().map(|(id, _)| id.as_u64()).collect();
        assert_eq!(ids, (1..=9).collect::<Vec<u64>>());
        assert!(cut.iter().all(|(_, score)| (*score - 0.5).abs() < 1e-12));
    }

    /// A NaN source score makes the node's weighted score NaN: it ranks
    /// after every number, not wherever the sort leaves it.
    #[test]
    fn weighted_ranks_nan_scores_last() {
        let text = vec![
            (NodeId::new(1), f64::NAN),
            (NodeId::new(2), 1.0),
            (NodeId::new(3), 0.5),
        ];
        let vector = vec![(NodeId::new(4), -0.1)];
        let results = fuse_results(
            &[text, vector],
            &FusionMethod::Weighted {
                weights: vec![0.5, 0.5],
            },
            10,
        );
        let ids: Vec<u64> = results.iter().map(|(id, _)| id.as_u64()).collect();
        // Nodes 2 and 4 tie at 0.5, node 3 scores 0.0, node 1 NaN.
        assert_eq!(ids, vec![2, 4, 3, 1]);
        assert!(results[3].1.is_nan());
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
