//! The order of scored search results.
//!
//! Text, vector and fused searches return `(NodeId, score)` lists and cut
//! them at `k`. Both orders here are total: equal scores rank by ascending
//! node id and NaN ranks after every number. Two identical searches then
//! return the same ordered list, and a cut inside a group of equal scores
//! keeps its lowest ids. Sorting by the score alone left ties in the order
//! the candidates came in (often a hash map's, which differs per call), and
//! `partial_cmp` with NaN as equal is no order at all: a sort may panic.

use grafeo_common::types::NodeId;
use std::cmp::Ordering;

/// Compares two scores, NaN after every number and equal to itself.
fn nan_last(a: f64, b: f64, numbers: impl FnOnce(f64, f64) -> Ordering) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => numbers(a, b),
    }
}

/// Orders results best first for scores where higher is better (BM25,
/// fused scores): by descending score, NaN last, ties by ascending node id.
#[cfg(feature = "text-index")]
pub(crate) fn by_descending_score(a: &(NodeId, f64), b: &(NodeId, f64)) -> Ordering {
    nan_last(a.1, b.1, |a, b| b.total_cmp(&a)).then(a.0.cmp(&b.0))
}

/// Orders results nearest first for distances: by ascending distance, NaN
/// last, ties by ascending node id.
pub(crate) fn by_ascending_distance<D: Copy + Into<f64>>(
    a: &(NodeId, D),
    b: &(NodeId, D),
) -> Ordering {
    nan_last(a.1.into(), b.1.into(), |a, b| a.total_cmp(&b)).then(a.0.cmp(&b.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: u64) -> NodeId {
        NodeId::new(id)
    }

    #[cfg(feature = "text-index")]
    #[test]
    fn descending_scores_rank_ties_by_id_and_nan_last() {
        let mut results = [
            (node(4), 1.0),
            (node(7), f64::NAN),
            (node(9), 2.0),
            (node(2), 1.0),
            (node(1), f64::NAN),
            (node(3), f64::NEG_INFINITY),
        ];
        results.sort_by(by_descending_score);
        let ids: Vec<u64> = results.iter().map(|(id, _)| id.as_u64()).collect();
        assert_eq!(ids, vec![9, 2, 4, 3, 1, 7]);
    }

    #[test]
    fn ascending_distances_rank_ties_by_id_and_nan_last() {
        // A negative NaN too: `total_cmp` alone would put it first.
        let mut results = [
            (node(5), 0.5_f32),
            (node(8), -f32::NAN),
            (node(6), 0.0),
            (node(2), 0.5),
            (node(1), f32::NAN),
        ];
        results.sort_by(by_ascending_distance);
        let ids: Vec<u64> = results.iter().map(|(id, _)| id.as_u64()).collect();
        assert_eq!(ids, vec![6, 2, 5, 1, 8]);
    }
}
