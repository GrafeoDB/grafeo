//! A search among an allowlist answers with the nearest allowed nodes, for every vector index
//! kind (the full-precision graph and each quantization, trained or not, with and without
//! rescoring), every metric and every shape of allowlist, against a brute force over the allowed
//! vectors.
//!
//! The kinds and metrics come from matches, so a new quantization or metric fails to compile until
//! its rows are decided here.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use grafeo_common::types::NodeId;

use super::{
    DistanceMetric, HnswConfig, HnswIndex, QuantizationType, QuantizedHnswIndex, VectorAccessor,
    VectorIndexKind, compute_distance,
};

const DIMENSIONS: usize = 8;
/// The nodes with a vector of their own.
const ORIGINALS: usize = 240;
/// Exact copies of the first originals' vectors, after them: duplicates of a graph node.
const COPIES: usize = 3;
/// Every indexed node, and a beam that covers the whole graph.
const NODES: usize = ORIGINALS + COPIES;
const K: usize = 5;
/// The training threshold of a trained quantizer: it trains before the last vectors arrive.
const TRAINED_AT: usize = 100;
/// Ids that no index holds.
const ABSENT: [u64; 2] = [10_000, 10_001];

/// How a quantized index rescores its candidates.
#[derive(Clone, Copy, Debug)]
enum Rescore {
    /// No rescoring: the candidates as the quantized search ranks them.
    Off,
    /// The default factor.
    Default,
    /// A factor that takes every node as a candidate.
    Every,
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    /// [`HnswIndex`] at full precision.
    Graph,
    /// [`QuantizedHnswIndex`].
    Quantized {
        quantization: QuantizationType,
        trained: bool,
        rescore: Rescore,
    },
}

/// Every metric.
fn metrics() -> Vec<DistanceMetric> {
    let all = [
        DistanceMetric::Cosine,
        DistanceMetric::Euclidean,
        DistanceMetric::DotProduct,
        DistanceMetric::Manhattan,
    ];
    for metric in all {
        // A new metric is a compile error here until it is in `all`.
        match metric {
            DistanceMetric::Cosine
            | DistanceMetric::Euclidean
            | DistanceMetric::DotProduct
            | DistanceMetric::Manhattan => {}
        }
    }
    all.to_vec()
}

/// Every index kind: the graph, and each quantization with each rescoring, trained and untrained
/// where it trains.
fn kinds() -> Vec<Kind> {
    let quantizations = [
        QuantizationType::None,
        QuantizationType::Scalar,
        QuantizationType::Binary,
        QuantizationType::Product { num_subvectors: 4 },
    ];
    let mut kinds = vec![Kind::Graph];
    for quantization in quantizations {
        // A new quantization is a compile error here until it is in `quantizations`.
        let trains = match quantization {
            QuantizationType::None | QuantizationType::Binary => false,
            QuantizationType::Scalar | QuantizationType::Product { .. } => true,
        };
        for trained in [false, true] {
            if trained && !trains {
                continue;
            }
            for rescore in [Rescore::Off, Rescore::Default, Rescore::Every] {
                kinds.push(Kind::Quantized {
                    quantization,
                    trained,
                    rescore,
                });
            }
        }
    }
    kinds
}

/// What a search with a beam over the whole graph promises: whether it finds the nearest allowed
/// nodes, and whether it reports their distances by the metric.
fn exactness(kind: Kind) -> (bool, bool) {
    let Kind::Quantized {
        quantization,
        trained,
        rescore,
    } = kind
    else {
        return (true, true);
    };
    match (quantization, trained, rescore) {
        // Ranked by an approximate Euclidean distance over the bits, which it reports.
        (QuantizationType::Binary, _, Rescore::Off) => (false, false),
        // The product codes choose k of the candidates before the exact rescoring reorders them.
        (QuantizationType::Product { .. }, true, Rescore::Default | Rescore::Every) => {
            (false, true)
        }
        // The graph's exact candidates, rescored exactly where at all; an untrained quantizer
        // searches at full precision.
        (
            QuantizationType::None
            | QuantizationType::Scalar
            | QuantizationType::Binary
            | QuantizationType::Product { .. },
            _,
            _,
        ) => (true, true),
    }
}

fn node(n: usize) -> NodeId {
    NodeId::new(u64::try_from(n).unwrap())
}

/// A fixed spread of vectors: `ORIGINALS` from a linear congruential sequence in [-1, 1), then
/// `COPIES` that repeat the first originals.
fn points() -> Vec<(NodeId, Vec<f32>)> {
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        f32::from(u16::try_from(state >> 48).unwrap()) / 32_768.0 - 1.0
    };
    let mut points: Vec<(NodeId, Vec<f32>)> = (0..ORIGINALS)
        .map(|n| (node(n), (0..DIMENSIONS).map(|_| next()).collect()))
        .collect();
    for copy in 0..COPIES {
        let vector = points[copy].1.clone();
        points.push((node(ORIGINALS + copy), vector));
    }
    points
}

const QUERY: [f32; DIMENSIONS] = [0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5];

/// The allowlists, by name: the nodes farthest from the query make the shapes a search must walk
/// the whole graph for.
fn shapes(
    points: &[(NodeId, Vec<f32>)],
    query: &[f32],
    metric: DistanceMetric,
) -> Vec<(&'static str, HashSet<NodeId>)> {
    let mut farthest: Vec<(f32, NodeId)> = points
        .iter()
        .take(ORIGINALS)
        .map(|(id, vector)| (compute_distance(query, vector, metric), *id))
        .collect();
    farthest.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let first = |n: usize| -> HashSet<NodeId> { farthest.iter().take(n).map(|f| f.1).collect() };
    let mut with_absent = first(1);
    with_absent.extend(ABSENT.map(NodeId::new));
    vec![
        ("no node", HashSet::new()),
        ("the farthest node", first(1)),
        ("the farthest k", first(K)),
        ("the farthest 3k", first(3 * K)),
        (
            "every other node",
            points
                .iter()
                .map(|p| p.0)
                .filter(|id| id.as_u64() % 2 == 0)
                .collect(),
        ),
        ("every node", points.iter().map(|p| p.0).collect()),
        ("the copies only", (ORIGINALS..NODES).map(node).collect()),
        ("the farthest node and ids no index holds", with_absent),
        (
            "ids no index holds only",
            ABSENT.map(NodeId::new).into_iter().collect(),
        ),
    ]
}

fn build(kind: Kind, metric: DistanceMetric, points: &[(NodeId, Vec<f32>)]) -> VectorIndexKind {
    let config = HnswConfig::new(DIMENSIONS, metric);
    let index = match kind {
        Kind::Graph => VectorIndexKind::Hnsw(HnswIndex::new(config)),
        Kind::Quantized {
            quantization,
            trained,
            rescore,
        } => {
            let index = QuantizedHnswIndex::new(config, quantization)
                .with_training_threshold(if trained { TRAINED_AT } else { usize::MAX });
            VectorIndexKind::Quantized(match rescore {
                Rescore::Off => index.without_rescore(),
                Rescore::Default => index,
                Rescore::Every => index.with_rescore_factor(NODES),
            })
        }
    };
    let accessor = accessor(points);
    for (id, vector) in points {
        index.insert(*id, vector, &accessor);
    }
    index
}

fn accessor(points: &[(NodeId, Vec<f32>)]) -> impl Fn(NodeId) -> Option<Arc<[f32]>> + Send + Sync {
    let vectors: HashMap<NodeId, Arc<[f32]>> = points
        .iter()
        .map(|(id, vector)| (*id, Arc::from(vector.as_slice())))
        .collect();
    move |id| vectors.get(&id).cloned()
}

/// The brute force over the vectors of one metric.
struct Oracle<'a> {
    points: &'a [(NodeId, Vec<f32>)],
    metric: DistanceMetric,
}

impl Oracle<'_> {
    fn distance(&self, vector: &[f32]) -> f32 {
        compute_distance(&QUERY, vector, self.metric)
    }

    /// What a search among `allowed` returned that it should not have, in `wrong`: a node
    /// outside the allowlist or the index, a node twice, distances out of order, fewer or more
    /// results than `min(k, allowed nodes)`; and with `exact` (set, distances) the nearest allowed
    /// distances and each node's distance by the metric. Ties make node sets ambiguous, so the
    /// nearest are compared by their distances.
    fn check(
        &self,
        wrong: &mut Vec<String>,
        case: &str,
        found: &[(NodeId, f32)],
        allowed: &HashSet<NodeId>,
        exact: (bool, bool),
    ) {
        let points = self.points;
        let mut nearest: Vec<f32> = points
            .iter()
            .filter(|(id, _)| allowed.contains(id))
            .map(|(_, vector)| self.distance(vector))
            .collect();
        nearest.sort_by(f32::total_cmp);
        nearest.truncate(K);
        let ids: Vec<u64> = found.iter().map(|(id, _)| id.as_u64()).collect();
        if found.len() != nearest.len() {
            wrong.push(format!(
                "{case}: {} results {ids:?}, want {}",
                found.len(),
                nearest.len()
            ));
        }
        let outside: Vec<u64> = ids
            .iter()
            .copied()
            .filter(|id| !allowed.contains(&NodeId::new(*id)) || *id >= node(NODES).as_u64())
            .collect();
        if !outside.is_empty() {
            wrong.push(format!(
                "{case}: {outside:?} outside the allowlist or the index"
            ));
        }
        if ids.iter().collect::<HashSet<_>>().len() != ids.len() {
            wrong.push(format!("{case}: a node twice in {ids:?}"));
        }
        if found.windows(2).any(|pair| pair[0].1 > pair[1].1) {
            wrong.push(format!("{case}: distances out of order in {found:?}"));
        }
        let (exact_set, exact_distances) = exact;
        let close = |a: f32, b: f32| (a - b).abs() <= 1e-4 * a.abs().max(1.0);
        if exact_distances {
            let vectors: HashMap<NodeId, &Vec<f32>> =
                points.iter().map(|(id, v)| (*id, v)).collect();
            for (id, distance) in found {
                if let Some(vector) = vectors.get(id) {
                    let want = self.distance(vector);
                    if !close(*distance, want) {
                        wrong.push(format!("{case}: node {id:?} at {distance}, want {want}"));
                    }
                }
            }
        }
        if exact_set && found.len() == nearest.len() {
            let vectors: HashMap<NodeId, &Vec<f32>> =
                points.iter().map(|(id, v)| (*id, v)).collect();
            let mut got: Vec<f32> = found
                .iter()
                .filter_map(|(id, _)| vectors.get(id))
                .map(|vector| self.distance(vector))
                .collect();
            got.sort_by(f32::total_cmp);
            if got.len() != nearest.len() || got.iter().zip(&nearest).any(|(a, b)| !close(*a, *b)) {
                wrong.push(format!(
                    "{case}: the nearest allowed are at {nearest:?}, found {got:?} ({ids:?})"
                ));
            }
        }
    }
}

/// [`VectorIndexKind::search_visiting`], the search every caller of an index goes through, with no
/// work bound.
fn search(
    index: &VectorIndexKind,
    accessor: &impl VectorAccessor,
    ef: Option<usize>,
    allowed: Option<&HashSet<NodeId>>,
) -> Vec<(NodeId, f32)> {
    index
        .search_visiting(&QUERY, K, ef, allowed, accessor, &mut || Ok::<(), ()>(()))
        .unwrap()
}

/// Each kind, metric and allowlist with a beam over the whole graph (the exact rows find the
/// nearest allowed nodes) and with the index's own beam, which it scales for the allowlist, as a
/// database's vector search calls it. Both return `min(k, allowed nodes)` nodes, only allowed ones.
#[test]
fn an_allowlist_search_finds_the_nearest_allowed_nodes() {
    let points = points();
    let accessor = accessor(&points);
    let mut wrong = Vec::new();
    let mut cases = 0;
    for metric in metrics() {
        let shapes = shapes(&points, &QUERY, metric);
        let oracle = Oracle {
            points: &points,
            metric,
        };
        for kind in kinds() {
            let index = build(kind, metric, &points);
            let exact = exactness(kind);
            for (shape, allowed) in &shapes {
                let case = |entry: &str| format!("{kind:?} {metric:?} {shape} {entry}");
                let covering = search(&index, &accessor, Some(NODES), Some(allowed));
                oracle.check(
                    &mut wrong,
                    &case("with a beam over the graph"),
                    &covering,
                    allowed,
                    exact,
                );
                let own = search(&index, &accessor, None, Some(allowed));
                oracle.check(
                    &mut wrong,
                    &case("with its own beam"),
                    &own,
                    allowed,
                    (false, exact.1),
                );
                cases += 2;
            }
        }
    }
    assert!(cases > 1000, "{cases} cases");
    assert!(
        wrong.is_empty(),
        "{} of {cases} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// The search without an allowlist and with every node allowed answer alike, for every kind:
/// an allowlist only narrows.
#[test]
fn allowing_every_node_answers_as_no_allowlist() {
    let points = points();
    let accessor = accessor(&points);
    let every: HashSet<NodeId> = points.iter().map(|p| p.0).collect();
    let mut wrong = Vec::new();
    for metric in metrics() {
        for kind in kinds() {
            let index = build(kind, metric, &points);
            let ids = |found: Vec<(NodeId, f32)>| -> Vec<u64> {
                found.iter().map(|(id, _)| id.as_u64()).collect()
            };
            let plain = ids(search(&index, &accessor, Some(NODES), None));
            let allowed = ids(search(&index, &accessor, Some(NODES), Some(&every)));
            if plain != allowed {
                wrong.push(format!(
                    "{kind:?} {metric:?}: {plain:?} without an allowlist, {allowed:?} with every node"
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
