//! The row-group store against the conformance suite, with its known gaps.

use super::{KnownGap, conformance_suite};
use crate::graph::rowgroup::RowGroupStore;

/// Values and labels are written in place until H2b's update chains.
const IN_PLACE: &str = "values and labels are written in place until H2b (#412)";

/// The cases the row-group store fails, each with its reason.
const GAPS: &[KnownGap] = &[
    KnownGap {
        case: "an_open_transactions_value_and_label_writes_are_invisible_to_others",
        reason: IN_PLACE,
    },
    KnownGap {
        case: "a_reader_at_an_earlier_epoch_keeps_its_values_and_labels",
        reason: IN_PLACE,
    },
];

conformance_suite!(RowGroupStore, RowGroupStore::new, GAPS);

/// Adjacency past its merges: thousands of edges, some merged into the
/// sorted lists while their transaction was open and then undone, others
/// deleted. Every access path agrees, no undone edge is listed, and an
/// expand of one type lists exactly the node's edges of that type.
#[test]
fn adjacency_past_its_merges_agrees_with_the_edges() {
    use std::collections::BTreeSet;

    use grafeo_common::types::{EdgeId, NodeId, Value};

    use super::{Rng, Tx, committed, counts, counts_of};
    use crate::graph::Direction;
    use crate::graph::traits::GraphStore;

    const TYPES: [&str; 3] = ["KNOWS", "LIKES", "LIVES_IN"];
    let store = RowGroupStore::new();
    let mut rng = Rng::new(19);
    let mut tx = Tx::begin(&store, 10);
    let nodes: Vec<NodeId> = (0..64)
        .map(|_| {
            tx.create_node(&["Person"], &[("name", Value::from("Alix"))])
                .unwrap()
        })
        .collect();
    let mut created = Vec::new();
    for _ in 0..4_000 {
        let edge_type = TYPES[rng.below(TYPES.len())];
        let src = *rng.pick(&nodes).unwrap();
        let dst = *rng.pick(&nodes).unwrap();
        created.push(tx.create_edge(src, dst, edge_type, &[]).unwrap());
    }
    tx.commit().unwrap();

    let mut undone = Tx::begin(&store, 11);
    let mut rolled_back = BTreeSet::new();
    for _ in 0..1_500 {
        let src = *rng.pick(&nodes).unwrap();
        let dst = *rng.pick(&nodes).unwrap();
        rolled_back.insert(undone.create_edge(src, dst, "KNOWS", &[]).unwrap().as_u64());
    }
    undone.roll_back().unwrap();

    let mut deleting = Tx::begin(&store, 12);
    for edge in created.iter().step_by(13) {
        deleting.delete_edge(*edge).unwrap();
    }
    deleting.commit().unwrap();

    let now = committed(&store);
    now.consistent("after the merges").unwrap();
    assert_eq!(now.edges.len(), 4_000 - created.iter().step_by(13).count());
    assert_eq!(counts(&store, &["Person"]), counts_of(&now, &["Person"]));
    for node in &nodes {
        for direction in [Direction::Outgoing, Direction::Incoming] {
            let listed: Vec<(NodeId, EdgeId)> = store.edges_from(*node, direction);
            assert!(
                listed
                    .iter()
                    .all(|(_, edge)| !rolled_back.contains(&edge.as_u64())),
                "an undone edge is listed"
            );
            for edge_type in TYPES {
                let typed: BTreeSet<u64> = store
                    .edges_of_type(*node, direction, edge_type)
                    .into_iter()
                    .map(|(_, edge)| edge.as_u64())
                    .collect();
                let filtered: BTreeSet<u64> = listed
                    .iter()
                    .filter(|(_, edge)| store.edge_type(*edge).as_deref() == Some(edge_type))
                    .map(|(_, edge)| edge.as_u64())
                    .collect();
                assert_eq!(typed, filtered, "node {node:?}, {direction:?}, {edge_type}");
            }
        }
    }
}
