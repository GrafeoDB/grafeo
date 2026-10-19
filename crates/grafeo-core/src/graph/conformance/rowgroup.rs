//! The row-group store against the conformance suite, with its known gaps.

use std::collections::BTreeMap;

use grafeo_common::types::{EpochId, NodeId, Value};

use super::{
    KnownGap, LABELS, Reader, Rng, Tx, committed, conformance_suite, counts, counts_of, image,
    people, random_write,
};
use crate::graph::Direction;
use crate::graph::rowgroup::RowGroupStore;
use crate::graph::traits::GraphStore;

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

/// The data an image holds and whether its access paths agree.
fn data(
    image: &super::Image,
) -> (
    &BTreeMap<u64, super::NodeView>,
    &BTreeMap<u64, super::EdgeView>,
) {
    image.consistent("the image").unwrap();
    (&image.nodes, &image.edges)
}

/// A seeded workload of commits, rollbacks and savepoint rollbacks, run on
/// two stores alike, one of which collects at its epoch after every
/// transaction: both hold the same committed image and counts after each,
/// and the collected one keeps no deleted row.
#[test]
fn collect_between_transactions_keeps_the_committed_image() {
    let mut gone = 0;
    for seed in 1..=12 {
        let (plain, collected) = (RowGroupStore::new(), RowGroupStore::new());
        let (mut rng_plain, mut rng_collected) = (Rng::new(seed), Rng::new(seed));
        for transaction in 0..12_u64 {
            let id = 100 + transaction;
            let (mut a, mut b) = (Tx::begin(&plain, id), Tx::begin(&collected, id));
            // The two generators draw alike while the two stores agree.
            let mut draw = |bound: usize| {
                let drawn = rng_plain.below(bound);
                assert_eq!(rng_collected.below(bound), drawn);
                drawn
            };
            let savepoint = draw(2) == 0;
            let roll_back = draw(3) == 0;
            for step in 0..12 {
                if step == 6 && savepoint {
                    let (mark_a, mark_b) = (a.mark(), b.mark());
                    random_write(&mut a, &mut rng_plain).unwrap();
                    random_write(&mut b, &mut rng_collected).unwrap();
                    a.roll_back_to(mark_a).unwrap();
                    b.roll_back_to(mark_b).unwrap();
                }
                random_write(&mut a, &mut rng_plain).unwrap();
                random_write(&mut b, &mut rng_collected).unwrap();
            }
            if roll_back {
                a.roll_back().unwrap();
                b.roll_back().unwrap();
            } else {
                a.commit().unwrap();
                b.commit().unwrap();
            }
            gone += collected.collect(collected.current_epoch());
            let what = format!("seed {seed}, after transaction {transaction}");
            let (expected, found) = (committed(&plain), committed(&collected));
            assert_eq!(data(&found), data(&expected), "{what}");
            assert_eq!(
                counts(&collected, &LABELS),
                counts(&plain, &LABELS),
                "{what}: the counts"
            );
            assert_eq!(
                collected.stamped_vectors(),
                0,
                "{what}: every vector collapsed"
            );
            let listed: Vec<u64> = collected
                .all_node_ids()
                .into_iter()
                .map(|id| id.as_u64())
                .collect();
            let visible: Vec<u64> = found.nodes.keys().copied().collect();
            assert_eq!(listed, visible, "{what}: no deleted node is left");
        }
    }
    assert!(gone > 0, "the workloads delete rows that collect takes out");
}

/// While a transaction is open, `collect` leaves its vectors stamped: it
/// still sees its own writes and no one else does. Once it commits, the
/// rows it deleted go with their adjacency entries.
#[test]
fn a_pending_transactions_versions_stay_through_collect() {
    let store = RowGroupStore::new();
    let people = people(&store).unwrap();
    assert!(store.stamped_vectors() > 0, "the replay stamped its rows");
    assert_eq!(store.collect(store.current_epoch()), 0);
    assert_eq!(
        store.stamped_vectors(),
        0,
        "nothing pending, nothing deleted"
    );
    let before = committed(&store);

    let mut tx = Tx::begin(&store, 10);
    let jules = tx
        .create_node(&["Person"], &[("name", Value::from("Jules"))])
        .unwrap();
    tx.delete_edge(people.gus_mia).unwrap();
    tx.delete_node(people.mia).unwrap();
    assert_eq!(
        store.collect(store.current_epoch()),
        0,
        "nothing committed is deleted"
    );
    assert_eq!(store.stamped_vectors(), 2, "the node and the edge vector");
    let own = tx.view();
    assert!(own.nodes.contains_key(&jules.as_u64()));
    assert!(!own.nodes.contains_key(&people.mia.as_u64()));
    assert!(!own.edges.contains_key(&people.gus_mia.as_u64()));
    assert_eq!(
        data(&committed(&store)),
        data(&before),
        "others see no change"
    );

    let epoch = tx.commit().unwrap();
    let listed = |store: &RowGroupStore| -> Vec<NodeId> { store.all_node_ids() };
    assert!(
        listed(&store).contains(&people.mia),
        "before the collect a deleted row stays for earlier readers"
    );
    assert_eq!(store.collect(epoch), 2, "Mia and her edge");
    assert_eq!(store.stamped_vectors(), 0);
    assert_eq!(listed(&store), [people.alix, people.gus, jules]);
    assert_eq!(
        store.edges_from(people.gus, Direction::Outgoing),
        Vec::new(),
        "the deleted edge left its source's adjacency"
    );
    assert_eq!(
        store.edges_from(people.gus, Direction::Incoming),
        [(people.alix, people.alix_gus)]
    );
    assert!(!store.is_edge_visible_at_epoch(people.gus_mia, EpochId::new(0)));
    let now = committed(&store);
    now.consistent("after the collect").unwrap();
    assert_eq!(
        now.nodes.keys().copied().collect::<Vec<_>>(),
        [people.alix, people.gus, jules].map(|id| id.as_u64())
    );
    assert_eq!(counts(&store, &["Person"]), counts_of(&now, &["Person"]));
}

/// After `collect(horizon)`, readers at the horizon and later, a
/// transaction that began there included, see what they saw before it; a
/// vector with a stamp after the horizon keeps its stamps; the rows deleted
/// by the horizon leave `all_node_ids`.
#[test]
fn a_reader_at_the_horizon_sees_the_same_after_collect() {
    use crate::graph::rowgroup::VECTOR_ROWS;

    let store = RowGroupStore::new();
    let mut tx = Tx::begin(&store, 10);
    let people: Vec<NodeId> = (0..VECTOR_ROWS)
        .map(|_| {
            tx.create_node(&["Person"], &[("name", Value::from("Vincent"))])
                .unwrap()
        })
        .collect();
    let first = tx.commit().unwrap();
    let mut tx = Tx::begin(&store, 11);
    for node in &people[..10] {
        tx.delete_node(*node).unwrap();
    }
    let horizon = tx.commit().unwrap();
    let reader = Tx::begin(&store, 12);

    let mut tx = Tx::begin(&store, 13);
    let jules = tx
        .create_node(&["Person"], &[("name", Value::from("Jules"))])
        .unwrap();
    let knows = tx.create_edge(jules, people[100], "KNOWS", &[]).unwrap();
    let created = tx.commit().unwrap();
    let mut tx = Tx::begin(&store, 14);
    tx.delete_edge(knows).unwrap();
    let deleted = tx.commit().unwrap();

    let epochs = [horizon, created, deleted];
    let images = |store: &RowGroupStore| -> Vec<super::Image> {
        epochs
            .iter()
            .map(|epoch| image(store, Reader::At(*epoch)))
            .collect()
    };
    let (before, view) = (images(&store), reader.view());
    assert_eq!(
        store.collect(horizon),
        10,
        "the people deleted by the horizon"
    );
    for (index, (found, expected)) in images(&store).iter().zip(&before).enumerate() {
        assert_eq!(data(found), data(expected), "at epoch {:?}", epochs[index]);
    }
    assert_eq!(
        data(&reader.view()),
        data(&view),
        "the transaction at the horizon"
    );
    assert!(
        image(&store, Reader::At(created))
            .edges
            .contains_key(&knows.as_u64()),
        "the edge deleted after the horizon stays for a reader before its delete"
    );
    assert_eq!(
        store.stamped_vectors(),
        2,
        "the vectors written after the horizon"
    );
    let listed = store.all_node_ids();
    assert_eq!(listed.len(), VECTOR_ROWS - 10 + 1);
    assert!(people[..10].iter().all(|node| !listed.contains(node)));
    assert!(
        !image(&store, Reader::At(first))
            .nodes
            .contains_key(&people[0].as_u64()),
        "a reader older than the horizon is not served: its deleted rows are gone"
    );

    drop(reader);
    assert_eq!(store.collect(deleted), 1, "the edge");
    assert_eq!(store.stamped_vectors(), 0);
    assert_eq!(data(&committed(&store)), data(&before[2]));
    assert_eq!(store.edges_from(jules, Direction::Outgoing), Vec::new());
    assert_eq!(
        store.edges_from(people[100], Direction::Incoming),
        Vec::new()
    );
}

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
