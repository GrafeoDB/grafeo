//! The row-group store against the conformance suite, with its known gaps.

use std::collections::BTreeMap;

use grafeo_common::types::{EpochId, NodeId, ReadView, Value};

use super::{
    KnownGap, LABELS, Reader, Rng, Tx, apply_all, committed, conformance_suite, counts, counts_of,
    image, people, random_write,
};
use crate::graph::Direction;
use crate::graph::rowgroup::RowGroupStore;
use crate::graph::traits::GraphStore;

/// The cases the row-group store fails, each with its reason: none since
/// H2b's update chains (#412).
const GAPS: &[KnownGap] = &[];

conformance_suite!(RowGroupStore, RowGroupStore::new, GAPS);

// ── Collect at the horizon, and replay, through every read path ──────

/// The collect and replay cases the row-group store fails, each with its
/// reason. They run through the suite's harness, so a gap that closes
/// fails its test too.
const COLLECT_GAPS: &[KnownGap] = &[];

/// Runs a collect or replay case of this module with [`COLLECT_GAPS`] on
/// stores from `$make`.
macro_rules! collect_cases {
    ($make:expr; $($case:ident),* $(,)?) => {
        $(
            #[test]
            fn $case() {
                $crate::graph::conformance::run::<RowGroupStore>(
                    stringify!($case),
                    $crate::graph::conformance::rowgroup::cases::$case,
                    $make,
                    $crate::graph::conformance::rowgroup::COLLECT_GAPS,
                );
            }
        )*
    };
}

collect_cases!(RowGroupStore::new;
    collect_keeps_every_read_path_at_the_horizon,
    collect_returns_the_version_info_to_its_flat_size,
    a_transactions_pending_writes_survive_collect,
    others_keep_the_committed_values_through_collect,
    replay_writes_no_version,
);

/// The suite and the collect cases again, on a store that keeps no incoming
/// edges (`backward_edges` off, 10b): every incoming read the cases check
/// (`edges_from` and `edges_from_at` incoming and both ways, the in-degree,
/// the refusal to delete a node an edge points to) is answered by a scan of
/// the edge table, at every view, with transactions open and after a
/// collect, and the store reports no backward adjacency.
mod without_backward_edges {
    use super::{GAPS, RowGroupStore};
    use crate::graph::conformance::conformance_suite;

    fn store() -> RowGroupStore {
        RowGroupStore::with_backward_edges(false)
    }

    conformance_suite!(RowGroupStore, store, GAPS);

    collect_cases!(store;
        collect_keeps_every_read_path_at_the_horizon,
        collect_returns_the_version_info_to_its_flat_size,
        a_transactions_pending_writes_survive_collect,
        others_keep_the_committed_values_through_collect,
        replay_writes_no_version,
    );
}

/// The suite and the collect cases again, on a store with property indexes
/// on the keys the reads at a view probe: every equality and maybe-equal
/// lookup at a view then goes through an index that holds old and pending
/// values, which it must recheck at the view (#412 S7).
mod indexed {
    use super::{GAPS, RowGroupStore};
    use crate::graph::conformance::conformance_suite;

    fn indexed_store() -> RowGroupStore {
        let store = RowGroupStore::new();
        for key in ["name", "age", "city"] {
            store.create_property_index(key);
        }
        store
    }

    conformance_suite!(RowGroupStore, indexed_store, GAPS);

    collect_cases!(indexed_store;
        collect_keeps_every_read_path_at_the_horizon,
        collect_returns_the_version_info_to_its_flat_size,
        a_transactions_pending_writes_survive_collect,
        others_keep_the_committed_values_through_collect,
        replay_writes_no_version,
    );
}

/// The cases of [`collect_cases`].
mod cases {
    use grafeo_common::change::DataOp;
    use grafeo_common::types::{ArcStr, PropertyKey, Value};

    use super::super::cases::{create_and_delete_writes, value_and_label_writes};
    use super::super::{
        Outcome, Reader, Tx, apply_all, committed, ensure, image, next_epoch, paths_at, people,
        same,
    };
    use crate::graph::apply::Writer;
    use crate::graph::rowgroup::RowGroupStore;

    /// After `collect(horizon)`, a reader at the horizon, a transaction that
    /// began there and a reader at a later commit see, through every read at
    /// their views, what they saw before the collect.
    pub(super) fn collect_keeps_every_read_path_at_the_horizon(
        make: &dyn Fn() -> RowGroupStore,
    ) -> Outcome {
        let store = make();
        let people = people(&store)?;
        let mut tx = Tx::begin(&store, 10);
        value_and_label_writes(&mut tx, &people)?;
        create_and_delete_writes(&mut tx, &people)?;
        let horizon = tx.commit()?;
        let reader = Tx::begin(&store, 11);
        let mut tx = Tx::begin(&store, 12);
        tx.set_node(people.mia, "city", Value::from("Prague"))?;
        tx.remove_label(people.gus, "Person")?;
        let later = tx.commit()?;

        let (at_horizon, at_later) = (
            image(&store, Reader::At(horizon)),
            image(&store, Reader::At(later)),
        );
        let gone = store.collect(horizon);
        ensure(gone > 0, || {
            "the commit at the horizon deleted rows".to_string()
        })?;
        paths_at(&store, Reader::At(horizon), &at_horizon, &[&at_later])?;
        paths_at(
            &store,
            Reader::Transaction(reader.id, reader.snapshot),
            &at_horizon,
            &[&at_later],
        )?;
        paths_at(&store, Reader::At(later), &at_later, &[&at_horizon])
    }

    /// Once every write is committed by the horizon and collected, the rows'
    /// version info is as small as for a store that was never written
    /// after its load, values and labels included.
    pub(super) fn collect_returns_the_version_info_to_its_flat_size(
        make: &dyn Fn() -> RowGroupStore,
    ) -> Outcome {
        let (written, untouched) = (make(), make());
        let people = people(&written)?;
        super::super::people(&untouched)?;
        let mut tx = Tx::begin(&written, 10);
        value_and_label_writes(&mut tx, &people)?;
        tx.commit()?;
        written.collect(written.current_epoch());
        untouched.collect(untouched.current_epoch());
        same(
            &written.version_bytes(),
            &untouched.version_bytes(),
            "the version bytes after the collect",
        )?;
        same(
            &written.stamped_vectors(),
            &0,
            "no vector keeps stamps after the collect",
        )
    }

    /// An open transaction's writes stay through a collect at the store's
    /// epoch: it still sees them through every read at its view.
    pub(super) fn a_transactions_pending_writes_survive_collect(
        make: &dyn Fn() -> RowGroupStore,
    ) -> Outcome {
        let store = make();
        let people = people(&store)?;
        let mut tx = Tx::begin(&store, 10);
        value_and_label_writes(&mut tx, &people)?;
        create_and_delete_writes(&mut tx, &people)?;
        let own = tx.view();
        store.collect(store.current_epoch());
        ensure(store.stamped_vectors() > 0, || {
            "the open transaction's vectors keep their stamps".to_string()
        })?;
        paths_at(
            &store,
            Reader::Transaction(tx.id, tx.snapshot),
            &own,
            &[&committed(&store)],
        )
    }

    /// While a transaction is open, a collect at the store's epoch leaves
    /// every other reader the committed values and labels (#412: the
    /// versions an open write keeps survive the collect).
    pub(super) fn others_keep_the_committed_values_through_collect(
        make: &dyn Fn() -> RowGroupStore,
    ) -> Outcome {
        let store = make();
        let people = people(&store)?;
        let before = committed(&store);
        let mut tx = Tx::begin(&store, 10);
        value_and_label_writes(&mut tx, &people)?;
        let own = tx.view();
        store.collect(store.current_epoch());
        paths_at(&store, Reader::At(store.current_epoch()), &before, &[&own])
    }

    /// Replay writes values and labels with no version: no reader runs
    /// during recovery, so the version info stays as small as before the
    /// replayed writes.
    pub(super) fn replay_writes_no_version(make: &dyn Fn() -> RowGroupStore) -> Outcome {
        let store = make();
        let people = people(&store)?;
        store.collect(store.current_epoch());
        let bytes = store.version_bytes();
        let ops = [
            DataOp::SetNodeProperty {
                id: people.alix,
                key: PropertyKey::new("city"),
                value: Value::from("Paris"),
            },
            DataOp::RemoveNodeProperty {
                id: people.gus,
                key: PropertyKey::new("age"),
            },
            DataOp::AddNodeLabel {
                id: people.mia,
                label: ArcStr::from("Traveller"),
            },
            DataOp::SetEdgeProperty {
                id: people.alix_gus,
                key: PropertyKey::new("since"),
                value: Value::Int64(2019),
            },
        ];
        for op in &ops {
            apply_all(
                &store,
                std::slice::from_ref(op),
                Writer::Replay {
                    epoch: next_epoch(&store),
                },
            )?;
        }
        same(
            &store.version_bytes(),
            &bytes,
            "the version bytes after the replayed writes",
        )?;
        same(
            &store.stamped_vectors(),
            &0,
            "no vector holds stamps after the replayed value and label writes",
        )
    }
}

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

/// Version info per vector as DuckDB keeps it: a vector whose rows one
/// transaction created holds one constant insert stamp, open and once
/// committed, also for a bulk range sharing a vector with a single create;
/// a second writer's row turns the inserts into a stamp per row; a vector
/// gets its delete stamps with its first delete, and holds none before.
#[test]
fn a_vector_filled_by_one_transaction_holds_one_constant_stamp() {
    use grafeo_common::change::{DataOp, Table};

    use crate::graph::apply::ChangeTarget;
    use crate::graph::rowgroup::VECTOR_ROWS;

    /// A stamp per row of a vector.
    const ARRAY: usize = VECTOR_ROWS * std::mem::size_of::<u64>();

    let store = RowGroupStore::new();
    let mut tx = Tx::begin(&store, 10);
    let nodes: Vec<NodeId> = (0..2 * VECTOR_ROWS)
        .map(|_| tx.create_node(&["Person"], &[]).unwrap())
        .collect();
    for pair in nodes.windows(2).take(VECTOR_ROWS) {
        tx.create_edge(pair[0], pair[1], "KNOWS", &[]).unwrap();
    }
    let open = store.version_bytes();
    assert!(open < 2_048, "three vectors while open: {open} bytes");
    tx.commit().unwrap();
    let committed_bytes = store.version_bytes();
    assert_eq!(
        committed_bytes, open,
        "committed per entry, still one constant stamp per vector"
    );

    // A bulk range, and a single create in the vector it ends in.
    let mut tx = Tx::begin(&store, 11);
    let ids = store.reserve_node_ids(3 * VECTOR_ROWS as u64 - 1).unwrap();
    for raw in ids.clone() {
        let op = DataOp::CreateNode {
            id: NodeId::new(raw),
            labels: super::labels(&["City"]),
            properties: super::properties(&[]),
        };
        store.apply(&op, tx.writer()).unwrap();
    }
    tx.create_node(&["City", "Traveller"], &[]).unwrap();
    tx.push_bulk(Table::Nodes, ids).unwrap();
    tx.commit().unwrap();
    let bulk = store.version_bytes() - committed_bytes;
    assert!(
        bulk < 2_048,
        "three more vectors, a constant each: {bulk} bytes"
    );
    assert_eq!(store.node_count(), 5 * VECTOR_ROWS);
    assert_eq!(store.nodes_by_label_count("City"), 3 * VECTOR_ROWS);
    assert_eq!(store.nodes_by_label_count("Traveller"), 1);
    assert_eq!(store.nodes_by_label_count("Person"), 2 * VECTOR_ROWS);
    assert_eq!(store.edge_count(), VECTOR_ROWS);

    // Jules opens a new vector; Mia, of another transaction, joins it.
    let mut tx = Tx::begin(&store, 12);
    let jules = tx.create_node(&["Person"], &[]).unwrap();
    tx.commit().unwrap();
    let before = store.version_bytes();
    let mut tx = Tx::begin(&store, 13);
    let mia = tx.create_node(&["Person"], &[]).unwrap();
    let array = store.version_bytes() - before;
    assert!(
        (ARRAY..2 * ARRAY).contains(&array),
        "a second writer's row turns the inserts into an array: {array} bytes"
    );
    let own = tx.view();
    assert!(own.nodes.contains_key(&jules.as_u64()) && own.nodes.contains_key(&mia.as_u64()));
    assert!(!committed(&store).nodes.contains_key(&mia.as_u64()));
    tx.commit().unwrap();

    let before = store.version_bytes();
    let mut tx = Tx::begin(&store, 14);
    tx.delete_node(jules).unwrap();
    let deletes = store.version_bytes() - before;
    assert!(
        (VECTOR_ROWS / 8..VECTOR_ROWS / 8 + 1_024).contains(&deletes),
        "the first delete allocates the vector's delete stamps, a bit per row \
         and a short list: {deletes} bytes"
    );
    let own = tx.view();
    assert!(!own.nodes.contains_key(&jules.as_u64()) && own.nodes.contains_key(&mia.as_u64()));
    assert!(
        committed(&store).nodes.contains_key(&jules.as_u64()),
        "others still see Jules"
    );
    tx.commit().unwrap();
    assert_eq!(store.collect(store.current_epoch()), 1);
    assert_eq!(store.stamped_vectors(), 0);
    assert_eq!(store.node_count(), 5 * VECTOR_ROWS + 1);
}

/// A transaction's first write of a value, or of a node's labels, creates
/// its version: an undo node keeps what the write replaced. Its next write
/// of the same value or labels replaces that version in place and keeps
/// nothing more; a write of a row it created keeps nothing (no one else sees
/// the row). A rollback takes every undo node out again. An immediate write
/// keeps what it replaced, for older readers, until a collect at its epoch;
/// replay keeps nothing.
#[test]
fn a_transactions_next_write_of_a_value_replaces_its_version() {
    use grafeo_common::change::{DataOp, PendingVersion};
    use grafeo_common::types::{ArcStr, PropertyKey};

    use crate::graph::apply::{Applied, ChangeTarget, Writer};

    let store = RowGroupStore::new();
    let people = people(&store).unwrap();
    store.collect(store.current_epoch());
    let flat = store.version_bytes();
    let set = |id: NodeId, key: &str, value: Value| DataOp::SetNodeProperty {
        id,
        key: PropertyKey::new(key),
        value,
    };
    let label = |id: NodeId, name: &str, add: bool| {
        let label = ArcStr::from(name);
        if add {
            DataOp::AddNodeLabel { id, label }
        } else {
            DataOp::RemoveNodeLabel { id, label }
        }
    };
    let version = |applied: Result<Applied, _>| match applied {
        Ok(Applied::Changed { version, .. }) => version,
        other => panic!("expected a change, got {other:?}"),
    };

    let mut tx = Tx::begin(&store, 10);
    let alix = people.alix;
    assert_eq!(
        version(tx.apply(set(alix, "age", Value::Int64(3)))),
        PendingVersion::Created
    );
    assert_eq!(
        version(tx.apply(set(alix, "age", Value::Int64(88)))),
        PendingVersion::Replaced
    );
    assert_eq!(
        version(tx.apply(set(alix, "city", Value::from("Paris")))),
        PendingVersion::Created,
        "another key, another version"
    );
    assert_eq!(
        version(tx.apply(label(alix, "Traveller", true))),
        PendingVersion::Created
    );
    assert_eq!(
        version(tx.apply(label(alix, "Person", false))),
        PendingVersion::Replaced
    );
    assert_eq!(store.update_nodes(), 3, "age, city and labels, a node each");
    let jules = tx.create_node(&["Person"], &[]).unwrap();
    assert_eq!(
        version(tx.apply(set(jules, "age", Value::Int64(19)))),
        PendingVersion::Created
    );
    assert_eq!(
        version(tx.apply(label(jules, "City", true))),
        PendingVersion::Created
    );
    assert_eq!(store.update_nodes(), 3, "nothing kept of Jules");
    tx.roll_back().unwrap();
    assert_eq!(store.update_nodes(), 0, "the rollback took every node out");
    store.collect(store.current_epoch());
    assert_eq!(store.version_bytes(), flat);

    let epoch = super::next_epoch(&store);
    store
        .apply(
            &set(people.gus, "age", Value::Int64(40)),
            Writer::Immediate {
                epoch,
                before_images: false,
            },
        )
        .unwrap();
    assert_eq!(
        store.node_property_at(
            people.gus,
            "age",
            ReadView::committed(EpochId::new(epoch.as_u64() - 1))
        ),
        Some(Value::Int64(3)),
        "an older reader keeps Gus's age"
    );
    assert_eq!(store.update_nodes(), 1);
    store.collect(epoch);
    assert_eq!(store.update_nodes(), 0);
    assert_eq!(store.version_bytes(), flat, "a row costs what it did");

    apply_all(
        &store,
        &[set(people.mia, "age", Value::Int64(19))],
        Writer::Replay {
            epoch: super::next_epoch(&store),
        },
    )
    .unwrap();
    assert_eq!(store.update_nodes(), 0, "replay keeps nothing");
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

/// The incoming reads of one store, per node: without a view and at each
/// of `views`, the edges listed incoming and both ways (as sets: the scan
/// lists them by edge id, the adjacency by type and other node), per edge
/// type, the neighbors, and the degrees.
fn incoming_reads(store: &RowGroupStore, nodes: &[NodeId], views: &[ReadView]) -> Vec<String> {
    use std::collections::BTreeSet;

    use grafeo_common::types::EdgeId;

    let set = |listed: Vec<(NodeId, EdgeId)>| -> BTreeSet<(u64, u64)> {
        listed
            .into_iter()
            .map(|(other, edge)| (other.as_u64(), edge.as_u64()))
            .collect()
    };
    let mut reads = Vec::new();
    for node in nodes {
        for direction in [Direction::Incoming, Direction::Both] {
            let mut neighbors: Vec<u64> = store
                .neighbors(*node, direction)
                .into_iter()
                .map(|id| id.as_u64())
                .collect();
            neighbors.sort_unstable();
            reads.push(format!(
                "{node:?} {direction:?}: {:?} neighbors {neighbors:?}",
                set(store.edges_from(*node, direction))
            ));
            for edge_type in ["KNOWS", "LIVES_IN", "NONE_SUCH"] {
                reads.push(format!(
                    "{node:?} {direction:?} {edge_type}: {:?}",
                    set(store.edges_of_type(*node, direction, edge_type))
                ));
            }
            for view in views {
                let mut neighbors: Vec<u64> = store
                    .neighbors_at(*node, direction, *view)
                    .into_iter()
                    .map(|id| id.as_u64())
                    .collect();
                neighbors.sort_unstable();
                reads.push(format!(
                    "{node:?} {direction:?} at {view:?}: {:?} neighbors {neighbors:?}",
                    set(store.edges_from_at(*node, direction, *view))
                ));
            }
        }
        reads.push(format!(
            "{node:?} degrees: in {} out {}",
            store.in_degree(*node),
            store.out_degree(*node)
        ));
    }
    reads
}

/// A store that keeps no incoming edges (`backward_edges` off, 10b) answers
/// every incoming read as one that keeps them: a seeded graph of 24 people
/// and 300 edges of two types (self-loops included), a commit that deletes
/// every seventh edge, a rolled-back transaction, and one still open that
/// creates edges and deletes others. Both stores give the same edges,
/// edges per type, neighbors and degrees without a view, at the epoch
/// before the deletes, at the latest commit and at the open transaction's
/// own view; again once the open one commits and both collect, which takes
/// the deleted edges out. Only the store that keeps them holds incoming
/// adjacency bytes.
#[test]
fn incoming_reads_agree_with_and_without_backward_edges() {
    use grafeo_common::types::EdgeId;

    const NAMES: [&str; 5] = ["Alix", "Gus", "Vincent", "Jules", "Mia"];
    const TYPES: [&str; 2] = ["KNOWS", "LIVES_IN"];

    /// The workload, alike on each store (same seed, same ids): the
    /// nodes, the epochs to read at, the transaction left open and the
    /// edges it deletes.
    fn workload(
        store: &RowGroupStore,
    ) -> (Vec<NodeId>, EpochId, EpochId, Tx<'_, RowGroupStore>, usize) {
        let mut rng = Rng::new(41);
        let mut tx = Tx::begin(store, 10);
        let nodes: Vec<NodeId> = (0..24)
            .map(|index| {
                tx.create_node(
                    &["Person"],
                    &[("name", Value::from(NAMES[index % NAMES.len()]))],
                )
                .unwrap()
            })
            .collect();
        let edges: Vec<EdgeId> = (0..300)
            .map(|_| {
                let (src, dst) = (*rng.pick(&nodes).unwrap(), *rng.pick(&nodes).unwrap());
                tx.create_edge(src, dst, TYPES[rng.below(TYPES.len())], &[])
                    .unwrap()
            })
            .collect();
        let created = tx.commit().unwrap();
        let mut tx = Tx::begin(store, 11);
        for edge in edges.iter().step_by(7) {
            tx.delete_edge(*edge).unwrap();
        }
        let deleted = tx.commit().unwrap();
        let mut undone = Tx::begin(store, 12);
        for _ in 0..40 {
            let (src, dst) = (*rng.pick(&nodes).unwrap(), *rng.pick(&nodes).unwrap());
            undone.create_edge(src, dst, "KNOWS", &[]).unwrap();
        }
        undone.roll_back().unwrap();
        let mut open = Tx::begin(store, 13);
        for _ in 0..20 {
            let (src, dst) = (*rng.pick(&nodes).unwrap(), *rng.pick(&nodes).unwrap());
            open.create_edge(src, dst, "LIVES_IN", &[]).unwrap();
        }
        let mut open_deletes = 0;
        for edge in edges.iter().skip(1).step_by(11) {
            if edges.iter().step_by(7).all(|gone| gone != edge) {
                open.delete_edge(*edge).unwrap();
                open_deletes += 1;
            }
        }
        (nodes, created, deleted, open, open_deletes)
    }

    let (kept, scanned) = (
        RowGroupStore::new(),
        RowGroupStore::with_backward_edges(false),
    );
    assert!(kept.has_backward_adjacency() && kept.keeps_backward_edges());
    assert!(!scanned.has_backward_adjacency() && !scanned.keeps_backward_edges());
    let (nodes, created, deleted, kept_open, open_deletes) = workload(&kept);
    let (_, _, _, scanned_open, _) = workload(&scanned);
    assert!(open_deletes > 0);
    let views = [
        ReadView::committed(created),
        ReadView::committed(deleted),
        ReadView::transaction(kept_open.id, kept_open.snapshot),
    ];
    let before = incoming_reads(&kept, &nodes, &views);
    assert_eq!(incoming_reads(&scanned, &nodes, &views), before);
    let listed: usize = nodes
        .iter()
        .map(|node| kept.edges_from(*node, Direction::Incoming).len())
        .sum();
    assert_eq!(
        listed,
        300 + 20,
        "every edge not collected, the open ones too"
    );
    let own: usize = nodes
        .iter()
        .map(|node| {
            kept.edges_from_at(*node, Direction::Incoming, views[2])
                .len()
        })
        .sum();
    assert_eq!(
        own,
        300 - 300_usize.div_ceil(7) + 20 - open_deletes,
        "the open transaction sees its creates and not its deletes"
    );

    let (kept_epoch, scanned_epoch) = (kept_open.commit().unwrap(), scanned_open.commit().unwrap());
    assert_eq!(kept_epoch, scanned_epoch);
    let gone = kept.collect(kept_epoch);
    assert_eq!(
        gone,
        300_usize.div_ceil(7) + open_deletes,
        "the deleted edges"
    );
    assert_eq!(scanned.collect(scanned_epoch), gone);
    let views = [ReadView::committed(kept_epoch)];
    assert_eq!(
        incoming_reads(&scanned, &nodes, &views),
        incoming_reads(&kept, &nodes, &views),
        "after the collect"
    );
    assert!(kept.memory().incoming_bytes > 0);
    assert_eq!(scanned.memory().incoming_bytes, 0, "no incoming adjacency");
    assert_eq!(
        scanned.memory().outgoing_bytes,
        kept.memory().outgoing_bytes
    );
}

/// A store without backward edges (10b) finds a node's incoming edges for
/// a delete: a raw delete of Mia, whom Gus's edge points to, is refused,
/// `delete_node_edges` (a detach delete's first step) deletes that edge,
/// and the delete then goes through; Alix keeps her edge to Gus.
#[test]
fn a_detach_delete_finds_the_incoming_edges_without_backward_edges() {
    use crate::graph::traits::GraphStoreMut;

    let store = RowGroupStore::with_backward_edges(false);
    let people = people(&store).unwrap();
    assert!(
        !GraphStoreMut::delete_node(&store, people.mia),
        "Gus's edge points to Mia"
    );
    assert_eq!(store.in_degree(people.mia), 1);
    store.delete_node_edges(people.mia);
    assert!(!store.is_edge_visible_at_epoch(people.gus_mia, store.current_epoch()));
    assert_eq!(store.in_degree(people.mia), 0);
    assert_eq!(store.out_degree(people.gus), 0);
    assert!(GraphStoreMut::delete_node(&store, people.mia));
    assert_eq!(
        store.edges_from(people.gus, Direction::Both).len(),
        2,
        "Alix's edge to Gus, and the deleted one to Mia until a collect"
    );
    assert_eq!(
        store.edges_from_at(
            people.gus,
            Direction::Both,
            ReadView::committed(store.current_epoch())
        ),
        vec![(people.alix, people.alix_gus)]
    );
}
